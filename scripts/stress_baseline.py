#!/usr/bin/env python3
"""
Comprehensive baseline stress test for oxydllm.

For each local model that fits in the machine's memory budget, measures:
    cold_load_ms        First request after fresh server start (includes
                        weight load + kernel compile + tokenizer init).
    warm_load_ms        Second request — pure load is over, only Metal
                        command-buffer warm-up remains.
    ttft_ms             Time-to-first-token on a 256-word prefill.
    decode_tps          Steady-state generation throughput, median of 5
                        runs with 150-token outputs.
    rss_mb              Resident memory after the model is fully warm.
    coherence_pass      Boolean — output of "Tokyo is the capital of"
                        contains "Japan" (case-insensitive).

Outputs a live table to stdout, plus:
    test-results/stress-baseline/<timestamp>/results.csv
    test-results/stress-baseline/<timestamp>/results.json
    test-results/stress-baseline/<timestamp>/server-*.log

Run:
    ./scripts/stress_baseline.py                  # default suite
    ./scripts/stress_baseline.py --models MODELS  # comma-separated override
    ./scripts/stress_baseline.py --quick          # fewer runs / shorter outputs
    ./scripts/stress_baseline.py --include-slow   # include Phi-3.5 (~8 tok/s)

Compare two runs with `diff` on the CSVs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib import error as urlerror
from urllib import request as urlrequest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = REPO_ROOT / "target" / "release" / "oxydllm"
DEFAULT_PORT = 11335
DEFAULT_RESULTS = REPO_ROOT / "test-results" / "stress-baseline"

# Model lists — `core` runs by default. `slow` (Phi-3.5) is opt-in.
CORE_MODELS = [
    # Small dense bf16 — sanity baseline.
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    # GGUF kernel suite spread across quant types.
    "Qwen/qwen2-1_5b-instruct-q4_0",
    "Qwen/Qwen2.5-1.5B-Instruct-Q2_K",
    "Qwen/Qwen2.5-1.5B-Instruct-Q3_K_M",
    "Qwen/Qwen2.5-1.5B-Instruct-Q4_0",
    "bartowski/Qwen2.5-1.5B-Instruct-Q4_0",
    "Qwen/Qwen2.5-1.5B-Instruct-Q4_K_M",
    "Qwen/Qwen3-1.7B-Q8_0",
    "Qwen/Qwen3-4B-Q4_K_M",
    "Qwen/Qwen3-4B-Q5_0",
    "Qwen/Qwen3-4B-Q5_K_M",
    "Qwen/Qwen3-4B-Q6_K",
    # Packed-int resident paths.
    "Qwen/Qwen3-4B-AWQ",
    "Qwen/Qwen3-0.6B-GPTQ-Int8",
    "Qwen/Qwen3-1.7B-GPTQ-Int8",
    # Gemma family — exercises different chat template + softcaps.
    "google/gemma-2b-it",
    "google/gemma-2-2b-it",
    "google/gemma-3-1b-it",
    "google/gemma-4-E2B-it",
    # Mistral family.
    "mistralai/Ministral-3-3B-Instruct-25",
    # MoE.
    "allenai/OLMoE-1B-7B-0924-Instruct",
    # FP8.
    "Qwen/Qwen3-4B-Instruct-2507-FP8",
]

SLOW_MODELS = [
    "microsoft/Phi-3.5-mini-instruct",  # ~8 tok/s on Metal (no Q4_K fast path for Phi-3)
]

WARMUP_PROMPT = "Reply with the single word ready."
COHERENCE_PROMPT = "Tokyo is the capital of"
COHERENCE_EXPECTED = "japan"
DECODE_PROMPTS = [
    "Write a 100-word essay on the topic of rivers.",
    "Write a 100-word essay on the topic of mountains.",
    "Write a 100-word essay on the topic of cities.",
    "Write a 100-word essay on the topic of forests.",
    "Write a 100-word essay on the topic of deserts.",
]
# Long prefill for TTFT — ~256 words ≈ 200-300 tokens per BPE.
TTFT_PROMPT_WORDS = 256
TTFT_PROMPT = " ".join(
    ["The quick brown fox jumps over the lazy dog near the riverside."]
    * (TTFT_PROMPT_WORDS // 12 + 1)
).split()
TTFT_PROMPT = " ".join(TTFT_PROMPT[:TTFT_PROMPT_WORDS])


@dataclass
class ModelResult:
    model: str
    cold_load_ms: Optional[float] = None
    warm_load_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    decode_tps: Optional[float] = None
    decode_tps_runs: list[float] = field(default_factory=list)
    rss_mb: Optional[float] = None
    coherence_pass: Optional[bool] = None
    coherence_output: str = ""
    error: Optional[str] = None


def color(s: str, c: str) -> str:
    if not sys.stdout.isatty():
        return s
    palette = {
        "green": "\033[0;32m",
        "yellow": "\033[0;33m",
        "red": "\033[0;31m",
        "cyan": "\033[0;36m",
        "bold": "\033[1m",
        "reset": "\033[0m",
        "dim": "\033[2m",
    }
    return f"{palette[c]}{s}{palette['reset']}"


def find_free_port(start: int) -> int:
    for p in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"no free port found in [{start}, {start + 200})")


def start_server(bin_path: Path, port: int, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "wb")
    env = os.environ.copy()
    env.setdefault("RUST_LOG", "info")
    proc = subprocess.Popen(
        [str(bin_path), "start", "--port", str(port)],
        stdout=fh,
        stderr=fh,
        env=env,
        start_new_session=True,
    )
    return proc


def wait_ready(port: int, timeout: float = 30.0) -> bool:
    health = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlrequest.urlopen(health, timeout=1.0):
                return True
        except (urlerror.URLError, ConnectionResetError, TimeoutError):
            time.sleep(0.2)
    return False


def chat(
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    timeout: float = 180.0,
) -> Optional[dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urlrequest.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
        return json.loads(body)
    except (urlerror.URLError, TimeoutError) as e:
        print(f"  {color('chat error: ' + str(e), 'red')}")
        return None
    except json.JSONDecodeError as e:
        print(f"  {color('chat decode error: ' + str(e), 'red')}")
        return None


def chat_stream_ttft(
    port: int, model: str, prompt: str, *, timeout: float = 180.0
) -> Optional[float]:
    """Return wall-clock ms to the first streamed delta with content."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
    }
    req = urlrequest.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:") :].strip()
                if body == "[DONE]":
                    break
                try:
                    ev = json.loads(body)
                except json.JSONDecodeError:
                    continue
                delta = ev.get("choices", [{}])[0].get("delta", {})
                if delta.get("content") or "tool_calls" in delta:
                    return (time.monotonic() - start) * 1000.0
    except (urlerror.URLError, TimeoutError) as e:
        print(f"  {color('ttft stream error: ' + str(e), 'red')}")
        return None
    return None


def server_rss_mb(pid: int) -> Optional[float]:
    """Resident memory of the server process in MB (macOS `ps -o rss=` is in KB)."""
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True
        ).strip()
        if not out:
            return None
        return int(out) / 1024.0
    except subprocess.CalledProcessError:
        return None


def parse_tps_from_log(log_path: Path, model: str) -> list[float]:
    """Scan server log for `tokens_per_second=X` lines for `model`."""
    if not log_path.exists():
        return []
    txt = log_path.read_text(errors="ignore")
    rx = re.compile(
        rf"model_id={re.escape(model)}.*?tokens_per_second=([\d.]+)", re.DOTALL
    )
    return [float(m) for m in rx.findall(txt)]


def measure_model(
    bin_path: Path,
    model: str,
    port: int,
    log_path: Path,
    *,
    decode_runs: int,
    decode_max_tokens: int,
) -> ModelResult:
    res = ModelResult(model=model)

    proc = start_server(bin_path, port, log_path)
    try:
        if not wait_ready(port, timeout=20.0):
            res.error = "server did not become ready"
            return res

        # Cold load — first request, includes weight load + kernel compile.
        t0 = time.monotonic()
        warmup = chat(port, model, WARMUP_PROMPT, max_tokens=4, timeout=300.0)
        cold_elapsed = (time.monotonic() - t0) * 1000.0
        if warmup is None or "choices" not in warmup:
            res.error = "warmup request failed"
            return res
        res.cold_load_ms = cold_elapsed

        # Warm load — second request, model is now loaded.
        t1 = time.monotonic()
        warm = chat(port, model, WARMUP_PROMPT, max_tokens=4, timeout=60.0)
        res.warm_load_ms = (time.monotonic() - t1) * 1000.0
        if warm is None:
            res.error = "warm request failed"
            return res

        # RSS snapshot.
        res.rss_mb = server_rss_mb(proc.pid)

        # TTFT — long prefill, max_tokens=1, stream to capture first delta.
        ttft = chat_stream_ttft(port, model, TTFT_PROMPT, timeout=120.0)
        res.ttft_ms = ttft

        # Decode TPS — run N times, take median (drops cold + tail outliers).
        tps_samples: list[float] = []
        for i in range(decode_runs):
            prompt = DECODE_PROMPTS[i % len(DECODE_PROMPTS)]
            t = time.monotonic()
            r = chat(port, model, prompt, max_tokens=decode_max_tokens, timeout=120.0)
            elapsed = time.monotonic() - t
            if r is None:
                continue
            usage = r.get("usage", {})
            n = usage.get("completion_tokens", 0)
            if n > 0 and elapsed > 0:
                tps_samples.append(n / elapsed)
        # Prefer the server's own measurement (excludes network) if available.
        server_tps = parse_tps_from_log(log_path, model)
        # Drop the cold-load entry — it includes load overhead.
        server_tps_steady = server_tps[1:] if len(server_tps) > 1 else server_tps
        if server_tps_steady:
            res.decode_tps_runs = server_tps_steady
            res.decode_tps = statistics.median(server_tps_steady)
        elif tps_samples:
            res.decode_tps_runs = tps_samples
            res.decode_tps = statistics.median(tps_samples)

        # Coherence: greedy decoding is usually deterministic, but some kernels
        # use non-associative reductions that can flip top-1 on boundary prompts.
        # Try up to 3 times so a single flake doesn't fail an otherwise-working model.
        for attempt in range(3):
            coh = chat(port, model, COHERENCE_PROMPT, max_tokens=96, timeout=60.0)
            if coh is None or "choices" not in coh:
                continue
            content = coh["choices"][0]["message"].get("content") or ""
            if attempt == 0:
                res.coherence_output = content[:120].replace("\n", " ")
            if COHERENCE_EXPECTED in content.lower():
                res.coherence_output = content[:120].replace("\n", " ")
                res.coherence_pass = True
                break
            res.coherence_pass = False
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=10)
    return res


def format_table(results: list[ModelResult]) -> str:
    rows = []
    header = (
        "MODEL",
        "COLD_LD_S",
        "WARM_LD_MS",
        "TTFT_MS",
        "TPS",
        "RSS_MB",
        "COH",
    )
    rows.append(header)
    for r in results:
        cold = f"{r.cold_load_ms / 1000:.2f}" if r.cold_load_ms is not None else "—"
        warm = f"{r.warm_load_ms:.0f}" if r.warm_load_ms is not None else "—"
        ttft = f"{r.ttft_ms:.0f}" if r.ttft_ms is not None else "—"
        tps = f"{r.decode_tps:.1f}" if r.decode_tps is not None else "—"
        rss = f"{r.rss_mb:.0f}" if r.rss_mb is not None else "—"
        coh = "✓" if r.coherence_pass else ("✗" if r.coherence_pass is False else "—")
        if r.error:
            coh = "ERR"
        rows.append((r.model, cold, warm, ttft, tps, rss, coh))
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(header))]
    lines = []
    for idx, r in enumerate(rows):
        line = "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
        if idx == 0:
            lines.append(color(line, "bold"))
            lines.append(color("─" * len(line), "dim"))
        else:
            lines.append(line)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=str(DEFAULT_BIN), help=f"oxydllm binary (default: {DEFAULT_BIN})")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"listen port (default: {DEFAULT_PORT})")
    ap.add_argument("--models", help="comma-separated model override (otherwise: default suite)")
    ap.add_argument("--quick", action="store_true", help="3 decode runs × 60 tokens (vs 5 × 150)")
    ap.add_argument("--include-slow", action="store_true", help="include the SLOW_MODELS list too")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    args = ap.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.is_file() or not os.access(bin_path, os.X_OK):
        print(f"{color('error', 'red')}: {bin_path} is not an executable file")
        print(f"build with: cargo build --release")
        return 1

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = list(CORE_MODELS)
        if args.include_slow:
            models += SLOW_MODELS

    decode_runs = 3 if args.quick else 5
    decode_max_tokens = 60 if args.quick else 150

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.results_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    port = args.port if args.port == DEFAULT_PORT else args.port
    # If default port is busy from a previous run, pick a free one.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            port = find_free_port(port + 1)

    print(color(f"oxydllm stress baseline — {len(models)} models — port {port}", "cyan"))
    print(color(f"Results: {out_dir}", "dim"))
    print(color(f"Decode: {decode_runs} runs × {decode_max_tokens} tokens", "dim"))
    print()

    results: list[ModelResult] = []
    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {color(model, 'cyan')}")
        log_path = out_dir / f"server-{model.replace('/', '_')}.log"
        try:
            r = measure_model(
                bin_path,
                model,
                port,
                log_path,
                decode_runs=decode_runs,
                decode_max_tokens=decode_max_tokens,
            )
        except KeyboardInterrupt:
            print(color("\ninterrupted by user", "yellow"))
            break
        except Exception as e:  # noqa: BLE001
            r = ModelResult(model=model, error=f"exception: {e}")
        results.append(r)
        if r.error:
            print(f"  {color('FAIL', 'red')} — {r.error}")
        else:
            summary = (
                f"cold={r.cold_load_ms / 1000:.2f}s "
                f"warm={r.warm_load_ms:.0f}ms "
                f"ttft={r.ttft_ms:.0f}ms "
                f"tps={r.decode_tps:.1f} "
                f"rss={r.rss_mb:.0f}MB "
                f"coh={'OK' if r.coherence_pass else 'FAIL'}"
            )
            print(f"  {summary}")
        # Brief pause so the OS reclaims pages before the next run.
        time.sleep(2)

    print()
    print(color("━" * 60, "bold"))
    print(format_table(results))
    print()

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w") as fh:
        fh.write("model,cold_load_ms,warm_load_ms,ttft_ms,decode_tps,rss_mb,coherence_pass,error\n")
        for r in results:
            fh.write(
                ",".join(
                    [
                        r.model,
                        f"{r.cold_load_ms:.1f}" if r.cold_load_ms is not None else "",
                        f"{r.warm_load_ms:.1f}" if r.warm_load_ms is not None else "",
                        f"{r.ttft_ms:.1f}" if r.ttft_ms is not None else "",
                        f"{r.decode_tps:.2f}" if r.decode_tps is not None else "",
                        f"{r.rss_mb:.1f}" if r.rss_mb is not None else "",
                        "1" if r.coherence_pass else ("0" if r.coherence_pass is False else ""),
                        r.error or "",
                    ]
                )
                + "\n"
            )
    json_path = out_dir / "results.json"
    with open(json_path, "w") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)

    n_ok = sum(1 for r in results if r.error is None)
    n_coh = sum(1 for r in results if r.coherence_pass)
    print(
        color(
            f"Done. {n_ok}/{len(results)} models loaded, {n_coh}/{len(results)} coherence-pass.",
            "green" if n_coh == len(results) else "yellow",
        )
    )
    print(color(f"CSV : {csv_path}", "dim"))
    print(color(f"JSON: {json_path}", "dim"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
