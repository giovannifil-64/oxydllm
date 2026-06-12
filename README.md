<p align="center">
  <img width="350" src=".github/res/oxydLLM_light.png#gh-light-mode-only">
  <img width="350" src=".github/res/oxydLLM_dark.png#gh-dark-mode-only">
</p>

<br>

<p align="center">
    <a href="https://github.com/giovannifil-64/oxydllm/actions/workflows/ci.yml">
        <img src="https://github.com/giovannifil-64/oxydllm/actions/workflows/ci.yml/badge.svg?branch=main" />
    </a>
    <a href="https://github.com/giovannifil-64/oxydllm/actions/workflows/nightly.yml">
        <img src="https://github.com/giovannifil-64/oxydllm/actions/workflows/nightly.yml/badge.svg?branch=main" />
    </a>
    <a href="https://github.com/giovannifil-64/oxydllm/releases">
        <img src="https://img.shields.io/github/v/release/giovannifil-64/oxydllm?include_prereleases&sort=semver&label=release&logo=github" />
    </a>
</p>

<p align="center">
    <img src="https://img.shields.io/github/repo-size/giovannifil-64/oxydllm" />
</p>

<br>

A rust-based inference engine for Large Language Models.

> [!NOTE]
> For transparency, the engine has been developed with the assistant of Claude. The code has been reviewed and edited, but may still contain inaccuracies, suboptimal implementations, or other kind of issues not yet identified.
> 
> The engine has been tested primarily on Apple Silicon, so Metal support is more mature than CUDA. Contributions to improve NVIDIA GPU support are welcome.
> 
> GGUF support is available, but compatibility still depends on architecture and quantization variant.
>
> At the moment it only supports text input/output and a limited set of models.

## Features
- OpenAI-compatible chat completions endpoint (`/v1/chat/completions`)
- Function calling / tool use (function tools): `tools`, `tool_choice` (`auto`, `required`, `none`, forced function, and `allowed_tools`), and `parallel_tool_calls`; assistant tool calls are returned as proper `tool_calls` with `finish_reason: "tool_calls"`, and streaming emits incremental tool-call deltas
- Structured output: `response_format` with `json_object` and `json_schema`; request-time schema validation includes strict-mode requirements plus recursive `$ref` / `$defs`, `anyOf`, enums, nested objects/arrays, and nullable unions
- Metal acceleration on Apple Silicon with fused attention, RMSNorm, RoPE, and Softmax kernels
- Paged KV cache with prefix caching for reduced redundant computation
- Batched scheduler, where all active sequences share each GPU forward pass so throughput scales with concurrent users rather than collapsing to serial execution. At startup the scheduler computes `max_num_seqs` automatically as `total_kv_blocks / ceil(context_len / block_size)`, capped to 256; the value is logged and can be overridden with `--max-num-seqs`. Incoming requests are held in a bounded `tokio::sync::mpsc` channel of capacity `--max-queued-requests` (default 200); once full, new arrivals receive HTTP 429 immediately, bounding memory consumption under sustained load.
- KV cache quantization (Lossless/Balanced/Aggressive) to reduce memory footprint by 2-4x
- Multi-model server: load several models simultaneously with LRU eviction and configurable memory budgets
- Thinking/reasoning model support with separated `reasoning_content` field: both `<think>`-style models (Qwen3, toggled with `enable_thinking`) and harmony channel models (gpt-oss, scaled with `reasoning_effort: low|medium|high`; reasoning cannot be disabled on that architecture)
- GGUF quantized model support: bf16 Metal fast path for ten quant types (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`); zero-copy mmap loader (Qwen3-4B-Q4_K_M loads in ~2.7 s); fused `mul_mm` for prefill so weights are never held twice. Sharded GGUF loading supported.
- AWQ 4-bit and 8-bit quantization: fused W4A16 / W8A16 GEMV kernels keep packed weights resident on Metal (Qwen3-4B-AWQ goes from ~7.5 GB to ~2.5 GB resident); QKV and gate+up are fused at load; auto-detected per checkpoint with no flag.
- GPTQ 4-bit and 8-bit quantization: `desc_act=false` checkpoints route through a dedicated Metal resident kernel family (Qwen3-0.6B-GPTQ-Int8 ~89 tok/s decode); CPU / non-bf16 paths still dequantize at load.
- FP8 (E4M3) block-wise weight loading: `Qwen3-4B-Instruct-2507-FP8` and similar checkpoints; the load-time `weight × scale_inv` multiply is performed in F32 to preserve precision across deep block-wise rescaling.
- MXFP4 (OCP microscaling FP4): GPT-OSS expert weights stay packed on Metal with fused dequant GEMV/GEMM kernels; `openai/gpt-oss-20b` (20.9B params) runs in ~13 GB resident on a 24 GB machine. `gpt-oss-120b` shares the same architecture and should load on machines with enough memory, but is untested.
- Mixture-of-Experts: `Qwen3MoeForCausalLM`, `OlmoeForCausalLM`, and `GptOssForCausalLM` (interleaved clamped-swiglu experts, attention sinks via a dedicated fused decode kernel, alternating sliding/full attention layers) with top-k routing and a hybrid sparse/naive dispatch (decode-friendly + prefill-friendly). Tested on `allenai/OLMoE-1B-7B-0924-Instruct` and `openai/gpt-oss-20b`.
- Hybrid linear-attention models (`Qwen3_5ForConditionalGeneration` and the GGUF `qwen35` arch, text-only): Gated DeltaNet layers run a chunked parallel scan for prefill and an O(1) recurrent step for decode, with per-sequence recurrent state managed alongside the paged KV cache; full-attention layers use gated attention (sigmoid output gate) with partial RoPE. Prefix caching and speculative decoding are automatically disabled for these models (recurrent state can neither skip tokens nor roll back). Vision tower and MTP weights are skipped at load. Supported checkpoint formats: BF16 safetensors, compressed-tensors pack-quantized INT4 (full or mixed BF16/INT4), and GGUF (llama.cpp `qwen35` tensor layout: tiled V-heads, pre-baked norm shift and `-exp(A_log)`).
- compressed-tensors "pack-quantized" INT4 (llm-compressor output, symmetric, group strategy): converted to the AWQ layout at load (nibbles transfer verbatim — the format is already offset-binary; zero-points are constant 8) so the resident W4A16 Metal kernels apply unchanged. Works for both fully-quantized and mixed-precision (`ignore` list) checkpoints.
- Streaming responses via Server-Sent Events
- Model download directly from HuggingFace with interactive variant selection

## Architecture
oxydLLM is built on top of the Candle tensor library. The model layer implements a unified transformer architecture that covers most supported model families with minimal per-architecture branching. The inference engine uses paged KV allocation with a shared block pool, a prefix cache keyed on rolling block hashes, and a scheduler that handles concurrent prefill and decode across multiple sequences.

KV cache quantization uses TurboQuant with MSE-based quantization during the decode phase, reducing memory overhead without significant quality loss. Metal kernels provide fused operations for attention, normalization, and positional embeddings on Apple Silicon.

> Note on `--kv-quant`: the quantization step currently runs on CPU, each KV write transfers the new K/V tensors from GPU to CPU and casts them to F32 before packing. On unified-memory Apple Silicon the transfer is cheap, but on discrete CUDA GPUs the per-step roundtrip can dominate. Enable `--kv-quant` for memory-constrained deployments; leave it `off` when throughput matters and KV memory is not the bottleneck. On-device kernels are on the roadmap.

## Tested Models
The following models all pass the deterministic coherence check in `scripts/stress_baseline.py` on the Apple Silicon reference machine (M5, 24 GB unified memory); the Qwen3.5 family is covered separately [below](#qwen35-hybrid-linear-attention) with its own adversarial battery. Decode `tok/s` is the steady-state median over five 150-token runs after one warm-up.

| Model | Architecture | Format | Decode tok/s |
|---|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | LlamaForCausalLM | BF16 safetensors | 33.1 |
| `Qwen/Qwen2.5-1.5B-Instruct` | Qwen2ForCausalLM | BF16 safetensors | 25.4 |
| `Qwen/Qwen2.5-3B-Instruct` | Qwen2ForCausalLM | BF16 safetensors | 13.6 |
| `Qwen/qwen2-1_5b-instruct-q4_0` | Qwen2 (GGUF) | Q4_0 | 67.3 |
| `Qwen/Qwen2.5-1.5B-Instruct-Q2_K` | Qwen2 (GGUF) | Q2_K | 105.2 |
| `Qwen/Qwen2.5-1.5B-Instruct-Q3_K_M` | Qwen2 (GGUF) | Q3_K_M | 85.1 |
| `Qwen/Qwen2.5-1.5B-Instruct-Q4_0` | Qwen2 (GGUF) | Q4_0 | 85.4 |
| `bartowski/Qwen2.5-1.5B-Instruct-Q4_0` | Qwen2 (GGUF) | Q4_0 | 64.6 |
| `Qwen/Qwen2.5-1.5B-Instruct-Q4_K_M` | Qwen2 (GGUF) | Q4_K_M | 80.4 |
| `Qwen/Qwen3-0.6B` | Qwen3ForCausalLM | BF16 safetensors | 52.7 |
| `Qwen/Qwen3-0.6B-GPTQ-Int8` | Qwen3ForCausalLM | GPTQ Int8 (W8A16 resident) | 86.8 |
| `Qwen/Qwen3-1.7B-Q8_0` | Qwen3 (GGUF) | Q8_0 | 38.0 |
| `Qwen/Qwen3-1.7B-GPTQ-Int8` | Qwen3ForCausalLM | GPTQ Int8 (W8A16 resident) | 41.5 |
| `Qwen/Qwen3-4B-Q4_K_M` | Qwen3 (GGUF) | Q4_K_M | 27.0 |
| `Qwen/Qwen3-4B-Q5_0` | Qwen3 (GGUF) | Q5_0 | 26.4 |
| `Qwen/Qwen3-4B-Q5_K_M` | Qwen3 (GGUF) | Q5_K_M | 25.5 |
| `Qwen/Qwen3-4B-Q6_K` | Qwen3 (GGUF) | Q6_K | 22.8 |
| `Qwen/Qwen3-4B-AWQ` | Qwen3ForCausalLM | AWQ 4-bit (W4A16 resident) | 38.5 |
| `Qwen/Qwen3-4B-Instruct-2507-FP8` | Qwen3ForCausalLM | FP8 (E4M3, block-wise) | 10.0 |
| `google/gemma-2b-it` | GemmaForCausalLM | BF16 safetensors | 16.6 |
| `google/gemma-2-2b-it` | Gemma2ForCausalLM | BF16 safetensors | 15.5 |
| `google/gemma-3-1b-it` | Gemma3ForCausalLM | BF16 safetensors | 36.1 |
| `google/gemma-4-E2B-it` | Gemma4ForConditionalGeneration | BF16 safetensors | 15.4 |
| `mistralai/Ministral-3-3B-Instruct-25` | Mistral3ForConditionalGeneration | BF16 safetensors | 12.1 |
| `allenai/OLMoE-1B-7B-0924-Instruct` | OlmoeForCausalLM (MoE) | BF16 safetensors, 64 experts × top-k 8 | 13.6 |
| `openai/gpt-oss-20b` | GptOssForCausalLM (MoE) | MXFP4 experts + BF16, 32 experts × top-k 4 | 14.3 |

> [!NOTE]
> All Qwen3 models have been tested with and without thinking enabled. Other checkpoints in the same architecture families (e.g. other Llama 3.2, Gemma 3, Qwen2.5 sizes) are likely to work but are not in the regression suite.

### Qwen3.5 (hybrid linear attention)

Qwen3.5 runs on a dedicated hybrid runtime (Gated DeltaNet + gated full attention). Thinking mode works with `enable_thinking` on/off, with reasoning separated into `reasoning_content` in both streaming and non-streaming responses.

| Model | Format | Resident | Decode tok/s* | Battery |
|---|---|---|---|---|
| `Qwen/Qwen3.5-4B` | BF16 safetensors | 8.7 GB | 8.9 | 13/13 |
| `cyankiwi/Qwen3.5-4B-AWQ-4bit` | compressed-tensors INT4 (W4A16 resident) | 3.1 GB | 19.5 | 13/13 |
| `cyankiwi/Qwen3.5-4B-AWQ-BF16-INT4` | mixed BF16 DeltaNet + INT4 attn/MLP | 4.4 GB | 14.7 | 13/13 |
| `unsloth/Qwen3.5-4B-GGUF` (Q4_K_M) | GGUF (`qwen35` arch) | 2.5 GB | 24.1 | 12/13** |

\* Median of three 150-token completions, prefill included.

\*\* Quality loss of the Q4_K_M quantization on one marginal reasoning prompt, not a runtime defect: the same weights answer correctly when prompted step-by-step, and batched-vs-single decode stays byte-identical.


## Unsupported Model Families
The following model families are not currently supported:
- Mixtral (`MixtralForCausalLM`): uses `block_sparse_moe.experts.{e}.{w1,w2,w3}` tensor naming, not yet routed in the loader (the MoE infrastructure itself is in place via `Qwen3MoeForCausalLM` and `OlmoeForCausalLM`).
- DeepSeek-V2/V3: Mixture-of-Experts plus Multi-head Latent Attention (MLA); MLA is not implemented yet.
- GGUF MoE checkpoints: quant-per-expert tensor layout not yet wired; safetensors MoE works.
- Multimodal inference (vision+language) is not supported yet; text-only paths from some multimodal checkpoints may work.
- Encoder-only models (BERT, etc.)

## Installation
For using oxydLLM, you can either build from source or use the provided installers.

### Building from source
Clone the repository and install the [Rust toolchain](https://rust-lang.org/tools/install/):
```bash
git clone https://github.com/giovannifil-64/oxydllm
cd oxydllm
```

#### Apple Silicon
```bash
cargo build --release --features metal
```

Requires macOS 14 (Sonoma) or newer: the Metal kernels rely on bfloat support introduced with Metal 3.1. Supported releases: macOS 14 (Sonoma), 15 (Sequoia), and 26 (Tahoe).

#### NVIDIA CUDA
```bash
CUDA_COMPUTE_CAP=<value> cargo build --release --features cuda
```

Replace `<value>` with the compute capability of your GPU:

| Compute Capability | Data Center | Workstation / Consumer | Jetson |
|---|---|---|---|
| 12.1 | | NVIDIA GB10 (DGX Spark) | |
| 12.0 | NVIDIA RTX PRO 6000 Blackwell Server Edition<br>NVIDIA RTX PRO 4500 Blackwell Server Edition | NVIDIA RTX PRO 6000/5000/4500/4000/2000 Blackwell<br>GeForce RTX 5090, 5080, 5070 Ti, 5070, 5060 Ti, 5060, 5050 | |
| 11.0 | | | Jetson T5000<br>Jetson T4000 |
| 10.3 | NVIDIA GB300<br>NVIDIA B300 | | |
| 10.0 | NVIDIA GB200<br>NVIDIA B200 | | |
| 9.0 | NVIDIA GH200<br>NVIDIA H200<br>NVIDIA H100 | | |
| 8.9 | NVIDIA L4<br>NVIDIA L40<br>NVIDIA L40S | NVIDIA RTX 6000/5000/4500/4000/2000 Ada<br>GeForce RTX 4090, 4080, 4070 Ti, 4070, 4060 Ti, 4060, 4050 | |

> [!NOTE]
> `CUDA_COMPUTE_CAP` is validated at compile time: passing an unsupported value is a build error. If not set, Candle attempts auto-detection via `nvidia-smi`.

Run the server

```bash
cargo run --release -- start
```

Run `cargo run --release -- help` for see all the options.

### Installers
Platform-specific installers are made available with pre-built binaries for supported configurations. The installers bundle the server executable and its dependencies, but **not** the models.

You can use the provided `install.sh` script to download and install the appropriate binary for your system. The script detects your platform and GPU (if applicable) to select the correct installer.

Simply run:

```bash
curl -fsSL https://github.com/giovannifil-64/oxydllm/raw/main/install.sh | sh
```

> [!TIP]
> You can override the automatic GPU detection by setting the `OXYDLLM_CUDA_TARGET` environment variable to one of the supported targets before running the installer script. This is useful if you want to install a specific CUDA variant or if automatic detection fails.
> 
> ```bash
> # x86_64
> OXYDLLM_CUDA_TARGET=ada|hopper|blackwell|blackwell-ultra|blackwell-desktop curl -fsSL https://github.com/giovannifil-64/oxydllm/raw/main/install.sh | sh
> # arm64
> OXYDLLM_CUDA_TARGET=hopper|blackwell|blackwell-ultra|thor|blackwell-desktop curl -fsSL https://github.com/giovannifil-64/oxydllm/raw/main/install.sh | sh
> ```

If you prefer to manually download the installer, you can find the latest releases on GitHub:

#### macOS (Apple Silicon)
> [!IMPORTANT]
> Intel-based Macs are not supported. Requires macOS 14 (Sonoma) or newer; supported releases are macOS 14, 15, and 26 (Tahoe). The installer refuses older versions because the Metal kernels need bfloat support (Metal 3.1, macOS 14+).
- `oxydllm-macos-arm64`

#### Linux (CUDA)

##### x86_64
- `oxydllm-linux-x86_64-cuda-ada.tar.gz` for Ada Lovelace (sm_89: RTX 40xx, L4, L40/L40S)
- `oxydllm-linux-x86_64-cuda-hopper.tar.gz` for Hopper (sm_90: H100, H200)
- `oxydllm-linux-x86_64-cuda-blackwell.tar.gz` for Blackwell datacenter (sm_100: B100, B200, GB200)
- `oxydllm-linux-x86_64-cuda-blackwell-ultra.tar.gz` for Blackwell Ultra (sm_103: B300, GB300)
- `oxydllm-linux-x86_64-cuda-blackwell-desktop.tar.gz` for Blackwell Desktop (sm_120: RTX 50xx, RTX PRO)

##### arm64 (GH200 / GB300 / Jetson / DGX Spark)
- `oxydllm-linux-arm64-cuda-hopper.tar.gz` for Hopper (sm_90: GH200)
- `oxydllm-linux-arm64-cuda-blackwell.tar.gz` for Blackwell datacenter (sm_100: B200, GB200)
- `oxydllm-linux-arm64-cuda-blackwell-ultra.tar.gz` for Blackwell Ultra (sm_103: GB300)
- `oxydllm-linux-arm64-cuda-thor.tar.gz` for Jetson GB (sm_110: T4000, T5000)
- `oxydllm-linux-arm64-cuda-blackwell-desktop.tar.gz` for Blackwell Desktop (sm_121: DGX Spark / GB10)

## Usage
Download a model from HuggingFace using the `user/model` repo ID. For GGUF repos, an interactive prompt lists available quantizations and lets you pick one; variants already on disk are shown with a check mark and excluded from the numbered choices. Use `--variant Q4_K_M` to skip the prompt, `--token` for gated models, and `--name` to save under a custom local name instead of the default `user/model` path.
```bash
oxydllm pull Qwen/Qwen3-4B-GGUF
oxydllm pull Qwen/Qwen3-4B-GGUF --variant Q4_K_M
oxydllm pull meta-llama/Llama-3.1-8B-Instruct --token hf_xxxxxxxxxxxx
```

List locally available models. Each model is identified by its HuggingFace `user/model` ID, which is the same string you pass to `run`, `estimate`, and `rm`, and the same one the API expects in the `model` field. Multiple GGUF quantizations stored in the same folder each appear as a separate entry.
```bash
oxydllm list
oxydllm list --models-dir /path/to/models
```

Estimate memory requirements for a model before downloading or running it. Accepts both local model IDs and HuggingFace repo IDs for remote estimation. Both `estimate` and `run` accept partial model names: `oxydllm run Qwen/Qwen3-4B` resolves to the first matching local model.
```bash
oxydllm estimate Qwen/Qwen3-4B-GGUF --context-len 8192 --num-sequences 4
```

Start an interactive chat session in the terminal, loading the model directly without starting an HTTP server.
```bash
oxydllm run Qwen/Qwen3-0.6B
```

Remove a model from disk and deregister it from the local registry.
```bash
oxydllm rm Qwen/Qwen3-0.6B
```

> [!IMPORTANT]
> Models are loaded on demand when the first request for that model arrives.

Update oxydllm to a newer release. Without flags the command queries the GitHub releases API for the latest stable non-pre-release build and compares the remote version tag against the installed binary. Pass `--pre` to target the most recent pre-release instead, or `--nightly` to compare the rolling nightly build against the compile-time Unix timestamp baked into the binary at build time. When the installed version is already current the command reports that and exits without making any changes. `update` is only available in binaries installed via `install.sh`; source builds receive an informational error and exit.
```bash
oxydllm update
oxydllm update --pre
oxydllm update --nightly
```

Remove oxydllm from the system. The command stops and removes the OS service (launchd agent on macOS, systemd unit on Linux), deletes the binary via self-removal, and then exits cleanly. A confirmation prompt is always shown before any changes are made. Pass `--purge` to also remove `~/.oxydllm/` and all downloaded models; this operation cannot be undone. `uninstall` is only available in binaries installed via `install.sh`.
```bash
oxydllm uninstall
oxydllm uninstall --purge
```

### API
The fastest way to interact with the server is through the OpenAI-compatible API. The following endpoints are available:
- `GET /health`
- `GET /metrics`
- `GET /v1/models`
- `GET /v1/models/running`
- `GET /v1/models/{model_id}`
- `POST /v1/chat/completions`

```bash
curl http://localhost:11313/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Thinking mode (for models that support it):

```bash
curl http://localhost:11313/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Q4_K_M",
    "messages": [{"role": "user", "content": "Explain quantum entanglement"}],
    "enable_thinking": true
  }'
```

Reasoning effort for harmony models (gpt-oss). These models cannot disable reasoning; `reasoning_effort` scales it (`low`, `medium`, `high`; default `medium`). The reasoning stream is returned separately in `reasoning_content`, the final answer in `content`:

```bash
curl http://localhost:11313/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Explain quantum entanglement"}],
    "reasoning_effort": "low"
  }'
```

### Observability

#### Prometheus metrics (`GET /metrics`)

Metrics are exposed in Prometheus text format at `GET /metrics`. Scrape this endpoint with Prometheus or any compatible agent (Vector, OpenTelemetry Collector, etc.).

```bash
curl http://localhost:11313/metrics
```

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `oxydllm_ttft_milliseconds` | Histogram | `model` | Time-to-first-token in ms, from request enqueue to first generated token. Includes prefill and queue wait. Buckets: 10, 50, 100, 200, 500, 1000, 2000, 5000 ms. |
| `oxydllm_tokens_per_second` | Histogram | `model` | Decode throughput in tokens/s from first token to completion. Buckets: 1, 5, 10, 20, 50, 100, 200 tok/s. |
| `oxydllm_requests_total` | Counter | `model`, `status` | Total completed requests. `status` is `ok` or `error`. |
| `oxydllm_queue_depth` | Gauge | - | Current number of sequences in the engine (waiting + running). Updated each engine step. |
| `oxydllm_prefix_cache_requests_total` | Counter | `model`, `result` | Prefix KV cache lookups by result (`hit` or `miss`). Compute hit ratio with `rate(hit[5m]) / rate(total[5m])`. |
| `oxydllm_model_weights_bytes` | Gauge | `model` | Weight memory in bytes, set at load and cleared at unload. |
| `oxydllm_kv_cache_allocated_bytes` | Gauge | `model` | KV cache memory reserved at load time per model. Not the dynamically occupied portion. |
| `oxydllm_vram_used_bytes` | Gauge | - | Total inference memory: `model_weights_bytes + kv_cache_allocated_bytes` across all loaded models. |

> Apple Silicon note: there is no discrete VRAM on Apple Silicon; all memory metrics measure unified system memory shared between CPU and GPU.

Example Prometheus queries:
```promql
# Average TTFT over the last 5 minutes
histogram_quantile(0.95, rate(oxydllm_ttft_milliseconds_bucket[5m]))

# Prefix cache hit ratio
rate(oxydllm_prefix_cache_requests_total{result="hit"}[5m])
  / rate(oxydllm_prefix_cache_requests_total[5m])

# Request throughput by status
rate(oxydllm_requests_total[1m])
```

#### Structured logs and request tracing

Every request is assigned a `request_id` (UUID v4) at the HTTP handler entry point. This ID appears in all log events for that request, from template rendering to the final token, making it possible to trace a single request end-to-end across concurrent traffic:

```bash
grep 'request_id=abc-123' app.log
```

By default logs use a compact human-readable format. Set `LOG_FORMAT=json` for machine-parseable output compatible with Loki, Datadog, AWS CloudWatch, and `jq`. The variable is read at startup and applies to all commands, not just `start`:

```bash
LOG_FORMAT=json oxydllm start
LOG_FORMAT=json oxydllm run Qwen/Qwen3-4B-Q4_K_M
```

Each log line becomes a self-contained JSON object:
```json
{"timestamp":"2024-01-01T12:00:00.123Z","level":"INFO","fields":{"request_id":"abc-123","ttft_ms":123.4,"model_id":"Qwen/Qwen3-4B-Q4_K_M"},"message":"first token emitted"}
```

Query a single request's lifecycle with Loki: `{app="oxydllm"} | json | request_id="abc-123"`, or with `jq`:
```bash
oxydllm start 2>&1 | jq 'select(.fields.request_id=="abc-123")'
```

## Server Options
Every option can be set via a CLI flag or an environment variable. CLI flags take priority when both are set. When running as a system service (launchd on macOS, systemd on Linux) you typically configure via env vars without touching the service unit file itself.

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--port <PORT>` | `OXYDLLM_PORT` | `11313` | Listen port |
| `--models-dir <DIR>` | `OXYDLLM_MODELS_DIR` | `~/.oxydllm/models` | Model storage directory |
| `--keep-alive <SECS>` | `OXYDLLM_KEEP_ALIVE` | `900` | Idle seconds before model eviction |
| `--memory-budget <MB>` | `OXYDLLM_MEMORY_BUDGET` | - | Max VRAM for loaded models; LRU eviction when exceeded |
| `--max-context-len <N>` | `OXYDLLM_MAX_CONTEXT_LEN` | `4096` | KV cache context length per sequence |
| `--max-num-seqs <N>` | `OXYDLLM_MAX_NUM_SEQS` | auto | Max concurrent sequences per model (auto-computed from KV block budget at load time) |
| `--max-queued-requests <N>` | `OXYDLLM_MAX_QUEUED_REQUESTS` | `200` | Request queue depth; returns HTTP 429 when full |
| `--devices <IDS>` | `OXYDLLM_DEVICES` | auto | Comma-separated CUDA device indices |
| `--kv-quant <MODE>` | `OXYDLLM_KV_QUANT` | `off` | KV cache quantization: `off`, `lossless`, `balanced`, `aggressive` |
| `--shutdown-timeout <SECS>` | `OXYDLLM_SHUTDOWN_TIMEOUT` | `30` | Grace period for in-flight requests on shutdown |
| `--qjl-quantization` | - | disabled | Enable Stage-2 QJL key residual quantization |
| `--allow-cpu` | `OXYDLLM_ALLOW_CPU` | disabled | Permit CPU fallback when no GPU is available. By default startup fails fast on a GPU-less host. |
| `--api-key <KEY>` | `OXYDLLM_API_KEY` | disabled | When set, every `/v1/*` and `/metrics` request must present the key via `Authorization: Bearer <KEY>` (or `X-API-Key: <KEY>`). `/health` remains unauthenticated for liveness probes. |
| `--request-timeout <SECS>` | `OXYDLLM_REQUEST_TIMEOUT` | `300` | Wall-clock timeout per `/v1/chat/completions` request. Non-streaming responses are returned as `408 Request Timeout`; streaming responses emit a final `request_timeout` error chunk followed by `[DONE]`. Set to `0` to disable. |

To produce machine-parseable JSON log output (useful with Loki, Datadog, or `jq`), set `LOG_FORMAT=json`. The variable is read at startup and applies to all commands. See the [Observability](#observability) section for details and examples.

### Configuration examples

**systemd (Linux)**: edit `/etc/default/oxydllm`, then `sudo systemctl restart oxydllm`:
```
OXYDLLM_MAX_CONTEXT_LEN=8192
OXYDLLM_MAX_NUM_SEQS=16
OXYDLLM_KV_QUANT=balanced
LOG_FORMAT=json
```

**launchd (macOS)**: edit `~/Library/LaunchAgents/com.oxydllm.oxydllmd.plist` under `EnvironmentVariables`, then reload the agent:
```xml
<key>OXYDLLM_MAX_CONTEXT_LEN</key><string>8192</string>
<key>OXYDLLM_MAX_NUM_SEQS</key><string>16</string>
<key>LOG_FORMAT</key><string>json</string>
```

**Docker**:
```bash
docker run -e OXYDLLM_MAX_CONTEXT_LEN=8192 -e OXYDLLM_MAX_NUM_SEQS=16 -e LOG_FORMAT=json \
  -p 11313:11313 ghcr.io/giovannifil-64/oxydllm:latest start --models-dir /root/.oxydllm/models
```

**docker compose**: set variables in your shell or a `.env` file:
```
OXYDLLM_MAX_CONTEXT_LEN=8192
OXYDLLM_MAX_NUM_SEQS=16
OXYDLLM_KV_QUANT=balanced
LOG_FORMAT=json
```

## Security

The HTTP API has **no authentication by default**. Without `--api-key` set, any client that can reach the port can list and invoke loaded models and read Prometheus metrics. For any deployment that is not a single-user local machine:

1. Set `OXYDLLM_API_KEY=<random-token>` (or pass `--api-key <KEY>`). Once configured, every request to `/v1/*` and `/metrics` must include the header `Authorization: Bearer <KEY>`; `X-API-Key: <KEY>` is also accepted. Missing or wrong keys return `401` with `error.type = "invalid_api_key"`. `/health` remains unauthenticated so liveness probes keep working.
2. Bind the listener to a private address or place it behind a reverse proxy (nginx, Caddy, Traefik). The default bind is `0.0.0.0`, which exposes the server on every interface.

Request-side hardening already enforced by the server (no configuration needed):

- Per-request wall-clock timeout (`--request-timeout`, default 300s) bounds the time a single chat-completion request can hold a slot. On expiry the engine sequence is aborted, the client receives `408` (non-streaming) or an error chunk + `[DONE]` (streaming).
- Sampling parameter ranges are validated up-front (`temperature ∈ [0, 2]`, `top_p ∈ [0, 1]`, `frequency_penalty`/`presence_penalty ∈ [-2, 2]`, `top_logprobs ∈ [0, 20]`, `repetition_penalty > 0`, `n ∈ [1, 128]`, `max_tokens ≥ 1`, `reasoning_effort ∈ {low, medium, high}`). Out-of-range values return `400 invalid_request_error` rather than silently degrading the sampler.

## Run Options
Options specific to the `oxydllm run` interactive chat command (not available in the server):
```
--temperature <T>      Sampling temperature (default: 0.7)
--top-k <K>            Top-k filtering (default: 0, disabled)
--top-p <P>            Nucleus sampling (default: 1.0)
--min-p <P>            Min-p filtering (default: 0.0)
--repeat-penalty <R>   Repetition penalty (default: 1.0)
--repeat-window <N>    Trailing token window for repetition penalty (default: 0 = full history)
```

The following options are shared between `start` and `run`:
`--models-dir`, `--devices`, `--max-context-len`, `--kv-quant`, `--qjl-quantization`, `--allow-cpu`.

## Known Limitations and Work in Progress
- GGUF compatibility: the bf16 Metal fast path covers ten quant types (`Q4_0/1`, `Q5_0/1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`); i-quants (`IQ*`) and ternary (`TQ*`) types fall back to candle's F32 path. MoE GGUFs are not yet wired.
- Thinking mode is template-dependent: `enable_thinking` is applied only when the tokenizer chat template supports it.
- Byte-level tokenizers: Streaming decode uses incremental buffering; occasional model-specific Unicode artifacts can still appear.
- Tool / schema adherence is model-dependent: the OpenAI-compatible request fields, response shapes, and streaming semantics are implemented server-side, but local models can still ignore tool instructions or emit invalid JSON / tool arguments.
- Only function tools are implemented: OpenAI custom tools are not supported on `/v1/chat/completions` yet.
- Gemma4 edge cases: Some checkpoints may require architecture-specific tuning.
- Metal softcap SDPA policy: The Metal SDPA path with attention softcap is currently hard-disabled in runtime (no experimental toggle) and falls back to the standard attention path.
- Attention-sink models (gpt-oss): decode runs a dedicated fused sink-aware SDPA kernel; prefill falls back to the standard attention path, so long-prompt TTFT is higher than on comparable non-sink models.
- Metal SDPA head-dim coverage: The fused Metal SDPA kernel supports head dimensions `32, 64, 72, 80, 96, 128, 256`. Models with other head dimensions remain functionally correct but fall back to the non-fused attention path with a measurable throughput cost.
- CUDA optimization: Support exists but is not optimized for production use.
- GPTQ act-order (`desc_act=true`) not supported: load fails fast; only sequential `desc_act=false` checkpoints are accepted. `g_idx` is loaded but ignored on the supported path.
- FP8 on Apple Silicon doubles resident memory: Metal has no FP8 compute kernels, so all FP8 checkpoints are dequanted to BF16 at load time. A 4B-FP8 model needs ~8 GB resident instead of the ~4 GB on-disk footprint. CUDA / CPU retain the Level-2 resident FP8 path.
- MoE perf is dispatch-bound: the hybrid sparse/naive path is correct and decode-competitive, but per-expert Metal command-buffer overhead caps prefill throughput. A custom fused MoE kernel would unlock the next ~2-3× speedup on long prompts.

## CUDA Status
CUDA is currently a functional compatibility path, not a performance-tuned backend.

> [!WARNING]
> The CUDA path has not been tested on real NVIDIA hardware. Builds are CI-verified (compile + CPU tests only). Runtime correctness and performance on actual GPUs are unvalidated.

- Build/runtime support is available via `--features cuda`.
- Core model execution works, but the CUDA path currently relies mostly on Candle generic kernels.
- Metal has additional fused kernels in this project (attention/RMSNorm/RoPE/softmax), so CUDA throughput will be lower than specialized CUDA stacks.
- Contributions and testing reports from NVIDIA hardware owners are welcome.

### Official CUDA Docker tags

> [!IMPORTANT]
> Every tag below is unvalidated, the images compile and pass CPU tests in CI, but no inference run has been verified on physical NVIDIA hardware. Treat the table as a build matrix, not a compatibility guarantee.

| Tag | Compute capability | Platform | Target |
|---|---:|---|---|
| `cuda-ada` | 89 | amd64 | Ada Lovelace (RTX 40xx, L4, L40/L40S) |
| `cuda-hopper` | 90 | amd64 | Hopper (H100, H200) |
| `cuda-blackwell` | 100 | amd64 | Blackwell datacenter (B100, B200, GB200) |
| `cuda-blackwell-ultra` | 103 | amd64 | Blackwell Ultra (B300, GB300) |
| `cuda-blackwell-desktop` | 120 | amd64 | Blackwell Desktop (RTX 50xx, RTX PRO) |
| `cuda-hopper-arm64` | 90 | arm64 | Hopper (GH200 Grace Hopper) |
| `cuda-blackwell-arm64` | 100 | arm64 | Blackwell datacenter (B200, GB200) |
| `cuda-blackwell-ultra-arm64` | 103 | arm64 | Blackwell Ultra (GB300 NVL72) |
| `cuda-thor-arm64` | 110 | arm64 | Jetson GB (T4000, T5000) |
| `cuda-blackwell-desktop-arm64` | 121 | arm64 | Blackwell Desktop (DGX Spark / GB10) |

- `latest` and `cuda` point to `cuda-ada` (stable default, widest x86_64 compatibility).
- `nightly` and `nightly-cuda` point to nightly `cuda-ada`.
- Cross-generation SASS is **not** compatible: a Hopper binary will not run on Blackwell and vice versa. Pick the tag that matches your GPU.


## Contributing
Contributions are welcome! If you want to contribute, please follow these steps:

1. Fork the repository and create a new branch for your feature or bug fix.
2. Make your changes and commit/push them with clear and descriptive messages.
3. Open a pull request against the `main` branch with a detailed description of your changes and the problem they solve.

Enable local pre-commit checks (recommended for contributors)

```bash
chmod +x .githooks/pre-commit
git config core.hooksPath .githooks
```

The pre-commit hook runs formatting and strict clippy checks before each commit, helping to maintain code quality and consistency. You can run the checks manually with `cargo fmt` and `cargo clippy -- -D warnings`.

## License
The code in this repository is made available under the Apache 2.0 license. See [LICENSE](LICENSE) for details.
