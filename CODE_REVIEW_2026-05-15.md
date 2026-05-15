# oxydllm — Code Review (2026-05-15)

> claude --resume 4823a1e8-f6c2-4c3d-8f0f-8aae7e28fc04

**Reviewer:** independent Rust code review pass.
**Scope:** all of `src/` (≈22k lines), `build.rs`, `Cargo.toml`, `README.md`, `install.sh`, `Dockerfile{,.cpu}`, `docker-compose.yml`, `http_compat_tests.rs` (711 lines, cross-checked against the OpenAI Chat Completions API spec), and all 357 non-mutex `.unwrap()` / `.expect()` calls in production code.

**Methodology:** every finding was re-verified against the actual source at least twice. Findings that turned out to be artifacts of an outdated `CODE_REVIEW/` snapshot or that didn't survive a second read are listed under "False positives" at the end.

---

## TL;DR

The codebase is in considerably better shape than `CODE_REVIEW/05_verified_final.md` suggests. **7 of its 10 source-code findings are already fixed in master.** Several other items tagged CRITICAL elsewhere in `CODE_REVIEW/` are not bugs at all when traced through the code. The first recommended action is to **archive `CODE_REVIEW/`** — it is now historical and misleading to anyone reading it cold.

Items below are organized by macroarea: server/API, performance, robustness, Apple Silicon, NVIDIA CUDA, model loading, engine/scheduler, README, and HTTP-compat test coverage.

Severity scale: **CRITICAL** (silent data loss / crash on common input) · **HIGH** (functional bug / API contract break) · **MEDIUM** (degradation under specific conditions) · **LOW** (polish / future-fragility).

---

## ~~A. Stale review docs~~ [CLOSED] All the changes have been implemented and verified.

`CODE_REVIEW/*.md` (dated 2026-05-14) lists 16 "verified final" source findings. Re-verifying each against current source on 2026-05-15:

| Old finding | Current state in code | Action |
|---|---|---|
| **#1** Missing `[DONE]` on stream error (chat.rs:1869) | **FIXED** — `chat.rs:1885` sends `[DONE]` after Error |
| **#2** Schema validation strict mode ignored | **FIXED** — `chat.rs:2177-2192` returns `content_filter` in strict mode |
| **#3** Orphan tasks on n>1 error with tools | **FIXED** — `chat.rs:1462-1474` calls `h.abort()` on remaining handles |
| **#4** `tool_type` default empty string | **FIXED** — `types.rs:48-49` has no `#[serde(default)]` |
| **#5** NaN logits → token 0 silently | **PARTIALLY FIXED** — guard exists at `sampling.rs:66` but only on non-greedy path (see S1) |
| **#6** Template fallback silent | **STILL VALID** — `chat.rs:1029-1033` still falls back to plain text on 200 (see S2) |
| **#7** Registry save at warn level | **FIXED** — `manager.rs:115` uses `tracing::error!` |
| **#8** AWQ first-load 4× underestimate | **FIXED** — `manager.rs:300-316` uses `awq_qweight_expansion` |
| **#9** CUDA compiled vs hardware cap warning | **FIXED** — `loader.rs:502-515` warns when `compiled_cap < hardware_cap` |
| **#10** `block.rs:327` reshape divisibility | not re-verified; low severity if still present |

**Recommended action:** delete or move `CODE_REVIEW/` to `docs/historical/` to prevent future contributors from chasing fixed bugs.

---

## B. New / still-valid findings (verified against current source)

### ~~B1. Server / API compatibility~~ [CLOSED 2026-05-15]

[✓] ~~**S1 — Greedy fast-path bypasses NaN logit check** · **HIGH** · `src/sampling.rs:50-62`~~

The NaN guard at line 66 is reached **only** when the greedy fast-path is skipped. The fast-path conditions are the production default (no penalties, no logit bias, no logprobs, `temperature == 0.0`), so most requests hit:

```rust
if params.temperature == 0.0 && params.top_logprobs_k == 0 && no_mods {
    let token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
    return Ok(SampleOutput { token, logprob: None, top_logprobs: Vec::new() });
}
```

with no NaN inspection. IEEE-754 argmax over a NaN-containing vector is implementation-defined; on Candle's CPU/Metal backends it typically returns index 0. A numerically unstable forward pass thus produces a silent stream of BOS tokens — exactly the failure mode the guard was meant to prevent.

*Fix:* move the NaN check above the fast-path return, or run it on the input tensor directly (cheap on small tensors via a quick all-finite scan).

---

[✓] ~~**S2 — Template-render failure returns 200 with degraded prompt** · **MEDIUM** · `src/server/routes/chat.rs:1013-1035`~~

When `apply_chat_template` fails and the no-system retry also fails, the code logs `tracing::error!` and silently calls `format_plain_chat(messages)`. The HTTP response is still 200, the client receives output produced from an un-templated prompt. For models that depend on their chat template — especially with `enable_thinking: true` — this disables thinking and degrades quality with no signal to the client.

*Fix:* when the request explicitly asked for thinking (`enable_thinking == true`) or both render attempts failed, return 500 with `error.code = "template_render_failed"`. Plain-text fallback should be reserved for models known to lack a template.

---

[✓] ~~**S3 — No authentication on the HTTP API** · **MEDIUM** (security/operational) · `src/server/routes/mod.rs:47-56`~~

```rust
Router::new()
    .route("/health", get(handlers::health))
    .route("/metrics", get(metrics::serve_metrics))
    .route("/v1/models", get(handlers::list_models))
    .route("/v1/models/running", get(handlers::list_running_models))
    .route("/v1/models/{*model_id}", get(handlers::get_model))
    .route("/v1/chat/completions", post(chat::chat_completions))
```

No middleware checks any `Authorization` header. The README documents `0.0.0.0:11313` as the default bind. Anyone on the network can invoke models, enumerate loaded weights, and scrape Prometheus metrics — including a CUDA-tagged Docker image marketed for production deployment.

*Fix:* add an optional `OXYDLLM_API_KEY` env var that, when set, requires `Authorization: Bearer <key>`. Bind `/metrics` to a separate address (e.g., `127.0.0.1:9090`) via a `--metrics-bind` flag, or restrict it to the same auth layer. At minimum, add a Security section to the README documenting the current exposure.

---

[✓] ~~**S4 — No request validation for sampling parameter ranges** · **MEDIUM** · `src/server/routes/chat.rs:1147-1182`~~

Validated: `messages` non-empty, `model` non-empty, `n >= 1`, `logit_bias` is object, `response_format` shape, `tool_config` shape. **Not validated:**

| Field | OpenAI range | oxydllm behavior on out-of-range |
|---|---|---|
| `temperature` | `[0, 2]` | accepted; negative → silently treated as ~0 by `temp.max(1e-8)` |
| `top_p` | `(0, 1]` | accepted; values > 1 → effectively `top_p = 1`; negative → keeps only argmax |
| `top_k` | `>= 0` | accepted; `0` disables, fine |
| `frequency_penalty` | `[-2, 2]` | accepted unrestricted |
| `presence_penalty` | `[-2, 2]` | accepted unrestricted |
| `top_logprobs` | `[0, 20]` | accepted unrestricted |
| `repetition_penalty` | n/a (oxydllm extension) | `0` produces division-by-zero → NaN logits |
| `n` | `[1, 128]` | only `>= 1` validated; client can send `n: 10000` |
| `seed` | any | not validated; accepted |
| `max_tokens` / `max_completion_tokens` | `>= 1` | precedence correct; `0` not validated |

A client sending `repetition_penalty: 0.0` with any repeated token produces NaN logits, which on the greedy fast-path (S1) leak through and on the non-greedy path correctly bail. Combined with S1, this is a user-triggerable silent-bad-output path.

*Fix:* add a `validate_sampling_params(&body)` helper called immediately after the model/n/messages checks at line 1146. Mirror OpenAI's ranges. Reject with `error.type = "invalid_request_error"`.

---

[✓] ~~**S5 — No per-request timeout / cancellation propagation** · **MEDIUM** · `src/server/routes/chat.rs` (`collect_one_completion`)~~

A disconnected client is detected by `tracker.tx.is_closed()` at `engine_loop.rs:393`, then aborted at the next step boundary. Non-streaming responses go through `collect_one_completion`, which awaits the engine receiver with no timeout. A stuck or runaway sequence (engine doesn't abort it, model emits no EOS, `max_tokens` is the full context window) holds a slot indefinitely.

*Fix:* add `--request-timeout` (default 300s). When exceeded, route through the existing `abort_sequences` path.

---

[✓] ~~**S6 — Streaming tools branch n>1: no `[DONE]` on internal task error** · **LOW** · `src/server/routes/chat.rs:1462-1475`~~

The non-tools n=1 path (line 1878-1887) and non-tools n>1 path (line 2086-2095) both send `[DONE]` after an error event. The **tools** streaming path at line 1461-1475 sends the error event and `return`s — no `[DONE]`. For consistency add the sentinel.

---

[✓] ~~**S7 — Strict-mode schema validation only catches *parseable* JSON failures** · **LOW** · `src/server/routes/chat.rs:2169-2176`~~

```rust
let schema_fail = json_schema_spec
    .and_then(|js| js.schema.as_ref())
    .and_then(|spec| {
        serde_json::from_str::<serde_json::Value>(&raw)
            .ok()
            .map(|val| !validate_against_schema(&val, spec))
    })
    .unwrap_or(false);
```

If the model returns non-JSON, `serde_json::from_str` is `Err`, `.ok()` is `None`, `.and_then` short-circuits, `schema_fail = false`, and the garbage is returned with `finish_reason: "stop"`. The strict-mode contract expects `content_filter` on any schema deviation, which includes "not JSON at all."

*Fix:* explicitly handle parse failure as schema fail in strict mode.

---

[✓] ~~**S8 — Per-request serde_json::to_string(...).unwrap() in streaming branches** · **LOW** (fragility) · `src/server/routes/chat.rs` (≈20 sites: 1467, 1503, 1546, 1585, 1613, 1642, 1671, 1696, 1733, 1780, 1809, 1841, 1865, 1883, 1933, 1985, 2014, 2047, 2074, 2091)**~~

`serde_json::to_string` fails on `f32::NAN` and `f32::INFINITY` (JSON has no representation for either). The structs being serialized (`ChatCompletionChunk`, `Logprobs`, `TokenLogprob`, `ToolCall`, …) contain `f32` fields. Today the upstream NaN guard at `sampling.rs:66` prevents NaN logprobs from entering these structs — but only on the non-greedy path (see S1). If S1 is fixed first, S8 stops being a chain risk; if not, a NaN propagates → serialization panics mid-stream → server thread aborts.

*Fix:* either fix S1 (preferred), or replace these unwraps with `?` propagation through an error chunk + `[DONE]`. The pattern is repeated enough that a `send_chunk!` macro would help.

---

### ~~B2. Performance~~

[✓] ~~**P1 — KV-cache quantization performs CPU↔GPU roundtrips per write** · **HIGH** (perf) · `src/common/paged.rs:374-379`~~

```rust
KvPool::Quantized { … } => {
    let n_tokens = data_k.dim(0)?;
    let k_f32 = data_k.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let v_f32 = data_v.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let k_vec: Vec<f32> = k_f32.flatten_all()?.to_vec1()?;
    let v_vec: Vec<f32> = v_f32.flatten_all()?.to_vec1()?;
```

On **every** KV write (every prefill chunk and every decode step) with `--kv-quant` enabled, K and V are cast to F32, moved to CPU, and materialized as `Vec<f32>`. The README markets KV quant as a memory-saving win; in practice on Metal the per-step GPU stall dwarfs the inference cost. On unified-memory Apple Silicon the copy itself is cheap but the F32 cast and `Vec` materialization are wasted work; on a discrete CUDA GPU the bus transfer would be catastrophic.

*Fix:* either implement on-device quantization (Metal kernel + CUDA kernel), or document the cost loudly in the README: "KV quantization currently runs on CPU; expect 2-5× per-token slowdown vs `kv-quant=off`."

---

[✓] ~~**P2 — `paged.rs` allocates fresh zero-filled contig buffers per sequence** · **MEDIUM** · `src/common/paged.rs:629-647`~~

```rust
let init_cap = (total_needed * 2).max(64);
let new_k_buf = Tensor::zeros((1, n_kv, new_cap, hd), dtype, &device)?;
let new_v_buf = Tensor::zeros((1, n_kv, new_cap, hd), dtype, &device)?;
```

Every fresh-prefill sequence triggers two zero-initialized tensor allocations. For Llama-3-8B (n_kv=8, hd=128) with an 8k prompt that's ~32 MB of GPU zero-fill per sequence on entry. Pool these per-sequence buffers, or reuse a single device-resident workspace.

---

[✓] ~~**P3 — `apply_top_k` does three passes over the probs vector** · **LOW** (perf) · `src/sampling.rs:296-326`~~

`Vec::to_vec`, `select_nth_unstable_by`, filter, then a linear tie-break pass. For Qwen-3 (152k vocab), that's ~600k float ops on CPU per decoded token before sampling. Not catastrophic, but the obvious next perf target after attention.

---

[✓] ~~**P4 — `metal_ops.rs` SDPA supports a fixed set of head dims** · **LOW** (perf coverage) · `src/common/metal_ops.rs:87`~~

`32 | 64 | 72 | 80 | 96 | 128 | 256`. Models with other head_dims fall back to non-fused attention (functionally correct, slower). Worth documenting in the README's Metal section.

---

### ~~B3. Robustness / correctness edge cases~~

[✓] ~~**R1 — Registry write is non-atomic** · **MEDIUM** · `src/models/manager.rs:110-117`~~

```rust
std::fs::write(&path, &json)
```

Two oxydllm processes sharing a `--models-dir` (e.g., a server plus an interactive `oxydllm run`) can race and produce a torn/empty `.oxydllm_registry.json`. The fact that `load_registry` falls back to `BTreeMap::new()` on parse failure (`manager.rs:107`) silently masks this — the user just loses all registry metadata.

*Fix:* `write_atomic_via_rename(path, content)` — write to `.tmp` then `std::fs::rename`. POSIX and Windows rename are atomic since Rust 1.61. Optionally add `fs2` for multi-process file locking.

---

[✓] ~~**R2 — `paged.rs:591` silent `usize → u32` narrowing** · **LOW** ·~~

```rust
let base = (block_id * block_size) as u32;
```

Realistic configs are safe (≤16M slot indices). The cast is silent; switch to `u32::try_from(block_id * block_size).expect("slot index overflow")` to fail loudly if the assumption is ever broken.

---

[✓] ~~**R3 — AWQ pack factor hardcoded** · **LOW** · `src/common/weights.rs:262-264`~~

`total += t.elem_count() * 8 * runtime_elem_bytes` assumes the qweight is always packed 8×4-bit per i32. AWQ-3bit / AWQ-8bit / alternate packings would silently report wrong memory sizes. Factor out a `pack_factor(bits)` helper.

---

[✓] ~~**R4 — `parse_compute_cap` formula breaks for two-digit minor versions** · **LOW** (future-fragility) · `build.rs:67-76`~~

```rust
Some(maj * 10 + min)
```

`"12.10"` → `12 * 10 + 10 = 130`, not 1210. NVIDIA hasn't shipped a two-digit minor yet, but the invariant ("flat int = major × 10 + minor") quietly breaks for `minor ≥ 10`. Reject `minor ≥ 10` explicitly, or compute `major * 100 + minor` and update `SUPPORTED_COMPUTE_CAPS` accordingly.

---

[✓] ~~**R5 — `linear.rs:55` `weight.t().expect("Linear weight must be 2D")` panics on malformed weights** · **LOW** ·~~

A malformed checkpoint with a non-2D weight tensor in a Linear layer panics at load instead of producing an anyhow error. Convert to `?` propagation.

---

[✓] ~~**R6 — CPU fallback is a warning, not an error** · **LOW** · `src/models/loader.rs:546-557`~~

When both CUDA and Metal init fail and `require_gpu == false` (default), the server runs on CPU after a `tracing::warn!`. For an inference server this is almost always a misconfiguration. Consider inverting: make `--require-gpu` the default and require `--allow-cpu` to opt out.

---

### B4. Apple Silicon / Metal

**M1 — SDPA softcap path hard-disabled (documented)** · **INFO** · `src/common/metal_ops.rs:91`, README "Known Limitations"

`supports_sdpa_full = … && self.softcapping == 1.0`. README:396 acknowledges this. No action.

**M2 — `cpu_fwd` on Metal SDPA bails instead of computing** · **DESIGN** · `src/common/metal_ops.rs:43-53`

A misconfigured test (Device::Cpu plus a model whose attention layer constructed the Metal SDPA op) panics with `"SDPA: Metal-only"`. Worth asserting `device.is_metal()` at SDPA construction time so the error surfaces at load.

**M3 — KV-quant CPU roundtrip on Apple Silicon** — covered under P1. Unified memory makes the copy cheap but the F32 cast and `Vec` materialization are still wasted work per step.

---

### B5. NVIDIA CUDA — the immature backend

The README is honest ("Status: not tested on real NVIDIA hardware," line 400-409). Most CUDA observations are gaps, not bugs:

**C1 — No CUDA-specific kernels exist** · **INFO** · 
The Metal path has fused SDPA, RMSNorm, RoPE, GatedSiLU, Softmax. The CUDA path has none — CUDA users get Candle's generic backends. README acknowledges this. Make sure CI keeps `--features cuda` compiling (I didn't read the workflows).

**C2 — `OXYDLLM_COMPILED_CAP` not propagated as `cfg`** · **INFO** · `build.rs:21`
Set via `cargo:rustc-env`. Downstream `#[cfg]` per arch isn't available, so kernels can't be conditionally compiled. Deliberate decision (Candle is generic), worth a comment in `build.rs`.

**C3 — Multi-device tensor placement isn't asserted** · **POTENTIAL** · 
`OXYDLLM_DEVICES` accepts a comma-separated list; the loader picks one device per model. If a future TP implementation adds cross-device tensors, the lack of `device_id` assertions in attention/KV paths will be hard to debug. Pre-emptive `debug_assert_eq!(tensor.device().location(), expected_device.location())` checks at the hot path entries would help.

---

### B6. Model loading / management

**L1 — `load_registry` silently drops malformed JSON** · **LOW** · `src/models/manager.rs:107`
```rust
serde_json::from_str(&raw).unwrap_or_default()
```
A corrupt registry → empty map → all sizes re-estimated. Fine as recovery, but should `tracing::warn!` so an operator notices.

**L2 — `MmapedSafetensors::multi(paths)` is unsafe** · **INFO** · `src/common/weights.rs:135-138`
Standard mmap caveat: "no one truncates this file while we hold it." For a server that owns its `models_dir`, fine. Document the assumption.

**L3 — `tie_word_embeddings` defaults** · **LOW** · `src/models/parsers/hf_parser.rs:101-107`
Defaults to `true` for Gemma family, `false` otherwise. Phi-3-mini sets it explicitly. The default only matters when a custom Gemma checkpoint omits the field. Consider failing instead.

**L4 — CPU GGUF expansion factor** · **LOW** · `src/models/manager.rs:316`
```rust
let corrected = if is_cpu { gpu_bytes * 2 } else { gpu_bytes };
```
The `× 2` is correct for BF16 → F32 (`disk_bytes` is BF16 file size). For GGUF Q2_K → BF16 it's ~4×; for Q8_0 → BF16 it's ~1×. First-load CPU OOM risk for very quantized GGUF models. The fix is parallel to the AWQ fix at line 300-316: read the GGUF metadata and apply the right expansion.

**L5 — `discover_models` runs on every `GET /v1/models{,/...}`** · **LOW** · `src/server/routes/handlers.rs:20, 61`
Each request filesystem-scans the entire `models_dir`. With many models this becomes the bottleneck. Cache the result with a debounce or invalidate on registry changes.

**L6 — `list_models` TOCTOU between lock release and discover** · **LOW** · `src/server/routes/handlers.rs:14-50`
Manager is locked → state copied → lock released → `discover_models` scans disk. A model added/removed in between produces inconsistent rows (e.g., a discovered model with zero `size_bytes`). Mild user-visible inconsistency, no real corruption.

---

### B7. Engine / scheduler

**E1 — `engine.rs::step()` is monolithic** · **MAINTAINABILITY** · `src/engine.rs:224-405`
~200 lines of nested prefill/decode/sampling/write-back logic. Not a bug — a refactor target.

**E2 — Sampling helpers allocate a fresh `Vec<f32>` per filter** · **LOW** (perf) · `src/sampling.rs:151-165`
`apply_min_p`, `apply_top_k`, `apply_top_p` each return a new `Vec<f32>` of vocab size. A single reusable buffer would cut allocation pressure.

**E3 — `apply_top_k` tie-break biased toward low token IDs** · **LOW** · `src/sampling.rs:312-322`
When multiple tokens tie at the top-k threshold, the tie-break iterates token IDs in ascending order. Marginal sampling bias. Random tie-break using the same `splitmix64` source as `categorical_sample` would be unbiased.

[✓] ~~**E4 — `categorical_sample` last-resort returns token 0** · **LOW** · `src/sampling.rs:373`~~
~~If all probs are zero (or NaN slipped past the guard), the function returns token 0. The combined `S1 → S8 → E4` chain is the "silent BOS storm" failure mode. Fixing S1 short-circuits it.~~

---

## C. Unwrap audit (357 non-mutex `.unwrap()` / `.expect()` calls)

After categorizing, the production unwraps fall into:

| Category | Count | Risk |
|---|---|---|
| `lock().unwrap()` on mutexes | 42 | Acceptable (poisoned-mutex panic is a process death anyway) |
| Test code (`#[cfg(test)]`, test modules) | ~200 | None |
| `LazyLock`/`OnceLock::new(NonZeroUsize::new(N).unwrap())` with constant `N > 0` | 5 | None (compile-time invariant) |
| `serde_json::to_string(&self_owned_struct).unwrap()` | ~20 (chat.rs) | Real risk if NaN/Inf ever enters a chunk struct — see **S8** |
| `scheduler.get_running(seq_id).unwrap()` and similar within `step()` | ~15 (engine.rs, scheduler) | Internal invariant; safe within a single step but fragile to refactors |
| Prometheus `register_*().expect(...)` at startup | 8 | Acceptable (fail-fast at boot) |
| Signal handler `.expect("failed to install ...")` | 2 (mod.rs:27, 37) | Acceptable (fail-fast at boot) |
| Misc internal invariants (`pop().unwrap()`, `last().unwrap()`) | ~15 | Internal invariants; verified case-by-case to be safe |
| `weight.t().expect("Linear weight must be 2D")` | 1 (linear.rs:55) | **R5** — should be `?` propagation |
| `chat_template.rs:30 env.get_template("chat").unwrap()` | 1 | Safe — template just added |

**Conclusions:**
1. The overall pattern is defensive: most production unwraps are guarded by surrounding logic.
2. The only user-facing risk concentration is the chat.rs serialization chain (S8), and it's contingent on the NaN-on-greedy gap (S1).
3. Linear's weight-shape panic (R5) is the one case that converts a malformed checkpoint into a process panic instead of a clean error.

---

## D. README

The README is in good shape — every CLI command I cross-checked exists with documented flags, every env var matches `mod.rs:88-104`, the metrics table matches `routes/metrics.rs`, the architecture list matches `arch_defaults.rs`.

**D1 — No mention of HTTP authentication absence** · **MEDIUM** · README (no Security section)
Links to S3 above. For a server marketed with systemd integration, Docker images, and Prometheus metrics, the lack of an auth section is a gap. Add a short Security subsection:
> oxydllm currently has no authentication on the HTTP API. Bind to `127.0.0.1` for single-user deployments, or place it behind a reverse proxy (nginx, Caddy, Traefik) that adds auth. Setting `OXYDLLM_API_KEY` is on the roadmap.

**D2 — "CUDA Status" + 10 Docker tags feels self-contradictory** · **LOW** · README:400-423
Line 400: "CUDA is currently a functional compatibility path, not a performance-tuned backend." Lines 412-423 then offer 10 named CUDA Docker tags. Pick one tone or add a "validation" column making explicit that all tags are unvalidated on hardware.

**D3 — `--memory-budget` unit ambiguity** · **LOW** · README:332
Documented as `<MB>` — verify against the CLI parser in `main.rs`. (I did not.)

**D4 — Tested Models list is hard to keep current** · **INFO** · 
Consider a nightly CI matrix that smoke-tests one model per architecture and updates a generated "last verified" line.

---

## E. OpenAI Chat Completions compat — `http_compat_tests.rs` coverage

I read all 711 lines and cross-checked against OpenAI's `/v1/chat/completions` spec (as of late 2025).

### What's tested well

- 4xx error envelope shape (`error.message`, `error.type`)
- 404 on unknown model id
- 400 on empty messages, missing model, `n: 0`
- 400/422 on malformed body
- Non-streaming response: `object`, `model`, `choices[0].message.role/content`, `finish_reason`, `usage` presence, `tool_calls: null` when no tools
- Streaming: role chunk, content concatenation, terminal `finish_reason`, `[DONE]`
- Tools: forced `tool_choice.type = "function"` wraps direct-JSON output, `parallel_tool_calls: false` truncates, `tool_choice.type = "allowed_tools"` filters, streaming emits header + arg chunks with proper `finish_reason: "tool_calls"`
- `/metrics` content-type, presence of expected metric names, per-model label

### Gaps vs the OpenAI spec

| Untested scenario | Expected OpenAI behavior |
|---|---|
| `n > 1` (e.g., `n: 2`) | response has 2 `choices`, each with distinct `index` |
| `tool_choice: "none"` | model should NOT call tools |
| `tool_choice: "required"` | response always has `tool_calls` set |
| `tool_choice: "auto"` (default) with tools that aren't called | content set, `tool_calls: null`, `finish_reason: "stop"` |
| `parallel_tool_calls: true` (default) | all parsed tool calls returned |
| `logprobs: true` + `top_logprobs: 3` | `choices[i].logprobs.content[k].top_logprobs[j]` populated |
| `response_format: {"type": "json_object"}` | output parseable as JSON |
| `response_format: {"type": "json_schema", strict: true}` with valid output | output matches schema |
| `response_format: {"type": "json_schema", strict: true}` with invalid output | `finish_reason: "content_filter"`, `content: null` |
| `stream_options.include_usage: true` | trailing chunk with `usage` and empty `choices` |
| `stop: ["END"]` | model stops at first match |
| Multi-turn with `{role: "tool", tool_call_id, content}` | template renders tool result properly |
| Parameter out of range (`temperature: 5`, `top_p: 2`, `top_logprobs: 100`) | 400 invalid_request_error (current code: silently accepts) |
| `max_completion_tokens` vs `max_tokens` precedence | `max_completion_tokens` wins (current code: correct, untested) |
| `seed` reproducibility (same seed, same prompt → same output) | bit-identical |
| Queue full → 429 | `error.type: "rate_limit_error"`, ideally `Retry-After` header |
| Concurrent same-model requests | proper isolation, no cross-completion bleeding |
| Method validation (`PUT /v1/chat/completions`) | 405 Method Not Allowed |
| Content-Type mismatch (`application/x-www-form-urlencoded`) | 415 Unsupported Media Type or 400 |
| Streaming error mid-stream | error chunk + `[DONE]` |
| `name` field on user/system messages | rendered into template when supported |
| `image_url` / `audio` in content | clean 400 with explanation (not implemented) |
| Unknown fields in body | silently ignored (current behavior) |
| Empty `content` on assistant turn with `tool_calls` set | accepted |

### Observations on existing tests

- **Test `allowed_tools_restricts_returned_tool_calls`** uses a nested `allowed_tools.{mode,tools}` structure that's an oxydllm-specific form. OpenAI's Responses API uses a flat `{type, mode, tools}`. The code at `chat.rs:826-868` accepts BOTH via `unwrap_or(raw_choice)` fallback at line 829 — so flat OpenAI form would work too, just isn't covered by the test.

- **Test `invalid_json_body_returns_422`** accepts both 400 and 422. Axum's default is 400 for malformed JSON; OpenAI returns 400. Test is loose-but-not-wrong.

- The scripted engine (line 66-88) emits `Token + Finish + StreamEnd` in immediate sequence. Tests covering edge timing (slow first token, mid-stream disconnect, partial chunk drop) cannot be exercised through this fixture.

### Recommended highest-value additions

In priority order, the tests most worth adding:
1. **Parameter validation**: `temperature: 5.0`, `top_p: 2.0`, `top_logprobs: 100` — would surface S4.
2. **Streaming error path**: scripted engine emits `EngineEvent::Error` mid-stream; expect both error chunk and `[DONE]`.
3. **`n > 1`** non-streaming: assert exactly N choices, each with distinct `index`.
4. **`tool_choice: "none"`** with a model output that's a JSON tool-call shape — expect plain content.
5. **`response_format: json_schema strict`** with model output that doesn't parse → `finish_reason: "content_filter"`. Would surface S7.
6. **`stream_options.include_usage: true`** — assert a trailing usage chunk.

---

## F. Recommended priority ordering

1. **S3 + D1 (HTTP auth)** — public-facing inference endpoint with no auth is the highest blast-radius issue. Either implement `OXYDLLM_API_KEY` or document the gap loudly.
2. **S1 (greedy NaN gap) + S8 (serde panic chain)** — single fix; eliminates a class of silent-bad-output and an in-stream panic.
3. **P1 (KV-quant CPU roundtrip)** — perf hit on a marketed feature. Either fix or set expectations in the README.
4. **S4 (parameter validation)** — small surface, fixes API-compat gap, surfaces S1 chain.
5. **S2 (template fallback)** — small, fixes silent quality regression for thinking-model requests.
6. **R1 (atomic registry write)** — three lines, eliminates a corruption class.
7. **A (delete CODE_REVIEW/)** — keeps future contributors from chasing fixed bugs.
8. **L4 (CPU GGUF expansion)** — first-load OOM is a bad first impression.
9. **Add the 6 highest-value http_compat_tests** — proves S4 and S7 fixes and adds streaming-error coverage.
10. Rest in any order.

---

## G. False positives — items I checked twice and chose not to include

These appear in `CODE_REVIEW/01_…` through `CODE_REVIEW/05_verified_final.md` but do not survive re-verification against current source:

1. **`Cargo.toml` `edition = "2024"` is invalid** — Rust 2024 edition was stabilized in 1.85; `rust-toolchain.toml` pins 1.94.1. The project compiles and tests pass (178 / 178). False positive.
2. **`mask.rs:8-17` causal mask formula inverted** — Walked through the affine/tanh/affine chain manually. For `col > row` (above diagonal), result is `-1e30` (masked); for `col <= row`, result is `0` (visible). Formula is correct.
3. **`loader.rs:193` UTF-8 panic on Unicode dir names** — Uses `to_ascii_lowercase`, which preserves byte length and char boundaries. Non-ASCII bytes pass through unchanged. `dir_base_lower.len() == dir_base.len()` always.
4. **`kv_quant.rs:511-513` 3-bit unpack reads garbage** — The `& 0x07` mask cuts the contribution of the next byte to exactly the 1 or 2 bits needed, and `packed.len() = (n*3).div_ceil(8)` guarantees the next-byte read is valid when `bit_idx > 5`. Enumerated `n=1..9` to confirm.
5. **`hf_parser.rs:31` `v["architectures"][0]` panic** — `serde_json::Value`'s `Index` impl returns `Value::Null` for missing keys or wrong types; `Null[0]` is `Null`; `.as_str().unwrap_or("Unknown")` handles it. No panic.
6. **`engine.rs:409` underflow on `uncached_len == 0`** — `max_cacheable = (seq_len.saturating_sub(1)) / block_size` at `engine.rs:269` caps `num_cached_tokens` to at most `seq_len - 1`, so `uncached_len >= 1` is guaranteed in the prefill path.
7. **`paged.rs:315-325` `write_staged` bounds panic** — Function trusts the caller-provided `block_id` and `offset`. Callers are internal (the block allocator). This is "internal API trusts internal invariants," not a user-triggerable bug.
8. **`attention.rs:74,77` `narrow()` without bounds** — Line 74 is inside `if … && kv_len > w`, so `kv_len - w` is valid. Line 77 is inside `if kv_len < k.dim(2)?`, so `narrow(2, 0, kv_len)` is valid.
9. **`prefix_cache.rs:117-120` block leak on `evicted.block_ids.len() < allocators.len()`** — Re-reading the constructor: `block_ids` is built one entry per allocator, length is invariant. The `if layer_idx < allocators.len()` check is defensive, never false in practice.
10. **`rope.rs:108-111` Yarn off-by-one (`dim/2 - 1`)** — This is the standard Yarn implementation (vLLM, llama.cpp, transformers all do this); the upper-frequency band is meant to be excluded from the ramp. Not a bug.

---

## H. What was *not* done in this pass

- No `cargo build --features cuda` (macOS host).
- No end-to-end run of the test suite under sanitizers (user confirms 178/178 pass on a normal run).
- No reading of `.github/workflows/` to verify CI keeps `--features cuda` green.
- No re-read of `docs/` content — user confirmed these are internal, not exposed.
- No deep audit of `arch_regression.rs` (test data) or `models/hub.rs` HuggingFace download path beyond the unwrap surface.

If you want a deeper pass on any of those, pick a target and I'll go.

---

## Appendix: All findings index

| #         | Title                                                    | File:line                        | Severity        | Category      |
| --------- | -------------------------------------------------------- | -------------------------------- | --------------- | ------------- |
| S1        | Greedy fast-path bypasses NaN logit check                | sampling.rs:50-62                | HIGH            | CORRECTNESS   |
| S2        | Template-render failure returns 200 with degraded prompt | chat.rs:1013-1035                | MEDIUM          | API-COMPAT    |
| S3        | No HTTP API authentication                               | routes/mod.rs:47-56              | MEDIUM          | SECURITY      |
| S4        | No sampling parameter range validation                   | chat.rs:1147-1376                | MEDIUM          | API-COMPAT    |
| S5        | No per-request timeout                                   | chat.rs (collect_one_completion) | MEDIUM          | RESOURCE      |
| S6        | Tools-streaming branch n>1: no `[DONE]` on error         | chat.rs:1462-1475                | LOW             | API-COMPAT    |
| S7        | Strict-mode schema only catches parseable JSON           | chat.rs:2169-2176                | LOW             | API-COMPAT    |
| S8        | serde_json::to_string unwraps on owned chunk structs     | chat.rs (20 sites)               | LOW             | FRAGILITY     |
| P1        | KV-quant CPU↔GPU roundtrip per write                     | paged.rs:374-379                 | HIGH            | PERF          |
| P2        | Fresh zero-filled contig buffers per sequence            | paged.rs:629-647                 | MEDIUM          | PERF          |
| P3        | apply_top_k: three vocab-size passes                     | sampling.rs:296-326              | LOW             | PERF          |
| P4        | Metal SDPA limited head dim coverage                     | metal_ops.rs:87                  | LOW             | PERF          |
| R1        | Non-atomic registry write                                | manager.rs:110-117               | MEDIUM          | ROBUSTNESS    |
| R2        | usize→u32 silent narrowing on slot index                 | paged.rs:591                     | LOW             | ROBUSTNESS    |
| R3        | AWQ pack factor hardcoded                                | weights.rs:262-264               | LOW             | ROBUSTNESS    |
| R4        | parse_compute_cap breaks on two-digit minor              | build.rs:67-76                   | LOW             | ROBUSTNESS    |
| R5        | Linear weight shape panic                                | linear.rs:55                     | LOW             | ROBUSTNESS    |
| R6        | CPU fallback is warning, not error                       | loader.rs:546-557                | LOW             | UX            |
| M2        | Metal SDPA cpu_fwd panics, not asserts                   | metal_ops.rs:43-53               | DESIGN          | DESIGN        |
| C2        | OXYDLLM_COMPILED_CAP not propagated as cfg               | build.rs                         | INFO            | CUDA          |
| C3        | Multi-device tensor placement not asserted               | (architecture-wide)              | INFO            | CUDA          |
| L1        | load_registry silently drops malformed JSON              | manager.rs:107                   | LOW             | OBSERVABILITY |
| L3        | tie_word_embeddings Gemma default                        | hf_parser.rs:101-107             | LOW             | CORRECTNESS   |
| L4        | CPU GGUF expansion factor                                | manager.rs:316                   | LOW             | RESOURCE      |
| L5        | discover_models runs per /v1/models request              | handlers.rs:20,61                | LOW             | PERF          |
| L6        | list_models TOCTOU                                       | handlers.rs:14-50                | LOW             | CORRECTNESS   |
| E1        | engine.rs::step() is monolithic                          | engine.rs:224-405                | MAINTAINABILITY | REFACTOR      |
| E2        | Sampling helpers allocate per call                       | sampling.rs:151-165              | LOW             | PERF          |
| E3        | top_k tie-break biased toward low IDs                    | sampling.rs:312-322              | LOW             | CORRECTNESS   |
| E4        | categorical_sample last-resort returns 0                 | sampling.rs:373                  | LOW             | CORRECTNESS   |
| D1        | README missing Security section                          | README.md                        | MEDIUM          | DOCS          |
| D2        | CUDA Status vs Docker tags inconsistency                 | README.md:400-423                | LOW             | DOCS          |
| D3        | --memory-budget unit ambiguity                           | README.md:332                    | LOW             | DOCS          |
| Test gaps | 6 high-value missing OpenAI compat tests                 | http_compat_tests.rs             | —               | COVERAGE      |
