# Changelog

All notable changes to this project will be documented in this file.

## 0.0.0-alpha.12.0.1

- Fixed nondeterministic garbage output on Metal under concurrent serving — including the intermittent `Qwen3-4B-Instruct-2507-FP8` gibberish — by pinning candle's Metal command-buffer pool to a single buffer; throughput is unchanged.
- Fixed every Docker image build (CPU and all CUDA architectures) failing on a missing compile-time `OXYDLLM_BUILD_TS`.
- Silenced a `dead_code` warning emitted on the non-Metal CI test builds.
- Removed a full-vocabulary F32 materialization from the greedy per-token NaN guard.

### Reliability and Correctness
- **Metal command-buffer pool corruption under concurrency** (`src/main.rs`): `candle-metal-kernels` 0.10.2 spreads GPU work across a pool of command buffers (`CANDLE_METAL_COMMAND_POOL_SIZE`, default 5) that is not safe for concurrent encoding. When more than one thread touches the device at once — the server's tokio workers, the model-load/warmup thread, and off-thread `Tensor` drops freeing `MTLBuffer`s — operations collide and the output is silently corrupted (gibberish or `NaN`). This is why `Qwen3-4B-Instruct-2507-FP8` decoded coherently on some loads and produced garbage on others, but it affected any model served under concurrency. `main()` now sets `CANDLE_METAL_COMMAND_POOL_SIZE=1` at startup, before any device or thread is created, serializing onto one command buffer. GPU work is already serialized per device by the internal GPU lock, so there is no throughput cost — measured 53.32 vs 53.30 tok/s on Qwen3-0.6B (pool size 1 vs 5). Set the environment variable explicitly to override. With the fix the full `stress_baseline.py` run passes 25/25 models on the coherence check (the FP8 model previously failed).

### Performance and Efficiency
- **Greedy decode NaN guard** (`src/sampling.rs`): the `temperature == 0` fast path no longer promotes the entire logit row (~152 K values on the Qwen vocab) to F32 just to check for NaN. It reduces in the native dtype and casts only the one-element sum, so the argmax token id is the only value copied off the GPU per step.

### CI / Infra
- **Docker / CUDA image builds fixed** (`src/main.rs`, `build.rs`): every Docker target (CPU and all CUDA architectures) failed with `error: environment variable OXYDLLM_BUILD_TS not defined at compile time`. The Dockerfile's dummy-build layer — build dependencies with a stub `main`, then rebuild against the real sources — does not reliably reapply the build script's compile env to the final compile. `env!("OXYDLLM_BUILD_TS")` is now `option_env!(…)`: it is optional build metadata and the code already defaulted to `0`. The `dist_build` cfg gating `oxydllm update` / `uninstall` was likewise replaced by an `OXYDLLM_DIST_BUILD` compile env (emitted by `build.rs`, read via `option_env!`), removing the custom cfg, its `cargo::rustc-check-cfg` declaration, and any need for an `#[allow(unexpected_cfgs)]`.
- **`dead_code` warning on non-Metal builds** (`src/common/awq.rs`): `QuantWeight::runtime_size_bytes` and `to_device` are Metal-only but were gated `#[cfg(any(feature = "metal", test))]`, so they compiled — unused — into the CUDA/CPU test binaries and emitted a `dead_code` warning on those CI jobs. Narrowed to `#[cfg(feature = "metal")]`.

### Tests
- **Metal command-buffer race reproducer** (`src/common/linear.rs::metal_pool_ordering_race_repro`, env-gated): several threads run the same deterministic candle-matmul + custom Metal `rms_norm` chain on one shared device; identical results across threads prove correctness, while divergence or `NaN` exposes the pool race. Documents the bug and confirms `CANDLE_METAL_COMMAND_POOL_SIZE=1` fixes it.

**Full Changelog**: https://github.com/giovannifil-64/oxydllm/compare/0.0.0-alpha.12...0.0.0-alpha.12.0.1

## 0.0.0-alpha.12

- Mixture-of-Experts support shipped end-to-end: `Qwen3MoeForCausalLM` and `OlmoeForCausalLM` are routed through a new `MoeFeedForward` module with top-k router, hybrid sparse/naive dispatch chosen per call from `n_tokens` vs `top_k`, and runtime-verified output coherence on `allenai/OLMoE-1B-7B-0924-Instruct`.
- AWQ checkpoints now stay packed in memory on Metal: a fused W4A16 GEMV kernel (`PackedQuantLinear`) cuts resident weight memory ~3× (Qwen3-4B-AWQ: 7.5 → 2.5 GB) while keeping decode throughput on par with the previous fp16 dequant-at-load path.
- W8A16 GEMV + dequant wrappers built off the same bit-parametric AWQ template; `hf_parser` now accepts AWQ with `bits ∈ {4, 8}` and dispatches the right kernel at runtime.
- GPTQ checkpoints (4-bit and 8-bit, `desc_act=false`) load and run end-to-end with a dedicated Metal resident kernel family (`gptq{4,8}_gemv_*` + `dequantize_gptq{4,8}_*`); `Qwen/Qwen3-{0.6B,1.7B}-GPTQ-Int8` reach 85+ tok/s decode on Apple Silicon.
- Bf16-aware GGUF Metal fast path now covers ten quant types — `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` — with a fused `mul_mm` for prefill that dequantizes inline and removes the previous 2× residency cost on the M=1 fast path.
- GGUF loader rewritten on top of `memmap2`: `Qwen3-4B-Q4_K_M` cold load drops from 9.4 s to 2.7 s (3.47×) with bit-identical outputs on the verified checkpoint set.
- FP8 dequant on Metal is now numerically correct on block-wise-scaled checkpoints (`Qwen3-4B-Instruct-2507-FP8`): the weight × scale_inv multiply is promoted to F32 to avoid the BF16 mantissa loss that produced coherent-but-wrong outputs through 36 layers of rescaling.
- HTTP API hardened with optional Bearer authentication, per-request wall-clock timeout, sampling parameter range validation, and queue-full backpressure.
- Batched scheduler now derives `max_num_seqs` automatically from the available KV cache and exposes a bounded request queue via `--max-queued-requests`.
- Prometheus metrics endpoint (`/metrics`) exposes TTFT, decode throughput, queue depth, prefix-cache hit ratio, model weight memory, and KV cache allocation.
- New CLI commands: `oxydllm update [--pre|--nightly]` for self-update via GitHub releases and `oxydllm uninstall [--purge]` for clean removal of binary + OS service.
- `CUDA_COMPUTE_CAP` is validated at compile time against a curated list of supported architectures; the loader also warns when the binary's compiled cap is below the hardware cap.
- GPU is now required by default; pass `--allow-cpu` (or `OXYDLLM_ALLOW_CPU=1`) to opt into the CPU fallback path.
- New Metal fused kernels (`GatedGeLU-Tanh`, `GeLU-Tanh-Mul`) for Gemma-family activations, plus a NaN-propagation fix in SDPA full mode for sequences past 16 tokens.
- Per-layer KV contig buffer pool (`ContigBufferPool`) recycles tensors across sequences, removing the per-fresh-prefill zero-fill allocation.
- A focused code review pass identified 34 findings and closed 31 of them across server, performance, robustness, model loading, and engine.

### New Features
- **Mixture-of-Experts (MoE)** (`src/common/moe.rs`): `MoeFeedForward { router, experts, top_k, activation, norm_topk }` with top-k softmax routing (`arg_sort_last_dim` + `gather` + `scatter_add`). The transformer block dispatches to a dense `FeedForward` or `MoeFeedForward` based on `BlockConfig.moe`. Two equivalent forward paths are picked per call:
    - *Naive* (`n_tokens ≤ top_k`): dense gate via `scatter_add`, run each non-empty expert on the full `x_flat`. Decode-friendly (no per-expert `index_select` / `index_add` overhead).
    - *Sparse* (`n_tokens > top_k`): group token indices per expert on the CPU (small: `n_tokens × top_k` ints), then `index_select` + FFN + `index_add` per non-empty expert. Per-expert compute drops from `n_tokens` to `~n_tokens × top_k / num_experts` — up to ~8× on OLMoE-1B-7B for long prefill.
- **OLMoE / Qwen3-MoE support** (`src/models/arch_defaults.rs`, `src/models/parsers/hf_parser.rs`): `OlmoeForCausalLM` and `Qwen3MoeForCausalLM` are now in `arch_defaults`. `hf_parser` reads `num_experts` / `num_local_experts`, `num_experts_per_tok`, and `norm_topk_prob` from `config.json`. Mixtral / DeepSeek-V2/V3 remain unsupported (different tensor naming or MLA attention).
- **`QkNormLayout` auto-detection** (`src/common/attention.rs`): `q_norm` / `k_norm` weight shape is inspected at load; `[head_dim]` selects per-head normalisation (Qwen3, Gemma3), `[n_heads × head_dim]` selects flat normalisation applied to `[B, T, q_dim]` before reshape (OLMoE). Forward branches on the detected layout; no config flag added because the two cases are mutually exclusive at the shape level.
- **AWQ W4A16 resident path on Metal** (`src/common/quant_kernels.metal`, `src/common/metal_ops.rs`, `src/common/linear.rs`): packed 4-bit weights stay resident; decode runs a fused split-K GEMV kernel (`w4a16_gemv_*`), prefill/M>1 uses an inline `dequantize_w4_*` plus matmul. Qwen3-4B-AWQ resident weight memory drops from ~7.5 GB (fp16 dequant-at-load) to ~2.5 GB.
- **W8A16 (8-bit AWQ) kernel wrappers**: `w8a16_gemv_{f16,bf16}` and `dequantize_w8_{f16,bf16}` are bit-parametric instantiations of the same template as the 4-bit path. `PackedQuantLinear` carries a `bits` field and dispatches to the right kernel; `validate_quantization_config` accepts `awq` with `bits ∈ {4, 8}`.
- **GPTQ end-to-end support**: `validate_quantization_config` now accepts `gptq` with `bits ∈ {4, 8}` and `desc_act=false`; `weights.rs::try_get_gptq` loads `qweight` (packed along `in_features`), `qzeros`, `scales`, and optional `g_idx`; `dequantize_gptq` implements the PlusOne zero-point convention. A `QuantWeight` struct unifies AWQ and GPTQ behind a single `AnyLinear::from_quant` factory with a `pack_dim` field that disambiguates the layouts.
- **GPTQ resident Metal kernel** (`src/common/quant_kernels.metal`): a dedicated `gptq_gemv_impl<T, BITS>` template handles the GPTQ `[in/pack_factor, out]` layout (1 thread per output column instead of AWQ's 1 thread per `pack_factor` columns) plus `gptq_dequantize_impl<T, BITS>`. Wrappers `gptq{4,8}_gemv_{f16,bf16}` + `dequantize_gptq{4,8}_{f16,bf16}` are dispatched by `PackedQuantLinear` based on `pack_dim`. Decode on `Qwen3-0.6B-GPTQ-Int8` improves from ~50 tok/s (dequant-at-load) to ~89 tok/s with the resident path.
- **GGUF kernel suite bf16-aware** (`src/common/quant_kernels.metal`): ten quant types — `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` — each with a `gguf_*_gemv_bf16` (M=1 decode) and a fused `gguf_*_mul_mm_bf16` (M>1 prefill, dequant inline). `GgufFastPath` in `linear.rs::QLinear` activates the fast path on Metal + BF16; `QMatMul` fallback is dropped when active so weights are not held twice (no 2× memory cost). Coverage spans ~98% of the GGUF mainstream parc.
- **GGUF mmap loader** (`src/common/gguf_weights.rs`): `GgufWeights::load` and `load_shards` open the file with `memmap2::Mmap`, parse the header from a `Cursor`, and build `QTensor` from slices of the mmap (zero copy in user space; one memcpy into `MTLBuffer`). `parallelise_tensor_load` materialises tensors in parallel via `rayon::par_iter`. End-to-end load + first inference on Qwen3-4B-Q4_K_M drops from 9.4 s to 2.7 s (3.47×).
- **Stress baseline script** (`scripts/stress_baseline.py`): per-model cold/warm load time, TTFT, median decode TPS, RSS, and deterministic coherence check (`"Tokyo is the capital of"` → must contain `"japan"`). Writes CSV + JSON per run under `test-results/stress-baseline/` for regression tracking.
- **HTTP API authentication** (`src/server/routes/mod.rs`): optional `OXYDLLM_API_KEY` / `--api-key` enables a Bearer-token middleware on `/v1/*` and `/metrics`; `X-API-Key` is accepted as an alternative header; `/health` stays unauthenticated for liveness probes; mismatches return `401 invalid_api_key` via a constant-time comparison.
- **Per-request wall-clock timeout** (`--request-timeout` / `OXYDLLM_REQUEST_TIMEOUT`, default 300 s, `0` disables): non-streaming responses return `408 Request Timeout`; streaming responses emit a final `request_timeout` error chunk followed by `[DONE]` via a watchdog that races the SSE task. Dropping the inner future closes the engine channel so the underlying sequence is aborted at the next step boundary.
- **Sampling parameter range validation** (`validate_sampling_params` in `chat.rs`): out-of-range values for `temperature`, `top_p`, `min_p`, `frequency_penalty`, `presence_penalty`, `top_logprobs`, `repetition_penalty`, `n`, `max_tokens`, and `max_completion_tokens` return `400 invalid_request_error` instead of silently degrading the sampler. Also closes the user-triggerable NaN-logits path through `repetition_penalty: 0`.
- **Batched scheduler with dynamic concurrency**: at model load, `max_num_seqs` is computed as `total_kv_blocks / ceil(max_context_len / block_size)`, capped at 256, and logged. Overridable via `--max-num-seqs` / `OXYDLLM_MAX_NUM_SEQS`. Requests are held in a bounded `tokio::sync::mpsc` channel of capacity `--max-queued-requests` (default 200); once full, new arrivals receive HTTP 429 immediately.
- **Prometheus metrics endpoint** (`/metrics`, `src/server/routes/metrics.rs`): `oxydllm_ttft_milliseconds` (histogram), `oxydllm_tokens_per_second` (histogram), `oxydllm_requests_total` (counter by status), `oxydllm_queue_depth` (gauge), `oxydllm_prefix_cache_requests_total` (counter by `hit`/`miss`), `oxydllm_model_weights_bytes` (gauge), `oxydllm_kv_cache_allocated_bytes` (gauge), `oxydllm_vram_used_bytes` (gauge). Every request is also assigned a UUID-v4 `request_id` that propagates through structured logs; setting `LOG_FORMAT=json` switches the logger to JSON-per-line for Loki / Datadog / `jq`.
- **`oxydllm update [--pre|--nightly]`**: queries the GitHub releases API for the latest stable / pre-release / nightly build, compares against the version compiled into the binary (or `OXYDLLM_BUILD_TS` for nightly), and re-runs `install.sh` with the appropriate channel set. Source builds print an error and exit cleanly.
- **`oxydllm uninstall [--purge]`**: stops and removes the OS service (launchd on macOS, systemd on Linux), then self-removes the binary. `--purge` also removes `~/.oxydllm/`, including all downloaded models and configuration data. Always shows a confirmation prompt before mutating anything.
- **CUDA compute capability validation** (`build.rs`): `CUDA_COMPUTE_CAP` is matched against a curated `SUPPORTED_COMPUTE_CAPS` list (Ada Lovelace 8.9, Hopper 9.0, Blackwell 10.0/10.3, Jetson GB 11.0, Blackwell Desktop 12.0, Blackwell Edge 12.1). Out-of-list or below-minimum values fail the build with an actionable message. At runtime, `loader::select_device_at` warns when the binary's compiled cap is below the actual GPU's hardware cap.
- **GPU required by default**: `select_device_at` returns an error if no GPU device is available unless `--allow-cpu` is passed (or `OXYDLLM_ALLOW_CPU=1` is set). Avoids silently serving requests at CPU-speed throughput.
- **`ContigBufferPool`** (`src/common/paged.rs`): each `BlockAllocator` owns a small pool of retired KV contig buffers (default 4 per layer). `PagedKvCache::clear` retires the buffer instead of dropping it, and `PagedKvCache::append` takes the smallest-fit buffer from the pool on grow or fresh-start. The "stale" tail of a reused buffer is never observed: all reads narrow to `contig_len`.
- **Metal `GatedGeLU-Tanh` and `GeLU-Tanh-Mul` kernels** (`src/common/metal_ops.rs`): single-dispatch fused activations for Gemma-family models (GeLU tanh-approximation with multiplicative gate). Mirror the existing `gated_silu` / `silu_mul` shapes for Llama / Qwen.
- **FP8 quantization in validation logic**: `validate_quantization_config` accepts `float8_e4m3fn` and routes the load through the FP8 path established in alpha.11.
- **Discovery cache with atomic snapshot**: `ModelManager::discovered_with_registry()` returns `(Vec<DiscoveredModel>, BTreeMap)` from a single mutex acquisition with a 5 s TTL on the filesystem scan. Removes both the TOCTOU between manager-lock release and `discover_models`, and the per-request disk scan on `GET /v1/models{,/:id}`.
- **GGUF CPU expansion factor**: `manager.rs` reads GGUF metadata via `estimate::gguf_cpu_expansion` to compute the dequantized memory footprint (`Q4_K_M` ~7×, `Q8_0` ~4×, etc.) instead of using a flat 2× for all formats. Eliminates first-load CPU OOM on very quantized GGUF models.
- **Discovered models metadata**: `DiscoveredModel` carries `created_at` and `owned_by` (HF namespace) fields exposed by `/v1/models` and `/v1/models/{id}`.
- **Tools-streaming `[DONE]` sentinel**: the tools n ≥ 1 streaming branch now emits `[DONE]` after error / `JoinError` events, matching the non-tools paths.
- **Strict-mode JSON schema catches non-parseable JSON**: a `response_format: json_schema` request with `strict: true` whose model output is not parseable JSON now ends with `finish_reason: "content_filter"` and `content: null` instead of slipping through as `stop`.

### Performance and Efficiency
- **AWQ W4A16 fused matmul on Metal**: Qwen3-4B-AWQ resident weight memory drops from ~7.5 GB to ~2.5 GB; QKV and gate+up are fused at load (matmul count per layer = 2, same as fp16); single-shot decode throughput catches up to Ollama's Q4_K_M reference within noise.
- **GPTQ Metal resident path**: Qwen3-0.6B-GPTQ-Int8 decode improves from ~50 tok/s (dequant-at-load) to ~89 tok/s; Qwen3-1.7B-GPTQ-Int8 reaches ~42 tok/s.
- **GGUF Metal fast path**: bf16-aware GEMV port of llama.cpp / candle templates removes the bf16↔f32 host-side casts. Measured deltas (steady-state, M=1 decode):
    - Qwen2.5-1.5B-Q4_K_M: 81.5 → 92.5 tok/s (+13.5%)
    - Qwen2.5-1.5B-Q3_K_M: ~89 tok/s (≈8% faster than Q4_K_M same model)
    - Qwen3-4B-Q4_K_M: 28.8 → 31.5 tok/s (+9.4%)
    - Qwen3-4B-Q5_0: 25.6 → 27.5 tok/s (+7.4%)
    - Qwen3-1.7B-Q8_0: ~43 tok/s
  All quant types converge at ~50% memory-bandwidth utilisation, on par with the host-side cast cost they eliminated.
- **GGUF mmap loader**: Qwen3-4B-Q4_K_M cold load 9.4 s → 2.7 s (3.47×); steady-state warm load matches mmap-only baselines. Zero copy in user space, one memcpy into `MTLBuffer`.
- **GGUF fused `mul_mm` (M>1 prefill)**: dequantizes inline in the matmul kernel; previously the prefill path materialised a transient bf16 weight matrix per call (up to ~192 MB for `gate_up_proj` on Qwen3-4B). The previous M=1-only fast path doubled resident memory; the fused kernel removes that — Qwen3-4B-Q4_K_M RSS measured at 2.48 GB (was ~4.6 GB).
- **MoE hybrid dispatch**: on OLMoE-1B-7B-0924-Instruct the hybrid is ~25% faster on TTFT (256-word prompt: 9.8 s → 7.3 s) and ~60% faster on decode (6.5 → 10.6 tok/s) compared to the naive ALL-experts variant.
- **AWQ load-time parallelism**: safetensors materialisation runs through `rayon::par_iter`, paired with a dtype short-circuit when the target matches the on-disk type. gemma-4 (2011 tensors): cold load 14 s → 9.2 s; Qwen3-0.6B: 1.5 s → 0.26 s.
- **In-place sampling filters**: `apply_min_p` / `apply_top_k` / `apply_top_p` mutate `&mut [f32]` instead of returning a fresh `Vec<f32>` per token. Auxiliary buffers for `select_nth_unstable` and the indices sort are thread-local and amortize to zero allocation after the first sample — saves roughly 1.8 MB of allocation/free traffic per decoded token on Qwen-3's 152 K vocab.
- **`ContigBufferPool` reuse**: sequences on the same allocator skip the `Tensor::zeros` for the contig KV buffer after the first one. On Apple Silicon with 32 layers × 4-buffer pool, this avoids hundreds of MB of zero-fill at the start of each fresh prefill under steady-state churn.
- **`apply_top_k` single-pass filtering**: the old three-pass (filter, count, renormalize) is now one in-place pass that tracks `sum` inline.
- **KV-quant dtype reorder**: `to_device(Cpu)` is applied before `to_dtype(F32)` when reading quantized K/V back to host, so the dtype cast runs on CPU and is skipped entirely when the tensor is already F32.
- **`discover_models` cache**: the per-request filesystem scan is replaced by a 5 s TTL cache shared between `/v1/models` and `/v1/models/{id}`.

### Reliability and Correctness
- **FP8 dequant precision fix** (`src/common/weights.rs::apply_weight_scale_inv`): on Metal, FP8 weights are dequanted at load (F8E4M3 → F32 → BF16) because Metal has no FP8 compute. The previous code did the `weight × scale_inv` multiply in BF16, which compounded mantissa rounding error across 36 layers of block-wise rescaling on `Qwen3-4B-Instruct-2507-FP8`, producing coherent-but-wrong outputs ("Tokyo is not the capital of Japan") or, at low `max_tokens`, token degeneracy. The multiply is now done in F32 with a single cast back to the target dtype.
- **MoE qk_norm layout auto-detection**: OLMoE applies `q_norm` / `k_norm` to the flat `[B, T, hidden]` tensor before reshape into heads (weight shape `[hidden]`), whereas Qwen3 / Gemma3 apply per-head on `[B, H, T, head_dim]` (weight shape `[head_dim]`). The variance differs, so the layouts are not interchangeable. `Attention::load` now inspects the loaded weight's `elem_count` and branches at forward time; no config flag because the two shapes are disjoint.
- **MoE routing correctness**: top-k routing uses `arg_sort_last_dim(false)` + `narrow` + `gather` to extract the top-k probabilities, optionally renormalises (controlled by `norm_topk_prob` from `config.json`), and dispatches via a hybrid sparse/naive path. Three unit tests cover the degenerate case (num_experts=1, top_k=1 must equal a single dense FFN), the routing-mass shape, and the top_k > num_experts rejection at load.
- **Greedy NaN guard**: the greedy fast-path in `sample()` now sums logits and bails on NaN before `argmax`. Closes the "silent BOS storm" failure mode where NaN logits made `argmax` deterministically return token 0.
- **Sampler-side NaN guard on the non-greedy path**: scans `logits_vec` for NaN after dtype conversion and bails before any filter touches the tensor. Combined with the greedy guard, eliminates the NaN → sampler → serde-panic chain that could abort a streaming connection mid-response.
- **Template-render failure surfaces**: a failed `apply_chat_template` no longer falls back to a plain-text render with a 200 OK; the error now propagates as a structured `500` so thinking-model requests can't be silently downgraded to non-thinking.
- **Metal SDPA NaN propagation in full mode** (`src/common/metal_ops.rs`): the full path (sequences > 16 tokens) had a mask-handling path that let NaN values leak into the attention output. The fix masks before softmax in a way that keeps the output NaN-free even when the input mask is `-inf`.
- **Model warmup hardening**: the per-model warmup step runs through more code paths to prevent first-request NaN logits on Metal, with JIT-compilation optimizations.
- **Tied-embeddings load-time check**: `load_standard_safetensors` warns when `tie_word_embeddings = true` is selected but the safetensors file also contains an explicit `lm_head.weight` — the file's `lm_head` would otherwise be silently ignored, producing wrong logits. The default for Gemma family stays `true` because the official Gemma checkpoints omit the field.
- **Randomized top-k tie-break**: when multiple tokens tie at the top-k threshold, the kept set is now picked uniformly at random via `splitmix64` priorities (seedable for reproducibility) instead of iterating in ascending token-ID order. Removes a small but real sampling bias toward low-ID tokens during ties.
- **Atomic registry write**: `save_registry` writes to `.oxydllm_registry.json.tmp` and renames atomically. Multi-process scenarios (e.g. the server plus an interactive `oxydllm run` sharing `--models-dir`) can no longer produce a torn or empty registry.
- **Malformed-JSON recovery in `load_registry`**: a corrupt registry now emits `tracing::warn!` (with path and parse error) before falling back to an empty map, instead of silently dropping the file.
- **`Linear::new` returns `Result`**: a malformed checkpoint with a non-2D weight tensor in a `Linear` layer now produces an anyhow error at load instead of a runtime panic.
- **`parse_compute_cap` rejects two-digit minor versions**: the `major*10+minor` flat encoding becomes ambiguous past `.9`, so values like `"12.10"` now fail explicitly instead of silently collapsing to `130`.
- **`u32::try_from` on slot indices**: replaces silent `as u32` narrowing on `block_id * block_size` in `PagedKvCache` with explicit overflow checking.
- **AWQ `pack_factor` helper**: the hard-coded `8` for 4-bit packing in `weights.rs` runtime-size accounting is now a `pack_factor(bits)` helper that future bit-widths (AWQ-3 / AWQ-8) can extend without silently mis-reporting memory.
- **Metal `sdpa()` device check at call site**: `metal_ops::sdpa()` rejects non-Metal inputs at the entry point with an actionable error instead of letting candle dispatch fall through to `cpu_fwd`'s opaque "Metal-only" bail.
- **Route parameter syntax fix**: the `/v1/models/{*model_id}` capture now correctly handles `user/model` ids with embedded slashes.
- **Assorted sampling / KV cache / loader hardening**: defensive cleanups across the inference path, including explicit error propagation from `AnyLinear::from_weight_with_scale_inv` callers in `load_standard_safetensors`.

### Refactors and Maintainability
- **`QuantWeight` generalisation** (`src/common/awq.rs`): the original `AwqRawTensors` struct grew into a unified `QuantWeight { bits, group_size, pack_dim, pack_order, zero_point, qweight, qzeros: Option, scales, g_idx: Option }` that carries enough metadata to disambiguate AWQ and GPTQ at runtime. `AwqRawTensors` stays as a type alias so existing call sites keep working; the new constructors are `QuantWeight::new_awq(bits, …)` and `QuantWeight::new_gptq(bits, sym, …)`. `AnyLinear::from_quant` dispatches based on the metadata.
- **`QuantScheme` plumbed model-wide**: `ModelWeights` carries `Option<QuantScheme>` (set by the loader from `hf_parser`); `try_get_quant(prefix)` returns the right `QuantWeight` shape without each caller needing to know the format. `concat_awq_along_out` checks `bits` consistency across fused parts.
- **`FeedForwardLayer` enum** in `block.rs` selects between dense `FeedForward` and `MoeFeedForward` polymorphically; the transformer block does not branch outside this enum's `forward()`.
- **`GgufFastQuant` enum** in `metal_ops.rs` carries each supported quant type's block size, GEMV kernel name, fused `mul_mm` kernel name, op-name, and dispatch geometry. Adding a new quant type is now a single match arm per method plus the Metal kernels.
- **`engine::step()` decomposition**: the ~280-line monolithic function is now a ~95-line orchestrator. Five private helpers own the phases (`plan_prefill_inputs`, `build_batch_input`, `run_forward_pass`, `sample_prefill_outputs`, `sample_decode_outputs`). `StopRules` and `PrefixRegistry` bundles drop the per-function argument count enough to remove every `#[allow(clippy::too_many_arguments)]` from the codebase.
- **Debug-only device asserts at hot-path entries**: `block::run_transformer_layers_batch` cross-checks `token_ids` ↔ `position_ids`; `attention::forward_batch_optional_rope` cross-checks `x` ↔ `position_ids` ↔ `mask`. Pre-empts the cross-device misroute that will become possible once tensor-parallel inference lands; no release-build cost.
- **Build script comment for `OXYDLLM_COMPILED_CAP`**: documents why the value is emitted as `cargo:rustc-env` rather than `cargo:rustc-cfg`, and what to add if per-arch CUDA kernels are ever introduced.
- **Simplified model directory resolution**: legacy fallback branches in `resolve_model_path` removed; discovery logic consolidated.
- **Outdated review documents removed**: the old `CODE_REVIEW/` snapshot was misleading after the alpha.12 fixes landed; replaced by a single living `CODE_REVIEW_2026-05-15.md` at the repo root.

### Tests
- **MoE unit tests** (`src/common/moe::tests`): `moe_single_expert_topk1_matches_single_ffn` (degeneracy check), `moe_topk_routing_uses_only_topk_experts` (output finiteness + convex-combination scale), and `moe_rejects_invalid_topk` (load-time guard).
- **GPTQ parity tests** (`src/common/awq::tests` + `src/common/metal_ops::fused_kernel_parity_tests`): `dequantize_gptq_int4_matches_reference` and `dequantize_gptq_int8_matches_reference` (CPU path against a hand-built reference), `gptq{4,8}_gemv_{bf16,f16}_matches_reference` and `gptq8_dequantize_bf16_matches_reference` (Metal kernels against the CPU reference).
- **W8A16 parity tests**: `w8a16_gemv_{bf16,f16}_g{64,128}_matches_reference` and `dequantize_w8_bf16_matches_reference` exercise the bit-parametric template instantiated with `BITS=8` against synthetic data (no real AWQ-8bit checkpoint locally).
- **GGUF Q3_K coverage**: Q3_K kernel + dispatch added with the same naive-tiled `mul_mm` pattern as Q4_K / Q5_K, plus a Qwen2.5-1.5B-Q3_K_M smoke run.
- **`scripts/stress_baseline.py` baseline run** committed: 18 local models pass the deterministic coherence prompt, including the FP8 fix and OLMoE-1B-7B.
- **26 new `http_compat_tests`** for the auth / timeout / validation / schema work: scripted-engine fixture covering auth on/off (Bearer + `X-API-Key`), `/health` exemption, `/metrics` + `/v1/models` gating, every sampling field at its range edge accepted plus every OOB value rejected, stuck-engine `408`, streaming error + `[DONE]`, fast-engine not falsely timed out, and strict / non-strict schema paths under both parseable-but-invalid and non-JSON output.
- **Three coverage gaps from the May review's §E**: streaming error mid-stream emits the error chunk + `[DONE]`; `n > 1` non-streaming returns exactly N choices with distinct indices; `stream_options.include_usage: true` emits a trailing usage chunk with empty `choices` before `[DONE]`.
- **`ContigBufferPool` unit tests**: recycle across sequences, evict smallest on overflow, smallest-fit selection, growth retires old buffer.
- **Engine cache & prefix-cache tests**: `KvFakeModel` test fixture plus a unit test asserting prefix cache hits on a repeated prompt.
- **Sampling tie-break tests**: unbiased over many seeds (within ±15 % over 4 000 trials with all-equal probs) and reproducibility — `apply_top_k` with the same seed/step yields the same kept set.
- **`hf_parser` `tie_word_embeddings` tests**: Gemma config without the field defaults to `true`; non-Gemma defaults to `false`; explicit value is respected on both.
- **Discovery cache tests**: repeated reads within TTL don't rescan; explicit invalidation forces refresh; snapshot stays consistent with the registry across mutations.

### Documentation
- **README — Security section**: documents `OXYDLLM_API_KEY` setup, the Bearer / `X-API-Key` flow, the `/health` exemption, and reverse-proxy recommendations.
- **README — CUDA Status**: added a `[!IMPORTANT]` callout above the Docker tag table to make the "no tag is validated on physical NVIDIA hardware" disclaimer impossible to miss.
- **README — Compute capability table**: updated to match `build.rs::SUPPORTED_COMPUTE_CAPS` and to document the compile-time validation.
- **Public documentation site** (`docs.html`): added a Security section mirroring the README; added a cache-TTL note under `GET /v1/models`; added `OXYDLLM_ALLOW_CPU` to the environment-variables table; fixed two anchor labels pointing at the "Advanced Topics → KV cache quantization" section.
- **Code review**: `CODE_REVIEW_2026-05-15.md` is a single living document tracking the 34 findings from the May 15 review pass; 31 are closed with commit hashes, the remaining four are all `INFO` / `LOW`.

### CI / Infra
- **`oxydllm update` / `oxydllm uninstall` inherit `install.sh`'s service-management story end to end**: the `update` command re-runs `install.sh` for the appropriate channel (stable / pre / nightly); the `uninstall` command stops the launchd agent or systemd unit before removing the binary.

### Dependencies
- `memmap2 = "0.9.10"` added for the GGUF zero-copy loader.

**Full Changelog**: https://github.com/giovannifil-64/oxydllm/compare/0.0.0-alpha.11...0.0.0-alpha.12

## 0.0.0-alpha.11

- Added AWQ quantization support across attention, FFN, and `lm_head`, with QKV and gate+up fused at load time.
- New Metal-accelerated fused kernels for `gated_silu`, `silu_mul`, and logit/attention `softcap`.
- Added bias loading for QKV and output projections in both safetensors and GGUF paths.
- Reworked streaming decode to always emit from a full-sequence canonical decode, eliminating BPE drift and double-spacing artifacts.
- Introduced chunked prefill for large batched prompts to reduce peak memory during the prefill phase.

### New Features
- AWQ weight loading (`src/common/awq.rs`): `AwqRawTensors`, `dequantize_awq`, `concat_awq_along_out`, and `AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]`. Auto-detected via `ModelWeights::try_get_awq(prefix)` for q/k/v/o, gate/up/down, and `lm_head`. Dequantization happens at load time, so no Metal kernel changes are required at inference time.
- AWQ attention loader: when `qkv_proj` is detected as AWQ, q/k/v are concatenated along the output dimension and loaded as a single fused projection (matmul count per layer = 2, identical to the fp16 path). Mixed-format layers are rejected with a descriptive error.
- AWQ FFN loader: `gate_proj` + `up_proj` AWQ tensors are fused into a single packed projection; `down_proj` is loaded as a separate AWQ linear.
- `AnyLinear::from_awq` constructor for building linear layers directly from AWQ raw tensors with optional bias.
- Quantization-config validation in the HF parser (`hf_parser.rs`) ensures only AWQ configurations supported by the runtime are accepted.
- Bias support for attention projections: `q_proj.bias`, `k_proj.bias`, `v_proj.bias`, and `o_proj.bias` are now loaded in both the safetensors and GGUF (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`, `attn_qkv.bias`, `attn_output.bias`) paths. Fused QKV biases are concatenated to match the fused weight layout.
- Chunked prefill (`pick_prefill_chunk_split` in `engine.rs`): batches with two or more prefill sequences and a combined uncached length ≥ 1024 tokens are split into two halves and forwarded sequentially, sharing the same KV cache slices. Single-sequence prefills and small batches keep the original single-pass path.
- `flush_caches` extracted as a reusable helper in `block.rs` so the engine can flush pending KV writes after each prefill chunk.

### Performance and Efficiency
- **Metal `gated_silu_fused`**: single-kernel `silu(x[..H]) * x[H..]` for packed gate+up activations (Phi-3 / Phi-3.5 and AWQ-fused FFN paths), avoiding two separate kernel launches and intermediate buffers.
- **Metal `silu_mul_fused`**: single-kernel `silu(gate) * up` for the standard split gate/up FFN layout.
- **Metal `softcap_fused`**: single-kernel `tanh(x / cap) * cap` used by Gemma2/3 attention softcap and final logit softcap; replaces the three-op `(x / cap).tanh() * cap` chain.
- **Attention scratch buffer**: added `out_buf: RefCell<Option<Tensor>>` to `Attention` for reusing the output projection tensor across decode steps.
- **KV-quant flush dtype reorder**: `to_device(Cpu)` is now applied before `to_dtype(F32)` when reading quantized K/V back to host memory, so the (potentially expensive) dtype conversion runs on CPU instead of issuing an extra GPU cast.
- AWQ runtime size estimation in `estimate.rs` excludes pre-dequantization scratch tensors so the reported in-memory footprint matches the actual fp16/bf16 weight size after dequantization.

### Reliability and Correctness
- **Streaming decode rewrite** (`src/server/routes/engine_loop.rs`, `src/main.rs`): removed the per-token fast path that re-inserted a leading space when the vocab piece started with `▁` or `Ġ`. The heuristic could double-space (`"assistente"` + `" "` + `"virtuale"` → `"assistente  virtuale"`) and produce character drift that later swallowed user-visible characters. Streaming now always decodes the full accumulated id list and emits only the suffix beyond `decoded_len`, with trailing `U+FFFD` held back until the continuation token arrives. The `piece` field on the decode cache is no longer needed and was removed.
- Output of fused QKV biases requires either all of q/k/v biases or none; mismatched bias presence falls back to a separate (non-fused) projection layout to avoid silent shape errors.
- AWQ projections validate that out-features (`scales.dim(1)`) match `n_heads * head_dim` (q) and `n_kv_heads * head_dim` (k/v) before fusing.

### Refactors and Maintainability
- Engine prefill body extracted into a small closure so the chunked and single-pass paths share cache-slice handling and `flush_caches` is called uniformly at the end.
- `engine_loop.rs` split the streaming decode into a pure `emit_suffix(full, decoded_len)` helper that is independently unit-tested.

### Tests
- Streaming decode unit tests (`streaming_decode_tests` in `engine_loop.rs`): cumulative-emission matches canonical decode, no double-spacing across `Ġ`-prefixed tokens, no character loss across partial-UTF-8 token boundaries, idempotency when `decoded_len == full.len()`, defensive clamping past end of `full`, and char-boundary clamping inside multibyte sequences.
- AWQ unit tests covering load round-trips and runtime-size accounting.
- Engine chunk-split tests (`pick_prefill_chunk_split`): small-batch skip, balanced two-sequence split, dominant-first-sequence skip, and three-sequence balancing.
- Architecture regression coverage extended with additional `run_prefill` slice setups; existing slice initialization simplified.

### Documentation
- README updated to clarify model-compatibility notes.

### CI / Infra
- **Release notes now include the `CHANGELOG.md` section for the tag**: previously the release workflow set `body:` to the Installation block alone, fully overwriting the release body and leaving the changelog content visible only in the repository file. A new `Extract changelog section for this tag` step in `release.yml` slices the relevant block out of `CHANGELOG.md` (from `## <version>` to the next `## ` heading) and prepends it to the body, with the Installation/checksums block following and the auto-generated `**Full Changelog**` link appended by `softprops/action-gh-release` at the end.
- **Heading match is literal, not regex**: the awk extractor matches the version heading via `index()` rather than interpolating the tag into a regex. This makes characters like `.`, `+` (semver build metadata), and `-` behave as plain text and removes the need to escape future tag formats.
- **Prefix-collision guard**: the heading match also requires either an exact length match or a trailing whitespace character, so a tag `1.2` cannot accidentally match a heading `## 1.2.3`.
- **Inline `**Full Changelog**` and `---` separators stripped from extracted sections**: the manual `**Full Changelog**` line at the bottom of each changelog section is dropped during extraction so it doesn't duplicate the auto-generated comparison link, and standalone `---` separators between historical sections are removed for the same reason.
- **Missing-entry fallback**: if a tag is pushed without a matching `## <version>` in `CHANGELOG.md`, the release proceeds with a placeholder body and a GitHub Actions `::warning::` instead of failing, so a forgotten changelog entry never blocks a hotfix.

**Full Changelog**: https://github.com/giovannifil-64/oxydllm/compare/0.0.0-alpha.10.0.1...0.0.0-alpha.11

## 0.0.0-alpha.10.0.1

- Fixed release workflow stalling indefinitely on tag push due to missing repository context and insufficient permissions.
- Replaced deprecated `rustsec/audit-check` with `taiki-e/install-action` + `cargo audit`.
- Migrated all CI and build workflow cache jobs from `actions/cache` to `Swatinem/rust-cache`.
- Restricted CI cache writes to `main` branch only, reducing total cache storage from ~9.2 GB to ~2.6 GB.
- Added path filters to CI so it only triggers on code changes, not on docs or workflow-only commits.

### CI / Infra
- **Release workflow — repository context**: `verify-ci` was calling `gh run list` without `--repo`, causing it to fail silently in the ephemeral runner environment (no git checkout in that job). Added `--repo ${{ github.repository }}`.
- **Release workflow — permissions**: The workflow-level `permissions` block (`contents: write`, `packages: write`) implicitly set `actions: read` to `none`. `gh run list` was failing with HTTP 403, silently caught by `2>/dev/null || echo "[]"`, making every poll return `status=<none>`. Fixed by adding `permissions: actions: read` at the `verify-ci` job level.
- **Release workflow — polling timeout**: Reduced polling attempts from 40 to 20 (10 minutes). CI completes well within this window under normal conditions.
- **Security audit**: Replaced deprecated `rustsec/audit-check@v2` (required `checks: write` and is no longer maintained) with `taiki-e/install-action@v2` + `cargo audit`. Installs a prebuilt binary in ~5 s with no extra permissions needed.
- **CI cache**: Replaced all `actions/cache@v5` blocks in `ci.yml` with `Swatinem/rust-cache@v2`, which automatically prunes `incremental/` and `.fingerprint/` directories before saving. CUDA cache size reduced from ~490 MB to ~265 MB per architecture. Added `save-if: ${{ github.ref == 'refs/heads/main' }}` — PR runs restore from existing caches but do not write new entries. Total cache storage reduced from ~9.2 GB to ~2.6 GB.
- **Build cache**: Replaced all `actions/cache@v5` blocks in `build.yml` with `Swatinem/rust-cache@v2`. Each native binary build job uses a distinct `prefix-key` to prevent cache collisions with CI jobs. No `save-if` restriction since build jobs are only triggered by nightly cron and release tags, never by PRs.
- **CI path filters**: Added `paths` filters to both `push` and `pull_request` triggers. CI now only runs when `src/**`, `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `install.sh`, or `.github/workflows/ci.yml` change. Commits that only touch docs, images, or other workflow files no longer trigger a full CI run.

**Full Changelog**: https://github.com/giovannifil-64/oxydllm/compare/0.0.0-alpha.10...0.0.0-alpha.10.0.1

## 0.0.0-alpha.10

- **Project renamed from `rllm` to `oxydllm`** — binary, data directory, registry file, and env vars all updated.
- Added Phi-3 / Phi-3.5 support (safetensors + GGUF), including fused weight projection and LongRoPE scaling.
- Added `Mistral3ForConditionalGeneration` architecture support.
- Introduced OpenAI-compatible function calling and structured output (JSON Schema validation).
- Added FP8 weight loading with runtime dequantization and per-tensor scale handling.
- Built out a full CI/CD pipeline: CUDA multi-arch builds, ARM64, multi-platform Docker images, and architecture regression tests.
- New CLI commands: `oxydllm list` and `oxydllm version`.
- Integrated `tracing` for structured logging across the codebase.

### New Features
- Project-wide rename: binary `oxydllm`, data dir `~/.oxydllm/`, registry `.oxydllm_registry.json`, env var `OXYDLLM_DEVICES`.
- Phi-3 / Phi-3.5 model support: fused `qkv_proj` / `gate_up_proj` weight handling in both safetensors and GGUF; `build_llama_tokenizer` for GGUF tokenizer type; LongRoPE (`RopeScaling::LongRoPE`) for Phi-3.5.
- `Mistral3ForConditionalGeneration` added to the supported architecture list; `text_config` nesting in HF parser handled for multimodal configs.
- OpenAI-compatible function calling: tool definitions, tool-call detection, `ToolCallDelta` streaming, `finish_reason: "tool_calls"`.
- Structured output with JSON Schema validation (`type`, `required`, `additionalProperties`, `properties`, `items`, `enum`); invalid output yields `finish_reason: "content_filter"`.
- FP8 weight loading with per-tensor `_scale_inv` / `scale` dequantization at load time.
- Hadamard transform in KV quantization with fallback for non-power-of-two `head_dim`.
- Alternating sliding-window attention support at the layer level.
- Per-device GPU locking mechanism replacing the global lock for finer-grained concurrency control.
- `oxydllm list` command: shows locally available models with size, architecture, and last-used date.
- `oxydllm version` command.
- Graceful server shutdown with configurable timeout and OS signal handling.
- Abort-sequence API: `abort_sequence` in engine, scheduler, and routes for cancelling in-flight requests.

### Performance and Efficiency
- FP8 runtime dequantization avoids storing full-precision weight copies, reducing peak memory at load time.
- `apply_qk_with_positions` in RoPE for optimised per-position tensor handling.
- KV quantization Hadamard transform improves quantization quality for large head dims.
- `GateUpProjection::Packed` variant for pre-fused gate+up tensors (Phi-3 / Phi-3.5) avoids splitting overhead.

### Reliability and Correctness
- Fixed Gemma2 arch defaults: FFN pre/post norms were inadvertently disabled.
- Fixed RoPE `dims4` handling on the Metal feature flag path.
- Added `require-gpu` guard and hardened attention/causal-mask paths against out-of-bounds indexing.
- Fixed case-insensitive model path resolution and improved FP8 tensor key matching.
- Fixed leading-whitespace stripping in token decoding for both server streaming and interactive mode.
- Fixed Cargo release profile settings that were causing oversized debug builds.
- Causal mask functions now accept an explicit `DType` to avoid implicit promotions.

### Refactors and Maintainability
- `routes.rs` split into focused sub-modules for improved readability.
- Device key representation migrated to `DeviceLocation` enum.
- `QkvProjection` now uses `AnyLinear` internally; `GateUpProjection` extended with `Packed` and `Simple` variants.
- Registry management switched from `HashMap` to `BTreeMap` for deterministic ordering.
- Removed unused `user` field from `ChatCompletionRequest`.
- `tracing` integrated for structured, levelled logging across engine, scheduler, and server.

### Tests
- Unit tests for `Attention` and `RotaryEmbedding`.
- Unit tests for `ModelManager` and tokenizer (error handling, encoding round-trips).
- Architecture regression tests for `StandardTransformer` (CPU, run in pre-push hook and CI).

### CI / Infra
- Full GitHub Actions pipeline: CPU, CUDA (multi-arch: Ampere, Ada, Hopper, Blackwell Ultra), ARM64, macOS.
- Multi-platform Docker images for CPU with manifest-list support.
- Nightly build workflow with GHCR image cleanup for untagged images.
- Architecture regression test step gated before release publishing.
- Docker fallback: rebuild without cache on image pull failure.
- Binary stripping for macOS and Linux release artifacts.
- Rust toolchain version pinned in `rust-toolchain.toml` and read by all workflows.

### Dependencies
- `candle-core` / `candle-metal-kernels` updated to 0.10.2.
- `candle` / `cudarc` updated to 0.10.1 / 0.19.4.
- Added `fastrand` and `tempfile`.
- `action-gh-release` upgraded to v3 in nightly and release workflows.
- General dependency updates (`Cargo.lock` bump).

**Full Changelog**: https://github.com/giovannifil-64/oxydllm/compare/0.0.0-alpha.9...0.0.0-alpha.10

---

## 0.0.0-alpha.9

- Added Gemma4 support with stronger per-layer transformer configuration.
- Introduced KV cache quantization, plus QJL quantization for key residuals.
- Expanded OpenAI API compatibility with missing endpoints, fields, response objects, and error formats.
- Added system fingerprint generation for chat completion model identification.
- Improved sampling controls with `logprobs`, `top_logprobs`, `logit_bias`, and repetition window support.

### New Features
- Gemma4 architecture support and related model-loading/config upgrades.
- KV cache quantization path for reduced memory usage.
- QJL key-residual quantization support in the KV pool (`--qjl-quantization`).
- Repetition-window control for improved anti-repetition behavior.
- Extended sampling outputs to return token logprobs and top-logprobs.
- System fingerprint in chat completion responses.
- Broader OpenAI-compatible API surface and schema-aligned responses.

### Performance and Efficiency
- Quantized KV pool handling for lower memory footprint.
- Deferred-write and allocator updates for quantized cache paths.
- Separate key/value quantization size handling for tighter memory control.
- End-to-end propagation of quantization settings through loader/manager/scheduler flow.

### Reliability and Correctness
- Improved OpenAI-style error response formatting and route behavior.
- Better tokenizer handling for special tokens and chat templates.
- Stronger parser/config handling for advanced per-layer model settings.

### Refactors and Maintainability
- Removed unused `bytes_per_head` from `KvQuantizer`.
- Internal cleanup around sampling output structures and KV quantization flow.

### Dependencies
- Updated `windows-sys` dependency.

**Full Changelog**: https://github.com/giovannifil-64/rllm/compare/0.0.0-alpha.8...0.0.0-alpha.9

---

## 0.0.0-alpha.8

- Added sliding-window support and improved normalization handling for model execution.
- Introduced Metal-accelerated ops for RMSNorm, Softmax, and RoPE, with SDPA logic refactoring.
- Expanded RopeScaling support (including additional YaRN parameters) and updated parsing.
- Added abort capabilities in engine/scheduler flows for running sequence control.
- Improved model lifecycle management with model removal and registry handling improvements.

### New Features
- Sliding-window attention and related cache/model-path improvements.
- Abort functionality in engine and scheduler paths.
- Completion token tracking in engine events.
- Optional bias support in attention-related linear projection.
- Support for known unsupported architectures in defaults/parsing, with better surfacing.
- Better message truncation behavior in interactive mode.
- Additional file-type support in model size estimation.

### Performance and Efficiency
- Metal kernel usage for key transformer primitives (RMSNorm, Softmax, RoPE).
- Attention and paged KV cache optimizations for tensor handling and memory efficiency.
- Ensured tensor contiguity before critical ops in attention/cache paths.
- Simplified attention path by removing unnecessary padding logic.
- Feed-forward path optimized via GateUpProjection enum restructuring.

### Reliability and Correctness
- Improved error handling across model loading, chat template application, engine loop, and registry save flow.
- Added abort mechanism for consecutive engine errors.
- Enforced max_tokens limit in chat completions.
- Corrected architecture display for Qwen2 and Qwen3 in GGUF discovery.
- Improved transformer layer validation logic.

### Refactors and Maintainability
- Core module maintainability refactors across attention/block/mask/prefix-cache/sampling/routes.
- Simplified token decoding logic in interactive and request enqueue flows.

### Dependencies
- Removed unused rayon dependency.
- Updated Candle package source/version in Cargo.toml and Cargo.lock.

---

## 0.0.0-alpha.7

### New Features
- **Batch Processing:** Implemented native support for batch processing in Attention and Transformer blocks, optimizing concurrent inference.
- **Reasoning Capabilities:** Added `enable_thinking` support in the chat template and tokenizer for enhanced reasoning (disabled by default).
- **Architecture Enhancements:** Expanded the architecture configuration to include options for sliding window and RoPE (Rotary Position Embedding) scaling.
- **Template Engine:** Added support for Jinja2 template rendering.

### Model and Hardware Support
- **New Models:** Added support for `LlamaForCausalLM` models and updated the HuggingFace parser to support `Llama-3` and `Mistral3` architectures.
- **CUDA Support:** Added support for CUDA device selection in server and model management.
- **Metal Acceleration:** Implemented Metal-accelerated SDPA (Scaled Dot-Product Attention) with integrated kernel support for improved performance on macOS.
- **Global GPU Lock:** Implemented a global GPU lock for cross-model serialization to prevent contention during inference.

### Memory Management and Thread Safety
- **Thread Safety:** Replaced `Rc` and `RefCell` with `Arc` and `Mutex` for thread-safe memory management across allocators.
- **Caching Mechanisms:** Implemented Prefix Cache and enhanced the block allocator with reference counting.
- **Memory Budgeting:** Introduced `GlobalKvBudget` for memory management and added strict memory budget enforcement during the model loading process.
- **LRU Cache:** Added LRU cache support and integrated it into the transformer model forward implementations.
- **Size Estimation:** Enhanced model loading with accurate in-memory size estimation and reporting.

### Performance and Optimizations
- **Parallel Processing:** Added `rayon` and `rustc-hash` dependencies to optimize tensor loading with parallel processing.
- **SDPA Caching:** Added thread-local caching for causal masks and log SDPA fallback.
- **GGUF Enhancements:** Optimized GGUF file handling and improved readability.
- **Telemetry and Tracking:** Added timing logs for model warmup and enhanced request tracking. Changed the metric `first_token_sent` to `first_token_at` for better timing tracking.

### Refactoring and Bug Fixes
- **Unified Architecture:** Refactored transformer models to unify architecture handling, streamline the forward pass, and enhance configuration management.
- **Token Handling Fixes:** Ensured default EOS token IDs are properly included when parsing HuggingFace configurations, and adjusted the default `max_tokens` to consume the remaining tokens without a minimum cap.
- **Layer Refactoring:** Updated Attention and FeedForward structures to support optional rotary embeddings and removed the activation dependency. Refactored input handling for feedforward layers in TransformerBlock.
- **Codebase Cleanup:** Conducted an extensive cleanup of the engine and scheduler components by removing dead code, unused fields, and duplicate implementations.
- **Project Structure:** Reorganized the project structure to make navigation easier. 

---

## 0.0.0-alpha.6

### Server
- Added HTTP inference server (`rllm start`) with OpenAI-compatible `/v1/chat/completions` endpoint
- Streaming responses via Server-Sent Events (SSE)
- Model auto-loading on first request; idle models evicted after configurable keep-alive timeout
- New endpoints: `GET /v1/models`, `GET /v1/models/running`, `GET /health`
- Per-request `keep_alive` override in the chat completions payload
- Optional `--memory-budget <MB>` flag: LRU eviction when total loaded model size exceeds the budget
- Model registry persisted to `.rllm_registry.json` (tracks size, architecture, last used)

### Model pulling
- New `rllm pull <user/model>` command to download models from HuggingFace
- Supports `--token` / `HF_TOKEN` env var for gated models
- Progress bar with per-file download speed and size
- `--name` flag to save under a custom folder name; `--force` to overwrite

### CLI
- Replaced single-shot inference mode with `rllm run <model-name>` for interactive multi-turn chat
- Unified `--models-dir` option across all subcommands
- Improved error messages and `--help` output

### Engine
- `finish_reason` field added to completed sequences (`stop` or `length`)
- EOS token no longer emitted as a generated token

### Internals
- Async model manager with concurrent loading and waiter queuing
- `kv_block_multiplier` exposed on `load_batch_model` for tuning KV cache size
- Upgraded `tokenizers` to 0.22.2; added `axum 0.8`, `tokio 1.49`, `reqwest 0.13`

**Full Changelog**: https://github.com/giovannifil-64/rllm/compare/0.0.0-alpha.5...0.0.0-alpha.6

---

## 0.0.0-alpha.5

### Engine Module
A new `Engine` struct has been introduced as the main entry point for running inference. It wraps the scheduler and model, exposing a clean API:

- **`add_request()`** — submit a prompt with sampling parameters and a token budget.
- **`step()`** — run one scheduling + inference step, returning newly generated tokens and any completed sequences.
- **`run_to_completion()`** — convenience method that drives the engine until all queued requests finish.
- **`has_pending_work()`** — query whether there is still work in flight.

### Scheduler
A new `Scheduler` module manages request lifecycle and memory:

- **Waiting → Running → Finished** state machine per sequence.
- **Prefill / Decode phase tracking** — sequences begin in prefill mode (full prompt processed at once) then transition to decode (one token per step).
- **Capacity limits** — `max_num_sequences` and `max_tokens_per_step` caps are enforced each step.
- **Preemption under memory pressure** — when KV-cache blocks are exhausted, running sequences are evicted back to the waiting queue and recomputed later.
- **Block conservation** — KV-cache blocks are released when a sequence is retired, returning them to the shared pool.

### Batch Model Trait
A new `BatchModel` trait separates single-sequence inference from batched, cache-managed inference:

- `forward_with_cache()` accepts an externally-owned `&mut [PagedKvCache]`, enabling per-sequence cache management.
- Exposes model metadata: `num_layers`, `n_kv_heads`, `head_dim`, `dtype`, `allocators`.

### `--engine` CLI Flag
Pass `--engine` to run inference through the new engine pipeline instead of the legacy `generate()` path. Output is streamed token-by-token.

#### Paged KV Cache Improvements
- `BlockAllocator::num_total_blocks()` — inspect total block capacity.
- `PagedKvCache::num_blocks_used()` / `num_tokens_cached()` — observability helpers.
- Several previously private types (`BlockTable`, `SharedBlockAllocator`) are now `pub`.

### Bug Fixes / Internal Changes

- `Qwen3::load` now accepts a `kv_block_multiplier` parameter; the engine uses `2×` the default block count to support multiple concurrent sequences.
- `Model::forward` on `Qwen3` uses `mem::take` to avoid a double-borrow on `self.caches`.

**Full Changelog**: https://github.com/giovannifil-64/rllm/compare/0.0.0-alpha.4...0.0.0-alpha.5

---

## 0.0.0-alpha.4

### Paged KV Cache

The KV cache has been rewritten to use a paged memory management system, inspired by vLLM's PagedAttention:
- Introduced `BlockAllocator`, a pool-based memory manager that pre-allocates fixed-size blocks of KV memory, avoiding dynamic tensor concatenation on every decode step.
- Introduced `PagedKvCache`, a per-layer cache backed by `BlockAllocator`, using a block table to track allocated slots and gather live KV entries efficiently via index selection.
- Default block size is **16 tokens** (`DEFAULT_BLOCK_SIZE`).
- Memory exhaustion now returns a descriptive error instead of panicking.
- The `KvCache` module (`src/model/common/kv_cache.rs`) has been removed. All references across `attention.rs`, `block.rs`, and `qwen3/model.rs` have been updated to use `PagedKvCache`.
- `Qwen3::load` now requires a `DType` parameter to correctly initialize the typed KV pool tensors.
- The number of allocated blocks is derived from `max_position_embeddings` and `DEFAULT_BLOCK_SIZE` at model load time.

**Full Changelog**: https://github.com/giovannifil-64/rllm/compare/0.0.0-alpha.3...0.0.0-alpha.4

---

## 0.0.0-alpha.3

### KV Cache

Dramatically faster autoregressive generation via key-value caching.
- Introduced a `KvCache` structure that accumulates past key and value tensors across decoding steps, eliminating redundant recomputation of the full sequence at each step.
- Each transformer layer now holds its own dedicated cache instance, correctly reset before each new generation.
- The generation loop now processes the full prompt in a single forward pass, then feeds only the latest token at each subsequent step.
- The `Model` trait and `generate` function updated to require mutable access, reflecting stateful inference.

**Full Changelog**: https://github.com/giovannifil-64/rllm/compare/0.0.0-alpha.2...0.0.0-alpha.3

---

## 0.0.0-alpha.2

### Sampling & CLI Improvements

Configurable text generation with advanced sampling strategies.
- Replaced greedy decoding with a flexible sampling pipeline supporting temperature scaling, Top-K, Top-P (nucleus), Min-P, and repetition penalty.
- Extended the CLI with optional flags: `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`.
- Added unit tests for all sampling methods (greedy, temperature, Top-K, repetition penalty, Min-P).
- Improved argument parsing to handle flags in any order alongside positional arguments.

**Full Changelog**: https://github.com/giovannifil-64/rllm/compare/0.0.0-alpha.1...0.0.0-alpha.2

---

## 0.0.0-alpha.1

### Initial Release

First working prototype of rllm with Qwen3 support.
- Implemented the core model architecture including attention, feed-forward networks, RMS normalization, rotary positional embeddings (RoPE), and causal masking.
- Added support for loading Qwen3 models from safetensors weight files (single file or sharded via index).
- Introduced a tokenizer wrapper for encoding/decoding text using `tokenizers`.
- Implemented greedy decoding for text generation.
- Added automatic device selection: CUDA → Metal → CPU fallback.
- Basic CLI: `rllm <model-dir> <prompt>`.

**Full Changelog**: https://github.com/giovannifil-64/rllm/commits/0.0.0-alpha.1