# Changelog

All notable changes to this project will be documented in this file.

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