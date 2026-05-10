<p align="center">
  <picture>
    <img src=".github/res/oxydLLM.png" width="350">
  </picture>
</p>

<br>

<p align="center">
    <a href="https://github.com/giovannifil-64/oxydllm/actions/workflows/ci.yml">
        <img src="https://github.com/giovannifil-64/oxydllm/actions/workflows/ci.yml/badge.svg?branch=main" />
    </a>
    <a href="https://github.com/giovannifil-64/oxydllm/actions/workflows/nightly.yml">
        <img src="https://github.com/giovannifil-64/oxydllm/actions/workflows/nightly.yml/badge.svg?branch=main" />
    </a>
    <a href="https://github.com/giovannifil-64/oxydllm/actions/workflows/release.yml">
        <img src="https://github.com/giovannifil-64/oxydllm/actions/workflows/release.yml/badge.svg?branch=main" />
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
- **Function calling / tool use (function tools)** — `tools`, `tool_choice` (`auto`, `required`, `none`, forced function, and `allowed_tools`), and `parallel_tool_calls`; assistant tool calls are returned as proper `tool_calls` with `finish_reason: "tool_calls"`, and streaming emits incremental tool-call deltas
- **Structured output** — `response_format` with `json_object` and `json_schema`; request-time schema validation includes strict-mode requirements plus recursive `$ref` / `$defs`, `anyOf`, enums, nested objects/arrays, and nullable unions
- Metal acceleration on Apple Silicon with fused attention, RMSNorm, RoPE, and Softmax kernels
- Paged KV cache with prefix caching for reduced redundant computation
- KV cache quantization (Lossless/Balanced/Aggressive) to reduce memory footprint by 2-4x
- Multi-model server: load several models simultaneously with LRU eviction and configurable memory budgets
- Thinking/reasoning model support with separated `reasoning_content` field
- GGUF quantized model support (Q4_K_M, Q5_0, Q8_0, and others), including sharded GGUF loading
- AWQ 4-bit quantized safetensors support (autoawq GEMM layout) with auto-detection, fused QKV/gate-up projections, and load-time dequantization
- Streaming responses via Server-Sent Events
- Model download directly from HuggingFace with interactive variant selection

## Architecture
oxydLLM is built on top of the Candle tensor library. The model layer implements a unified transformer architecture that covers most supported model families with minimal per-architecture branching. The inference engine uses paged KV allocation with a shared block pool, a prefix cache keyed on rolling block hashes, and a scheduler that handles concurrent prefill and decode across multiple sequences.

KV cache quantization uses TurboQuant with MSE-based quantization during the decode phase, reducing memory overhead without significant quality loss. Metal kernels provide fused operations for attention, normalization, and positional embeddings on Apple Silicon.

## Tested Models
Here you can find a list of models that have been tested, divided by architecture. This is ***not*** an exhaustive list of compatible models.

### LlamaForCausalLM
- `Llama-3.2-1B-Instruct`

### Mistral3ForConditionalGeneration
- `Ministral-3-3B-Instruct-2512`

### Qwen2ForCausalLM
- `Qwen2.5-1.5B-Instruct` (including Q2_K and Q4_K_M quantized variants)
- `Qwen2.5-3B-Instruct`

### Qwen3ForCausalLM
> [!NOTE]
> All Qwen3 models have been tested with and without thinking enabled.

- `Qwen3-0.6B` (including the Q8_0 quantized variant)
- `Qwen3-1.7B-Q8_0`
- `Qwen3-4B` (Q4_K_M, Q5_0, and AWQ 4-bit autoawq GEMM variants)

### GemmaForCausalLM
- `gemma-2b-it`

### Gemma2ForCausalLM
- `gemma-2-2b-it`

### Gemma3ForCausalLM
- `gemma-3-270m-it`
- `gemma-3-1b-it`

### Gemma4ForConditionalGeneration (Minor issues)
- `gemma-4-E2B-it` - Known edge cases on some checkpoints/configurations

### Phi3ForCausalLM
- `Phi-3.5-mini-instruct`

## Unsupported Model Families
The following model families are not currently supported:
- Mixture-of-Experts models (Mixtral, Deepseek-V2/V3)
- Hybrid linear-attention models (Qwen3.5)
- Multimodal inference (vision+language) is not supported yet; text-only paths from some multimodal checkpoints may work
- Encoder-only models (BERT, etc.)

## Installation
For using oxydLLM, you can either build from source or use the provided installers.

### Building from source
Clone the repository
```bash
git clone https://github.com/giovannifil-64/oxydllm
cd oxydllm
```

Build the project (requires [Rust toolchain](https://rust-lang.org/tools/install/)) with the appropriate feature base on your platform

```bash
# For Apple Silicon (Metal backend)
cargo build --release --features metal

# NVIDIA CUDA (set the target compute capability explicitly)
CUDA_COMPUTE_CAP=89  cargo build --release --features cuda  # Ada (RTX 4090/L40S)
CUDA_COMPUTE_CAP=90  cargo build --release --features cuda  # Hopper (H100/H200/GH200)
CUDA_COMPUTE_CAP=100 cargo build --release --features cuda  # Blackwell datacenter (B100/B200/DGX Spark)
CUDA_COMPUTE_CAP=103 cargo build --release --features cuda  # Blackwell Ultra (B300/GB300) — requires CUDA 12.9+
CUDA_COMPUTE_CAP=110 cargo build --release --features cuda  # Thor / Jetson Thor — requires CUDA 13.0+
CUDA_COMPUTE_CAP=120 cargo build --release --features cuda  # Blackwell consumer (RTX 50xx)
```

> [!NOTE]
> `CUDA_COMPUTE_CAP` is consumed by Candle's CUDA kernel build scripts (`candle-kernels`), not by a direct `oxydllm` build flag. If not set, Candle tries to auto-detect compute capability from `nvidia-smi`.

Run the server

```bash
cargo run --release start
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
> OXYDLLM_CUDA_TARGET=ada|hopper|blackwell|blackwell-ultra|blackwell-consumer curl -fsSL https://github.com/giovannifil-64/oxydllm/raw/main/install.sh | sh
> # arm64
> OXYDLLM_CUDA_TARGET=hopper|blackwell|blackwell-ultra|thor curl -fsSL https://github.com/giovannifil-64/oxydllm/raw/main/install.sh | sh
> ```

If you prefer to manually download the installer, you can find the latest releases on GitHub:

#### macOS (Apple Silicon)
> [!IMPORTANT]
> Intel-based Macs are not supported
- `oxydllm-macos-arm64`

#### Linux (CUDA)

##### x86_64
- `oxydllm-linux-x86_64-cuda-ada.tar.gz` for Ada (sm_89, compute 8.9)
- `oxydllm-linux-x86_64-cuda-hopper.tar.gz` for Hopper (sm_90, compute 9.x)
- `oxydllm-linux-x86_64-cuda-blackwell.tar.gz` for Blackwell datacenter (sm_100, compute 10.x)
- `oxydllm-linux-x86_64-cuda-blackwell-ultra.tar.gz` for Blackwell Ultra (sm_103, compute 10.3+)
- `oxydllm-linux-x86_64-cuda-blackwell-consumer.tar.gz` for Blackwell consumer (sm_120, compute 12.x, RTX 50xx)

##### arm64 (GH200 / DGX Spark / GB300 / Jetson Thor)
- `oxydllm-linux-arm64-cuda-hopper.tar.gz` for Hopper (GH200, sm_90)
- `oxydllm-linux-arm64-cuda-blackwell.tar.gz` for Blackwell datacenter (DGX Spark/B200, sm_100)
- `oxydllm-linux-arm64-cuda-blackwell-ultra.tar.gz` for Blackwell Ultra (GB300, sm_103)
- `oxydllm-linux-arm64-cuda-thor.tar.gz` for Thor / Jetson Thor (sm_110)

## Usage
Download a model
```bash
oxydllm pull Qwen/Qwen3-0.6B
```

For GGUF repos, the variant selection prompt shows which quantizations are already downloaded (marked ✓) and only numbers the ones that aren't.

List locally available models
```bash
oxydllm list
```

Displays a table with NAME, ARCHITECTURE, and SIZE for each model found in the models directory, sorted alphabetically.

You can also estimate memory requirements before downloading
```bash
oxydllm estimate Qwen/Qwen3-4B-GGUF --context-len 8192 --num-sequences 4
```

`estimate` and `run` both accept partial model names — `oxydllm run Qwen3-4B` resolves to the first matching local model.

Interactive chat
```bash
oxydllm run Qwen3-0.6B
```

Remove a model
```bash
oxydllm rm Qwen3-0.6B
```

> [!IMPORTANT]
> Models are loaded on demand when the first request for that model arrives.

### API
The fastest way to interact with the server is through the OpenAI-compatible API. The following endpoints are available:
- `GET /health`
- `GET /v1/models`
- `GET /v1/models/running`
- `GET /v1/models/{model_id}`
- `POST /v1/chat/completions`

```bash
curl http://localhost:11313/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Thinking mode (for models that support it):

```bash
curl http://localhost:11313/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B-Q4_K_M",
    "messages": [{"role": "user", "content": "Explain quantum entanglement"}],
    "enable_thinking": true
  }'
```

## Server Options
```
--port <PORT>          Listen port (default: 11313)
--models-dir <DIR>     Models directory (default: ~/.oxydllm/models)
--keep-alive <SECS>    Idle timeout before model eviction (default: 900)
--memory-budget <MB>   Maximum VRAM for loaded models
--max-context-len <N>  KV cache context length per sequence (default: 4096)
--devices <IDS>        Comma-separated CUDA device indices
--kv-quant <MODE>      KV cache quantization: off, lossless, balanced, aggressive
--qjl-quantization     Enable Stage-2 QJL key residual quantization
--require-gpu          Fail startup if no GPU device is available (default: disabled)
```

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
`--models-dir`, `--devices`, `--max-context-len`, `--kv-quant`, `--qjl-quantization`, `--require-gpu`.

## Known Limitations and Work in Progress
- **GGUF compatibility**: Support is broad, but some architecture/quantization combinations may still fail depending on checkpoint format.
- **Thinking mode is template-dependent**: `enable_thinking` is applied only when the tokenizer chat template supports it.
- **Byte-level tokenizers**: Streaming decode uses incremental buffering; occasional model-specific Unicode artifacts can still appear.
- **Tool / schema adherence is model-dependent**: the OpenAI-compatible request fields, response shapes, and streaming semantics are implemented server-side, but local models can still ignore tool instructions or emit invalid JSON / tool arguments.
- **Only function tools are implemented**: OpenAI custom tools are not supported on `/v1/chat/completions` yet.
- **Gemma4 edge cases**: Some checkpoints may require architecture-specific tuning.
- **Metal softcap SDPA policy**: The Metal SDPA path with attention softcap is currently hard-disabled in runtime (no experimental toggle) and falls back to the standard attention path.
- **CUDA optimization**: Support exists but is not optimized for production use.
- **AWQ runtime memory footprint**: AWQ checkpoints currently dequantize to fp16/bf16 at load time, so resident weight memory matches an equivalent fp16 model rather than the on-disk 4-bit footprint. Inference throughput matches fp16 thanks to fused QKV/gate-up projections.

## CUDA Status
CUDA is currently a functional compatibility path, not a performance-tuned backend.

> [!WARNING]
> The CUDA path has not been tested on real NVIDIA hardware. Builds are CI-verified (compile + CPU tests only). Runtime correctness and performance on actual GPUs are unvalidated.

- Build/runtime support is available via `--features cuda`.
- Core model execution works, but the CUDA path currently relies mostly on Candle generic kernels.
- Metal has additional fused kernels in this project (attention/RMSNorm/RoPE/softmax), so CUDA throughput will be lower than specialized CUDA stacks.
- Contributions and testing reports from NVIDIA hardware owners are welcome.

### Official CUDA Docker tags
| Tag | Compute capability | Platform | Target |
|---|---:|---|---|
| `cuda-ada` | 89 | amd64 | Ada Lovelace (RTX 40xx, L40S) |
| `cuda-hopper` | 90 | amd64 | Hopper (H100, H200) |
| `cuda-blackwell` | 100 | amd64 | Blackwell datacenter (B100, B200) |
| `cuda-blackwell-ultra` | 103 | amd64 | Blackwell Ultra (B300, GB300) |
| `cuda-blackwell-consumer` | 120 | amd64 | Blackwell consumer (RTX 50xx) |
| `cuda-hopper-arm64` | 90 | arm64 | Hopper (GH200 Grace Hopper) |
| `cuda-blackwell-arm64` | 100 | arm64 | Blackwell datacenter (DGX Spark, B200) |
| `cuda-blackwell-ultra-arm64` | 103 | arm64 | Blackwell Ultra (GB300 NVL72) |
| `cuda-thor-arm64` | 110 | arm64 | Thor / Jetson Thor |

- `latest` and `cuda` point to `cuda-ada` (stable default — widest x86_64 compatibility).
- `nightly` and `nightly-cuda` point to nightly `cuda-ada`.
- Cross-generation SASS is **not** compatible: a Hopper binary will not run on Blackwell and vice versa. Pick the tag that matches your GPU.
- No CUDA target has been validated on real hardware — all targets carry the same caveat.


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
