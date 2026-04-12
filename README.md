<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/res/rllm_dark.png" width="250">
    <source media="(prefers-color-scheme: light)" srcset=".github/res/rllm_light.png" width="250">
    <img src=".github/res/rllm_white.png" width="250">
  </picture>
</p>

<br>

A rust-based inference engine for large language models.

> [!IMPORTANT]
> This project is under development and not yet ready for production use. At the moment it only supports text input/output and a limited set of models.

> [!NOTE]
> For transparency, the engine has been developed with the assistant of Claude. The code has been reviewed and edited, but may still contain inaccuracies, suboptimal implementations, or other kind of issues not yet identified.
> The engine has been tested primarily on Apple Silicon, so Metal support is more mature than CUDA. Contributions to improve NVIDIA GPU support are welcome.
> GGUF support is available, but compatibility still depends on architecture and quantization variant.

## Features

- OpenAI-compatible chat completions endpoint (`/v1/chat/completions`)
- Metal acceleration on Apple Silicon with fused attention, RMSNorm, RoPE, and Softmax kernels
- Paged KV cache with prefix caching for reduced redundant computation
- KV cache quantization (Lossless/Balanced/Aggressive) to reduce memory footprint by 2-4x
- Multi-model server: load several models simultaneously with LRU eviction and configurable memory budgets
- Thinking/reasoning model support with separated `reasoning_content` field
- GGUF quantized model support (Q4_K_M, Q5_0, Q8_0, and others), including sharded GGUF loading
- Streaming responses via Server-Sent Events
- Model download directly from HuggingFace with interactive variant selection

## Architecture

rLLM is built on the Candle tensor library. The model layer implements a unified transformer architecture that covers most supported model families with minimal per-architecture branching. The inference engine uses paged KV allocation with a shared block pool, a prefix cache keyed on rolling block hashes, and a scheduler that handles concurrent prefill and decode across multiple sequences.

KV cache quantization uses TurboQuant with MSE-based quantization during the decode phase, reducing memory overhead without significant quality loss. Metal kernels provide fused operations for attention, normalization, and positional embeddings on Apple Silicon.

## Tested Models
Here you can find a list of models that have been tested with rLLM, divided by architecture. Status indicates production readiness. This is not an exhaustive list of compatible models.

### LlamaForCausalLM
- `Llama-3.2-1B-Instruct`

### Mistral3ForConditionalGeneration
- `Ministral-3-3B-Instruct-2512`

### Qwen2ForCausalLM
- `Qwen2.5-1.5B-Instruct`

### Qwen3ForCausalLM

> [!NOTE]
> All Qwen3 models have been tested with and without thinking mode.

- `Qwen3-0.6B`
- `Qwen3-0.6B-Q8_0`
- `Qwen3-1.7B-Q8_0`
- `Qwen3-4B-Q4_K_M`
- `Qwen3-4B-Q5_0`

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
- `Phi-3-mini-4k-instruct-gguf`
- `Phi-3.5-mini-instruct`

## Unsupported Model Families
The following model families are not currently supported:
- Mixture-of-Experts models (Mixtral, Deepseek-V2/V3)
- Hybrid linear-attention models (Qwen3.5)
- Multimodal inference (vision+language) is not supported yet; text-only paths from some multimodal checkpoints may work
- Encoder-only models (BERT, etc.)

## Installation

Clone the repository
```bash
git clone https://github.com/giovannifil-64/rllm.git
cd rllm
```

Build the project (requires [Rust toolchain](https://rust-lang.org/tools/install/))

```bash
cargo build --release --features metal   # Apple Silicon
cargo build --release --features cuda    # NVIDIA
```

Enable local pre-commit checks (recommended for contributors)

```bash
chmod +x .githooks/pre-commit
git config core.hooksPath .githooks
```

The pre-commit hook runs formatting and strict clippy checks before each commit.

## Usage

Start the server

```bash
rllm start
```
> By default, the server listens on port 11313. You can change this with the `--port` option.

Download a model

```bash
rllm pull Qwen/Qwen3-0.6B
```

You can also estimate memory requirements

```bash
rllm estimate Qwen/Qwen3-4B-GGUF --context-len 8192 --num-sequences 4
```

Interactive chat

```bash
rllm run Qwen3-0.6B
```

Remove a model

```bash
rllm rm Qwen3-0.6B
```

## API

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
--port <PORT>             Listen port (default: 11313)
--models-dir <DIR>        Models directory (default: ~/.rllm/models)
--keep-alive <SECS>       Idle timeout before model eviction (default: 900)
--memory-budget <MB>      Maximum VRAM for loaded models
--max-context-len <N>     KV cache context length per sequence (default: 4096)
--devices <IDS>           Comma-separated CUDA device indices
--kv-quant <MODE>         KV cache quantization: off, lossless, balanced, aggressive
--qjl-quantization        Enable Stage-2 QJL key residual quantization
```

## Known Limitations and Work in Progress

- **GGUF compatibility**: Support is broad, but some architecture/quantization combinations may still fail depending on checkpoint format.
- **Thinking mode is template-dependent**: `enable_thinking` is applied only when the tokenizer chat template supports it.
- **Byte-level tokenizers**: Streaming decode uses incremental buffering; occasional model-specific Unicode artifacts can still appear.
- **Function calling/tools**: `tools` and `tool_choice` request fields are accepted but not executed server-side yet.
- **Gemma4 edge cases**: Some checkpoints may require architecture-specific tuning.
- **CUDA optimization**: Support exists but is not optimized for production use.

## CUDA Status

CUDA is currently a functional compatibility path, not a performance-tuned backend.

- Build/runtime support is available via `--features cuda`.
- Core model execution works, but the CUDA path currently relies mostly on Candle generic kernels.
- Metal has additional fused kernels in this project (attention/RMSNorm/RoPE/softmax), so CUDA throughput can be lower than specialized CUDA stacks.
- Performance claims for NVIDIA should be treated as hardware-dependent until validated on target GPUs.

## License
The code in this repository is made available under the Apache 2.0 license. See [LICENSE](LICENSE) for details.
