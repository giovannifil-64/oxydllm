# rLLM

A rust-based inference engine for large language models.

> [!IMPORTANT]
> This project is under development and not yet ready for production use. At the moment it only supports text input/output and a limited set of models.


## Tested Models
Here you can find a list of models that have been tested with rLLM, divided by architecture. This is not an exhaustive list of compatible models, but it can give you an idea of what has been verified to work.

<details>
<summary>LlamaForCausalLM</summary>

#### Llama-3.2-1B-Instruct
- Metal: MacBook Pro M5 with 24GB of unified memory.

</details>


<details>
<summary>Qwen3ForCausalLM</summary>

> [!NOTE]
> All models have been tested with and without thinking, with the same prompt.

#### Qwen3-0.6B
- Metal: MacBook Pro M5 with 24GB of unified memory.

#### Qwen3-0.6B-Q8_0
- Metal: MacBook Pro M5 with 24GB of unified memory.

#### Qwen3-1.7B-Q8_0
- Metal: MacBook Pro M5 with 24GB of unified memory.

#### Qwen3-4B-Q4_K_M
- Metal: MacBook Pro M5 with 24GB of unified memory.
    
#### Qwen3-4B-Q5_0
- Metal: MacBook Pro M5 with 24GB of unified memory.
</details>

<details>
<summary>GemmaForCausalLM</summary>

#### gemma-2b-it
- Metal: MacBook Pro M5 with 24GB of unified memory.
</details>

<details>
<summary>Gemma3ForCausalLM</summary>

#### gemma-3-1b-it
- Metal: MacBook Pro M5 with 24GB of unified memory.

#### gemma-3-270m-it
- Metal: MacBook Pro M5 with 24GB of unified memory.
</details>
