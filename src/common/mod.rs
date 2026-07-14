//! Architecture-agnostic transformer building blocks.
//!
//! Every model oxydllm runs (Llama, Qwen, Gemma, Phi, the MoE families and the
//! hybrid linear-attention variants) is assembled from the components here.
//! There is no bespoke struct per architecture: a model is a
//! [`config::StandardTransformerConfig`] (parsed once from the source
//! `config.json` or GGUF metadata) fed through the single generic forward pass
//! in [`block::run_transformer_layers_batch`]. Supporting a new architecture is
//! usually a matter of describing it in config, not writing new compute.
//!
//! ## Map
//!
//! **Compute graph** (what runs for every token):
//! - [`block`]: the transformer layer ([`block::TransformerBlock`]) and the
//!   top-level batched forward pass.
//! - [`attention`]: softmax multi-head attention (GQA, qk-norm, sliding
//!   window, Metal SDPA / FlashAttention).
//! - [`gdn`]: Gated DeltaNet, the linear-attention token mixer that replaces
//!   attention on the linear layers of hybrid models (Qwen3.5).
//! - [`ffn`] / [`moe`]: dense and Mixture-of-Experts feed-forward sub-layers.
//! - [`norm`]: RMSNorm (Standard and Gemma conventions).
//! - [`rope`]: rotary position embeddings and their scaling schemes.
//! - [`linear`]: [`linear::AnyLinear`], the dispatch over dense and quantized
//!   weight kinds, plus token embeddings.
//!
//! **Weights & quantization**:
//! - [`weights`]: unified weight access over safetensors and GGUF.
//! - [`awq`] / [`mxfp4`] / [`kv_quant`]: packed-int and FP4 formats.
//! - [`gguf_weights`]: typed access to GGUF tensors.
//!
//! **KV cache & runtime**:
//! - [`paged`]: paged KV cache, block allocator, and the recurrent state GDN
//!   carries in place of a KV cache.
//! - [`prefix_cache`]: shared-prefix reuse across requests.
//! - [`mask`]: causal and sliding-window attention masks.
//! - [`config`]: the schema that drives all of the above.
//! - [`decode_profile`]: per-phase latency profiling hooks.
//! - `metal_ops`: fused Metal kernels (norm, quantized matmul, attention);
//!   compiled only under the `metal` feature.
//!
//! ## Mental model
//!
//! A model goes from config to logits in four steps:
//!
//! 1. The parser (`models::parsers::hf_parser`) turns a `config.json` or GGUF
//!    metadata into one [`config::StandardTransformerConfig`].
//! 2. [`config::StandardTransformerConfig::block_config`] projects the per-layer
//!    subset into a [`config::BlockConfig`], one per layer.
//! 3. The loader builds a [`block::TransformerBlock`] per layer and gathers them,
//!    with the embeddings, final norm, lm-head and ropes, into a
//!    [`block::TransformerComponents`].
//! 4. [`block::run_transformer_layers_batch`] runs that bundle over a batch of
//!    tokens and returns the logits.
//!
//! Read [`config`] first to learn the vocabulary, then [`block`] to see how the
//! pieces are wired into a forward pass.

pub mod attention;
pub mod awq;
pub mod block;
pub mod config;
pub mod decode_profile;
pub mod expert_stream;
pub mod ffn;
pub mod gdn;
pub mod gguf_weights;
pub mod kv_quant;
pub mod linear;
pub mod mask;
#[cfg(feature = "metal")]
pub mod metal_ops;
pub mod moe;
pub mod mxfp4;
pub mod norm;
pub mod paged;
pub mod prefix_cache;
pub mod rope;
pub mod weights;
