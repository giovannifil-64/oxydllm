//! The configuration schema that describes a model to the generic runtime.
//!
//! [`StandardTransformerConfig`] is the single source of truth for a model's
//! shape and behaviour. It is produced once by the parser
//! (`models::parsers::hf_parser`) from a HuggingFace `config.json` or GGUF
//! metadata; from then on the architecture-agnostic forward pass reads only
//! this struct. [`StandardTransformerConfig::block_config`] projects the
//! per-layer subset into a [`BlockConfig`] that each
//! [`super::block::TransformerBlock`] consumes.
//!
//! In the common case, supporting a new architecture means populating these
//! fields correctly in the parser: there is no new compute code to write.

use crate::common::rope::RopeScaling;

/// Feed-forward activation function.
///
/// Selects the non-linearity in the gated FFN ([`super::ffn`]) and in MoE
/// experts. `SiLU` (swish) covers Llama / Qwen / Mistral / Phi; `GeLUTanh`
/// (the tanh approximation of GeLU) is the Gemma family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    SiLU,
    GeLUTanh,
}

/// RMSNorm weight convention.
///
/// `Standard` applies the learned weight as stored. `Gemma` stores weights
/// centred at zero, so the effective scale is `1 + w`; the offset is folded
/// into the weight at construction time by [`super::norm::RMSNorm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    Standard,
    Gemma,
}

/// Per-layer token-mixer kind for hybrid architectures (Qwen3.5 / Qwen3-Next).
///
/// Hybrid models interleave layers that use softmax attention with layers that
/// use a [`super::gdn::GatedDeltaNet`]. [`StandardTransformerConfig::layer_types`]
/// lists the kind for each layer; the loader uses it to pick the token mixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

/// Gated DeltaNet (linear-attention) geometry, shared by every linear layer.
///
/// Linear layers run a [`super::gdn::GatedDeltaNet`] instead of attention; this
/// is its head layout. Keys and values may have different head counts and
/// per-head dimensions, and `conv_kernel` is the width of the short causal
/// depthwise convolution applied to q/k/v before the recurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearAttnConfig {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel: usize,
}

/// The per-layer slice of [`StandardTransformerConfig`] that a single
/// [`super::block::TransformerBlock`] is built from.
///
/// Produced by [`StandardTransformerConfig::block_config`]. Most fields are
/// plain geometry: head counts, `head_dim`, `rms_norm_eps`, `activation`,
/// `norm_type`. The ones that change behaviour rather than size are:
///
/// - `n_kv_heads < n_heads` selects grouped-query attention (GQA).
/// - `qk_norm` applies RMSNorm to per-head queries and keys before RoPE (Qwen3,
///   Gemma3); `v_norm` does the same for values.
/// - `attention_scale` overrides the softmax scale; `None` defaults to `1/sqrt(head_dim)`.
/// - `attn_softcap` tanh-caps the attention logits (Gemma2: 50.0).
/// - `sliding_window` restricts attention to the last `n` tokens; `None` is full
///   causal.
/// - `has_ffn_norms` adds Gemma's "sandwich" norms around the feed-forward
///   sub-layer.
///
/// Three fields select the layer's *shape* rather than tune it, and are set per
/// layer by the loader: `moe` (`Some` means a Mixture-of-Experts FFN), `linear_attn`
/// (`Some` means the token mixer is a [`super::gdn::GatedDeltaNet`], not attention),
/// and `attn_output_gate` / `rotary_dim` (Qwen3.5 gated attention, where q_proj
/// emits per-head `[query | gate]`, and partial RoPE over the first `rotary_dim`
/// dims of each head).
pub struct BlockConfig {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub qk_norm: bool,
    pub attention_scale: Option<f64>,
    pub activation: Activation,
    pub norm_type: NormType,
    pub attn_softcap: Option<f64>,
    pub v_norm: bool,
    pub has_ffn_norms: bool,
    pub sliding_window: Option<usize>,
    pub moe: Option<MoeConfig>,
    pub linear_attn: Option<LinearAttnConfig>,
    pub attn_output_gate: bool,
    pub rotary_dim: Option<usize>,
}

/// Mixture-of-Experts routing parameters for one layer.
///
/// Each token is routed to its `num_experts_per_tok` highest-scoring experts
/// (the top-k) out of `num_experts`. `norm_topk_prob` renormalises those top-k
/// gate weights to sum to 1: Qwen3-MoE and Mixtral set it, OLMoE does not.
/// `gpt_oss` selects the GPT-OSS expert format (MXFP4-packed, interleaved
/// gate/up, SwiGLU clamped to `swiglu_limit`, e.g. 7.0 on gpt-oss-20b). See
/// [`super::moe::MoeFeedForward`] for the routing and dispatch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub gpt_oss: bool,
    pub swiglu_limit: f64,
}

/// The complete description of a model: the single source of truth that the
/// loader and the generic forward pass read.
///
/// One of these is built per model by the parser (`models::parsers::hf_parser`)
/// and never mutated afterwards. The fields fall into groups:
///
/// - **Core geometry**: `vocab_size`, `num_hidden_layers`, head counts,
///   `head_dim`, `rope_theta`, `rope_scaling`.
/// - **Behaviour flags**: `qk_norm`, `attention_scale`, `attn_softcap` /
///   `logit_softcap`, `norm_type`, `activation`, `embed_scale`,
///   `sliding_window`, `tie_word_embeddings`.
/// - **Per-layer overrides** (`per_layer_*`, `kv_shared_layer_map`): `Some`
///   only when layers differ from one another; `None` means every layer uses
///   the scalar field above. `kv_shared_layer_map[i] = Some(j)` makes layer `i`
///   reuse layer `j`'s KV cache, and the `per_layer_input_*` set is Gemma 3n's
///   second embedding table, gated into each layer.
/// - **Quantization**: `quant_scheme`.
/// - **MoE** (`moe_*`): present when `moe_num_experts` is set;
///   `moe_norm_topk_prob` defaults to `true` (Qwen3-MoE / Mixtral), OLMoE sets
///   `false`. Folded into a [`MoeConfig`] by [`block_config`](Self::block_config).
/// - **Hybrid linear attention**: `layer_types` (per-layer mixer kind),
///   `linear_attn` (shared DeltaNet geometry), `attn_output_gate`, `rotary_dim`;
///   all `None` / `false` for standard transformers.
///
/// Use [`block_config`](Self::block_config) to get the per-layer
/// [`BlockConfig`].
pub struct StandardTransformerConfig {
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub max_position_embeddings: usize,
    pub qk_norm: bool,
    pub tie_word_embeddings: bool,
    pub attention_scale: Option<f64>,
    pub eos_token_ids: Vec<u32>,
    pub activation: Activation,
    pub norm_type: NormType,
    pub embed_scale: Option<f64>,
    pub attn_softcap: Option<f64>,
    pub logit_softcap: Option<f64>,
    pub v_norm: bool,
    pub has_ffn_norms: bool,
    pub sliding_window: Option<usize>,

    pub per_layer_num_key_value_heads: Option<Vec<usize>>,
    pub per_layer_head_dims: Option<Vec<usize>>,
    pub per_layer_sliding_windows: Option<Vec<Option<usize>>>,
    pub per_layer_rope_thetas: Option<Vec<f64>>,
    pub kv_shared_layer_map: Option<Vec<Option<usize>>>,

    pub per_layer_input_hidden_size: Option<usize>,
    pub per_layer_input_vocab_size: Option<usize>,
    pub per_layer_input_embed_scale: Option<f64>,
    pub per_layer_model_projection_scale: Option<f64>,
    pub per_layer_input_scale: Option<f64>,

    pub quant_scheme: Option<crate::common::weights::QuantScheme>,

    pub moe_num_experts: Option<usize>,
    pub moe_num_experts_per_tok: Option<usize>,
    pub moe_norm_topk_prob: Option<bool>,
    pub moe_gpt_oss: bool,
    pub moe_swiglu_limit: Option<f64>,

    pub layer_types: Option<Vec<LayerType>>,
    pub linear_attn: Option<LinearAttnConfig>,
    pub attn_output_gate: bool,
    pub rotary_dim: Option<usize>,
}

impl StandardTransformerConfig {
    /// Projects the model-wide config into the per-layer [`BlockConfig`] every
    /// [`super::block::TransformerBlock`] is built from.
    ///
    /// The scattered `moe_*` fields are folded into a single [`MoeConfig`] (or
    /// `None`). `linear_attn` is deliberately left `None`: it is per-layer state
    /// that the loader fills in only for the linear-attention layers of a hybrid
    /// model, using [`layer_types`](Self::layer_types).
    pub fn block_config(&self) -> BlockConfig {
        let moe = match (self.moe_num_experts, self.moe_num_experts_per_tok) {
            (Some(n), Some(k)) => Some(MoeConfig {
                num_experts: n,
                num_experts_per_tok: k,
                norm_topk_prob: self.moe_norm_topk_prob.unwrap_or(true),
                gpt_oss: self.moe_gpt_oss,
                swiglu_limit: self.moe_swiglu_limit.unwrap_or(7.0),
            }),
            _ => None,
        };
        BlockConfig {
            n_heads: self.num_attention_heads,
            n_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            rms_norm_eps: self.rms_norm_eps,
            qk_norm: self.qk_norm,
            attention_scale: self.attention_scale,
            activation: self.activation,
            norm_type: self.norm_type,
            attn_softcap: self.attn_softcap,
            v_norm: self.v_norm,
            has_ffn_norms: self.has_ffn_norms,
            sliding_window: self.sliding_window,
            moe,
            linear_attn: None,
            attn_output_gate: self.attn_output_gate,
            rotary_dim: self.rotary_dim,
        }
    }
}
