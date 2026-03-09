use crate::common::rope::RopeScaling;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    SiLU,
    GeLUTanh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    Standard,
    Gemma,
}

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
    pub has_ffn_norms: bool,
    pub sliding_window: Option<usize>,
}

/// Architecture-independent config for standard pre-norm transformer models
/// (Llama, Qwen3, Mistral, Phi-3, …).  Architecture-specific loaders convert
/// their JSON config into this struct; the loader then uses it uniformly.
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
    pub has_ffn_norms: bool,
    pub sliding_window: Option<usize>,
}

impl StandardTransformerConfig {
    pub fn block_config(&self) -> BlockConfig {
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
            has_ffn_norms: self.has_ffn_norms,
            sliding_window: self.sliding_window,
        }
    }
}
