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
    pub v_norm: bool,
    pub has_ffn_norms: bool,
    pub sliding_window: Option<usize>,

    pub moe: Option<MoeConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// True for Qwen3-MoE / Mixtral; false for OLMoE.
    pub norm_topk_prob: bool,
    /// GPT-OSS experts: MXFP4 stacked tensors, interleaved gate/up, clamped
    /// swiglu. `swiglu_limit` is the clamp bound (gpt-oss-20b: 7.0).
    pub gpt_oss: bool,
    pub swiglu_limit: f64,
}

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
    /// `None` defaults to `true` (Qwen3-MoE / Mixtral); OLMoE sets `false`.
    pub moe_norm_topk_prob: Option<bool>,
    pub moe_gpt_oss: bool,
    pub moe_swiglu_limit: Option<f64>,
}

impl StandardTransformerConfig {
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
        }
    }
}
