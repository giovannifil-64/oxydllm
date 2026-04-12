use crate::common::config::{Activation, NormType};

pub struct ArchDefaults {
    pub activation: Activation,
    pub norm_type: NormType,
    pub qk_norm: bool,
    pub v_norm: bool,
    pub has_ffn_norms: bool,
    pub embed_scale_from_hidden: bool,
    pub attn_softcap: Option<f64>,
    pub logit_softcap: Option<f64>,
    pub default_rope_theta: f64,
    pub extra_eos_ids: &'static [u32],
    pub default_sliding_window: Option<usize>,
}

impl Default for ArchDefaults {
    fn default() -> Self {
        Self {
            activation: Activation::SiLU,
            norm_type: NormType::Standard,
            qk_norm: false,
            v_norm: false,
            has_ffn_norms: false,
            embed_scale_from_hidden: false,
            attn_softcap: None,
            logit_softcap: None,
            default_rope_theta: 10_000.0,
            extra_eos_ids: &[],
            default_sliding_window: None,
        }
    }
}

pub fn llama_defaults() -> ArchDefaults {
    ArchDefaults {
        activation: Activation::SiLU,
        norm_type: NormType::Standard,
        qk_norm: false,
        v_norm: false,
        has_ffn_norms: false,
        embed_scale_from_hidden: false,
        attn_softcap: None,
        logit_softcap: None,
        default_rope_theta: 500_000.0,
        extra_eos_ids: &[128009, 128008],
        default_sliding_window: None,
    }
}

pub fn known_unsupported_reason(arch: &str) -> Option<&'static str> {
    match arch {
        // Mixture-of-Experts models — MoE routing not yet implemented.
        "Qwen3MoeForCausalLM"
        | "MixtralForCausalLM"
        | "DeepseekV2ForCausalLM"
        | "DeepseekV3ForCausalLM" => {
            Some("Mixture-of-Experts (MoE) architectures are not yet supported")
        }
        "Qwen3_5ForConditionalGeneration" => {
            Some("Hybrid linear+full attention models are not yet supported")
        }
        _ => None,
    }
}

pub fn arch_defaults(arch: &str) -> Option<ArchDefaults> {
    match arch {
        "llama" | "LlamaForCausalLM" => Some(llama_defaults()),

        "mistral" | "MistralForCausalLM" | "Mistral3ForConditionalGeneration" => {
            Some(ArchDefaults {
                default_rope_theta: 10_000.0,
                ..Default::default()
            })
        }
        "phi3" | "Phi3ForCausalLM" => Some(ArchDefaults {
            default_rope_theta: 10_000.0,
            extra_eos_ids: &[],
            ..llama_defaults()
        }),

        "qwen2" | "Qwen2ForCausalLM" => Some(ArchDefaults {
            default_rope_theta: 1_000_000.0,
            extra_eos_ids: &[],
            ..llama_defaults()
        }),
        "qwen3" | "Qwen3ForCausalLM" => Some(ArchDefaults {
            qk_norm: true,
            default_rope_theta: 1_000_000.0,
            extra_eos_ids: &[],
            ..llama_defaults()
        }),

        "gemma" | "GemmaForCausalLM" => Some(ArchDefaults {
            activation: Activation::GeLUTanh,
            norm_type: NormType::Gemma,
            embed_scale_from_hidden: true,
            ..Default::default()
        }),
        "gemma2" | "gemma-2" | "Gemma2ForCausalLM" => Some(ArchDefaults {
            activation: Activation::GeLUTanh,
            norm_type: NormType::Gemma,
            has_ffn_norms: true,
            embed_scale_from_hidden: true,
            attn_softcap: Some(50.0),
            logit_softcap: Some(30.0),
            default_sliding_window: Some(4096),
            ..Default::default()
        }),
        "gemma3" | "gemma-3" | "Gemma3ForCausalLM" => Some(ArchDefaults {
            activation: Activation::GeLUTanh,
            norm_type: NormType::Gemma,
            qk_norm: true,
            has_ffn_norms: true,
            embed_scale_from_hidden: true,
            extra_eos_ids: &[1, 106],
            ..Default::default()
        }),
        "gemma4"
        | "gemma-4"
        | "gemma4_text"
        | "Gemma4ForCausalLM"
        | "Gemma4ForConditionalGeneration" => Some(ArchDefaults {
            activation: Activation::GeLUTanh,
            norm_type: NormType::Standard,
            qk_norm: true,
            v_norm: true,
            has_ffn_norms: true,
            embed_scale_from_hidden: true,
            logit_softcap: Some(30.0),
            extra_eos_ids: &[1, 106],
            ..Default::default()
        }),
        _ => None,
    }
}
