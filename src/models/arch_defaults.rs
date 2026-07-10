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
    pub alternating_sliding_window: bool,
    /// GPT-OSS MoE: MXFP4 stacked experts, interleaved gate/up, clamped swiglu,
    /// attention sinks (the sink tensor itself is detected by presence).
    pub gpt_oss_moe: bool,
    /// Qwen3.5 gated attention (q_proj emits per-head [query | gate]). Only
    /// consulted by the GGUF loader; safetensors reads `attn_output_gate`
    /// from config.json.
    pub attn_output_gate: bool,
    /// GGUF q/k projections are stored with llama.cpp's per-head row
    /// interleave (Llama-family converter) and must be de-interleaved at load
    /// to match our NeoX/HF RoPE. Only consulted by the GGUF loader.
    pub gguf_qk_permuted: bool,
}

impl ArchDefaults {
    pub fn resolve_sliding_window_for_layer(
        &self,
        sliding_window: Option<usize>,
        layer_idx: usize,
    ) -> Option<usize> {
        if self.alternating_sliding_window && layer_idx % 2 == 1 {
            None
        } else {
            sliding_window
        }
    }

    pub fn per_layer_sliding_windows(
        &self,
        sliding_window: Option<usize>,
        num_layers: usize,
    ) -> Option<Vec<Option<usize>>> {
        if !self.alternating_sliding_window {
            return None;
        }
        sliding_window.map(|_| {
            (0..num_layers)
                .map(|layer_idx| self.resolve_sliding_window_for_layer(sliding_window, layer_idx))
                .collect()
        })
    }
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
            gpt_oss_moe: false,
            alternating_sliding_window: false,
            attn_output_gate: false,
            gguf_qk_permuted: false,
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
        alternating_sliding_window: false,
        gpt_oss_moe: false,
        attn_output_gate: false,
        gguf_qk_permuted: false,
    }
}

pub fn known_unsupported_reason(arch: &str) -> Option<&'static str> {
    match arch {
        // Mixtral + DeepSeek-V2/V3 use a different MoE tensor naming
        // (`block_sparse_moe.experts.*` and shared-expert paths) and the
        // DeepSeek variants add latent attention. Qwen3-MoE and OLMoE are now
        // supported via the `mlp.experts.{e}.{gate,up,down}_proj` convention.
        "MixtralForCausalLM" | "DeepseekV2ForCausalLM" | "DeepseekV3ForCausalLM" => {
            Some("This MoE variant (Mixtral / DeepSeek) uses a tensor naming we don't load yet")
        }
        // GraniteMoe stores experts as fused 3D tensors
        // (`block_sparse_moe.input_linear.weight`, shape [n_experts, 2*ffn, hidden])
        // that our per-expert `mlp.experts.{e}.*` loader cannot consume yet;
        // the Hybrid variant additionally interleaves Mamba2 layers.
        "GraniteMoeForCausalLM" | "GraniteMoeSharedForCausalLM" => Some(
            "GraniteMoe uses fused 3D expert tensors (block_sparse_moe.input_linear) we don't load yet",
        ),
        "GraniteMoeHybridForCausalLM" => {
            Some("Granite 4.0 hybrid interleaves Mamba2 layers, which have no runtime yet")
        }
        _ => None,
    }
}

pub fn arch_defaults(arch: &str) -> Option<ArchDefaults> {
    match arch {
        // GGUF "llama" files (Llama, Mistral, ...) come from the converter's
        // LlamaModel, which interleaves q/k rows for llama.cpp's paired RoPE.
        "llama" | "LlamaForCausalLM" => Some(ArchDefaults {
            gguf_qk_permuted: true,
            ..llama_defaults()
        }),

        "mistral" | "MistralForCausalLM" | "Mistral3ForConditionalGeneration" => {
            Some(ArchDefaults {
                default_rope_theta: 10_000.0,
                ..Default::default()
            })
        }
        // Granite 3.x dense: Llama-shaped blocks plus four scalar multipliers
        // (embedding / attention / residual / logits), all read from the
        // config or GGUF metadata rather than defaulted here. Granite GGUFs
        // are converted through the Llama-family path, so q/k are interleaved.
        "granite" | "GraniteForCausalLM" => Some(ArchDefaults {
            default_rope_theta: 10_000_000.0,
            extra_eos_ids: &[],
            gguf_qk_permuted: true,
            ..llama_defaults()
        }),
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
        // Qwen3.5: hybrid Gated DeltaNet + gated full attention. HF
        // checkpoints store all RMSNorms zero-centered (Gemma-style ×(1+w));
        // activation SiLU; embeddings unscaled. Hybrid layout comes from
        // `layer_types`; gate/partial-RoPE from
        // `attn_output_gate`/`partial_rotary_factor`.
        "qwen3_5" | "qwen3_5_text" | "Qwen3_5ForConditionalGeneration" | "Qwen3_5ForCausalLM" => {
            Some(ArchDefaults {
                qk_norm: true,
                norm_type: NormType::Gemma,
                default_rope_theta: 10_000_000.0,
                extra_eos_ids: &[],
                attn_output_gate: true,
                ..llama_defaults()
            })
        }
        // GGUF flavour of the same arch: llama.cpp's converter bakes the +1
        // into every norm weight (except the DeltaNet gated norm), so the
        // runtime must NOT re-shift; Standard norms here. Hybrid layout
        // comes from `full_attention_interval` + ssm.* metadata.
        "qwen35" => Some(ArchDefaults {
            qk_norm: true,
            norm_type: NormType::Standard,
            default_rope_theta: 10_000_000.0,
            extra_eos_ids: &[],
            attn_output_gate: true,
            ..llama_defaults()
        }),
        // Qwen3-MoE: same attention defaults as Qwen3, MoE FFN handled by
        // `BlockConfig.moe` (parsed from `num_experts` / `num_experts_per_tok`).
        "qwen3_moe" | "Qwen3MoeForCausalLM" => Some(ArchDefaults {
            qk_norm: true,
            default_rope_theta: 1_000_000.0,
            extra_eos_ids: &[],
            ..llama_defaults()
        }),
        // OLMoE (1B-7B/7B-A1B family): Llama-style attention with per-head
        // q_norm/k_norm (qk_norm=true), MoE FFN, rope_theta=10k. `clip_qkv`
        // (sometimes set in OLMoE configs) is currently ignored; on the
        // 0924-Instruct checkpoint it's `null` so this is a no-op.
        "gpt_oss" | "GptOssForCausalLM" => Some(ArchDefaults {
            default_rope_theta: 150_000.0,
            gpt_oss_moe: true,
            ..Default::default()
        }),

        "olmoe" | "OlmoeForCausalLM" => Some(ArchDefaults {
            qk_norm: true,
            default_rope_theta: 10_000.0,
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
            alternating_sliding_window: true,
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
