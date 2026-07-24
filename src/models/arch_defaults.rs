//! Per-architecture mechanism defaults, keyed by architecture name.
//!
//! [`arch_defaults`] is the single lookup consulted by both checkpoint
//! frontends: [`crate::models::parsers::hf_parser`] passes the HF
//! `architectures[0]` string, the GGUF loader
//! ([`crate::models::gguf_model::StandardTransformer::load_gguf`]) passes the
//! `general.architecture` metadata string. Each entry encodes the mechanisms a
//! family uses (activation, norm variant, qk-norm, softcaps, RoPE base, ...)
//! that its checkpoints do not spell out themselves; values the checkpoint
//! does publish override these defaults in the respective loader.
//! [`known_unsupported_reason`] rejects recognised-but-unloadable families
//! with an actionable message instead of a missing-tensor error.

use crate::common::config::{Activation, NormType};

/// Mechanism defaults for one architecture family.
///
/// Most fields mirror their [`crate::common::config::BlockConfig`] or
/// [`crate::common::config::StandardTransformerConfig`] counterparts and act
/// only as fallbacks for values a checkpoint may omit: `activation`,
/// `norm_type`, `qk_norm`, `v_norm`, `has_ffn_norms`, `attn_softcap`,
/// `logit_softcap`, `default_rope_theta`, `default_sliding_window`.
/// `embed_scale_from_hidden` scales embeddings by sqrt(hidden_size)
/// (Gemma family); `extra_eos_ids` adds stop tokens the config omits;
/// `alternating_sliding_window` gives odd layers global attention (Gemma2,
/// see [`resolve_sliding_window_for_layer`](Self::resolve_sliding_window_for_layer)).
///
/// Three flags encode checkpoint-format quirks rather than defaults:
///
/// * `gpt_oss_moe`: GPT-OSS MoE layout, meaning MXFP4 stacked experts,
///   interleaved gate/up rows, clamped swiglu, and attention sinks (the sink
///   tensor itself is detected by presence).
/// * `attn_output_gate`: Qwen3.5 gated attention, where `q_proj` emits
///   per-head `[query | gate]`. Only consulted by the GGUF loader;
///   safetensors reads `attn_output_gate` from config.json.
/// * `gguf_qk_permuted`: GGUF q/k projections are stored with llama.cpp's
///   per-head row interleave (Llama-family converter) and must be
///   de-interleaved at load to match our NeoX/HF RoPE. Only consulted by the
///   GGUF loader.
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
    pub gpt_oss_moe: bool,
    pub attn_output_gate: bool,
    pub gguf_qk_permuted: bool,
}

impl ArchDefaults {
    /// Sliding window for one layer under the alternating pattern: odd layers
    /// become global (`None`) when `alternating_sliding_window` is set
    /// (Gemma2), otherwise every layer keeps `sliding_window` as given.
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

    /// Expands `sliding_window` into a per-layer vector via
    /// [`resolve_sliding_window_for_layer`](Self::resolve_sliding_window_for_layer),
    /// or `None` when the architecture does not alternate (the caller then
    /// applies the window uniformly).
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

/// Llama-family baseline: SiLU, standard RMSNorm, rope_theta 500k, and the
/// Llama 3 extra EOS ids (128009 `<|eot_id|>`, 128008 `<|eom_id|>`). Most
/// entries in [`arch_defaults`] start from this and override a few fields.
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

/// Rejection reason for architectures we recognise but cannot load, or `None`
/// when the architecture is either supported or simply unknown.
///
/// The MoE loader only consumes the per-expert
/// `mlp.experts.{e}.{gate,up,down}_proj` convention (Qwen3-MoE, OLMoE), which
/// is why these families are rejected upfront:
///
/// * Mixtral and DeepSeek-V2/V3 name their experts
///   `block_sparse_moe.experts.*` (plus shared-expert paths), and the
///   DeepSeek variants additionally need latent (MLA) attention.
/// * GraniteMoe stores experts as fused 3D tensors
///   (`block_sparse_moe.input_linear.weight`, shape
///   `[n_experts, 2*ffn, hidden]`).
/// * Granite 4.0 hybrid additionally interleaves Mamba2 layers, which have no
///   runtime here.
pub fn known_unsupported_reason(arch: &str) -> Option<&'static str> {
    match arch {
        "MixtralForCausalLM" | "DeepseekV2ForCausalLM" | "DeepseekV3ForCausalLM" => {
            Some("This MoE variant (Mixtral / DeepSeek) uses a tensor naming we don't load yet")
        }
        "GraniteMoeForCausalLM" | "GraniteMoeSharedForCausalLM" => Some(
            "GraniteMoe uses fused 3D expert tensors (block_sparse_moe.input_linear) we don't load yet",
        ),
        "GraniteMoeHybridForCausalLM" => {
            Some("Granite 4.0 hybrid interleaves Mamba2 layers, which have no runtime yet")
        }
        _ => None,
    }
}

/// [`ArchDefaults`] for a supported architecture name, or `None` if unknown.
///
/// Accepts both spellings of each family: the HF `architectures[0]` class
/// name and the GGUF `general.architecture` string. The per-arm comments
/// below record checkpoint-format facts (converter permutations, baked-in
/// norm shifts, scale-key semantics) that exist nowhere else; keep them with
/// their entries.
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
        "qwen3_5"
        | "qwen3_5_text"
        | "Qwen3_5ForConditionalGeneration"
        | "Qwen3_5ForCausalLM"
        | "qwen3_5_moe"
        | "Qwen3_5MoeForConditionalGeneration"
        | "Qwen3_5MoeForCausalLM" => Some(ArchDefaults {
            qk_norm: true,
            norm_type: NormType::Gemma,
            default_rope_theta: 10_000_000.0,
            extra_eos_ids: &[],
            attn_output_gate: true,
            ..llama_defaults()
        }),
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
        // GPT-OSS: MoE layout quirks (MXFP4 stacked experts, interleaved
        // gate/up, clamped swiglu, sinks) ride on the gpt_oss_moe flag.
        "gpt_oss" | "GptOssForCausalLM" => Some(ArchDefaults {
            default_rope_theta: 150_000.0,
            gpt_oss_moe: true,
            ..Default::default()
        }),

        // OLMoE (1B-7B/7B-A1B family): Llama-style attention with per-head
        // q_norm/k_norm (qk_norm=true), MoE FFN, rope_theta=10k. `clip_qkv`
        // (sometimes set in OLMoE configs) is currently ignored; on the
        // 0924-Instruct checkpoint it's `null` so this is a no-op.
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
