use anyhow::{Context, Result};
use serde_json::Value;
use crate::common::config::{Activation, NormType, StandardTransformerConfig};

/// Parse a HuggingFace `config.json` into a `StandardTransformerConfig`.
///
/// Uses `serde_json::Value` instead of per-architecture serde structs so that
/// adding support for a new architecture only requires adding a match arm here,
/// with no new struct, no `From` impl, and no change to `loader.rs`.
pub fn parse(config_path: &str) -> Result<StandardTransformerConfig> {
    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("Cannot read {config_path}"))?;
    let v: Value = serde_json::from_str(&raw)
        .with_context(|| format!("Cannot parse JSON from {config_path}"))?;

    let arch = v["architectures"][0].as_str().unwrap_or("Unknown");

    // Bail early for architectures that need special handling before touching
    // common fields (e.g. nested text_config).
    match arch {
        "Qwen3_5ForConditionalGeneration" => {
            anyhow::bail!(
                "Qwen3_5ForConditionalGeneration is not yet supported: it uses hybrid \
linear/full-attention layers that require a dedicated model implementation."
            );
        }
        _ => {}
    }

    let hidden_size          = req_usize(&v, "hidden_size")?;
    let num_attention_heads  = req_usize(&v, "num_attention_heads")?;
    let num_key_value_heads  = v["num_key_value_heads"].as_u64()
        .map(|x| x as usize)
        .unwrap_or(num_attention_heads);
    let head_dim = v["head_dim"].as_u64()
        .map(|x| x as usize)
        .unwrap_or(hidden_size / num_attention_heads);

    let mut eos_token_ids = parse_eos(&v["eos_token_id"]);

    // --- architecture-specific defaults & overrides ---
    let mut activation       = Activation::SiLU;
    let mut norm_type        = NormType::Standard;
    let mut qk_norm          = false;
    let mut tie_word_embeddings = false;
    let mut embed_scale: Option<f64>   = None;
    let mut attn_softcap: Option<f64>  = None;
    let mut logit_softcap: Option<f64> = None;
    let mut attention_scale: Option<f64> = None;
    let mut has_ffn_norms = false;
    let mut default_rope_theta: f64 = 10_000.0;

    match arch {
        // Llama family — add Llama-3 extra EOS tokens
        "LlamaForCausalLM"
        | "MistralForCausalLM"
        | "Mistral3ForConditionalGeneration" => {
            default_rope_theta = 500_000.0;
            for &e in &[128009u32, 128008u32] {
                if !eos_token_ids.contains(&e) {
                    eos_token_ids.push(e);
                }
            }
        }

        // Qwen2 (full attention, no qk_norm)
        "Qwen2ForCausalLM" | "Qwen2_5ForCausalLM" => {
            default_rope_theta = 1_000_000.0;
        }

        // Qwen3 — only difference from Qwen2 is qk_norm
        "Qwen3ForCausalLM" => {
            qk_norm = true;
            default_rope_theta = 1_000_000.0;
        }

        // Gemma 1 — GeLU-tanh, (1+w)*rms_norm, embed scaling, always ties embeddings
        "GemmaForCausalLM" => {
            activation = Activation::GeLUTanh;
            norm_type  = NormType::Gemma;
            tie_word_embeddings = true;
            embed_scale = Some((hidden_size as f64).sqrt());
        }

        // Gemma 2 — same as Gemma 1 + softcapping (always present in config) + query_pre_attn_scalar
        "Gemma2ForCausalLM" => {
            activation = Activation::GeLUTanh;
            norm_type  = NormType::Gemma;
            tie_word_embeddings = true;
            embed_scale = Some((hidden_size as f64).sqrt());
            // Gemma 2 always has these fields; fall back to canonical values if missing.
            attn_softcap  = Some(v["attn_logit_softcapping"].as_f64().unwrap_or(50.0));
            logit_softcap = Some(v["final_logit_softcapping"].as_f64().unwrap_or(30.0));
            if let Some(scalar) = v["query_pre_attn_scalar"].as_f64() {
                attention_scale = Some(1.0 / scalar.sqrt());
            }
        }

        // Gemma 3 — like Gemma 1 but NO softcapping; only apply if explicitly set in config.
        "Gemma3ForCausalLM" => {
            activation = Activation::GeLUTanh;
            norm_type  = NormType::Gemma;
            tie_word_embeddings = true;
            has_ffn_norms = true;
            qk_norm = true;
            embed_scale = Some((hidden_size as f64).sqrt());
            // Only enable softcapping if the fields are actually present and positive.
            if let Some(s) = v["attn_logit_softcapping"].as_f64().filter(|&x| x > 0.0) {
                attn_softcap = Some(s);
            }
            if let Some(s) = v["final_logit_softcapping"].as_f64().filter(|&x| x > 0.0) {
                logit_softcap = Some(s);
            }
            if let Some(scalar) = v["query_pre_attn_scalar"].as_f64() {
                attention_scale = Some(1.0 / scalar.sqrt());
            }
        }

        other => anyhow::bail!(
            "Architecture not supported: '{other}'. \
             Supported: LlamaForCausalLM, MistralForCausalLM, Mistral3ForConditionalGeneration, \
             Qwen2ForCausalLM, Qwen2_5ForCausalLM, \
             Qwen3ForCausalLM, GemmaForCausalLM, Gemma2ForCausalLM, Gemma3ForCausalLM."
        ),
    }

    if eos_token_ids.is_empty() {
        eos_token_ids = vec![2]; // generic <eos>
    }

    Ok(StandardTransformerConfig {
        vocab_size:               req_usize(&v, "vocab_size")?,
        num_hidden_layers:        req_usize(&v, "num_hidden_layers")?,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps:             v["rms_norm_eps"].as_f64().unwrap_or(1e-5),
        rope_theta:               v["rope_theta"].as_f64().unwrap_or(default_rope_theta),
        max_position_embeddings:  v["max_position_embeddings"].as_u64().unwrap_or(131_072) as usize,
        qk_norm,
        tie_word_embeddings:      v["tie_word_embeddings"].as_bool().unwrap_or(tie_word_embeddings),
        attention_scale,
        eos_token_ids,
        activation,
        norm_type,
        embed_scale,
        attn_softcap,
        logit_softcap,
        has_ffn_norms,
    })
}

fn req_usize(v: &Value, key: &str) -> Result<usize> {
    v[key]
        .as_u64()
        .map(|x| x as usize)
        .with_context(|| format!("Missing or invalid field '{key}' in config.json"))
}

fn parse_eos(v: &Value) -> Vec<u32> {
    match v {
        Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Value::Array(arr) => arr.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect(),
        _ => vec![],
    }
}
