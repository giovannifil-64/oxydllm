use anyhow::{Context, Result};
use serde_json::Value;
use crate::common::config::StandardTransformerConfig;

use crate::common::rope::RopeScaling;

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

    // For multimodal models the LLM parameters are nested under "text_config".
    // Merge those fields into the root so the rest of the parser is uniform.
    let v = if let Some(text_cfg) = v.get("text_config").and_then(|tc| tc.as_object()) {
        let mut merged = v.clone();
        let root = merged.as_object_mut().unwrap();
        for (k, val) in text_cfg {
            root.entry(k.clone()).or_insert_with(|| val.clone());
        }
        merged
    } else {
        v
    };

    let arch = v["architectures"][0].as_str().unwrap_or("Unknown");

    let arch_def = crate::models::arch_defaults::arch_defaults(arch)
        .with_context(|| format!("Architecture '{arch}' not supported"))?;

    let hidden_size          = req_usize(&v, "hidden_size")?;
    let num_attention_heads  = req_usize(&v, "num_attention_heads")?;
    let num_key_value_heads  = v["num_key_value_heads"].as_u64()
        .map(|x| x as usize)
        .unwrap_or(num_attention_heads);
    let head_dim = v["head_dim"].as_u64()
        .map(|x| x as usize)
        .unwrap_or(hidden_size / num_attention_heads);

    let mut eos_token_ids = parse_eos(&v["eos_token_id"]);

    for &e in arch_def.extra_eos_ids {
        if !eos_token_ids.contains(&e) {
            eos_token_ids.push(e);
        }
    }
    if eos_token_ids.is_empty() {
        eos_token_ids = vec![2]; // generic <eos>
    }

    let embed_scale = if arch_def.embed_scale_from_hidden {
        Some((hidden_size as f64).sqrt())
    } else {
        None
    };

    let mut attn_softcap = arch_def.attn_softcap;
    let mut logit_softcap = arch_def.logit_softcap;
    let mut attention_scale = None;

    if let Some(s) = v["attn_logit_softcapping"].as_f64().filter(|&x| x > 0.0) {
        attn_softcap = Some(s);
    }
    if let Some(s) = v["final_logit_softcapping"].as_f64().filter(|&x| x > 0.0) {
        logit_softcap = Some(s);
    }
    if let Some(scalar) = v["query_pre_attn_scalar"].as_f64() {
        attention_scale = Some(1.0 / scalar.sqrt());
    }

    let tie_word_embeddings = v["tie_word_embeddings"].as_bool()
        .unwrap_or(arch == "GemmaForCausalLM" || arch == "Gemma2ForCausalLM" || arch == "Gemma3ForCausalLM"); // Temporary fallback logic for tying

    let rope_scaling = parse_rope_scaling(&v["rope_scaling"]);
    let sliding_window = v["sliding_window"].as_u64().map(|x| x as usize);

    Ok(StandardTransformerConfig {
        vocab_size:               req_usize(&v, "vocab_size")?,
        num_hidden_layers:        req_usize(&v, "num_hidden_layers")?,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps:             v["rms_norm_eps"].as_f64().unwrap_or(1e-5),
        rope_theta:               v["rope_theta"].as_f64().unwrap_or(arch_def.default_rope_theta),
        rope_scaling,
        max_position_embeddings:  v["max_position_embeddings"].as_u64().unwrap_or(131_072) as usize,
        qk_norm:                  arch_def.qk_norm,
        tie_word_embeddings,
        attention_scale,
        eos_token_ids,
        activation:               arch_def.activation,
        norm_type:                arch_def.norm_type,
        embed_scale,
        attn_softcap,
        logit_softcap,
        has_ffn_norms:            arch_def.has_ffn_norms,
        sliding_window,
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

fn parse_rope_scaling(v: &Value) -> RopeScaling {
    if v.is_null() {
        return RopeScaling::None;
    }
    
    let rope_type = v["rope_type"].as_str().or_else(|| v["type"].as_str()).unwrap_or("");
    let factor = v["factor"].as_f64().unwrap_or(1.0);
    
    match rope_type {
        "linear" => RopeScaling::Linear { factor },
        "llama3" => {
            let low_freq_factor = v["low_freq_factor"].as_f64().unwrap_or(1.0);
            let high_freq_factor = v["high_freq_factor"].as_f64().unwrap_or(1.0);
            let original_max_pos = v["original_max_position_embeddings"].as_u64().unwrap_or(8192) as usize;
            RopeScaling::Llama3 { factor, low_freq_factor, high_freq_factor, original_max_pos }
        }
        "yarn" => RopeScaling::Yarn { factor },
        _ => RopeScaling::None,
    }
}
