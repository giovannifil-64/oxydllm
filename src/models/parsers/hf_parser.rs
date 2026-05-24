use crate::common::config::StandardTransformerConfig;
use anyhow::{Context, Result};
use serde_json::Value;

use crate::common::rope::RopeScaling;

pub fn parse(config_path: &str) -> Result<StandardTransformerConfig> {
    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("Cannot read {config_path}"))?;
    let v: Value = serde_json::from_str(&raw)
        .with_context(|| format!("Cannot parse JSON from {config_path}"))?;

    // Multimodal configs nest LLM params under "text_config"; merge them up,
    // letting text_config win on key collision.
    let v = if let Some(text_cfg) = v.get("text_config").and_then(|tc| tc.as_object()) {
        let mut merged = v.clone();
        let root = merged.as_object_mut().unwrap();
        for (k, val) in text_cfg {
            root.insert(k.clone(), val.clone());
        }
        merged
    } else {
        v
    };

    let arch = v["architectures"][0].as_str().unwrap_or("Unknown");
    maybe_log_torch_dtype(&v);
    let quant_scheme = validate_quantization_config(&v["quantization_config"])?;

    if let Some(reason) = crate::models::arch_defaults::known_unsupported_reason(arch) {
        anyhow::bail!("Architecture '{arch}' is not supported: {reason}");
    }

    let arch_def = crate::models::arch_defaults::arch_defaults(arch)
        .with_context(|| format!("Architecture '{arch}' not supported"))?;

    let hidden_size = req_usize(&v, "hidden_size")?;
    let num_hidden_layers = req_usize(&v, "num_hidden_layers")?;
    let num_attention_heads = req_usize(&v, "num_attention_heads")?;
    let num_key_value_heads = v["num_key_value_heads"]
        .as_u64()
        .map(|x| x as usize)
        .unwrap_or(num_attention_heads);
    let head_dim = v["head_dim"]
        .as_u64()
        .map(|x| x as usize)
        .unwrap_or(hidden_size / num_attention_heads);

    let mut eos_token_ids = parse_eos(&v["eos_token_id"]);

    for &e in arch_def.extra_eos_ids {
        if !eos_token_ids.contains(&e) {
            eos_token_ids.push(e);
        }
    }

    for e in parse_generation_eos(config_path) {
        if !eos_token_ids.contains(&e) {
            eos_token_ids.push(e);
        }
    }

    if eos_token_ids.is_empty() {
        eos_token_ids = vec![2];
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
    if attention_scale.is_none()
        && (arch == "gemma4"
            || arch == "gemma-4"
            || arch == "gemma4_text"
            || arch == "Gemma4ForCausalLM"
            || arch == "Gemma4ForConditionalGeneration")
    {
        attention_scale = Some(1.0);
    }

    // Several Gemma checkpoints omit `tie_word_embeddings`; default to true
    // for the Gemma family (canonical setting) and false otherwise. Loader
    // warns if a file ships both tie=true and an explicit lm_head.weight.
    let tie_word_embeddings = v["tie_word_embeddings"].as_bool().unwrap_or(
        arch == "GemmaForCausalLM"
            || arch == "Gemma2ForCausalLM"
            || arch == "Gemma3ForCausalLM"
            || arch == "Gemma4ForCausalLM"
            || arch == "Gemma4ForConditionalGeneration",
    );

    let mut rope_scaling = parse_rope_scaling(&v["rope_scaling"]);
    if matches!(rope_scaling, RopeScaling::None)
        && let Some(rp) = v.get("rope_parameters")
        && (rp.get("rope_type").is_some() || rp.get("type").is_some())
    {
        rope_scaling = parse_rope_scaling(rp);
    }
    let sliding_window = v["sliding_window"].as_u64().map(|x| x as usize);

    // Qwen3-MoE/OLMoE use `num_experts`, Mixtral uses `num_local_experts`.
    let moe_num_experts = v["num_experts"]
        .as_u64()
        .or_else(|| v["num_local_experts"].as_u64())
        .map(|x| x as usize)
        .filter(|&n| n > 1);
    let moe_num_experts_per_tok = v["num_experts_per_tok"].as_u64().map(|x| x as usize);
    let moe_norm_topk_prob = v["norm_topk_prob"].as_bool();

    let layer_types =
        parse_string_array(&v["layer_types"]).filter(|x| x.len() == num_hidden_layers);
    let global_head_dim = v["global_head_dim"]
        .as_u64()
        .map(|x| x as usize)
        .filter(|&x| x > 0);
    let num_global_key_value_heads = v["num_global_key_value_heads"]
        .as_u64()
        .map(|x| x as usize)
        .filter(|&x| x > 0);

    let rope_parameters = v.get("rope_parameters").unwrap_or(&Value::Null);
    let full_rope_theta = rope_parameters
        .get("full_attention")
        .and_then(|x| x.get("rope_theta"))
        .and_then(Value::as_f64);
    let sliding_rope_theta = rope_parameters
        .get("sliding_attention")
        .and_then(|x| x.get("rope_theta"))
        .and_then(Value::as_f64)
        .or_else(|| rope_parameters.get("rope_theta").and_then(Value::as_f64));

    let rope_theta = v["rope_theta"]
        .as_f64()
        .or(sliding_rope_theta)
        .or(full_rope_theta)
        .unwrap_or(arch_def.default_rope_theta);

    let per_layer_head_dims = layer_types.as_ref().map(|types| {
        types
            .iter()
            .map(|layer_type| {
                if layer_type == "full_attention" {
                    global_head_dim.unwrap_or(head_dim)
                } else {
                    head_dim
                }
            })
            .collect::<Vec<_>>()
    });

    let per_layer_num_key_value_heads = layer_types.as_ref().map(|types| {
        types
            .iter()
            .map(|layer_type| {
                if layer_type == "full_attention" {
                    num_global_key_value_heads.unwrap_or(num_key_value_heads)
                } else {
                    num_key_value_heads
                }
            })
            .collect::<Vec<_>>()
    });

    let per_layer_sliding_windows = layer_types
        .as_ref()
        .and_then(|types| {
            sliding_window.map(|w| {
                types
                    .iter()
                    .map(|layer_type| {
                        if layer_type == "full_attention" {
                            None
                        } else {
                            Some(w)
                        }
                    })
                    .collect::<Vec<_>>()
            })
        })
        .or_else(|| arch_def.per_layer_sliding_windows(sliding_window, num_hidden_layers));

    let per_layer_rope_thetas = layer_types.as_ref().map(|types| {
        types
            .iter()
            .map(|layer_type| {
                if layer_type == "full_attention" {
                    full_rope_theta.unwrap_or(rope_theta)
                } else {
                    sliding_rope_theta.unwrap_or(rope_theta)
                }
            })
            .collect::<Vec<_>>()
    });

    let num_kv_shared_layers = v["num_kv_shared_layers"].as_u64().unwrap_or(0) as usize;
    let kv_shared_layer_map = layer_types.as_ref().and_then(|types| {
        if num_kv_shared_layers == 0 {
            return None;
        }
        let first_shared = num_hidden_layers.saturating_sub(num_kv_shared_layers);
        let mut map = vec![None; num_hidden_layers];
        for layer_idx in first_shared..num_hidden_layers {
            if let Some(ref_layer) = (0..first_shared)
                .rev()
                .find(|&j| types[j] == types[layer_idx])
            {
                map[layer_idx] = Some(ref_layer);
            }
        }
        Some(map)
    });

    let per_layer_input_hidden_size = v["hidden_size_per_layer_input"]
        .as_u64()
        .map(|x| x as usize)
        .filter(|&x| x > 0);
    let per_layer_input_vocab_size = v["vocab_size_per_layer_input"]
        .as_u64()
        .map(|x| x as usize)
        .filter(|&x| x > 0);
    let (per_layer_input_embed_scale, per_layer_model_projection_scale, per_layer_input_scale) =
        if let Some(h) = per_layer_input_hidden_size {
            (
                Some((h as f64).sqrt()),
                Some(1.0 / (hidden_size as f64).sqrt()),
                Some(1.0 / (2.0f64).sqrt()),
            )
        } else {
            (None, None, None)
        };

    Ok(StandardTransformerConfig {
        vocab_size: req_usize(&v, "vocab_size")?,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-5),
        rope_theta,
        rope_scaling,
        max_position_embeddings: v["max_position_embeddings"].as_u64().unwrap_or(131_072) as usize,
        qk_norm: arch_def.qk_norm,
        tie_word_embeddings,
        attention_scale,
        eos_token_ids,
        activation: arch_def.activation,
        norm_type: arch_def.norm_type,
        embed_scale,
        attn_softcap,
        logit_softcap,
        v_norm: arch_def.v_norm,
        has_ffn_norms: arch_def.has_ffn_norms,
        sliding_window,
        per_layer_num_key_value_heads,
        per_layer_head_dims,
        per_layer_sliding_windows,
        per_layer_rope_thetas,
        kv_shared_layer_map,
        per_layer_input_hidden_size,
        per_layer_input_vocab_size,
        per_layer_input_embed_scale,
        per_layer_model_projection_scale,
        per_layer_input_scale,
        quant_scheme,
        moe_num_experts,
        moe_num_experts_per_tok,
        moe_norm_topk_prob,
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
        Value::Array(arr) => arr
            .iter()
            .filter_map(|x| x.as_u64())
            .map(|x| x as u32)
            .collect(),
        _ => vec![],
    }
}

fn parse_string_array(v: &Value) -> Option<Vec<String>> {
    v.as_array().map(|arr| {
        arr.iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect::<Vec<_>>()
    })
}

fn parse_generation_eos(config_path: &str) -> Vec<u32> {
    let config_path = std::path::Path::new(config_path);
    let Some(parent) = config_path.parent() else {
        return Vec::new();
    };
    let generation_config_path = parent.join("generation_config.json");
    let raw = match std::fs::read_to_string(generation_config_path) {
        Ok(raw) => raw,
        Err(_) => return Vec::new(),
    };
    let v: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    parse_eos(&v["eos_token_id"])
}

fn validate_quantization_config(v: &Value) -> Result<Option<crate::common::weights::QuantScheme>> {
    if v.is_null() {
        return Ok(None);
    }
    let method = v["quant_method"].as_str().unwrap_or("");
    let bits = v["bits"].as_u64();
    let group_size = v["group_size"].as_u64();
    let version = v["version"].as_str().unwrap_or("gemm");

    match method.to_ascii_lowercase().as_str() {
        "awq" => {
            let bits = bits.ok_or_else(|| {
                anyhow::anyhow!(
                    "AWQ checkpoint missing required 'bits' field in quantization_config"
                )
            })? as u32;
            if !matches!(bits, 4 | 8) {
                anyhow::bail!(
                    "AWQ checkpoint bits={bits} not supported. Only 4 and 8 bits are recognised."
                );
            }
            if !version.eq_ignore_ascii_case("gemm") {
                anyhow::bail!(
                    "AWQ version '{version}' is not supported (only 'gemm' has a runtime path). \
                     Re-export the checkpoint with autoawq's GEMM kernel or use a different model."
                );
            }
            tracing::info!(
                quant = "awq",
                version = "gemm",
                bits,
                group_size = group_size.unwrap_or(128),
                "AWQ checkpoint detected (W{bits}A16 fused matmul on Metal)"
            );
            Ok(Some(crate::common::weights::QuantScheme::Awq { bits }))
        }
        "fp8" => {
            tracing::info!(
                quant = "fp8",
                "FP8 checkpoint detected; weights will be dequantized at load time (CPU path on Metal)"
            );
            Ok(None)
        }
        "gptq" => {
            let bits = bits.ok_or_else(|| {
                anyhow::anyhow!(
                    "GPTQ checkpoint missing required 'bits' field in quantization_config"
                )
            })?;
            if bits != 4 && bits != 8 {
                anyhow::bail!(
                    "GPTQ checkpoint bits={bits} not supported. Only 4 and 8 bits are recognised."
                );
            }
            let sym = v["sym"].as_bool().unwrap_or(true);
            let desc_act = v["desc_act"].as_bool().unwrap_or(false);
            if desc_act {
                anyhow::bail!(
                    "GPTQ checkpoint has desc_act=true (act-order). The runtime loader \
                     currently only supports desc_act=false (sequential g_idx)."
                );
            }
            let checkpoint_format = v["checkpoint_format"].as_str().unwrap_or("gptq");
            if !checkpoint_format.eq_ignore_ascii_case("gptq") {
                anyhow::bail!(
                    "GPTQ checkpoint_format='{checkpoint_format}' not supported \
                     (only the canonical 'gptq' format is wired)."
                );
            }
            tracing::info!(
                quant = "gptq",
                bits = bits,
                group_size = group_size.unwrap_or(128),
                sym = sym,
                desc_act = desc_act,
                "GPTQ checkpoint detected (dequant-at-load CPU path)"
            );
            Ok(Some(crate::common::weights::QuantScheme::Gptq {
                bits: bits as u32,
                sym,
            }))
        }
        "" => Ok(None),
        other => anyhow::bail!(
            "Unknown quantization method '{other}' in quantization_config. \
             Supported: awq (gemm, 4-bit), gptq (4/8-bit, desc_act=false), fp8."
        ),
    }
}

fn maybe_log_torch_dtype(v: &Value) {
    let Some(torch_dtype) = v["torch_dtype"].as_str() else {
        return;
    };

    if torch_dtype.eq_ignore_ascii_case("float8_e4m3fn") {
        tracing::info!(
            torch_dtype,
            "FP8 checkpoint detected; weights will be dequantized at runtime using *_scale_inv tensors"
        );
    }
}

fn parse_rope_scaling(v: &Value) -> RopeScaling {
    if v.is_null() {
        return RopeScaling::None;
    }

    let rope_type = v["rope_type"]
        .as_str()
        .or_else(|| v["type"].as_str())
        .unwrap_or("");
    let factor = v["factor"].as_f64().unwrap_or(1.0);

    match rope_type {
        "linear" => RopeScaling::Linear { factor },
        "llama3" => {
            let low_freq_factor = v["low_freq_factor"].as_f64().unwrap_or(1.0);
            let high_freq_factor = v["high_freq_factor"].as_f64().unwrap_or(1.0);
            let original_max_pos = v["original_max_position_embeddings"]
                .as_u64()
                .unwrap_or(8192) as usize;
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_pos,
            }
        }
        "yarn" => {
            let original_max_pos = v["original_max_position_embeddings"]
                .as_u64()
                .unwrap_or(8192) as usize;
            let beta_fast = v["beta_fast"].as_f64().unwrap_or(32.0);
            let beta_slow = v["beta_slow"].as_f64().unwrap_or(1.0);
            RopeScaling::Yarn {
                factor,
                original_max_pos,
                beta_fast,
                beta_slow,
            }
        }
        "longrope" => {
            let original_max_pos = v["original_max_position_embeddings"]
                .as_u64()
                .unwrap_or(4096) as usize;
            let short_factor = parse_rope_factor(&v["short_factor"], 1.0);
            let long_factor = parse_rope_factor(&v["long_factor"], factor.max(1.0));
            RopeScaling::LongRope {
                short_factor,
                long_factor,
                original_max_pos,
            }
        }
        _ => {
            if !rope_type.is_empty() {
                tracing::warn!(
                    rope_type = %rope_type,
                    "unknown rope_scaling type, falling back to no scaling"
                );
            }
            RopeScaling::None
        }
    }
}

fn parse_rope_factor(v: &Value, default: f64) -> Vec<f64> {
    match v {
        Value::Number(n) => n.as_f64().map(|x| vec![x]).unwrap_or_else(|| vec![default]),
        Value::Array(arr) => {
            let vals: Vec<f64> = arr.iter().filter_map(Value::as_f64).collect();
            if vals.is_empty() { vec![default] } else { vals }
        }
        _ => vec![default],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn text_config_overrides_root_fields() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.json");

        let config = json!({
            "architectures": ["Mistral3ForConditionalGeneration"],
            "hidden_size": 4096,
            "num_hidden_layers": 6,
            "num_attention_heads": 16,
            "vocab_size": 32000,
            "text_config": {
                "hidden_size": 3072,
                "num_hidden_layers": 24,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "vocab_size": 128256
            }
        });

        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let cfg = parse(config_path.to_string_lossy().as_ref()).unwrap();

        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.vocab_size, 128256);
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn parse_longrope_scaling_from_config() {
        let v = json!({
            "type": "longrope",
            "short_factor": [1.0, 1.5],
            "long_factor": [2.0, 2.5],
            "original_max_position_embeddings": 8192
        });

        match parse_rope_scaling(&v) {
            RopeScaling::LongRope {
                short_factor,
                long_factor,
                original_max_pos,
            } => {
                assert_eq!(short_factor, vec![1.0, 1.5]);
                assert_eq!(long_factor, vec![2.0, 2.5]);
                assert_eq!(original_max_pos, 8192);
            }
            _ => panic!("expected longrope variant"),
        }
    }

    #[test]
    fn parse_unknown_rope_scaling_falls_back_to_none() {
        let v = json!({
            "type": "mystery_rope",
            "factor": 2.0
        });
        assert!(matches!(parse_rope_scaling(&v), RopeScaling::None));
    }

    #[test]
    fn mistral_v3_rope_parameters_yarn_picked_up() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        let config = json!({
            "architectures": ["Mistral3ForConditionalGeneration"],
            "text_config": {
                "hidden_size": 3072,
                "num_hidden_layers": 24,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "vocab_size": 131072,
                "max_position_embeddings": 262144,
                "rope_parameters": {
                    "rope_type": "yarn",
                    "factor": 16.0,
                    "rope_theta": 1000000.0,
                    "original_max_position_embeddings": 16384,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0
                }
            }
        });
        fs::write(&config_path, config.to_string()).unwrap();
        let cfg = parse(&config_path).unwrap();
        assert!(matches!(
            cfg.rope_scaling,
            RopeScaling::Yarn { factor, original_max_pos, beta_fast, beta_slow }
                if factor == 16.0
                && original_max_pos == 16384
                && beta_fast == 32.0
                && beta_slow == 1.0
        ));
        assert_eq!(cfg.rope_theta, 1_000_000.0);
    }

    #[test]
    fn gemma2_has_alternating_sliding_windows_without_layer_types() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.json");

        let config = json!({
            "architectures": ["Gemma2ForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "vocab_size": 256000,
            "sliding_window": 4096
        });

        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let cfg = parse(config_path.to_string_lossy().as_ref()).unwrap();

        assert_eq!(cfg.sliding_window, Some(4096));
        assert_eq!(
            cfg.per_layer_sliding_windows,
            Some(vec![Some(4096), None, Some(4096), None])
        );
    }

    #[test]
    fn gemma3_does_not_force_alternating_sliding_windows() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.json");

        let config = json!({
            "architectures": ["Gemma3ForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "vocab_size": 256000,
            "sliding_window": 4096
        });

        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let cfg = parse(config_path.to_string_lossy().as_ref()).unwrap();

        assert_eq!(cfg.sliding_window, Some(4096));
        assert!(cfg.per_layer_sliding_windows.is_none());
    }

    #[test]
    fn missing_tie_word_embeddings_on_gemma_defaults_to_true() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.json");

        let config = json!({
            "architectures": ["Gemma2ForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "vocab_size": 256000,
        });
        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let cfg = parse(config_path.to_string_lossy().as_ref()).unwrap();
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn missing_tie_word_embeddings_on_non_gemma_defaults_to_false() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.json");

        let config = json!({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "vocab_size": 32000,
        });
        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let cfg = parse(config_path.to_string_lossy().as_ref()).unwrap();
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn explicit_tie_word_embeddings_is_respected_on_gemma() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.json");

        let config = json!({
            "architectures": ["GemmaForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "vocab_size": 256000,
            "tie_word_embeddings": false
        });
        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let cfg = parse(config_path.to_string_lossy().as_ref()).unwrap();
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn parse_accepts_fp8_torch_dtype() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.json");

        let config = json!({
            "architectures": ["Mistral3ForConditionalGeneration"],
            "hidden_size": 3072,
            "num_hidden_layers": 24,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "torch_dtype": "float8_e4m3fn"
        });

        fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let cfg = parse(config_path.to_string_lossy().as_ref()).unwrap();

        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 8);
    }
}
