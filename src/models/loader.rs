use std::sync::{Arc, Mutex};
use candle_core::{DType, Device};
use crate::common::{
    block::TransformerBlock,
    config::StandardTransformerConfig,
    gguf_weights::GgufWeights,
    kv_quant::{self, KvQuantMode, KvQuantizer},
    linear::{AnyLinear, Embedding, Linear},
    norm::RMSNorm,
    paged::{BlockAllocator, DEFAULT_BLOCK_SIZE, GlobalKvBudget, SharedBlockAllocator, SharedGlobalKvBudget},
    rope::RotaryEmbedding,
    weights::ModelWeights,
};
use crate::models::traits::BatchModel;
use crate::models::gguf_model::StandardTransformer;
use crate::models::parsers::hf_parser;

use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct DiscoveredModel {
    pub id: String,
    pub architecture: String,
    pub vocab_size: usize,
    pub num_layers: usize,
}

pub fn discover_models(models_dir: &Path) -> Vec<DiscoveredModel> {
    let mut models = Vec::new();
    let entries = match std::fs::read_dir(models_dir) {
        Ok(e) => e,
        Err(_) => return models,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let id = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let config_path = path.join("config.json");
        if config_path.exists() {
            let raw = match std::fs::read_to_string(&config_path) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let value: serde_json::Value = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let architecture = value["architectures"][0]
                .as_str()
                .unwrap_or("Unknown")
                .to_string();
            let vocab_size = value["vocab_size"].as_u64().unwrap_or(0) as usize;
            let num_layers = value["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
            models.push(DiscoveredModel {
                id,
                architecture,
                vocab_size,
                num_layers,
            });
            continue;
        }

        if let Some(gguf_paths) = find_gguf_files(&path) {
            for gguf_path in &gguf_paths {
                let raw_stem = gguf_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let gguf_id = strip_gguf_split_suffix(&raw_stem).to_string();
                let effective_id = if gguf_id.is_empty() { id.clone() } else { gguf_id };
                if models.iter().any(|m: &DiscoveredModel| m.id == effective_id) {
                    continue;
                }
                if let Some(info) = discover_gguf_model(&effective_id, gguf_path) {
                    models.push(info);
                }
            }
        }
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));
    models
}

fn strip_gguf_split_suffix(stem: &str) -> &str {
    let parts: Vec<&str> = stem.split('-').collect();
    let n = parts.len();
    if n >= 3
        && parts[n - 2] == "of"
        && parts[n - 1].chars().all(|c| c.is_ascii_digit())
        && parts[n - 3].chars().all(|c| c.is_ascii_digit())
    {
        let trim = 1 + parts[n-1].len() + 1 + parts[n-2].len() + 1 + parts[n-3].len();
        &stem[..stem.len() - trim]
    } else {
        stem
    }
}

pub fn resolve_model_path(models_dir: &Path, model_id: &str) -> Option<PathBuf> {
    let direct = models_dir.join(model_id);
    if direct.is_dir() {
        let ok = direct.join("config.json").exists() || find_gguf_file(&direct).is_some();
        if ok {
            return Some(direct);
        }
    }
    let entries = std::fs::read_dir(models_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if let Some(gguf_paths) = find_gguf_files(&path) {
            for gguf_path in &gguf_paths {
                let raw_stem = gguf_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let stem = strip_gguf_split_suffix(&raw_stem);
                if stem.eq_ignore_ascii_case(model_id) {
                    return Some(path);
                }
            }
        }
    }
    None
}

fn resolve_weight_paths(model_dir: &str) -> anyhow::Result<Vec<String>> {
    let index_path = format!("{}/model.safetensors.index.json", model_dir);

    if std::path::Path::new(&index_path).exists() {
        let raw = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&raw)?;

        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Missing weight_map in {}", index_path))?;

        let mut seen = std::collections::HashSet::new();
        let mut files: Vec<String> = Vec::new();
        for filename in weight_map.values() {
            let name = filename
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Expected string value in weight_map, got {:?}", filename))?;
            if seen.insert(name.to_string()) {
                files.push(format!("{}/{}", model_dir, name));
            }
        }
        files.sort();
        println!("Total shared weight files: {}", files.len());
        Ok(files)
    } else {
        // No index file: try the standard single-file names in order.
        // Some repos (e.g. Mistral-7B-Instruct-v0.3) ship a `consolidated.safetensors`
        // as an alternative to the sharded layout.
        for name in &["model.safetensors", "consolidated.safetensors"] {
            let path = format!("{}/{}", model_dir, name);
            if std::path::Path::new(&path).exists() {
                return Ok(vec![path]);
            }
        }
        Ok(vec![format!("{}/model.safetensors", model_dir)])
    }
}

pub fn find_gguf_files(dir: &Path) -> Option<Vec<std::path::PathBuf>> {
    let index_path = dir.join("gguf.index");
    if index_path.exists()
        && let Ok(content) = std::fs::read_to_string(&index_path) {
            let files: Vec<std::path::PathBuf> = content
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(|l| dir.join(l))
                .collect();
            if !files.is_empty() {
                return Some(files);
            }
        }
    find_gguf_file(dir).map(|p| vec![p])
}

pub fn find_gguf_file(dir: &Path) -> Option<std::path::PathBuf> {
    let index_path = dir.join("gguf.index");
    if index_path.exists()
        && let Ok(content) = std::fs::read_to_string(&index_path)
            && let Some(first) = content
                .lines()
                .map(|l| l.trim())
                .find(|l| !l.is_empty() && !l.starts_with('#'))
            {
                return Some(dir.join(first));
            }
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Some(p);
        }
    }
    None
}

fn discover_gguf_model(id: &str, gguf_path: &Path) -> Option<DiscoveredModel> {
    use candle_core::quantized::gguf_file;
    let mut file = std::fs::File::open(gguf_path).ok()?;
    let content = gguf_file::Content::read(&mut file).ok()?;

    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok()).cloned()
        .unwrap_or_else(|| "unknown".to_string());

    let prefix = &arch;
    let num_layers = content
        .metadata
        .get(&format!("{prefix}.block_count"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;
    let _hidden_size = content
        .metadata
        .get(&format!("{prefix}.embedding_length"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;

    let vocab_size = content
        .tensor_infos
        .get("token_embd.weight")
        .map(|info| info.shape.dims()[0])
        .unwrap_or(0);

    let arch_display = match arch.as_str() {
        "llama" => "LlamaForCausalLM (GGUF)".to_string(),
        "qwen2" => "Qwen2ForCausalLM (GGUF)".to_string(),
        "qwen3" => "Qwen3ForCausalLM (GGUF)".to_string(),
        "gemma" => "GemmaForCausalLM (GGUF)".to_string(),
        "gemma2" => "Gemma2ForCausalLM (GGUF)".to_string(),
        "gemma3" => "Gemma3ForCausalLM (GGUF)".to_string(),
        "gemma4" => "Gemma4ForConditionalGeneration (GGUF)".to_string(),
        other => format!("{} (GGUF)", other),
    };

    Some(DiscoveredModel {
        id: id.to_string(),
        architecture: arch_display,
        vocab_size,
        num_layers,
    })
}

pub fn is_gguf_model(model_dir: &str) -> bool {
    let dir = Path::new(model_dir);
    if dir.join("config.json").exists() {
        return false;
    }
    find_gguf_file(dir).is_some()
}

pub fn select_device_at(_cuda_idx: usize) -> anyhow::Result<Device> {
    #[cfg(feature = "cuda")]
    match Device::new_cuda(_cuda_idx) {
        Ok(d) => {
            println!("Device: CUDA:{}", _cuda_idx);
            return Ok(d);
        }
        Err(e) => eprintln!("CUDA:{} not available: {e}", _cuda_idx),
    }

    #[cfg(feature = "metal")]
    match Device::new_metal(0) {
        Ok(d) => {
            println!("Device: Metal");
            return Ok(d);
        }
        Err(e) => eprintln!("Metal not available: {e}"),
    }

    println!("Device: CPU");
    Ok(Device::Cpu)
}

pub fn load_batch_model(
    model_dir: &str,
    model_id: &str,
    device: &Device,
    max_context_len: usize,
    max_num_sequences: usize,
    kv_budget: &SharedGlobalKvBudget,
    kv_quant: KvQuantMode,
) -> anyhow::Result<(Box<dyn BatchModel>, usize)> {
    if is_gguf_model(model_dir) {
        return load_batch_model_gguf(model_dir, model_id, device, max_context_len, max_num_sequences, kv_budget, kv_quant);
    }

    let dtype = if matches!(device, Device::Cpu) { DType::F32 } else { DType::BF16 };
    let cfg = hf_parser::parse(&format!("{}/config.json", model_dir))?;
    load_standard_safetensors(cfg, model_dir, device, dtype, max_context_len, max_num_sequences, kv_budget, kv_quant)
}

fn load_standard_safetensors(
    cfg: StandardTransformerConfig,
    model_dir: &str,
    device: &Device,
    dtype: DType,
    max_context_len: usize,
    max_num_sequences: usize,
    kv_budget: &SharedGlobalKvBudget,
    kv_quant: KvQuantMode,
) -> anyhow::Result<(Box<dyn BatchModel>, usize)> {
    let weight_paths = resolve_weight_paths(model_dir)?;
    let weight_path_refs: Vec<&str> = weight_paths.iter().map(|s| s.as_str()).collect();
    let weights = ModelWeights::load(&weight_path_refs, device, dtype)?;
    let weights_size = weights.total_size_bytes();

    let num_layers = cfg.num_hidden_layers;
    let per_layer_head_dims = cfg
        .per_layer_head_dims
        .clone()
        .filter(|v| v.len() == num_layers)
        .unwrap_or_else(|| vec![cfg.head_dim; num_layers]);
    let per_layer_kv_heads = cfg
        .per_layer_num_key_value_heads
        .clone()
        .filter(|v| v.len() == num_layers)
        .unwrap_or_else(|| vec![cfg.num_key_value_heads; num_layers]);
    let per_layer_sliding_windows = cfg
        .per_layer_sliding_windows
        .clone()
        .filter(|v| v.len() == num_layers)
        .unwrap_or_else(|| vec![cfg.sliding_window; num_layers]);
    let per_layer_rope_thetas = cfg
        .per_layer_rope_thetas
        .clone()
        .filter(|v| v.len() == num_layers)
        .unwrap_or_else(|| vec![cfg.rope_theta; num_layers]);

    let layer_kv_specs: Vec<(usize, usize)> = per_layer_kv_heads
        .iter()
        .copied()
        .zip(per_layer_head_dims.iter().copied())
        .collect();

    let ctx = max_context_len.min(cfg.max_position_embeddings);
    let (num_blocks, acquired_kv_bytes) = compute_kv_blocks(
        &KvBlockParams {
            layer_kv_specs: layer_kv_specs.clone(),
            max_context_len: ctx,
            max_num_sequences,
            dtype,
            kv_quant,
        },
        kv_budget,
    )?;

    let layer_quantizers: Vec<Option<Arc<KvQuantizer>>> = match kv_quant {
        KvQuantMode::Off => vec![None; num_layers],
        mode => per_layer_head_dims
            .iter()
            .map(|&hd| Some(Arc::new(KvQuantizer::new(mode.bit_width(), hd))))
            .collect(),
    };

    let result = (|| -> anyhow::Result<(Box<dyn BatchModel>, usize)> {
        let embed_weight = weights.get("model.embed_tokens.weight")
            .map_err(|e| anyhow::anyhow!("{e}"))?.clone();
        let lm_head = if cfg.tie_word_embeddings {
            AnyLinear::Float(Linear::new(embed_weight.clone(), None))
        } else {
            AnyLinear::Float(Linear::new(
                weights.get("lm_head.weight").map_err(|e| anyhow::anyhow!("{e}"))?.clone(),
                None,
            ))
        };
        let embed_tokens = Embedding::new(embed_weight);

        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| {
                let mut block_cfg = cfg.block_config();
                block_cfg.head_dim = per_layer_head_dims[i];
                block_cfg.n_kv_heads = per_layer_kv_heads[i];
                block_cfg.sliding_window = per_layer_sliding_windows[i];
                TransformerBlock::load(&block_cfg, i, &weights)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let norm = RMSNorm::load(&weights, "model.norm", cfg.rms_norm_eps, cfg.norm_type)?;

        let ropes = (0..cfg.num_hidden_layers)
            .map(|i| {
                RotaryEmbedding::new_with_scaling(
                    per_layer_head_dims[i],
                    ctx,
                    per_layer_rope_thetas[i],
                    cfg.rope_scaling.clone(),
                    dtype,
                    device,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let allocators = (0..cfg.num_hidden_layers)
            .map(|i| -> candle_core::Result<SharedBlockAllocator> {
                Ok(Arc::new(Mutex::new(BlockAllocator::new(
                    num_blocks,
                    DEFAULT_BLOCK_SIZE,
                    per_layer_kv_heads[i],
                    per_layer_head_dims[i],
                    dtype,
                    device,
                    layer_quantizers[i].clone(),
                )?)))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let has_per_layer_stream = cfg.per_layer_input_hidden_size.is_some()
            && cfg.per_layer_input_vocab_size.is_some()
            && weights.try_get("model.embed_tokens_per_layer.weight").is_some()
            && weights.try_get("model.per_layer_model_projection.weight").is_some()
            && weights.try_get("model.per_layer_projection_norm.weight").is_some();

        let per_layer_input_embed = if has_per_layer_stream {
            Some(Embedding::new(
                weights
                    .get("model.embed_tokens_per_layer.weight")
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .clone(),
            ))
        } else {
            None
        };
        let per_layer_model_projection = if has_per_layer_stream {
            Some(Linear::new(
                weights
                    .get("model.per_layer_model_projection.weight")
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .clone(),
                None,
            ))
        } else {
            None
        };
        let per_layer_projection_norm = if has_per_layer_stream {
            Some(RMSNorm::load(
                &weights,
                "model.per_layer_projection_norm",
                cfg.rms_norm_eps,
                cfg.norm_type,
            )?)
        } else {
            None
        };

        Ok((Box::new(StandardTransformer {
            embed_tokens,
            blocks,
            norm,
            lm_head,
            ropes,
            allocators,
            device: device.clone(),
            stop_token_ids: cfg.eos_token_ids,
            vocab_size: cfg.vocab_size,
            max_seq_len: ctx,
            embed_scale: cfg.embed_scale,
            logit_softcap: cfg.logit_softcap,
            per_layer_input_embed,
            per_layer_input_embed_scale: cfg.per_layer_input_embed_scale,
            per_layer_model_projection,
            per_layer_model_projection_scale: cfg.per_layer_model_projection_scale,
            per_layer_projection_norm,
            per_layer_input_scale: cfg.per_layer_input_scale,
            kv_shared_layer_map: cfg.kv_shared_layer_map.clone(),
        }), weights_size))
    })();

    if result.is_err() {
        kv_budget.release(acquired_kv_bytes);
    }
    result
}

fn load_batch_model_gguf(
    model_dir: &str,
    model_id: &str,
    device: &Device,
    max_context_len: usize,
    max_num_sequences: usize,
    kv_budget: &SharedGlobalKvBudget,
    kv_quant: KvQuantMode,
) -> anyhow::Result<(Box<dyn BatchModel>, usize)> {
    let dir = Path::new(model_dir);
    let all_gguf_paths = find_gguf_files(dir)
        .ok_or_else(|| anyhow::anyhow!("No .gguf file found in {}", model_dir))?;

    let gguf_paths: Vec<PathBuf> = all_gguf_paths
        .iter()
        .filter(|p| {
            let raw_stem = p
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let stem = strip_gguf_split_suffix(&raw_stem);
            stem.eq_ignore_ascii_case(model_id)
        })
        .cloned()
        .collect();

    let gguf_paths = if gguf_paths.is_empty() { all_gguf_paths } else { gguf_paths };

    if gguf_paths.len() == 1 {
        println!("Loading GGUF model from '{}'", gguf_paths[0].display());
    } else {
        println!(
            "Loading GGUF model from '{}' ({} shards)",
            gguf_paths[0].display(),
            gguf_paths.len()
        );
    }
    let gguf_path_strs: Vec<&str> = gguf_paths
        .iter()
        .map(|p| p.to_str().expect("non-UTF8 path"))
        .collect();
    let gguf = GgufWeights::load_shards(&gguf_path_strs, device)?;

    let arch = gguf.architecture()?;
    println!("[gguf] Architecture: {}", arch);

    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };

    let weights_size = gguf.total_size_bytes();

    let topo = crate::models::gguf_model::parse_gguf_topology(&gguf)?;
    let ctx = max_context_len.min(topo.context_length);
    let layer_kv_specs = vec![(topo.num_key_value_heads, topo.head_dim); topo.num_hidden_layers];
    let (num_blocks, acquired_kv_bytes) = compute_kv_blocks(
        &KvBlockParams {
            layer_kv_specs,
            max_context_len: ctx,
            max_num_sequences,
            dtype,
            kv_quant,
        },
        kv_budget,
    )?;

    let quantizer: Option<Arc<KvQuantizer>> = match kv_quant {
        KvQuantMode::Off => None,
        mode => Some(Arc::new(KvQuantizer::new(mode.bit_width(), topo.head_dim))),
    };

    let model = match StandardTransformer::load_gguf(&gguf, device, dtype, num_blocks, quantizer) {
        Ok(m) => m,
        Err(e) => {
            kv_budget.release(acquired_kv_bytes);
            return Err(e.into());
        }
    };
    Ok((Box::new(model), weights_size))
}

struct KvBlockParams {
    layer_kv_specs: Vec<(usize, usize)>,
    max_context_len: usize,
    max_num_sequences: usize,
    dtype: DType,
    kv_quant: KvQuantMode,
}

fn compute_kv_blocks(p: &KvBlockParams, kv_budget: &GlobalKvBudget) -> anyhow::Result<(usize, usize)> {
    let total_slots = p.max_num_sequences * p.max_context_len;
    let desired_blocks = total_slots.div_ceil(DEFAULT_BLOCK_SIZE);
    let min_blocks: usize = 256; // ~4 096 token minimum context

    // Bytes for one block summed across all layers (K + V).
    let per_block_bytes = match p.kv_quant {
        KvQuantMode::Off => p
            .layer_kv_specs
            .iter()
            .map(|(n_kv_heads, head_dim)| {
                DEFAULT_BLOCK_SIZE * (*n_kv_heads) * (*head_dim) * p.dtype.size_in_bytes() * 2
            })
            .sum::<usize>(),
        mode => p
            .layer_kv_specs
            .iter()
            .map(|(n_kv_heads, head_dim)| {
                let bph = kv_quant::quantized_bytes_per_head(*head_dim, mode.bit_width());
                DEFAULT_BLOCK_SIZE * (*n_kv_heads) * bph * 2
            })
            .sum::<usize>(),
    };

    if per_block_bytes == 0 {
        return Ok((desired_blocks, 0));
    }

    let desired_bytes = desired_blocks.max(min_blocks) * per_block_bytes;
    let granted_bytes = kv_budget.acquire(desired_bytes);
    let granted_blocks = granted_bytes / per_block_bytes;

    if granted_blocks < min_blocks {
        kv_budget.release(granted_bytes);
        anyhow::bail!(
            "KV cache budget exhausted: requested {} blocks ({:.2} GB minimum) \
             but only {} blocks ({:.2} GB) available",
            min_blocks,
            min_blocks as f64 * per_block_bytes as f64 / 1_073_741_824.0,
            granted_blocks,
            granted_blocks as f64 * per_block_bytes as f64 / 1_073_741_824.0,
        );
    }

    if granted_blocks < desired_blocks {
        println!(
            "[kv-pool] KV cache capped: {} → {} blocks ({:.2} → {:.2} GB), \
             pool remaining: {:.2} GB",
            desired_blocks,
            granted_blocks,
            desired_blocks as f64 * per_block_bytes as f64 / 1_073_741_824.0,
            granted_blocks as f64 * per_block_bytes as f64 / 1_073_741_824.0,
            kv_budget.available_bytes() as f64 / 1_073_741_824.0,
        );
    }

    Ok((granted_blocks, granted_bytes))
}
