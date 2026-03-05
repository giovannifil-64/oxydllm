use candle_core::{DType, Device};
use crate::common::paged::DEFAULT_BLOCK_SIZE;
use crate::common::weights::ModelWeights;
use crate::models::traits::BatchModel;
use crate::models::qwen3::{config::Qwen3Config, model::Qwen3};
use crate::models::llama::{config::LlamaConfig, model::Llama};

use std::path::Path;

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
        let config_path = path.join("config.json");
        if !config_path.exists() {
            continue;
        }
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
        let id = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        models.push(DiscoveredModel {
            id,
            architecture,
            vocab_size,
            num_layers,
        });
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));
    models
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
        Ok(vec![format!("{}/model.safetensors", model_dir)])
    }
}

pub fn select_device_at(_cuda_idx: usize) -> anyhow::Result<Device> {
    #[cfg(feature = "cuda")]
    match Device::new_cuda(_cuda_idx) {
        Ok(d) => {
            println!("Device: CUDA:{}", cuda_idx);
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

/// Loads a model and returns both the model and its **real** in-memory footprint.
/// The footprint is computed from the tensors after dtype conversion (BF16 on GPU,
/// F32 on CPU), which is more accurate than reading raw file sizes from disk.
///
/// `max_context_len` is the maximum number of tokens per sequence the KV cache should
/// support.  `max_num_sequences` is the scheduler's concurrency limit.  Together they
/// determine how many KV blocks are pre-allocated.
/// A safety cap based on available system RAM prevents OOM.
pub fn load_batch_model(
    model_dir: &str,
    device: &Device,
    max_context_len: usize,
    max_num_sequences: usize,
) -> anyhow::Result<(Box<dyn BatchModel>, usize)> {
    let raw = std::fs::read_to_string(format!("{}/config.json", model_dir))?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let arch = value["architectures"][0].as_str().unwrap_or("Unknown");

    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };

    match arch {
        "Qwen3ForCausalLM" => {
            let cfg = Qwen3Config::from_file(&format!("{}/config.json", model_dir))?;
            let weight_paths = resolve_weight_paths(model_dir)?;
            let weight_path_refs: Vec<&str> = weight_paths.iter().map(|s| s.as_str()).collect();
            let weights = ModelWeights::load(&weight_path_refs, device, dtype)?;
            let weights_size = weights.total_size_bytes();
            let ctx = max_context_len.min(cfg.max_position_embeddings);
            let num_blocks = compute_kv_blocks(
                cfg.num_hidden_layers, cfg.num_key_value_heads,
                cfg.head_dim(), ctx, max_num_sequences,
                dtype, weights_size, device,
            );
            Ok((Box::new(Qwen3::load(cfg, &weights, device, dtype, num_blocks)?), weights_size))
        }
        "LlamaForCausalLM" => {
            let cfg = LlamaConfig::from_file(&format!("{}/config.json", model_dir))?;
            let weight_paths = resolve_weight_paths(model_dir)?;
            let weight_path_refs: Vec<&str> = weight_paths.iter().map(|s| s.as_str()).collect();
            let weights = ModelWeights::load(&weight_path_refs, device, dtype)?;
            let weights_size = weights.total_size_bytes();
            let ctx = max_context_len.min(cfg.max_position_embeddings);
            let num_blocks = compute_kv_blocks(
                cfg.num_hidden_layers, cfg.num_key_value_heads,
                cfg.head_dim(), ctx, max_num_sequences,
                dtype, weights_size, device,
            );
            Ok((Box::new(Llama::load(cfg, &weights, device, dtype, num_blocks)?), weights_size))
        }
        other => anyhow::bail!("Architecture not supported: {}", other),
    }
}

// ---------------------------------------------------------------------------
// KV cache memory-aware block computation
// ---------------------------------------------------------------------------

/// Detects total system memory in bytes.
/// On macOS uses `sysctl hw.memsize`; on Linux reads `/proc/meminfo`.
fn detect_system_memory_bytes() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .ok()?;
        let s = std::str::from_utf8(&output.stdout).ok()?.trim();
        return s.parse::<usize>().ok();
    }
    #[cfg(target_os = "linux")]
    {
        let content = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                let kb: usize = parts.get(1)?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        return None;
    }
    #[allow(unreachable_code)]
    None
}

/// Computes the number of KV blocks to allocate, capping based on available
/// system memory so that a single model doesn't exhaust all RAM/VRAM.
///
/// The allocation strategy:
///   1. Compute the *desired* blocks from `max_num_sequences × max_context_len`.
///   2. Detect system memory and reserve 65% for the model (weights + KV).
///   3. Cap KV blocks so they fit in the remaining budget after weights.
///   4. Guarantee a minimum of 256 blocks (~4096 tokens of context).
fn compute_kv_blocks(
    num_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_context_len: usize,
    max_num_sequences: usize,
    dtype: DType,
    weights_size: usize,
    device: &Device,
) -> usize {
    // Total token slots needed = sequences × context per sequence.
    let total_slots = max_num_sequences * max_context_len;
    let desired_blocks = (total_slots + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;

    // Cost of one KV block summed across all layers (K + V pools).
    let per_block_bytes =
        DEFAULT_BLOCK_SIZE * n_kv_heads * head_dim * dtype.size_in_bytes() * 2 * num_layers;

    if per_block_bytes == 0 {
        return desired_blocks;
    }

    let total_mem = match detect_system_memory_bytes() {
        Some(m) => m,
        None => {
            println!("[memory] Could not detect system memory — using full KV allocation");
            return desired_blocks;
        }
    };

    // On Metal (macOS unified memory) the GPU shares RAM with the OS/apps.
    // Use 65% as a safe upper bound for the model working set.
    // On CUDA with dedicated VRAM the system-memory heuristic is conservative
    // but still prevents obviously absurd allocations.
    let usable_fraction = if matches!(device, Device::Cpu) { 0.80 } else { 0.65 };
    let usable = (total_mem as f64 * usable_fraction) as usize;
    // Reserve 512 MB for activations, intermediates, etc.
    let headroom: usize = 512 * 1024 * 1024;
    let available_for_kv = usable.saturating_sub(weights_size).saturating_sub(headroom);

    let max_blocks = available_for_kv / per_block_bytes;
    let min_blocks: usize = 256; // ~4096 tokens minimum context

    let capped = desired_blocks.min(max_blocks).max(min_blocks);

    if capped < desired_blocks {
        let desired_kv = desired_blocks as u64 * per_block_bytes as u64;
        let capped_kv = capped as u64 * per_block_bytes as u64;
        let capped_ctx = capped * DEFAULT_BLOCK_SIZE;
        println!(
            "[memory] KV cache capped by available memory: {:.2} GB → {:.2} GB \
             ({} → {} blocks, ~{} effective token slots, {:.1} GB system RAM detected)",
            desired_kv as f64 / 1_073_741_824.0,
            capped_kv as f64 / 1_073_741_824.0,
            desired_blocks,
            capped,
            capped_ctx,
            total_mem as f64 / 1_073_741_824.0,
        );
    }

    capped
}
