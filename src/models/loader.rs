use candle_core::{DType, Device};
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

pub fn load_batch_model(
    model_dir: &str,
    device: &Device,
    kv_block_multiplier: usize,
) -> anyhow::Result<Box<dyn BatchModel>> {
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
            Ok(Box::new(Qwen3::load(cfg, &weights, device, dtype, kv_block_multiplier)?))
        }
        "LlamaForCausalLM" => {
            let cfg = LlamaConfig::from_file(&format!("{}/config.json", model_dir))?;
            let weight_paths = resolve_weight_paths(model_dir)?;
            let weight_path_refs: Vec<&str> = weight_paths.iter().map(|s| s.as_str()).collect();
            let weights = ModelWeights::load(&weight_path_refs, device, dtype)?;
            Ok(Box::new(Llama::load(cfg, &weights, device, dtype, kv_block_multiplier)?))
        }
        other => anyhow::bail!("Architecture not supported: {}", other),
    }
}
