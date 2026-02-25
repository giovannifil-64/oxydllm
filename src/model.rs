pub mod traits;
pub mod common;
mod qwen3;

pub use traits::{generate, Model};

use candle_core::{DType, Device};
use common::weights::ModelWeights;
use qwen3::{config::Qwen3Config, model::Qwen3};

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

pub fn select_device() -> anyhow::Result<Device> {
    #[cfg(feature = "cuda")]
    match Device::new_cuda(0) {
        Ok(d) => {
            println!("Device: CUDA");
            return Ok(d);
        }
        Err(e) => eprintln!("CUDA not available: {e}"),
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

pub fn load_model(model_dir: &str, device: &Device) -> anyhow::Result<Box<dyn Model>> {
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
            Ok(Box::new(Qwen3::load(cfg, &weights, device)?))
        }
        other => anyhow::bail!("Architecture not supported: {}", other),
    }
}
