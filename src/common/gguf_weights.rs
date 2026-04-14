use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use candle_core::Device;
use candle_core::quantized::QTensor;
use candle_core::quantized::gguf_file;
use rustc_hash::FxHashMap;

pub struct GgufWeights {
    tensors: FxHashMap<String, Arc<QTensor>>,
    pub metadata: HashMap<String, gguf_file::Value>,
}

impl GgufWeights {
    pub fn load(path: &str, device: &Device) -> anyhow::Result<Self> {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file: {}", path))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF header: {}", e))?;

        tracing::info!(
            tensors = content.tensor_infos.len(),
            metadata_entries = content.metadata.len(),
            "GGUF metadata parsed"
        );

        let mut tensors = FxHashMap::default();
        for name in content.tensor_infos.keys() {
            let qt = content
                .tensor(&mut file, name, device)
                .map_err(|e| anyhow::anyhow!("Failed to load GGUF tensor '{}': {}", name, e))?;
            tensors.insert(name.clone(), Arc::new(qt));
        }

        Ok(Self {
            tensors,
            metadata: content.metadata,
        })
    }

    pub fn get(&self, name: &str) -> candle_core::Result<Arc<QTensor>> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("GGUF tensor not found: {}", name)))
    }

    pub fn try_get(&self, name: &str) -> Option<Arc<QTensor>> {
        self.tensors.get(name).cloned()
    }

    pub fn total_size_bytes(&self) -> usize {
        self.tensors
            .values()
            .map(|qt| qt.storage_size_in_bytes())
            .sum()
    }

    pub fn metadata_u32(&self, key: &str) -> anyhow::Result<u32> {
        self.metadata
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("Missing GGUF metadata key: {}", key))
            .and_then(|v| {
                v.to_u32()
                    .map_err(|e| anyhow::anyhow!("Bad u32 for '{}': {}", key, e))
            })
    }

    pub fn metadata_f32(&self, key: &str) -> anyhow::Result<f32> {
        self.metadata
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("Missing GGUF metadata key: {}", key))
            .and_then(|v| {
                v.to_f32()
                    .map_err(|e| anyhow::anyhow!("Bad f32 for '{}': {}", key, e))
            })
    }

    pub fn metadata_string(&self, key: &str) -> anyhow::Result<String> {
        self.metadata
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("Missing GGUF metadata key: {}", key))
            .and_then(|v| {
                v.to_string()
                    .cloned()
                    .map_err(|e| anyhow::anyhow!("Bad string for '{}': {}", key, e))
            })
    }

    pub fn metadata_u32_or(&self, key: &str, default: u32) -> u32 {
        self.metadata_u32(key).unwrap_or(default)
    }

    pub fn metadata_f32_or(&self, key: &str, default: f32) -> f32 {
        self.metadata_f32(key).unwrap_or(default)
    }

    pub fn load_shards(paths: &[&str], device: &Device) -> anyhow::Result<Self> {
        anyhow::ensure!(!paths.is_empty(), "load_shards: paths must be non-empty");
        if paths.len() == 1 {
            return Self::load(paths[0], device);
        }
        let mut tensors = FxHashMap::default();
        let mut metadata = HashMap::new();
        let total_shards = paths.len();
        let mut total_tensors = 0usize;
        for (shard_idx, path) in paths.iter().enumerate() {
            let mut file = std::fs::File::open(path)
                .with_context(|| format!("Failed to open GGUF shard: {}", path))?;
            let content = gguf_file::Content::read(&mut file)
                .map_err(|e| anyhow::anyhow!("Failed to parse GGUF shard '{}': {}", path, e))?;
            if shard_idx == 0 {
                metadata = content.metadata.clone();
                tracing::info!(
                    shard = shard_idx + 1,
                    total_shards,
                    tensors = content.tensor_infos.len(),
                    metadata_entries = content.metadata.len(),
                    "GGUF shard parsed"
                );
            } else {
                tracing::info!(
                    shard = shard_idx + 1,
                    total_shards,
                    tensors = content.tensor_infos.len(),
                    "GGUF shard parsed"
                );
            }
            total_tensors += content.tensor_infos.len();
            for name in content.tensor_infos.keys() {
                let qt = content.tensor(&mut file, name, device).map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to load tensor '{}' from shard '{}': {}",
                        name,
                        path,
                        e
                    )
                })?;
                tensors.insert(name.clone(), Arc::new(qt));
            }
        }
        tracing::info!(
            total_tensors,
            total_shards,
            "GGUF tensors loaded from shards"
        );
        Ok(Self { tensors, metadata })
    }

    pub fn architecture(&self) -> anyhow::Result<String> {
        self.metadata_string("general.architecture")
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        if let Ok(eos) = self.metadata_u32("tokenizer.ggml.eos_token_id") {
            ids.push(eos);
        }
        if let Some(gguf_file::Value::Array(arr)) =
            self.metadata.get("tokenizer.ggml.eos_token_ids")
        {
            for v in arr {
                if let gguf_file::Value::U32(id) = v
                    && !ids.contains(id)
                {
                    ids.push(*id);
                }
            }
        }
        ids
    }
}
