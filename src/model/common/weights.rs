use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

pub struct ModelWeights {
    pub tensors: HashMap<String, Tensor>,
}

impl ModelWeights {
    pub fn load(paths: &[&str], device: &Device, dtype: DType) -> Result<Self> {
        let mut tensors = HashMap::new();
        for path in paths {
            let loaded = candle_core::safetensors::load(path, device)
                .with_context(|| format!("Errore caricamento {}", path))?;
            for (name, tensor) in loaded {
                let tensor = tensor.to_dtype(dtype)?;
                tensors.insert(name, tensor);
            }
        }
        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str) -> candle_core::Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Peso non trovato: {}", name)))
    }
}
