use anyhow::{Context, Result};
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Tensor};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

pub struct ModelWeights {
    tensors: FxHashMap<String, Tensor>,
}

impl ModelWeights {
    pub fn load(paths: &[&str], device: &Device, dtype: DType) -> Result<Self> {
        let mmap = unsafe {
            MmapedSafetensors::multi(paths)
                .context("Failed to memory-map weight files")?
        };

        let names: Vec<String> = mmap.tensors().into_iter().map(|(n, _)| n).collect();

        let tensors: FxHashMap<String, Tensor> = names
            .par_iter()
            .map(|name| {
                let t = mmap
                    .load(name, device)
                    .with_context(|| format!("Failed to load tensor {}", name))?;
                let t = t.to_dtype(dtype)?;
                Ok((name.clone(), t))
            })
            .collect::<Result<_>>()?;

        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str) -> candle_core::Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Tensor not found: {}", name)))
    }
}
