use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, safetensors::MmapedSafetensors};
use rustc_hash::FxHashMap;

pub struct ModelWeights {
    tensors: FxHashMap<String, Tensor>,
}

fn load_tensor_with_dtype(
    mmap: &MmapedSafetensors,
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let t = mmap
        .load(name, device)
        .with_context(|| format!("Failed to load tensor {}", name))?;

    match t.to_dtype(dtype) {
        Ok(t) => Ok(t),
        Err(device_cast_err) => {
            let t_cpu = mmap
                .load(name, &Device::Cpu)
                .with_context(|| format!("Failed to reload tensor {} on CPU", name))?;
            let t_cpu_f32 = if t_cpu.dtype() == DType::F32 {
                t_cpu
            } else {
                t_cpu
                    .to_dtype(DType::F32)
                    .with_context(|| format!("Failed to cast tensor {} to F32 on CPU", name))?
            };
            let t_on_device = t_cpu_f32.to_device(device).with_context(|| {
                format!(
                    "Failed to move tensor {} back to target device after CPU fallback",
                    name
                )
            })?;
            t_on_device.to_dtype(dtype).with_context(|| {
                format!(
                    "Failed to cast tensor {} to {:?} after CPU fallback (original error: {})",
                    name, dtype, device_cast_err
                )
            })
        }
    }
}

fn apply_weight_scale_inv(tensors: &mut FxHashMap<String, Tensor>) -> Result<()> {
    let weight_names: Vec<String> = tensors
        .keys()
        .filter(|name| name.ends_with(".weight"))
        .cloned()
        .collect();

    for weight_name in weight_names {
        let scale_name = format!("{}_scale_inv", weight_name);
        let (Some(weight), Some(scale_inv)) = (
            tensors.get(&weight_name).cloned(),
            tensors.get(&scale_name).cloned(),
        ) else {
            continue;
        };

        // FP8 checkpoints (e.g. Mistral/Ministral variants) store
        // per-weight inverse scales next to quantized matrices.
        let scaled = weight.broadcast_mul(&scale_inv).with_context(|| {
            format!(
                "Failed to apply '{}' dequantization factor to '{}'",
                scale_name, weight_name
            )
        })?;
        tensors.insert(weight_name, scaled);
    }

    Ok(())
}

impl ModelWeights {
    pub fn load(paths: &[&str], device: &Device, dtype: DType) -> Result<Self> {
        let mmap = unsafe {
            MmapedSafetensors::multi(paths).context("Failed to memory-map weight files")?
        };

        let names: Vec<String> = mmap.tensors().into_iter().map(|(n, _)| n).collect();

        let mut tensors: FxHashMap<String, Tensor> = names
            .iter()
            .map(|name| {
                let t = load_tensor_with_dtype(&mmap, name, device, dtype)?;
                Ok((name.clone(), t))
            })
            .collect::<Result<_>>()?;

        apply_weight_scale_inv(&mut tensors)?;

        Ok(Self { tensors })
    }

    fn resolve_name<'a>(&'a self, name: &str) -> Option<&'a Tensor> {
        if let Some(t) = self.tensors.get(name) {
            return Some(t);
        }

        // Multimodal checkpoints may nest the text model under wrappers
        // like `model.language_model.model.*` while this runtime expects
        // `model.*` names.
        if let Some(rest) = name.strip_prefix("model.") {
            for alias in [
                format!("model.language_model.{rest}"),
                format!("model.language_model.model.{rest}"),
                format!("language_model.{rest}"),
                format!("language_model.model.{rest}"),
                format!("model.model.{rest}"),
            ] {
                if let Some(t) = self.tensors.get(&alias) {
                    return Some(t);
                }
            }
        }

        if name == "lm_head.weight" {
            for alias in [
                "model.lm_head.weight",
                "model.language_model.lm_head.weight",
                "model.language_model.model.lm_head.weight",
                "language_model.lm_head.weight",
                "language_model.model.lm_head.weight",
            ] {
                if let Some(t) = self.tensors.get(alias) {
                    return Some(t);
                }
            }
        }

        None
    }

    pub fn get(&self, name: &str) -> candle_core::Result<&Tensor> {
        self.resolve_name(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Tensor not found: {}", name)))
    }

    pub fn try_get(&self, name: &str) -> Option<&Tensor> {
        self.resolve_name(name)
    }

    /// Returns the actual memory footprint of all loaded tensors in bytes.
    /// This reflects the dtype used at load time (BF16 on GPU, F32 on CPU),
    /// which may differ from the on-disk size of the safetensors files.
    pub fn total_size_bytes(&self) -> usize {
        self.tensors
            .values()
            .map(|t| t.dtype().size_in_bytes() * t.elem_count())
            .sum()
    }
}

#[cfg(test)]
impl ModelWeights {
    /// Build ModelWeights from a pre-built tensor map (test-only).
    /// Allows regression tests to construct synthetic models without safetensors files.
    pub fn from_tensors(tensors: FxHashMap<String, Tensor>) -> Self {
        Self { tensors }
    }
}
