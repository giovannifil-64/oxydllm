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
    preserve_fp8_weight: bool,
    force_f32: bool,
) -> Result<Tensor> {
    let t = mmap
        .load(name, device)
        .with_context(|| format!("Failed to load tensor {}", name))?;

    if preserve_fp8_weight && t.dtype() == DType::F8E4M3 {
        // Level-2 FP8 path: keep FP8 weights resident and dequantize on-the-fly in linear layers.
        return Ok(t);
    }

    let target_dtype = if force_f32 { DType::F32 } else { dtype };

    if t.dtype() == DType::F8E4M3 {
        // Some backends fail or produce unstable results on direct FP8 -> BF16 casts.
        // Route through F32 explicitly for consistent checkpoint compatibility.
        return t
            .to_dtype(DType::F32)
            .and_then(|t| t.to_dtype(target_dtype))
            .with_context(|| {
                format!(
                    "Failed to cast FP8 tensor {} to {:?} via F32 conversion path",
                    name, target_dtype
                )
            });
    }

    match t.to_dtype(target_dtype) {
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
            t_on_device.to_dtype(target_dtype).with_context(|| {
                format!(
                    "Failed to cast tensor {} to {:?} after CPU fallback (original error: {})",
                    name, target_dtype, device_cast_err
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

        if weight.dtype() == DType::F8E4M3 {
            // Level-2 FP8 path keeps quantized weights + scales for runtime dequantization.
            continue;
        }

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
        let scale_inv_names: std::collections::HashSet<String> = names
            .iter()
            .filter(|name| name.ends_with(".weight_scale_inv"))
            .cloned()
            .collect();

        let mut tensors: FxHashMap<String, Tensor> = names
            .iter()
            .map(|name| {
                let preserve_fp8_weight = name.ends_with(".weight")
                    && scale_inv_names.contains(&format!("{}_scale_inv", name));
                let force_f32 = name.ends_with(".weight_scale_inv");
                let t = load_tensor_with_dtype(
                    &mmap,
                    name,
                    device,
                    dtype,
                    preserve_fp8_weight,
                    force_f32,
                )?;
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

        if name == "lm_head.weight_scale_inv" {
            for alias in [
                "model.lm_head.weight_scale_inv",
                "model.language_model.lm_head.weight_scale_inv",
                "model.language_model.model.lm_head.weight_scale_inv",
                "language_model.lm_head.weight_scale_inv",
                "language_model.model.lm_head.weight_scale_inv",
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

    pub fn try_get_scale_inv(&self, weight_name: &str) -> Option<&Tensor> {
        self.try_get(&format!("{}_scale_inv", weight_name))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_weight_scale_inv_scales_non_fp8_weights() -> Result<()> {
        let device = Device::Cpu;
        let mut tensors = FxHashMap::default();

        tensors.insert(
            "layer.weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?,
        );
        tensors.insert(
            "layer.weight_scale_inv".to_string(),
            Tensor::from_vec(vec![2.0f32, 0.5], (2, 1), &device)?,
        );

        apply_weight_scale_inv(&mut tensors)?;

        let scaled = tensors
            .get("layer.weight")
            .expect("scaled tensor should exist")
            .to_vec2::<f32>()?;
        assert_eq!(scaled, vec![vec![2.0, 4.0], vec![1.5, 2.0]]);
        Ok(())
    }

    #[test]
    fn apply_weight_scale_inv_skips_fp8_weights() -> Result<()> {
        let device = Device::Cpu;
        let mut tensors = FxHashMap::default();

        let weight_fp8 = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, -4.0], (2, 2), &device)?
            .to_dtype(DType::F8E4M3)?;
        let before = weight_fp8.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        tensors.insert("layer.weight".to_string(), weight_fp8);
        tensors.insert(
            "layer.weight_scale_inv".to_string(),
            Tensor::from_vec(vec![10.0f32, 10.0], (2, 1), &device)?,
        );

        apply_weight_scale_inv(&mut tensors)?;

        let after_weight = tensors
            .get("layer.weight")
            .expect("fp8 tensor should exist after scaling pass");
        assert_eq!(after_weight.dtype(), DType::F8E4M3);

        let after = after_weight.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        assert_eq!(before, after);
        Ok(())
    }
}
