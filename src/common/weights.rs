use crate::common::awq::{AwqRawTensors, QuantWeight};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, safetensors::MmapedSafetensors};
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantScheme {
    Awq { bits: u32 },
    Gptq { bits: u32, sym: bool },
}

pub struct ModelWeights {
    tensors: FxHashMap<String, Tensor>,
    quant_scheme: Option<QuantScheme>,
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

    let target_dtype = if force_f32 { DType::F32 } else { dtype };

    if matches!(
        t.dtype(),
        DType::U8 | DType::U32 | DType::I16 | DType::I32 | DType::I64
    ) {
        return Ok(t);
    }

    if t.dtype() == target_dtype {
        return Ok(t);
    }

    if t.dtype() == DType::F8E4M3 {
        // Metal has no F8E4M3 compute kernels, so Fp8Linear's on-the-fly dequant
        // is unavailable there; dequantize at load on CPU and move to device.
        let use_cpu_path = matches!(device, Device::Metal(_));

        if preserve_fp8_weight && !use_cpu_path {
            return Ok(t);
        }

        let t_f32 = if use_cpu_path {
            mmap.load(name, &Device::Cpu)
                .with_context(|| format!("Failed to reload FP8 tensor {} on CPU", name))?
                .to_dtype(DType::F32)
                .with_context(|| format!("Failed to convert FP8 tensor {} to F32 on CPU", name))?
                .to_device(device)
                .with_context(|| {
                    format!(
                        "Failed to move tensor {} to device after CPU conversion",
                        name
                    )
                })?
        } else {
            t.to_dtype(DType::F32)
                .with_context(|| format!("Failed to cast FP8 tensor {} to F32", name))?
        };
        return t_f32.to_dtype(target_dtype).with_context(|| {
            format!(
                "Failed to cast FP8 tensor {} from F32 to {:?}",
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

/// Per-tensor / per-channel scales broadcast directly; block-wise scales
/// (DeepSeek- / Qwen3-FP8 style) are tiled up to the weight shape first.
pub(crate) fn apply_scale_inv(weight: &Tensor, scale_inv: &Tensor) -> candle_core::Result<Tensor> {
    if let Ok(scaled) = weight.broadcast_mul(scale_inv) {
        return Ok(scaled);
    }

    let (out, inn) = weight.dims2()?;
    let Ok((s_out, s_in)) = scale_inv.dims2() else {
        candle_core::bail!(
            "weight_scale_inv {:?} does not broadcast onto weight [{out}, {inn}] \
             and is not a 2-D block-scale grid",
            scale_inv.dims(),
        );
    };
    if s_out == 0 || s_in == 0 || out % s_out != 0 || inn % s_in != 0 {
        candle_core::bail!(
            "block-wise FP8 weight_scale_inv [{s_out}, {s_in}] does not tile \
             weight [{out}, {inn}] evenly"
        );
    }

    let (block_out, block_in) = (out / s_out, inn / s_in);
    scale_inv
        .reshape((s_out, 1, s_in, 1))?
        .broadcast_as((s_out, block_out, s_in, block_in))?
        .contiguous()?
        .reshape((out, inn))?
        .mul(weight)
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
            continue;
        }

        // Multiply in F32: BF16's 7-bit mantissa accumulates perceptible error
        // across 36+ layers of block-wise rescaling (gibberish vs coherent).
        let weight_dtype = weight.dtype();
        let weight_f32 = weight.to_dtype(DType::F32).with_context(|| {
            format!("Failed to promote '{weight_name}' to F32 for scale_inv multiply")
        })?;
        let scale_inv_f32 = if scale_inv.dtype() == DType::F32 {
            scale_inv
        } else {
            scale_inv.to_dtype(DType::F32).with_context(|| {
                format!("Failed to cast '{scale_name}' to F32 for scale multiply")
            })?
        };
        let scaled_f32 = apply_scale_inv(&weight_f32, &scale_inv_f32).with_context(|| {
            format!(
                "Failed to apply '{}' dequantization factor to '{}'",
                scale_name, weight_name
            )
        })?;
        let scaled = scaled_f32.to_dtype(weight_dtype).with_context(|| {
            format!("Failed to cast scaled '{weight_name}' back to {weight_dtype:?}")
        })?;
        tensors.insert(weight_name, scaled);
    }

    Ok(())
}

impl ModelWeights {
    pub fn load(paths: &[&str], device: &Device, dtype: DType) -> Result<Self> {
        // SAFETY: the server owns models_dir exclusively; no external process
        // will truncate or replace these files while the mmap is live.
        let mmap = unsafe {
            MmapedSafetensors::multi(paths).context("Failed to memory-map weight files")?
        };

        let names: Vec<String> = mmap.tensors().into_iter().map(|(n, _)| n).collect();
        let scale_inv_names: std::collections::HashSet<String> = names
            .iter()
            .filter(|name| name.ends_with(".weight_scale_inv"))
            .cloned()
            .collect();

        use rayon::prelude::*;
        let entries: Vec<(String, Tensor)> = names
            .par_iter()
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
                Ok::<_, anyhow::Error>((name.clone(), t))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut tensors: FxHashMap<String, Tensor> =
            FxHashMap::with_capacity_and_hasher(entries.len(), Default::default());
        for (n, t) in entries {
            tensors.insert(n, t);
        }

        apply_weight_scale_inv(&mut tensors)?;

        Ok(Self {
            tensors,
            quant_scheme: None,
        })
    }

    pub fn with_quant_scheme(mut self, scheme: Option<QuantScheme>) -> Self {
        self.quant_scheme = scheme;
        self
    }

    pub fn quant_scheme(&self) -> Option<QuantScheme> {
        self.quant_scheme
    }

    fn resolve_name<'a>(&'a self, name: &str) -> Option<&'a Tensor> {
        if let Some(t) = self.tensors.get(name) {
            return Some(t);
        }

        // Multimodal checkpoints nest the text model under `model.language_model.*`.
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

    pub fn try_get_awq(&self, prefix: &str, bits: u32) -> Option<AwqRawTensors> {
        let qweight = self.try_get(&format!("{prefix}.qweight"))?.clone();
        let qzeros = self.try_get(&format!("{prefix}.qzeros"))?.clone();
        let scales = self.try_get(&format!("{prefix}.scales"))?.clone();
        Some(QuantWeight::new_awq(bits, qweight, qzeros, scales))
    }

    pub fn try_get_gptq(&self, prefix: &str, bits: u32, sym: bool) -> Option<QuantWeight> {
        let qweight = self.try_get(&format!("{prefix}.qweight"))?.clone();
        let scales = self.try_get(&format!("{prefix}.scales"))?.clone();
        let qzeros = self.try_get(&format!("{prefix}.qzeros")).cloned();
        let g_idx = self.try_get(&format!("{prefix}.g_idx")).cloned();
        Some(QuantWeight::new_gptq(
            bits, sym, qweight, qzeros, scales, g_idx,
        ))
    }

    pub fn try_get_quant(&self, prefix: &str) -> Option<QuantWeight> {
        match self.quant_scheme {
            Some(QuantScheme::Awq { bits }) => self.try_get_awq(prefix, bits),
            Some(QuantScheme::Gptq { bits, sym }) => self.try_get_gptq(prefix, bits, sym),
            None => None,
        }
    }

    pub fn has_packed_quantized_weights(&self) -> bool {
        self.tensors.keys().any(|k| k.ends_with(".qweight"))
    }

    pub fn runtime_size_bytes(&self) -> usize {
        self.tensors
            .values()
            .map(|t| t.dtype().size_in_bytes() * t.elem_count())
            .sum()
    }
}

#[cfg(test)]
impl ModelWeights {
    pub fn from_tensors(tensors: FxHashMap<String, Tensor>) -> Self {
        Self {
            tensors,
            quant_scheme: None,
        }
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
    fn apply_weight_scale_inv_expands_block_wise_scales() -> Result<()> {
        let device = Device::Cpu;
        let mut tensors = FxHashMap::default();

        tensors.insert(
            "layer.weight".to_string(),
            Tensor::from_vec(
                (1..=16).map(|v| v as f32).collect::<Vec<_>>(),
                (4, 4),
                &device,
            )?,
        );
        tensors.insert(
            "layer.weight_scale_inv".to_string(),
            Tensor::from_vec(vec![10.0f32, 100.0, 1000.0, 10000.0], (2, 2), &device)?,
        );

        apply_weight_scale_inv(&mut tensors)?;

        let scaled = tensors
            .get("layer.weight")
            .expect("scaled tensor should exist")
            .to_vec2::<f32>()?;
        assert_eq!(
            scaled,
            vec![
                vec![10.0, 20.0, 300.0, 400.0],
                vec![50.0, 60.0, 700.0, 800.0],
                vec![9_000.0, 10_000.0, 110_000.0, 120_000.0],
                vec![13_000.0, 14_000.0, 150_000.0, 160_000.0],
            ]
        );
        Ok(())
    }

    #[test]
    fn runtime_size_bytes_sums_packed_awq_tensors() -> Result<()> {
        let device = Device::Cpu;
        let mut tensors = FxHashMap::default();

        let qweight = Tensor::zeros((1, 4), DType::I32, &device)?;
        let qzeros = Tensor::zeros((1, 4), DType::I32, &device)?;
        let scales = Tensor::zeros((1, 32), DType::BF16, &device)?;
        let bias = Tensor::zeros((32,), DType::BF16, &device)?;
        let ln = Tensor::zeros((8,), DType::BF16, &device)?;

        tensors.insert("layer.0.q_proj.qweight".into(), qweight);
        tensors.insert("layer.0.q_proj.qzeros".into(), qzeros);
        tensors.insert("layer.0.q_proj.scales".into(), scales);
        tensors.insert("layer.0.q_proj.bias".into(), bias);
        tensors.insert("layer.0.input_layernorm.weight".into(), ln);

        let weights = ModelWeights::from_tensors(tensors);

        assert_eq!(weights.runtime_size_bytes(), 16 + 16 + 64 + 64 + 16);
        Ok(())
    }

    #[test]
    fn runtime_size_bytes_sums_tensor_bytes_for_non_awq_models() -> Result<()> {
        let device = Device::Cpu;
        let mut tensors = FxHashMap::default();
        tensors.insert(
            "layer.0.self_attn.q_proj.weight".into(),
            Tensor::zeros((4, 4), DType::BF16, &device)?,
        );
        tensors.insert(
            "layer.0.input_layernorm.weight".into(),
            Tensor::zeros((4,), DType::BF16, &device)?,
        );
        let weights = ModelWeights::from_tensors(tensors);
        assert_eq!(weights.runtime_size_bytes(), 40);
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
