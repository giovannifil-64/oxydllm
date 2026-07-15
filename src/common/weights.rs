//! Unified weight access over safetensors checkpoints.
//!
//! [`ModelWeights`] memory-maps safetensors files, casts each tensor to the
//! runtime dtype (handling FP8 and block-wise scales), and serves tensors by
//! name to the loaders. Name resolution is alias-aware, so the same keys work
//! across plain and multimodal-nested checkpoints. [`QuantScheme`] records which
//! packed-int format a checkpoint uses so [`ModelWeights::try_get_quant`]
//! returns the right [`QuantWeight`].

use crate::common::awq::{AwqRawTensors, QuantWeight};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, safetensors::MmapedSafetensors};
use rustc_hash::FxHashMap;

/// The packed-int quantization format of a checkpoint.
///
/// `Awq` and `Gptq` carry the bit width (GPTQ also its symmetric flag).
/// `CompressedTensors4` is llm-compressor's pack-quantized symmetric INT4, which
/// is converted to the AWQ layout at load and runs on the resident W4A16 path.
/// A `None` scheme on [`ModelWeights`] means dense / non-packed weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantScheme {
    Awq { bits: u32 },
    Gptq { bits: u32, sym: bool },
    CompressedTensors4,
}

/// All tensors of a model, loaded and ready to serve to the layer constructors.
///
/// Built by [`load`](Self::load) from safetensors files; tensors are looked up
/// by canonical name via [`get`](Self::get) / [`try_get`](Self::try_get), with
/// alias fallbacks for multimodal-nested layouts. When a packed-int
/// [`QuantScheme`] is attached, [`try_get_quant`](Self::try_get_quant) assembles
/// the matching [`QuantWeight`] from the per-projection packed tensors.
pub struct ModelWeights {
    tensors: FxHashMap<String, Tensor>,
    quant_scheme: Option<QuantScheme>,
    expert_pool: Option<std::sync::Arc<crate::common::expert_stream::StreamedExperts>>,
}

/// Gated DeltaNet scalar parameters are stored F32 and consumed F32 (the
/// decay recurrence is precision-sensitive); never round-trip them via BF16.
fn keeps_file_dtype(name: &str) -> bool {
    name.ends_with(".linear_attn.A_log")
        || name.ends_with(".linear_attn.dt_bias")
        || name.ends_with(".linear_attn.norm.weight")
}

/// Serializes Metal tensor creation across the rayon loader threads: candle
/// 0.11 mutates its `MTLResidencySet` without a lock, so concurrent creation
/// races and leaves buffers non-resident (the GPU reads zeros from them).
pub(crate) fn metal_alloc_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    &LOCK
}

/// Drains the queued Metal work after a device transfer during loading; see
/// the two-phase note in [`ModelWeights::load`] for why this is mandatory.
pub(crate) fn drain_metal(device: &Device, what: &str) -> Result<()> {
    if device.is_metal() {
        device
            .synchronize()
            .map_err(|e| anyhow::anyhow!("synchronize after {what}: {e:#}"))?;
    }
    Ok(())
}

/// Loads one tensor from the mmap and casts it to the runtime dtype, on the
/// CPU for the load phase (the device transfer happens afterwards, see
/// [`ModelWeights::load`]).
///
/// Integer tensors and the F32-pinned GatedDeltaNet scalars
/// ([`keeps_file_dtype`]) are returned as stored. FP8 weights are dequantized
/// on CPU unless `preserve_fp8_weight` is set (non-Metal devices that can
/// consume F8 directly). `force_f32` overrides the target dtype (used for
/// scale tensors). A failed device-side cast retries via CPU.
///
/// ## Errors
/// Propagates load, cast, or device-transfer failures.
fn load_tensor_with_dtype(
    mmap: &MmapedSafetensors,
    name: &str,
    device: &Device,
    dtype: DType,
    preserve_fp8_weight: bool,
    force_f32: bool,
    file_is_f8: bool,
) -> Result<Tensor> {
    let target_dtype = if force_f32 { DType::F32 } else { dtype };

    // FP8 on Metal: dequantize entirely on CPU and transfer the final tensor
    // once, so the F8 original is never staged on the device (see the
    // two-phase note in ModelWeights::load).
    if file_is_f8 && device.is_metal() && !keeps_file_dtype(name) {
        let t_cpu = mmap
            .load(name, &Device::Cpu)
            .with_context(|| format!("Failed to load FP8 tensor {} on CPU", name))?
            .to_dtype(DType::F32)
            .with_context(|| format!("Failed to convert FP8 tensor {} to F32 on CPU", name))?
            .to_dtype(target_dtype)
            .with_context(|| {
                format!(
                    "Failed to cast FP8 tensor {} to {:?} on CPU",
                    name, target_dtype
                )
            })?;
        let _guard = metal_alloc_lock().lock().unwrap();
        let t_dev = t_cpu.to_device(device).with_context(|| {
            format!(
                "Failed to move tensor {} to device after CPU conversion",
                name
            )
        })?;
        drain_metal(device, name)?;
        return Ok(t_dev);
    }

    let t = {
        let _guard = device
            .is_metal()
            .then(|| metal_alloc_lock().lock().unwrap());
        let t = mmap
            .load(name, device)
            .with_context(|| format!("Failed to load tensor {}", name))?;
        drain_metal(device, name)?;
        t
    };

    if keeps_file_dtype(name) {
        return Ok(t);
    }

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
        if preserve_fp8_weight {
            return Ok(t);
        }
        let t_f32 = t
            .to_dtype(DType::F32)
            .with_context(|| format!("Failed to cast FP8 tensor {} to F32", name))?;
        return t_f32.to_dtype(target_dtype).with_context(|| {
            format!(
                "Failed to cast FP8 tensor {} from F32 to {:?}",
                name, target_dtype
            )
        });
    }

    let cast_result = {
        let _guard = device
            .is_metal()
            .then(|| metal_alloc_lock().lock().unwrap());
        let r = t.to_dtype(target_dtype);
        // The queued GPU cast still reads `t` after this function drops it;
        // drain before returning or the buffer is recycled under the command.
        if r.is_ok() {
            drain_metal(device, name)?;
        }
        r
    };
    match cast_result {
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
            let t_cpu_target = t_cpu_f32.to_dtype(target_dtype).with_context(|| {
                format!(
                    "Failed to cast tensor {} to {:?} on CPU (device cast error: {})",
                    name, target_dtype, device_cast_err
                )
            })?;
            let _guard = device
                .is_metal()
                .then(|| metal_alloc_lock().lock().unwrap());
            let t_dev = t_cpu_target.to_device(device).with_context(|| {
                format!(
                    "Failed to move tensor {} back to target device after CPU fallback",
                    name
                )
            })?;
            drain_metal(device, name)?;
            Ok(t_dev)
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

/// Folds every `*.weight_scale_inv` factor into its `*.weight` in place,
/// dequantizing block-wise FP8 weights that were not kept packed.
///
/// The multiply is done in F32 even for BF16 weights: BF16's 7-bit mantissa
/// accumulates perceptible error across the dozens of layers of block-wise
/// rescaling (coherent vs gibberish output). FP8-typed weights are skipped; they
/// dequantize later on their own path.
///
/// ## Errors
/// Propagates promotion, scale-application, or cast failures.
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
        // Drain per weight or the queue accumulates an F32 copy of every
        // scaled weight (see the two-phase note in ModelWeights::load).
        drain_metal(scaled.device(), &weight_name)?;
        tensors.insert(weight_name, scaled);
    }

    Ok(())
}

impl ModelWeights {
    /// Loads and memory-maps the safetensors files at `paths`, casting tensors to
    /// `dtype` on `device`.
    ///
    /// Vision-tower (`model.visual.*`) and multi-token-prediction (`mtp.*`)
    /// tensors are skipped (text-only runtime), saving the gigabytes they would
    /// cost on Qwen3.5-class checkpoints. After loading, every
    /// `*.weight_scale_inv` factor is folded into its weight.
    ///
    /// With `stream` set, MoE expert tensors (see
    /// [`is_streamed_expert_tensor`](crate::common::expert_stream::is_streamed_expert_tensor))
    /// are not loaded; the checkpoint mmap is retained in a
    /// [`StreamedExperts`](crate::common::expert_stream::StreamedExperts) pool
    /// that serves them on demand, exposed via
    /// [`expert_pool`](Self::expert_pool).
    ///
    /// ## Errors
    /// Fails if the files cannot be mapped, or any tensor cannot be loaded, cast,
    /// or scaled.
    pub fn load(
        paths: &[&str],
        device: &Device,
        dtype: DType,
        stream: Option<crate::common::expert_stream::ExpertStreamConfig>,
    ) -> Result<Self> {
        // SAFETY: the server owns models_dir exclusively; no external process
        // will truncate or replace these files while the mmap is live.
        let mmap = unsafe {
            MmapedSafetensors::multi(paths).context("Failed to memory-map weight files")?
        };

        let names: Vec<String> = mmap
            .tensors()
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| !n.starts_with("model.visual.") && !n.starts_with("mtp."))
            .filter(|n| {
                stream.is_none() || !crate::common::expert_stream::is_streamed_expert_tensor(n)
            })
            .collect();
        let scale_inv_names: std::collections::HashSet<String> = names
            .iter()
            .filter(|name| name.ends_with(".weight_scale_inv"))
            .cloned()
            .collect();

        // File dtypes from the safetensors metadata, so FP8 tensors never get
        // staged on the device just to discover their dtype.
        let f8_names: std::collections::HashSet<String> = mmap
            .tensors()
            .into_iter()
            .filter(|(_, view)| format!("{:?}", view.dtype()) == "F8_E4M3")
            .map(|(n, _)| n)
            .collect();

        // Two-phase load. Phase 1 (parallel, CPU only): read, cast to the
        // runtime dtype and fold the FP8 scale factors, all in host memory.
        // Phase 2 (sequential): transfer to the device with periodic drains.
        //
        // candle 0.11's Metal device tolerates neither concurrent tensor
        // creation (unsynchronized MTLResidencySet mutation), nor a load's
        // worth of queued transfers (staging buffers are only reclaimed at
        // sync points, so the peak blows past the working-set limit and
        // command buffers fail with kIOGPUCommandBufferCallbackError-
        // OutOfMemory), nor synchronizing from one thread while another
        // encodes. Keeping the device work sequential and drained sidesteps
        // all three; the heavy lifting (mmap reads, dequant casts, scale
        // multiplies) stays parallel on CPU.
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
                    &Device::Cpu,
                    dtype,
                    preserve_fp8_weight && !device.is_metal(),
                    force_f32,
                    f8_names.contains(name),
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

        if !matches!(device, Device::Cpu) {
            let names: Vec<String> = tensors.keys().cloned().collect();
            for (i, name) in names.iter().enumerate() {
                let cpu_t = tensors.remove(name).expect("key from the same map");
                let dev_t = cpu_t
                    .to_device(device)
                    .with_context(|| format!("Failed to move tensor {} to device", name))?;
                drop(cpu_t);
                tensors.insert(name.clone(), dev_t);
                if i % 16 == 15 {
                    drain_metal(device, name)?;
                }
            }
            drain_metal(device, "post-load")?;
        }

        let expert_pool = match stream {
            Some(cfg) => Some(std::sync::Arc::new(
                crate::common::expert_stream::StreamedExperts::new(
                    paths,
                    cfg,
                    device.clone(),
                    dtype,
                )
                .map_err(|e| anyhow::anyhow!("expert stream pool: {e:#}"))?,
            )),
            None => None,
        };

        Ok(Self {
            tensors,
            quant_scheme: None,
            expert_pool,
        })
    }

    /// The expert streaming pool, when this model was loaded with expert
    /// streaming enabled. MoE layers switch to the streamed bank when present.
    pub(crate) fn expert_pool(
        &self,
    ) -> Option<std::sync::Arc<crate::common::expert_stream::StreamedExperts>> {
        self.expert_pool.clone()
    }

    /// Attaches (or clears) the packed-int [`QuantScheme`] and returns `self`.
    pub fn with_quant_scheme(mut self, scheme: Option<QuantScheme>) -> Self {
        self.quant_scheme = scheme;
        self
    }

    /// The attached [`QuantScheme`], if any.
    #[cfg(feature = "metal")]
    pub fn quant_scheme(&self) -> Option<QuantScheme> {
        self.quant_scheme
    }

    /// Resolves a canonical tensor name, trying multimodal aliases when the
    /// direct lookup misses.
    ///
    /// Multimodal checkpoints nest the text model under `model.language_model.*`
    /// (and similar) and place `lm_head` outside `model.`; this maps the
    /// canonical names the loaders use onto those layouts.
    fn resolve_name<'a>(&'a self, name: &str) -> Option<&'a Tensor> {
        if let Some(t) = self.tensors.get(name) {
            return Some(t);
        }

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

    /// Returns the tensor for canonical `name` (with multimodal alias fallback).
    ///
    /// ## Errors
    /// Fails if no tensor resolves to that name.
    pub fn get(&self, name: &str) -> candle_core::Result<&Tensor> {
        self.resolve_name(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Tensor not found: {}", name)))
    }

    /// Returns the tensor for canonical `name`, or `None` if absent.
    pub fn try_get(&self, name: &str) -> Option<&Tensor> {
        self.resolve_name(name)
    }

    /// Returns the `{weight_name}_scale_inv` companion tensor, if present.
    pub fn try_get_scale_inv(&self, weight_name: &str) -> Option<&Tensor> {
        self.try_get(&format!("{}_scale_inv", weight_name))
    }

    /// Assembles the AWQ packed tensors at `prefix` (`qweight` + `qzeros` +
    /// `scales`), or `None` if any is missing.
    pub fn try_get_awq(&self, prefix: &str, bits: u32) -> Option<AwqRawTensors> {
        let qweight = self.try_get(&format!("{prefix}.qweight"))?.clone();
        let qzeros = self.try_get(&format!("{prefix}.qzeros"))?.clone();
        let scales = self.try_get(&format!("{prefix}.scales"))?.clone();
        Some(QuantWeight::new_awq(bits, qweight, qzeros, scales))
    }

    /// Assembles a GPTQ [`QuantWeight`] at `prefix` (`qweight` + `scales`, with
    /// optional `qzeros`), or `None` if a required tensor is missing.
    pub fn try_get_gptq(&self, prefix: &str, bits: u32, sym: bool) -> Option<QuantWeight> {
        let qweight = self.try_get(&format!("{prefix}.qweight"))?.clone();
        let scales = self.try_get(&format!("{prefix}.scales"))?.clone();
        let qzeros = self.try_get(&format!("{prefix}.qzeros")).cloned();
        Some(QuantWeight::new_gptq(bits, sym, qweight, qzeros, scales))
    }

    /// Assembles a [`QuantWeight`] from compressed-tensors `weight_packed` +
    /// `weight_scale` at `prefix`, converting to the AWQ layout. Returns `None`
    /// if the tensors are missing or the conversion fails (logged).
    pub fn try_get_compressed(&self, prefix: &str) -> Option<QuantWeight> {
        let packed = self.try_get(&format!("{prefix}.weight_packed"))?;
        let scale = self.try_get(&format!("{prefix}.weight_scale"))?;
        match crate::common::awq::compressed_to_awq(packed, scale) {
            Ok(raw) => Some(raw),
            Err(e) => {
                tracing::error!("compressed-tensors conversion failed at {prefix}: {e:#}");
                None
            }
        }
    }

    /// Assembles the packed [`QuantWeight`] at `prefix` using the attached
    /// [`QuantScheme`] (AWQ / GPTQ / compressed-tensors). Returns `None` when no
    /// scheme is set or a tensor is missing.
    pub fn try_get_quant(&self, prefix: &str) -> Option<QuantWeight> {
        match self.quant_scheme {
            Some(QuantScheme::Awq { bits }) => self.try_get_awq(prefix, bits),
            Some(QuantScheme::Gptq { bits, sym }) => self.try_get_gptq(prefix, bits, sym),
            Some(QuantScheme::CompressedTensors4) => self.try_get_compressed(prefix),
            None => None,
        }
    }

    /// Whether any tensor is packed-quantized (`*.qweight` or `*.weight_packed`).
    #[cfg(feature = "metal")]
    pub fn has_packed_quantized_weights(&self) -> bool {
        self.tensors
            .keys()
            .any(|k| k.ends_with(".qweight") || k.ends_with(".weight_packed"))
    }

    /// Total resident size of all tensors in bytes (packed tensors counted at
    /// their on-device size, not the dequantized size).
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
            expert_pool: None,
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
