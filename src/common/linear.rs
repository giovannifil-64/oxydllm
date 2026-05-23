use crate::common::awq::{AwqRawTensors, dequantize_awq};
use crate::common::weights::apply_scale_inv;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use std::sync::Arc;

pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (batch, seq) = tokens.dims2()?;
        let hidden = self.weight.dims()[1];
        let flat = tokens.flatten_all()?;
        let embedded = self.weight.index_select(&flat, 0)?;
        embedded.reshape((batch, seq, hidden))
    }

    pub fn from_qtensor(qtensor: &QTensor, device: &Device, dtype: DType) -> Result<Self> {
        let weight = qtensor.dequantize(device)?.to_dtype(dtype)?;
        Ok(Self::new(weight))
    }
}
pub struct Linear {
    weight_t: Tensor,
    bias: Option<Tensor>,
}

fn matmul_with_bias(x: &Tensor, weight_t: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let out = if x.rank() > 2 {
        let original_dims = x.dims().to_vec();
        let in_features = *original_dims.last().unwrap();
        let batch_flat: usize = original_dims[..original_dims.len() - 1].iter().product();
        let o = x.reshape((batch_flat, in_features))?.matmul(weight_t)?;
        let out_features = o.dim(1)?;
        let mut new_dims = original_dims;
        *new_dims.last_mut().unwrap() = out_features;
        o.reshape(new_dims)?
    } else {
        x.matmul(weight_t)?
    };

    match bias {
        Some(b) => out.broadcast_add(b),
        None => Ok(out),
    }
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let weight_t = weight.t()?;
        Ok(Self { weight_t, bias })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        matmul_with_bias(x, &self.weight_t, self.bias.as_ref())
    }
}

pub struct Fp8Linear {
    weight: Tensor,
    scale_inv: Option<Tensor>,
    bias: Option<Tensor>,
}

impl Fp8Linear {
    pub fn new(weight: Tensor, scale_inv: Option<Tensor>, bias: Option<Tensor>) -> Self {
        Self {
            weight,
            scale_inv,
            bias,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Level-2 FP8 path: dequantize on-the-fly right before matmul.
        // Compute remains in runtime dtype (BF16 on GPU, F32 on CPU).
        let mut weight_f32 = self.weight.to_dtype(DType::F32)?;

        if let Some(scale_inv) = &self.scale_inv {
            let scale_inv_f32 = if scale_inv.dtype() == DType::F32 {
                scale_inv.clone()
            } else {
                scale_inv.to_dtype(DType::F32)?
            };

            let scale_for_mul = match scale_inv_f32.rank() {
                // Common checkpoints store per-output scales as [out_features, 1].
                2 => scale_inv_f32,
                // Some checkpoints may encode scales as [out_features] or [1].
                1 => {
                    let out_features = weight_f32.dim(0)?;
                    let n = scale_inv_f32.dim(0)?;
                    if n == out_features {
                        scale_inv_f32.reshape((out_features, 1))?
                    } else if n == 1 {
                        scale_inv_f32.reshape((1, 1))?
                    } else {
                        candle_core::bail!(
                            "invalid FP8 scale_inv shape {:?}: expected [out_features, 1], [out_features], or scalar",
                            scale_inv_f32.shape().dims()
                        )
                    }
                }
                // Scalar scale_inv also works via broadcasting.
                0 => scale_inv_f32,
                _ => {
                    candle_core::bail!(
                        "invalid FP8 scale_inv rank {}: expected rank 0, 1, or 2",
                        scale_inv_f32.rank()
                    )
                }
            };

            weight_f32 = apply_scale_inv(&weight_f32, &scale_for_mul)?;
        }

        let weight_t = weight_f32.to_dtype(x.dtype())?.t()?;
        matmul_with_bias(x, &weight_t, self.bias.as_ref())
    }
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    x.silu()
}

pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    x.gelu()
}

pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if x.device().is_metal() {
        let x_c = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        return super::metal_ops::softmax_fused(&x_c);
    }
    let max = x.max_keepdim(candle_core::D::Minus1)?;
    let x = x.broadcast_sub(&max)?;
    let exp_x = x.exp()?;
    let sum = exp_x.sum_keepdim(candle_core::D::Minus1)?;
    exp_x.broadcast_div(&sum)
}

pub struct QLinear {
    /// Candle `QMatMul` fallback. `None` when `gguf_fast` is set — the bf16
    /// path owns its own packed weight stream (M=1 GEMV + M>1 fused GEMM,
    /// both dequant-inline) and the original `QTensor` is released, removing
    /// the 2× memory residency that the M=1-only pilot incurred.
    inner: Option<QMatMul>,
    #[cfg(feature = "metal")]
    gguf_fast: Option<GgufFastPath>,
    bias: Option<Tensor>,
    out_dtype: DType,
}

#[cfg(feature = "metal")]
struct GgufFastPath {
    quant: super::metal_ops::GgufFastQuant,
    weight_bytes: Tensor,
    in_features: usize,
    out_features: usize,
}

#[cfg(feature = "metal")]
impl GgufFastPath {
    fn build(qt: &Arc<QTensor>, out_dtype: DType) -> Result<Option<Self>> {
        if !matches!(qt.device(), Device::Metal(_)) || out_dtype != DType::BF16 {
            return Ok(None);
        }
        // Map candle's GgmlDType → our fast-path enum. Returning None for
        // unsupported quants keeps the existing QMatMul path as fallback.
        let quant = match qt.dtype() {
            GgmlDType::Q5_0 => super::metal_ops::GgufFastQuant::Q5_0,
            GgmlDType::Q8_0 => super::metal_ops::GgufFastQuant::Q8_0,
            GgmlDType::Q4K => super::metal_ops::GgufFastQuant::Q4K,
            GgmlDType::Q5K => super::metal_ops::GgufFastQuant::Q5K,
            GgmlDType::Q6K => super::metal_ops::GgufFastQuant::Q6K,
            _ => return Ok(None),
        };
        let shape_dims = qt.shape().dims();
        if shape_dims.len() != 2 {
            return Ok(None);
        }
        let (out_features, in_features) = (shape_dims[0], shape_dims[1]);
        let (block_elems, block_bytes) = quant.block_layout();
        if !in_features.is_multiple_of(block_elems) {
            return Ok(None);
        }

        let device = qt.device();
        let bytes = qt
            .data()
            .map_err(|e| candle_core::Error::Msg(format!("GGUF fast path: qt.data() failed: {e}")))?
            .into_owned();
        let expected = out_features * (in_features / block_elems) * block_bytes;
        if bytes.len() != expected {
            candle_core::bail!(
                "GGUF fast path: qt.data() returned {} bytes, expected {} for shape [{}, {}] dtype {:?}",
                bytes.len(),
                expected,
                out_features,
                in_features,
                qt.dtype()
            );
        }
        let weight_bytes = Tensor::from_vec(bytes, (expected,), &device)?;
        Ok(Some(Self {
            quant,
            weight_bytes,
            in_features,
            out_features,
        }))
    }

    /// M=1 decode: dispatch the bf16 GEMV kernel.
    fn forward_decode(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let in_features = *dims.last().unwrap();
        let m: usize = dims[..dims.len() - 1].iter().product();
        debug_assert_eq!(
            m, 1,
            "GgufFastPath::forward_decode must only be called for M=1"
        );
        debug_assert_eq!(in_features, self.in_features);
        let x_2d = x.reshape((1, in_features))?.contiguous()?;
        let y_2d = super::metal_ops::gguf_quant_matmul(
            &x_2d,
            &self.weight_bytes,
            self.in_features,
            self.out_features,
            self.quant,
        )?;
        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = self.out_features;
        y_2d.reshape(out_dims)
    }

    /// M>1 prefill: fused dequant + GEMM in a single Metal kernel — no
    /// intermediate bf16 weight tensor is ever materialised.
    fn forward_prefill(&self, x: &Tensor) -> Result<Tensor> {
        let original_dims = x.dims().to_vec();
        let in_features = *original_dims.last().unwrap();
        debug_assert_eq!(in_features, self.in_features);
        let batch_flat: usize = original_dims[..original_dims.len() - 1].iter().product();
        let x_2d = x.reshape((batch_flat, in_features))?.contiguous()?;
        let y_2d = super::metal_ops::gguf_quant_mul_mm(
            &x_2d,
            &self.weight_bytes,
            self.in_features,
            self.out_features,
            self.quant,
        )?;
        let mut new_dims = original_dims;
        *new_dims.last_mut().unwrap() = self.out_features;
        y_2d.reshape(new_dims)
    }
}

impl QLinear {
    pub fn from_arc(qtensor: Arc<QTensor>, out_dtype: DType) -> Result<Self> {
        Self::from_arc_with_bias(qtensor, None, out_dtype)
    }

    pub fn from_arc_with_bias(
        qtensor: Arc<QTensor>,
        bias: Option<Tensor>,
        out_dtype: DType,
    ) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            let gguf_fast = GgufFastPath::build(&qtensor, out_dtype)?;
            // When the bf16 fast path owns the packed weight, drop the
            // candle-side QTensor to free its Metal buffer (~2× pesi saved).
            let inner = if gguf_fast.is_some() {
                drop(qtensor);
                None
            } else {
                Some(QMatMul::from_arc(qtensor)?)
            };
            Ok(Self {
                inner,
                gguf_fast,
                bias,
                out_dtype,
            })
        }
        #[cfg(not(feature = "metal"))]
        {
            let inner = Some(QMatMul::from_arc(qtensor)?);
            Ok(Self {
                inner,
                bias,
                out_dtype,
            })
        }
    }

    /// When `gguf_fast` is available, both M=1 (decode → bf16 GEMV) and M>1
    /// (prefill → fused dequant+GEMM in a single kernel) take the bf16-aware
    /// path. The candle `QMatMul` fallback handles unsupported dtypes (Q2_K,
    /// Q6_K, …) and non-Metal builds — that path runs in F32, adding two
    /// dtype casts (BF16→F32 input, F32→BF16 output) per call.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_dims = x.dims().to_vec();
        let in_features = *original_dims.last().unwrap();
        let m: usize = original_dims[..original_dims.len() - 1].iter().product();

        #[cfg(feature = "metal")]
        if let Some(ref fast) = self.gguf_fast
            && in_features == fast.in_features
            && x.dtype() == DType::BF16
        {
            let out = if m == 1 {
                fast.forward_decode(x)?
            } else {
                fast.forward_prefill(x)?
            };
            let out = if out.dtype() != self.out_dtype {
                out.to_dtype(self.out_dtype)?
            } else {
                out
            };
            return match &self.bias {
                Some(b) => out.broadcast_add(b),
                None => Ok(out),
            };
        }

        let inner = self.inner.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "QLinear: candle QMatMul was released but the GGUF fast path is not engaged"
                    .to_string(),
            )
        })?;

        let x_f32_owned;
        let x_f32: &Tensor = if x.dtype() != DType::F32 {
            x_f32_owned = x.to_dtype(DType::F32)?;
            &x_f32_owned
        } else {
            x
        };

        let x_2d = if x_f32.rank() > 2 {
            let batch_flat: usize = original_dims[..original_dims.len() - 1].iter().product();
            x_f32.reshape((batch_flat, in_features))?
        } else {
            x_f32.clone()
        };

        let out = candle_core::Module::forward(inner, &x_2d)?;

        let out = if original_dims.len() > 2 {
            let out_features = out.dim(candle_core::D::Minus1)?;
            let mut new_dims = original_dims;
            *new_dims.last_mut().unwrap() = out_features;
            out.reshape(new_dims)?
        } else {
            out
        };

        let out = if out.dtype() != self.out_dtype {
            out.to_dtype(self.out_dtype)?
        } else {
            out
        };

        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}

#[cfg(feature = "metal")]
pub struct PackedQuantLinear {
    qweight: Tensor,
    qzeros: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

#[cfg(feature = "metal")]
impl PackedQuantLinear {
    pub fn new(raw: AwqRawTensors, bias: Option<Tensor>, dtype: DType) -> Result<Self> {
        let (in_features, _packed_out) = raw.qweight.dims2()?;
        let (_groups, out_features) = raw.scales.dims2()?;
        // The kernel reads `scales` in the activation dtype.
        let scales = raw.scales.to_dtype(dtype)?;
        Ok(Self {
            qweight: raw.qweight,
            qzeros: raw.qzeros,
            scales,
            bias,
            in_features,
            out_features,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let in_features = *dims.last().unwrap();
        if in_features != self.in_features {
            candle_core::bail!(
                "PackedQuantLinear: input last dim {in_features} != in_features {}",
                self.in_features
            );
        }
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape((m, in_features))?.contiguous()?;

        // M=1 (decode) → fused GEMV kernel; M>1 (prefill / batched) → dequantize
        // to a transient weight and use the tuned matmul.
        let y_2d = if m == 1 {
            super::metal_ops::w4a16_matmul(&x_2d, &self.qweight, &self.qzeros, &self.scales)?
        } else {
            let weight =
                super::metal_ops::dequantize_w4(&self.qweight, &self.qzeros, &self.scales)?;
            x_2d.matmul(&weight)?
        };

        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = self.out_features;
        let y = y_2d.reshape(out_dims)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

pub enum AnyLinear {
    Float(Linear),
    Fp8(Fp8Linear),
    Quantized(QLinear),
    #[cfg(feature = "metal")]
    PackedQuant(PackedQuantLinear),
}

impl AnyLinear {
    pub fn from_weight(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Self::from_weight_with_scale_inv(weight, None, bias)
    }

    pub fn from_weight_with_scale_inv(
        weight: Tensor,
        scale_inv: Option<Tensor>,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        if weight.dtype() == DType::F8E4M3 {
            Ok(Self::Fp8(Fp8Linear::new(weight, scale_inv, bias)))
        } else {
            Ok(Self::Float(Linear::new(weight, bias)?))
        }
    }

    pub fn from_awq(
        raw: &AwqRawTensors,
        bias: Option<Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        #[cfg(feature = "metal")]
        if device.is_metal() && matches!(dtype, DType::F16 | DType::BF16) {
            let resident = raw
                .to_device(device)
                .map_err(|e| candle_core::Error::Msg(format!("AWQ → Metal failed: {e:#}")))?;
            return Ok(Self::PackedQuant(PackedQuantLinear::new(
                resident, bias, dtype,
            )?));
        }
        let weight = dequantize_awq(raw, device, dtype)
            .map_err(|e| candle_core::Error::Msg(format!("AWQ dequantization failed: {e:#}")))?;
        Ok(Self::Float(Linear::new(weight, bias)?))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Float(l) => l.forward(x),
            Self::Fp8(l) => l.forward(x),
            Self::Quantized(q) => q.forward(x),
            #[cfg(feature = "metal")]
            Self::PackedQuant(q) => q.forward(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_linear_from_awq_matches_reference_linear() -> Result<()> {
        use crate::common::awq::{AWQ_PACK_FACTOR, AWQ_PACK_ORDER, AwqRawTensors};

        let device = Device::Cpu;
        let in_features = 4;
        let out_features = 8;
        let group_size = 2;
        let groups = in_features / group_size;
        let packed_out = out_features / AWQ_PACK_FACTOR;

        let mut iweight: Vec<Vec<u8>> = Vec::with_capacity(in_features);
        for i in 0..in_features {
            iweight.push(
                (0..out_features)
                    .map(|j| ((i * 5 + j) & 0xF) as u8)
                    .collect(),
            );
        }
        let izero: Vec<Vec<u8>> = (0..groups)
            .map(|g| (0..out_features).map(|j| ((g + j) & 0xF) as u8).collect())
            .collect();
        let scales: Vec<f32> = (0..groups)
            .flat_map(|g| (0..out_features).map(move |j| 0.05 * (g as f32 + 1.0) + 0.01 * j as f32))
            .collect();

        let mut qweight_words: Vec<u32> = vec![0; in_features * packed_out];
        for (i, row) in iweight.iter().enumerate().take(in_features) {
            for j in 0..packed_out {
                let mut word: u32 = 0;
                for (k, &offset) in AWQ_PACK_ORDER.iter().enumerate() {
                    let orig_col = j * AWQ_PACK_FACTOR + offset;
                    word |= (row[orig_col] as u32 & 0xF) << (4 * k as u32);
                }
                qweight_words[i * packed_out + j] = word;
            }
        }
        let mut qzero_words: Vec<u32> = vec![0; groups * packed_out];
        for (g, row) in izero.iter().enumerate().take(groups) {
            for j in 0..packed_out {
                let mut word: u32 = 0;
                for (k, &offset) in AWQ_PACK_ORDER.iter().enumerate() {
                    let orig_col = j * AWQ_PACK_FACTOR + offset;
                    word |= (row[orig_col] as u32 & 0xF) << (4 * k as u32);
                }
                qzero_words[g * packed_out + j] = word;
            }
        }

        let qweight = Tensor::from_vec(
            qweight_words
                .into_iter()
                .map(|w| w as i32)
                .collect::<Vec<_>>(),
            (in_features, packed_out),
            &device,
        )?;
        let qzeros = Tensor::from_vec(
            qzero_words
                .into_iter()
                .map(|w| w as i32)
                .collect::<Vec<_>>(),
            (groups, packed_out),
            &device,
        )?;
        let scales_t = Tensor::from_vec(scales.clone(), (groups, out_features), &device)?;

        let raw = AwqRawTensors {
            qweight,
            qzeros,
            scales: scales_t,
        };
        let bias_vec: Vec<f32> = (0..out_features).map(|j| 0.1 * j as f32).collect();
        let bias = Tensor::from_vec(bias_vec.clone(), (out_features,), &device)?;
        let awq_linear = AnyLinear::from_awq(&raw, Some(bias.clone()), &device, DType::F32)?;

        // Build reference [out, in] weight from the unpacked int4 matrix.
        let mut ref_weight = vec![0f32; out_features * in_features];
        for i in 0..in_features {
            let g = i / group_size;
            for j in 0..out_features {
                let w = iweight[i][j] as i32 - izero[g][j] as i32;
                ref_weight[j * in_features + i] = w as f32 * scales[g * out_features + j];
            }
        }
        let ref_weight_t = Tensor::from_vec(ref_weight, (out_features, in_features), &device)?;
        let ref_linear = Linear::new(ref_weight_t, Some(bias))?;

        let x = Tensor::from_vec(
            (0..in_features)
                .map(|v| 0.1 * (v as f32) - 0.05)
                .collect::<Vec<_>>(),
            (1, in_features),
            &device,
        )?;
        let out_awq = awq_linear.forward(&x)?.to_vec2::<f32>()?;
        let out_ref = ref_linear.forward(&x)?.to_vec2::<f32>()?;

        for (a_row, r_row) in out_awq.iter().zip(out_ref.iter()) {
            for (a, r) in a_row.iter().zip(r_row.iter()) {
                assert!((a - r).abs() < 1e-6, "awq={a} ref={r}");
            }
        }
        Ok(())
    }

    #[test]
    fn any_linear_selects_fp8_variant_for_fp8_weights() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![0.5f32, -1.0, 1.5, 0.25], (2, 2), &device)?
            .to_dtype(DType::F8E4M3)?;

        let linear = AnyLinear::from_weight(weight, None)?;
        assert!(matches!(linear, AnyLinear::Fp8(_)));
        Ok(())
    }

    #[test]
    fn fp8_linear_matches_reference_dequant_path() -> Result<()> {
        let device = Device::Cpu;

        let weight_fp8 =
            Tensor::from_vec(vec![1.0f32, -2.0, 3.0, 0.5, -0.25, 2.5], (2, 3), &device)?
                .to_dtype(DType::F8E4M3)?;
        let scale_inv = Tensor::from_vec(vec![0.5f32, 2.0], (2, 1), &device)?;
        let bias = Tensor::from_vec(vec![0.1f32, -0.2], (2,), &device)?;
        let x = Tensor::from_vec(vec![0.75f32, -1.0, 0.25, 1.5, 0.5, -0.5], (2, 3), &device)?;

        let out = AnyLinear::from_weight_with_scale_inv(
            weight_fp8.clone(),
            Some(scale_inv.clone()),
            Some(bias.clone()),
        )?
        .forward(&x)?;

        let ref_weight = weight_fp8.to_dtype(DType::F32)?.broadcast_mul(&scale_inv)?;
        let expected = Linear::new(ref_weight, Some(bias))?.forward(&x)?;

        let out_vals = out.to_vec2::<f32>()?;
        let expected_vals = expected.to_vec2::<f32>()?;

        for (out_row, exp_row) in out_vals.iter().zip(expected_vals.iter()) {
            for (o, e) in out_row.iter().zip(exp_row.iter()) {
                assert!((o - e).abs() < 1e-5, "o={o}, e={e}");
            }
        }

        Ok(())
    }

    #[test]
    fn fp8_linear_accepts_rank1_per_row_scale_inv() -> Result<()> {
        let device = Device::Cpu;

        let weight_fp8 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?
            .to_dtype(DType::F8E4M3)?;
        let scale_inv_rank1 = Tensor::from_vec(vec![0.5f32, 2.0], (2,), &device)?;
        let x = Tensor::from_vec(vec![1.0f32, -1.0, 0.5, 2.0], (2, 2), &device)?;

        let out = AnyLinear::from_weight_with_scale_inv(
            weight_fp8.clone(),
            Some(scale_inv_rank1.clone()),
            None,
        )?
        .forward(&x)?;

        let ref_scale = scale_inv_rank1.reshape((2, 1))?;
        let ref_weight = weight_fp8.to_dtype(DType::F32)?.broadcast_mul(&ref_scale)?;
        let expected = Linear::new(ref_weight, None)?.forward(&x)?;

        let out_vals = out.to_vec2::<f32>()?;
        let expected_vals = expected.to_vec2::<f32>()?;
        for (out_row, exp_row) in out_vals.iter().zip(expected_vals.iter()) {
            for (o, e) in out_row.iter().zip(exp_row.iter()) {
                assert!((o - e).abs() < 1e-5, "o={o}, e={e}");
            }
        }

        Ok(())
    }

    #[test]
    fn fp8_linear_accepts_scalar_scale_inv() -> Result<()> {
        let device = Device::Cpu;

        let weight_fp8 = Tensor::from_vec(vec![1.0f32, -2.0, 0.25, 4.0], (2, 2), &device)?
            .to_dtype(DType::F8E4M3)?;
        let scalar_scale_inv = Tensor::from_vec(vec![0.5f32], (1,), &device)?;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, -1.0, 0.5], (2, 2), &device)?;

        let out = AnyLinear::from_weight_with_scale_inv(
            weight_fp8.clone(),
            Some(scalar_scale_inv.clone()),
            None,
        )?
        .forward(&x)?;

        let ref_weight = weight_fp8
            .to_dtype(DType::F32)?
            .broadcast_mul(&scalar_scale_inv)?;
        let expected = Linear::new(ref_weight, None)?.forward(&x)?;

        let out_vals = out.to_vec2::<f32>()?;
        let expected_vals = expected.to_vec2::<f32>()?;
        for (out_row, exp_row) in out_vals.iter().zip(expected_vals.iter()) {
            for (o, e) in out_row.iter().zip(exp_row.iter()) {
                assert!((o - e).abs() < 1e-5, "o={o}, e={e}");
            }
        }

        Ok(())
    }

    #[test]
    fn fp8_linear_rejects_invalid_rank1_scale_inv_shape() -> Result<()> {
        let device = Device::Cpu;

        let weight_fp8 = Tensor::from_vec(vec![1.0f32, -2.0, 0.25, 4.0], (2, 2), &device)?
            .to_dtype(DType::F8E4M3)?;
        let invalid_scale_inv = Tensor::from_vec(vec![0.5f32, 1.0, 2.0], (3,), &device)?;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, -1.0, 0.5], (2, 2), &device)?;

        let err = AnyLinear::from_weight_with_scale_inv(weight_fp8, Some(invalid_scale_inv), None)?
            .forward(&x)
            .expect_err("expected invalid rank-1 scale_inv shape to fail");
        assert!(
            err.to_string().contains("invalid FP8 scale_inv shape"),
            "unexpected error: {err}"
        );

        Ok(())
    }

    #[test]
    fn fp8_linear_dequantizes_block_wise_scale_grid() -> Result<()> {
        let device = Device::Cpu;

        let weight_fp8 = Tensor::from_vec(
            (1..=16).map(|v| v as f32).collect::<Vec<_>>(),
            (4, 4),
            &device,
        )?
        .to_dtype(DType::F8E4M3)?;
        let scale_inv = Tensor::from_vec(vec![0.5f32, 2.0, 4.0, 0.25], (2, 2), &device)?;
        let x = Tensor::ones((1, 4), DType::F32, &device)?;

        let out = AnyLinear::from_weight_with_scale_inv(
            weight_fp8.clone(),
            Some(scale_inv.clone()),
            None,
        )?
        .forward(&x)?;

        let expanded = scale_inv
            .reshape((2, 1, 2, 1))?
            .broadcast_as((2, 2, 2, 2))?
            .contiguous()?
            .reshape((4, 4))?;
        let ref_weight = weight_fp8.to_dtype(DType::F32)?.mul(&expanded)?;
        let ref_out = x.matmul(&ref_weight.t()?)?;

        let diff = (out - ref_out)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4, "block-wise FP8 output diverged: {diff}");
        Ok(())
    }
}
