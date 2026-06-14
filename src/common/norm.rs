//! Root-mean-square normalization, with the Standard and Gemma weight
//! conventions folded behind one [`RMSNorm`] type.

use super::config::NormType;
use super::weights::ModelWeights;
use candle_core::quantized::QTensor;
use candle_core::{D, DType, Device, Result, Tensor};

/// Root-mean-square layer normalization.
///
/// Normalizes each row of the input by its RMS and scales by a learned
/// per-channel weight: `x / sqrt(mean(x²) + eps) · weight`. Unlike LayerNorm
/// there is no mean subtraction and no bias.
///
/// Two details matter for correctness and speed:
///
/// - The reduction always runs in F32, whatever the input dtype. Computing
///   `mean(x²)` over a bf16/f16 row loses enough precision to visibly shift the
///   logits; the result is cast back to the input dtype at the end.
/// - On Metal the whole operation is one fused kernel
///   ([`super::metal_ops::rms_norm_fused`]); elsewhere it is plain candle ops.
///
/// The [`NormType`] is applied to the weight once, at construction, so
/// [`forward`](Self::forward) is convention-agnostic. Build one with [`load`]
/// (safetensors), [`from_qtensor`] (GGUF), or [`new`] from a ready tensor.
///
/// [`load`]: Self::load
/// [`from_qtensor`]: Self::from_qtensor
/// [`new`]: Self::new
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    /// Builds a norm from a weight tensor, folding the [`NormType`] into it.
    ///
    /// For [`NormType::Gemma`] the stored weight is centred at zero, so `1.0` is
    /// added here once; [`NormType::Standard`] keeps the weight as-is. After
    /// this, the scale used by [`forward`](Self::forward) is just `weight`.
    pub fn new(weight: Tensor, eps: f64, variant: NormType) -> Result<Self> {
        let weight = match variant {
            NormType::Gemma => weight.affine(1.0, 1.0)?,
            NormType::Standard => weight,
        };
        Ok(Self { weight, eps })
    }

    /// Loads the weight `"{name}.weight"` from `weights` (safetensors path).
    ///
    /// ## Errors
    /// Fails if `weights` has no tensor under that key.
    pub fn load(weights: &ModelWeights, name: &str, eps: f64, variant: NormType) -> Result<Self> {
        let weight = weights.get(&format!("{}.weight", name))?.clone();
        Self::new(weight, eps, variant)
    }

    /// Builds a norm from a GGUF [`QTensor`], dequantizing it to `dtype` on
    /// `device` first.
    ///
    /// ## Errors
    /// Fails if dequantization or the dtype cast fails.
    pub fn from_qtensor(
        qtensor: &QTensor,
        device: &Device,
        dtype: DType,
        eps: f64,
        variant: NormType,
    ) -> Result<Self> {
        let weight = qtensor.dequantize(device)?.to_dtype(dtype)?;
        Self::new(weight, eps, variant)
    }

    /// Normalizes `x` over its last dimension; the output keeps `x`'s shape and
    /// dtype.
    ///
    /// On Metal this is a single fused kernel. Otherwise the RMS is computed in
    /// F32 (see the type docs for why) and the result cast back to `x`'s dtype.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if x.device().is_metal() {
            let x_c = x.contiguous()?;
            let w_c = self.weight.contiguous()?;
            return super::metal_ops::rms_norm_fused(&x_c, &w_c, self.eps as f32);
        }
        let dtype = x.dtype();
        let x_f32 = x.contiguous()?.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_f32
            .broadcast_div(&(variance + self.eps)?.sqrt()?)?
            .to_dtype(dtype)?;
        x_normed.broadcast_mul(&self.weight)
    }
}
