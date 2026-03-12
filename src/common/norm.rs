use candle_core::{DType, Device, D, Result, Tensor};
use candle_core::quantized::QTensor;
use super::weights::ModelWeights;
use super::config::NormType;

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f64, variant: NormType) -> Self {
        let weight = match variant {
            NormType::Gemma => weight.affine(1.0, 1.0).expect("RMSNorm Gemma weight+1 failed"),
            NormType::Standard => weight,
        };
        Self { weight, eps }
    }
    pub fn load(weights: &ModelWeights, name: &str, eps: f64, variant: NormType) -> Result<Self> {
        let weight = weights.get(&format!("{}.weight", name))?.clone();
        Ok(Self::new(weight, eps, variant))
    }
    pub fn from_qtensor(qtensor: &QTensor, device: &Device, dtype: DType, eps: f64, variant: NormType) -> Result<Self> {
        let weight = qtensor.dequantize(device)?.to_dtype(dtype)?;
        Ok(Self::new(weight, eps, variant))
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if x.device().is_metal() {
            let x_c = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
            let w_c = if self.weight.is_contiguous() { self.weight.clone() } else { self.weight.contiguous()? };
            return super::metal_ops::rms_norm_fused(&x_c, &w_c, self.eps as f32);
        }
        let dtype = x.dtype();
        let x_f32 = x.contiguous()?.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?.to_dtype(dtype)?;
        x_normed.broadcast_mul(&self.weight)
    }
}
