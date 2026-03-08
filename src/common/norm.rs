use candle_core::{DType, Device, D, Result, Tensor};
use candle_core::quantized::QTensor;
use super::weights::ModelWeights;
use super::config::NormType;

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
    variant: NormType,
}
impl RMSNorm {
    pub fn new(weight: Tensor, eps: f64, variant: NormType) -> Self {
        Self { weight, eps, variant }
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
        let dtype = x.dtype();
        let x_c = x.contiguous()?;
        let x_f32 = x_c.to_dtype(candle_core::DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(dtype)?;
        match self.variant {
            NormType::Standard => x_normed.broadcast_mul(&self.weight),
            NormType::Gemma => x_normed.broadcast_mul(&self.weight)? + x_normed,
        }
    }
}
