use candle_core::{D, Result, Tensor};
use super::weights::ModelWeights;

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}
impl RMSNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
    pub fn load(weights: &ModelWeights, name: &str, eps: f64) -> Result<Self> {
        let weight = weights.get(&format!("{}.weight", name))?.clone();
        Ok(Self::new(weight, eps))
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(dtype)?;
        x_normed.broadcast_mul(&self.weight)
    }
}
