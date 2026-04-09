use candle_core::quantized::{QMatMul, QTensor};
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

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        let weight_t = weight.t().expect("Linear weight must be 2D");
        Self { weight_t, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = if x.rank() > 2 {
            let original_dims = x.dims().to_vec();
            let in_features = *original_dims.last().unwrap();
            let batch_flat: usize = original_dims[..original_dims.len() - 1].iter().product();
            let o = x
                .reshape((batch_flat, in_features))?
                .matmul(&self.weight_t)?;
            let out_features = o.dim(1)?;
            let mut new_dims = original_dims;
            *new_dims.last_mut().unwrap() = out_features;
            o.reshape(new_dims)?
        } else {
            x.matmul(&self.weight_t)?
        };

        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
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
    inner: QMatMul,
    out_dtype: DType,
}

impl QLinear {
    pub fn from_arc(qtensor: Arc<QTensor>, out_dtype: DType) -> Result<Self> {
        Ok(Self {
            inner: QMatMul::from_arc(qtensor)?,
            out_dtype,
        })
    }

    /// Note: candle's QMatMul operates in F32 only, so on BF16/GPU this performs
    /// BF16→F32 (input) then F32→BF16 (output) — two dtype casts per call.
    /// This is an inherent candle limitation, not something we can avoid here.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_dims = x.dims().to_vec();

        let x_f32_owned;
        let x_f32: &Tensor = if x.dtype() != DType::F32 {
            x_f32_owned = x.to_dtype(DType::F32)?;
            &x_f32_owned
        } else {
            x
        };

        let x_2d = if x_f32.rank() > 2 {
            let in_features = *original_dims.last().unwrap();
            let batch_flat: usize = original_dims[..original_dims.len() - 1].iter().product();
            x_f32.reshape((batch_flat, in_features))?
        } else {
            x_f32.clone()
        };

        let out = candle_core::Module::forward(&self.inner, &x_2d)?;

        let out = if original_dims.len() > 2 {
            let out_features = out.dim(candle_core::D::Minus1)?;
            let mut new_dims = original_dims;
            *new_dims.last_mut().unwrap() = out_features;
            out.reshape(new_dims)?
        } else {
            out
        };

        if out.dtype() != self.out_dtype {
            out.to_dtype(self.out_dtype)
        } else {
            Ok(out)
        }
    }
}

pub enum AnyLinear {
    Float(Linear),
    Quantized(QLinear),
}

impl AnyLinear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Float(l) => l.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }
}
