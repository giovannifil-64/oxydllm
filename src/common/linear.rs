use candle_core::{Result, Tensor};

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
}
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out_features = self.weight.dim(0)?;
        let w_t = self.weight.t()?;

        let out = if x.rank() > 2 {
            let original_dims = x.dims().to_vec();
            let in_features = *original_dims.last().unwrap();
            let batch_flat: usize = original_dims[..original_dims.len() - 1].iter().product();
            let o = x.reshape((batch_flat, in_features))?.matmul(&w_t)?;
            let mut new_dims = original_dims;
            *new_dims.last_mut().unwrap() = out_features;
            o.reshape(new_dims)?
        } else {
            x.matmul(&w_t)?
        };

        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}


pub fn silu(x: &Tensor) -> Result<Tensor> {
    x.div(&x.neg()?.exp()?.affine(1.0, 1.0)?)
}

pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(candle_core::D::Minus1)?;
    let x = x.broadcast_sub(&max)?;
    let exp_x = x.exp()?;
    let sum = exp_x.sum_keepdim(candle_core::D::Minus1)?;
    exp_x.broadcast_div(&sum)
}
