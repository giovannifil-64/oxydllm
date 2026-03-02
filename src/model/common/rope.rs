use candle_core::{Tensor, Device, Result, D};

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| {
                1.0 / (rope_theta as f32).powf(2.0 * i as f32 / head_dim as f32)
            })
            .collect();
        
        let inv_freq = Tensor::from_vec(
            inv_freq, 
            (1, head_dim / 2),
            device
        )?;
        
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(
            positions,
            (max_seq_len, 1),
            device
        )?;
        
        let freqs = positions.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        
        Ok(Self { cos, sin })
    }

    pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_b, _h, seq, d) = x.dims4()?;
        
        let cos = self.cos.narrow(0, start_pos, seq)?;
        let sin = self.sin.narrow(0, start_pos, seq)?;
        
        let cos = cos.reshape((1, 1, seq, d / 2))?;
        let sin = sin.reshape((1, 1, seq, d / 2))?;
        
        let cos = cos.to_dtype(x.dtype())?;
        let sin = sin.to_dtype(x.dtype())?;

        let x1 = x.narrow(D::Minus1, 0, d / 2)?;
        let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
        
        let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let out2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
        
        Tensor::cat(&[&out1, &out2], D::Minus1)
    }

    pub fn apply_with_positions(&self, x: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (_b, _h, _seq, d) = x.dims4()?;

        let cos = self.cos.index_select(position_ids, 0)?;
        let sin = self.sin.index_select(position_ids, 0)?;

        let cos = cos.unsqueeze(0)?.unsqueeze(0)?.to_dtype(x.dtype())?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?.to_dtype(x.dtype())?;

        let x1 = x.narrow(D::Minus1, 0, d / 2)?;
        let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;

        let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let out2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;

        Tensor::cat(&[&out1, &out2], D::Minus1)
    }
}
