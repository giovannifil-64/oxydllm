use candle_core::{DType, Tensor, Device, Result, D};

#[derive(Debug, Clone)]
pub enum RopeScaling {
    None,
    Linear { factor: f64 },
    Llama3 { factor: f64, low_freq_factor: f64, high_freq_factor: f64, original_max_pos: usize },
    Yarn { factor: f64 },
}

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_scaling(head_dim, max_seq_len, rope_theta, RopeScaling::None, dtype, device)
    }

    pub fn new_with_scaling(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        scaling: RopeScaling,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| {
                1.0 / (rope_theta as f32).powf(2.0 * i as f32 / head_dim as f32)
            })
            .collect();
            
        match scaling {
            RopeScaling::Llama3 { factor, low_freq_factor, high_freq_factor, original_max_pos } => {
                let low_freq_wavelen = original_max_pos as f32 / low_freq_factor as f32;
                let high_freq_wavelen = original_max_pos as f32 / high_freq_factor as f32;
                
                for i in 0..head_dim / 2 {
                    let wavelen = 2.0 * std::f32::consts::PI / inv_freq[i];
                    if wavelen > low_freq_wavelen {
                        inv_freq[i] /= factor as f32;
                    } else if wavelen > high_freq_wavelen {
                        let smooth = (original_max_pos as f32 / wavelen - low_freq_factor as f32) / (high_freq_factor as f32 - low_freq_factor as f32);
                        inv_freq[i] /= (1.0 - smooth) * factor as f32 + smooth;
                    }
                }
            }
            RopeScaling::Linear { factor } => {
                for f in &mut inv_freq {
                    *f /= factor as f32;
                }
            }
            RopeScaling::Yarn { factor, .. } => {
                for f in &mut inv_freq {
                    *f /= factor as f32;
                }
            }
            RopeScaling::None => {}
        }
        
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
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    pub fn apply_with_positions(&self, x: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (_b, _h, _seq, d) = x.dims4()?;

        let cos = self.cos.index_select(position_ids, 0)?;
        let sin = self.sin.index_select(position_ids, 0)?;

        #[cfg(feature = "metal")]
        if x.device().is_metal() {
            let x_c   = if x.is_contiguous()   { x.clone()   } else { x.contiguous()?   };
            let cos_c = if cos.is_contiguous() { cos.clone() } else { cos.contiguous()? };
            let sin_c = if sin.is_contiguous() { sin.clone() } else { sin.contiguous()? };
            return super::metal_ops::rope_fused(&x_c, &cos_c, &sin_c);
        }

        // CPU / non-Metal fallback
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let x1 = x.narrow(D::Minus1, 0, d / 2)?;
        let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;

        let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let out2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;

        Tensor::cat(&[&out1, &out2], D::Minus1)
    }
}
