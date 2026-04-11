use candle_core::{D, DType, Device, Result, Tensor};

#[derive(Debug, Clone)]
pub enum RopeScaling {
    None,
    Linear {
        factor: f64,
    },
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_pos: usize,
    },
    Yarn {
        factor: f64,
        original_max_pos: usize,
        beta_fast: f64,
        beta_slow: f64,
    },
    LongRope {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        original_max_pos: usize,
    },
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
        Self::new_with_scaling(
            head_dim,
            max_seq_len,
            rope_theta,
            RopeScaling::None,
            dtype,
            device,
        )
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
            .map(|i| 1.0 / (rope_theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        match scaling {
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_pos,
            } => {
                let low_freq_wavelen = original_max_pos as f32 / low_freq_factor as f32;
                let high_freq_wavelen = original_max_pos as f32 / high_freq_factor as f32;
                let denom = high_freq_factor as f32 - low_freq_factor as f32;

                for inv in inv_freq.iter_mut().take(head_dim / 2) {
                    let wavelen = 2.0 * std::f32::consts::PI / *inv;
                    if wavelen > low_freq_wavelen {
                        *inv /= factor as f32;
                    } else if wavelen > high_freq_wavelen {
                        if denom.abs() < 1e-6 {
                            continue;
                        }
                        let smooth =
                            (original_max_pos as f32 / wavelen - low_freq_factor as f32) / denom;
                        *inv /= (1.0 - smooth) * factor as f32 + smooth;
                    }
                }
            }
            RopeScaling::Linear { factor } => {
                for f in &mut inv_freq {
                    *f /= factor as f32;
                }
            }
            RopeScaling::Yarn {
                factor,
                original_max_pos,
                beta_fast,
                beta_slow,
            } => {
                let dim = head_dim;
                let base = rope_theta;

                let find_correction_dim = |num_rotations: f64| -> f64 {
                    (dim as f64
                        * (original_max_pos as f64 / (num_rotations * 2.0 * std::f64::consts::PI))
                            .ln())
                        / (2.0 * base.ln())
                };
                let low = (find_correction_dim(beta_fast).floor().max(0.0)) as usize;
                let high = (find_correction_dim(beta_slow)
                    .ceil()
                    .min((dim / 2 - 1) as f64)) as usize;

                for (i, inv) in inv_freq.iter_mut().enumerate().take(dim / 2) {
                    let freq_extra = *inv;
                    let freq_inter = *inv / factor as f32;

                    let ramp = if high <= low {
                        if i < low { 0.0f32 } else { 1.0 }
                    } else {
                        ((i as f32 - low as f32) / (high as f32 - low as f32)).clamp(0.0, 1.0)
                    };

                    // ramp=0 → keep original freq (high-freq dims, small index)
                    // ramp=1 → scale by factor  (low-freq dims, large index)
                    *inv = freq_inter * ramp + freq_extra * (1.0 - ramp);
                }
            }
            RopeScaling::LongRope {
                short_factor,
                long_factor,
                original_max_pos,
            } => {
                let use_long = max_seq_len > original_max_pos;
                let factors = if use_long {
                    &long_factor
                } else {
                    &short_factor
                };
                let fallback = factors.last().copied().unwrap_or(1.0);

                for (i, inv) in inv_freq.iter_mut().enumerate().take(head_dim / 2) {
                    let factor = factors.get(i).copied().unwrap_or(fallback);
                    if factor > 0.0 {
                        *inv /= factor as f32;
                    }
                }
            }
            RopeScaling::None => {}
        }

        let inv_freq = Tensor::from_vec(inv_freq, (1, head_dim / 2), device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq_len, 1), device)?;

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
            let x_c = if x.is_contiguous() {
                x.clone()
            } else {
                x.contiguous()?
            };
            let cos_c = if cos.is_contiguous() {
                cos.clone()
            } else {
                cos.contiguous()?
            };
            let sin_c = if sin.is_contiguous() {
                sin.clone()
            } else {
                sin.contiguous()?
            };
            return super::metal_ops::rope_fused(&x_c, &cos_c, &sin_c);
        }

        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let x1 = x.narrow(D::Minus1, 0, d / 2)?;
        let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;

        let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let out2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;

        Tensor::cat(&[&out1, &out2], D::Minus1)
    }

    pub fn apply_qk_with_positions(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (bq, qh, tq, d_q) = q.dims4()?;
        let (bk, kh, tk, d_k) = k.dims4()?;

        if d_q != d_k {
            candle_core::bail!("RoPE q/k head_dim mismatch: q={d_q}, k={d_k}");
        }

        #[cfg(feature = "metal")]
        if q.device().is_metal()
            && k.device().is_metal()
            && q.dtype() == k.dtype()
            && bq == bk
            && tq == tk
        {
            let cos = self.cos.index_select(position_ids, 0)?;
            let sin = self.sin.index_select(position_ids, 0)?;

            let qk = Tensor::cat(&[q, k], 1)?;
            let qk_c = if qk.is_contiguous() {
                qk
            } else {
                qk.contiguous()?
            };
            let cos_c = if cos.is_contiguous() {
                cos
            } else {
                cos.contiguous()?
            };
            let sin_c = if sin.is_contiguous() {
                sin
            } else {
                sin.contiguous()?
            };

            let rotated = super::metal_ops::rope_fused(&qk_c, &cos_c, &sin_c)?;
            let q_rot = rotated.narrow(1, 0, qh)?;
            let k_rot = rotated.narrow(1, qh, kh)?;
            return Ok((q_rot, k_rot));
        }

        Ok((
            self.apply_with_positions(q, position_ids)?,
            self.apply_with_positions(k, position_ids)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_equal_freq_factors_do_not_produce_nan() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new_with_scaling(
            8,
            32,
            10_000.0,
            RopeScaling::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 1.0,
                original_max_pos: 4,
            },
            DType::F32,
            &device,
        )
        .expect("llama3 rope construction should succeed");

        let cos = rope
            .cos
            .to_vec2::<f32>()
            .expect("cos should be materializable for test");
        let sin = rope
            .sin
            .to_vec2::<f32>()
            .expect("sin should be materializable for test");

        assert!(cos.iter().flatten().all(|v| v.is_finite()));
        assert!(sin.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn longrope_scales_frequencies_for_long_context() {
        let device = Device::Cpu;
        let base = RotaryEmbedding::new_with_scaling(
            8,
            32,
            10_000.0,
            RopeScaling::None,
            DType::F32,
            &device,
        )
        .expect("base rope should construct");

        let long = RotaryEmbedding::new_with_scaling(
            8,
            32,
            10_000.0,
            RopeScaling::LongRope {
                short_factor: vec![1.0],
                long_factor: vec![2.0],
                original_max_pos: 4,
            },
            DType::F32,
            &device,
        )
        .expect("longrope should construct");

        let base_cos = base
            .cos
            .to_vec2::<f32>()
            .expect("base cos should be materializable");
        let long_cos = long
            .cos
            .to_vec2::<f32>()
            .expect("longrope cos should be materializable");

        // At position=1, dim=0 has frequency 1.0 in baseline and 0.5 in the longrope case.
        // Therefore cos should be larger in the longrope case (cos(0.5) > cos(1.0)).
        assert!(long_cos[1][0] > base_cos[1][0]);
    }

    #[test]
    fn apply_qk_with_positions_matches_individual_cpu() -> Result<()> {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(8, 16, 10_000.0, DType::F32, &device)?;

        let q_data: Vec<f32> = (0..(4 * 3 * 8)).map(|v| v as f32 * 0.01).collect();
        let k_data: Vec<f32> = (0..(2 * 3 * 8)).map(|v| 1.0 + v as f32 * 0.02).collect();
        let q = Tensor::from_vec(q_data, (1, 4, 3, 8), &device)?;
        let k = Tensor::from_vec(k_data, (1, 2, 3, 8), &device)?;
        let position_ids = Tensor::from_vec(vec![0u32, 1, 2], (3,), &device)?;

        let (q_pair, k_pair) = rope.apply_qk_with_positions(&q, &k, &position_ids)?;
        let q_single = rope.apply_with_positions(&q, &position_ids)?;
        let k_single = rope.apply_with_positions(&k, &position_ids)?;

        let q_pair_v: Vec<f32> = q_pair.flatten_all()?.to_vec1()?;
        let q_single_v: Vec<f32> = q_single.flatten_all()?.to_vec1()?;
        let k_pair_v: Vec<f32> = k_pair.flatten_all()?.to_vec1()?;
        let k_single_v: Vec<f32> = k_single.flatten_all()?.to_vec1()?;

        assert_eq!(q_pair_v.len(), q_single_v.len());
        assert_eq!(k_pair_v.len(), k_single_v.len());
        for (a, b) in q_pair_v.iter().zip(q_single_v.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        for (a, b) in k_pair_v.iter().zip(k_single_v.iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        Ok(())
    }
}
