use super::config::BlockConfig;
use super::gguf_weights::GgufWeights;
use super::linear::{AnyLinear, QLinear, softmax_last_dim};
use super::norm::RMSNorm;
use super::paged::PagedKvCache;
use super::rope::RotaryEmbedding;
use super::weights::ModelWeights;
use candle_core::{D, DType, Result, Tensor};
use std::cell::Cell;

thread_local! {
    static SDPA_FALLBACK_LOGGED: Cell<bool> = const { Cell::new(false) };
}

#[cfg(feature = "metal")]
fn log_sdpa_fallback_once(head_dim: usize, dtype: candle_core::DType) {
    SDPA_FALLBACK_LOGGED.with(|logged| {
        if !logged.get() {
            tracing::warn!(
                head_dim,
                dtype = ?dtype,
                "Metal SDPA unavailable, using standard attention"
            );
            logged.set(true);
        }
    });
}

#[cfg(feature = "metal")]
fn log_softcap_sdpa_policy_once() {
    static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    LOGGED.get_or_init(|| {
        tracing::warn!("Metal SDPA softcap path disabled; using standard attention fallback");
    });
}

pub struct SegmentInfo<'a> {
    pub num_tokens: usize,
    pub cache: &'a mut PagedKvCache,
    pub reuse_cache: bool,
}

enum QkvProjection {
    Fused(AnyLinear),
    Separate {
        q: AnyLinear,
        k: AnyLinear,
        v: AnyLinear,
    },
}

pub struct Attention {
    qkv: QkvProjection,
    o_proj: AnyLinear,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    scale: f64,
    attn_softcap: Option<f64>,
    sliding_window: Option<usize>,
    v_norm: bool,
    rms_norm_eps: f64,
}

fn truncate_kv_window(
    k: candle_core::Tensor,
    v: candle_core::Tensor,
    kv_len: usize,
    window: Option<usize>,
    num_tokens: usize,
) -> Result<(candle_core::Tensor, candle_core::Tensor, usize)> {
    if let Some(w) = window
        && num_tokens == 1
        && kv_len > w
    {
        return Ok((k.narrow(2, kv_len - w, w)?, v.narrow(2, kv_len - w, w)?, w));
    }
    if kv_len < k.dim(2)? {
        Ok((k.narrow(2, 0, kv_len)?, v.narrow(2, 0, kv_len)?, kv_len))
    } else {
        Ok((k, v, kv_len))
    }
}

fn compute_sliding_window(cfg: &BlockConfig) -> Option<usize> {
    cfg.sliding_window
}

fn rms_norm_no_weight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.contiguous()?.to_dtype(candle_core::DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
    x_f32
        .broadcast_div(&(variance + eps)?.sqrt()?)?
        .to_dtype(dtype)
}

impl Attention {
    pub fn load(cfg: &BlockConfig, layer_idx: usize, weights: &ModelWeights) -> Result<Self> {
        let p = format!("model.layers.{}.self_attn", layer_idx);
        let hd = cfg.head_dim;
        let q_dim = cfg.n_heads * hd;
        let kv_dim = cfg.n_kv_heads * hd;
        let qkv_weight_name = format!("{}.qkv_proj.weight", p);
        let q_weight_name = format!("{}.q_proj.weight", p);
        let k_weight_name = format!("{}.k_proj.weight", p);
        let v_weight_name = format!("{}.v_proj.weight", p);
        let o_weight_name = format!("{}.o_proj.weight", p);

        let (qkv_w, qkv_scale_inv, qkv_bias) = if let Some(qkv_w) =
            weights.try_get(&qkv_weight_name)
        {
            let expected = q_dim + 2 * kv_dim;
            let got = qkv_w.dim(0)?;
            if got != expected {
                candle_core::bail!(
                    "qkv_proj shape mismatch at {}: expected dim0={}, got {}",
                    p,
                    expected,
                    got
                );
            }
            (
                qkv_w.clone(),
                weights.try_get_scale_inv(&qkv_weight_name).cloned(),
                weights.try_get(&format!("{}.qkv_proj.bias", p)).cloned(),
            )
        } else {
            let q_w = weights.get(&q_weight_name)?;
            let k_w = weights.get(&k_weight_name)?;
            let v_w = weights.get(&v_weight_name)?;
            let qkv_w = Tensor::cat(&[q_w, k_w, v_w], 0)?;
            let qkv_bias = match (
                weights.try_get(&format!("{}.q_proj.bias", p)),
                weights.try_get(&format!("{}.k_proj.bias", p)),
                weights.try_get(&format!("{}.v_proj.bias", p)),
            ) {
                (Some(qb), Some(kb), Some(vb)) => Some(Tensor::cat(&[qb, kb, vb], 0)?),
                _ => None,
            };
            let q_is_fp8 = q_w.dtype() == DType::F8E4M3;
            let k_is_fp8 = k_w.dtype() == DType::F8E4M3;
            let v_is_fp8 = v_w.dtype() == DType::F8E4M3;

            let qkv_scale_inv = if q_is_fp8 || k_is_fp8 || v_is_fp8 {
                let q_scale = weights.try_get_scale_inv(&q_weight_name).cloned();
                let k_scale = weights.try_get_scale_inv(&k_weight_name).cloned();
                let v_scale = weights.try_get_scale_inv(&v_weight_name).cloned();

                match (q_scale, k_scale, v_scale) {
                    (Some(qs), Some(ks), Some(vs)) => Some(Tensor::cat(&[&qs, &ks, &vs], 0)?),
                    _ => {
                        candle_core::bail!(
                            "missing q/k/v *_scale_inv tensors required by FP8 projections at {}",
                            p
                        )
                    }
                }
            } else {
                None
            };

            (qkv_w, qkv_scale_inv, qkv_bias)
        };
        if qkv_w.dtype() == DType::F8E4M3 && qkv_scale_inv.is_none() {
            candle_core::bail!(
                "missing '{}' required by FP8 tensor '{}'",
                format!("{}_scale_inv", qkv_weight_name),
                qkv_weight_name
            );
        }
        let qkv_proj = AnyLinear::from_weight_with_scale_inv(qkv_w, qkv_scale_inv, qkv_bias);

        let o_weight = weights.get(&o_weight_name)?.clone();
        let o_scale_inv = weights.try_get_scale_inv(&o_weight_name).cloned();
        if o_weight.dtype() == DType::F8E4M3 && o_scale_inv.is_none() {
            candle_core::bail!(
                "missing '{}' required by FP8 tensor '{}'",
                format!("{}_scale_inv", o_weight_name),
                o_weight_name
            );
        }
        let o_proj = AnyLinear::from_weight_with_scale_inv(o_weight, o_scale_inv, None);

        let q_norm = if cfg.qk_norm {
            Some(RMSNorm::new(
                weights.get(&format!("{}.q_norm.weight", p))?.clone(),
                cfg.rms_norm_eps,
                cfg.norm_type,
            )?)
        } else {
            None
        };
        let k_norm = if cfg.qk_norm {
            Some(RMSNorm::new(
                weights.get(&format!("{}.k_norm.weight", p))?.clone(),
                cfg.rms_norm_eps,
                cfg.norm_type,
            )?)
        } else {
            None
        };

        let actual_window = compute_sliding_window(cfg);

        Ok(Self {
            qkv: QkvProjection::Fused(qkv_proj),
            o_proj,
            q_norm,
            k_norm,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: hd,
            q_dim,
            kv_dim,
            scale: cfg.attention_scale.unwrap_or(1.0 / (hd as f64).sqrt()),
            attn_softcap: cfg.attn_softcap,
            sliding_window: actual_window,
            v_norm: cfg.v_norm,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    pub fn load_gguf(
        cfg: &BlockConfig,
        layer_idx: usize,
        gguf: &GgufWeights,
        device: &candle_core::Device,
        dtype: candle_core::DType,
    ) -> Result<Self> {
        let prefix = format!("blk.{}", layer_idx);
        let hd = cfg.head_dim;
        let qkv = if let Some(qkv_qt) = gguf.try_get(&format!("{prefix}.attn_qkv.weight")) {
            let q_dim = cfg.n_heads * hd;
            let kv_dim = cfg.n_kv_heads * hd;
            let expected = q_dim + 2 * kv_dim;
            let got = qkv_qt.shape().dims()[0];
            if got != expected {
                candle_core::bail!(
                    "GGUF attn_qkv shape mismatch at {}: expected dim0={}, got {}",
                    prefix,
                    expected,
                    got
                );
            }
            QkvProjection::Fused(AnyLinear::Quantized(QLinear::from_arc(qkv_qt, dtype)?))
        } else {
            let q_proj = QLinear::from_arc(gguf.get(&format!("{prefix}.attn_q.weight"))?, dtype)?;
            let k_proj = QLinear::from_arc(gguf.get(&format!("{prefix}.attn_k.weight"))?, dtype)?;
            let v_proj = QLinear::from_arc(gguf.get(&format!("{prefix}.attn_v.weight"))?, dtype)?;
            QkvProjection::Separate {
                q: AnyLinear::Quantized(q_proj),
                k: AnyLinear::Quantized(k_proj),
                v: AnyLinear::Quantized(v_proj),
            }
        };
        let o_proj = QLinear::from_arc(gguf.get(&format!("{prefix}.attn_output.weight"))?, dtype)?;

        let q_norm = if cfg.qk_norm {
            let qt = gguf.get(&format!("{prefix}.attn_q_norm.weight"))?;
            Some(RMSNorm::from_qtensor(
                &qt,
                device,
                dtype,
                cfg.rms_norm_eps,
                cfg.norm_type,
            )?)
        } else {
            None
        };
        let k_norm = if cfg.qk_norm {
            let qt = gguf.get(&format!("{prefix}.attn_k_norm.weight"))?;
            Some(RMSNorm::from_qtensor(
                &qt,
                device,
                dtype,
                cfg.rms_norm_eps,
                cfg.norm_type,
            )?)
        } else {
            None
        };

        let q_dim = cfg.n_heads * hd;
        let kv_dim = cfg.n_kv_heads * hd;

        let actual_window = compute_sliding_window(cfg);

        Ok(Self {
            qkv,
            o_proj: AnyLinear::Quantized(o_proj),
            q_norm,
            k_norm,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: hd,
            q_dim,
            kv_dim,
            scale: cfg.attention_scale.unwrap_or(1.0 / (hd as f64).sqrt()),
            attn_softcap: cfg.attn_softcap,
            sliding_window: actual_window,
            v_norm: cfg.v_norm,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    fn qkv_split(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        match &self.qkv {
            QkvProjection::Fused(qkv) => {
                let out = qkv.forward(x)?;
                let q = out.narrow(D::Minus1, 0, self.q_dim)?;
                let k = out.narrow(D::Minus1, self.q_dim, self.kv_dim)?;
                let v = out.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?;
                Ok((q, k, v))
            }
            QkvProjection::Separate { q, k, v } => {
                Ok((q.forward(x)?, k.forward(x)?, v.forward(x)?))
            }
        }
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.n_heads / self.n_kv_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (b, n_kv_h, seq, hd) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b, n_kv_h, n_rep, seq, hd))?
            .reshape((b, n_kv_h * n_rep, seq, hd))
    }

    pub fn forward_batch(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        position_ids: &Tensor,
        mask: Option<&Tensor>,
        segments: &mut [SegmentInfo],
    ) -> Result<Tensor> {
        self.forward_batch_optional_rope(x, Some((rope, position_ids)), mask, segments)
    }

    pub fn forward_batch_optional_rope(
        &self,
        x: &Tensor,
        rope: Option<(&RotaryEmbedding, &Tensor)>,
        mask: Option<&Tensor>,
        segments: &mut [SegmentInfo],
    ) -> Result<Tensor> {
        let (b, total_seq, _) = x.dims3()?;
        let hd = self.head_dim;

        let (q_raw, k_raw, v_raw) = self.qkv_split(x)?;
        let q = q_raw
            .reshape((b, total_seq, self.n_heads, hd))?
            .transpose(1, 2)?;
        let k = k_raw
            .reshape((b, total_seq, self.n_kv_heads, hd))?
            .transpose(1, 2)?;
        let v = v_raw
            .reshape((b, total_seq, self.n_kv_heads, hd))?
            .transpose(1, 2)?;

        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };
        let v = if self.v_norm {
            rms_norm_no_weight(&v, self.rms_norm_eps)?
        } else {
            v
        };

        let (q, k) = if let Some((r, position_ids)) = rope {
            r.apply_qk_with_positions(&q, &k, position_ids)?
        } else {
            (q, k)
        };

        let mut k_parts: Vec<Tensor> = Vec::with_capacity(segments.len());
        let mut v_parts: Vec<Tensor> = Vec::with_capacity(segments.len());
        let mut kv_lengths: Vec<usize> = Vec::with_capacity(segments.len());
        let mut offset = 0usize;

        for seg in segments.iter_mut() {
            let (full_k, full_v) = if seg.reuse_cache {
                seg.cache.current()?
            } else {
                let seg_k = k.narrow(2, offset, seg.num_tokens)?;
                let seg_v = v.narrow(2, offset, seg.num_tokens)?;
                seg.cache.append(&seg_k, &seg_v)?
            };
            let kv_len = full_k.dim(2)?;
            kv_lengths.push(kv_len);
            k_parts.push(full_k);
            v_parts.push(full_v);
            offset += seg.num_tokens;
        }

        let device = x.device();
        let out_buf = Tensor::zeros((b, self.n_heads, total_seq, hd), q.dtype(), device)?;
        let mut q_offset = 0usize;

        // ── Metal SDPA path for batch ────────────────────────────────
        #[cfg(feature = "metal")]
        if self.attn_softcap.is_some() {
            log_softcap_sdpa_policy_once();
        }
        #[cfg(feature = "metal")]
        let use_sdpa = self.attn_softcap.is_none()
            && super::metal_ops::sdpa_available(&q, self.head_dim)
            && !kv_lengths
                .iter()
                .zip(segments.iter())
                .any(|(&kv_len, seg)| seg.num_tokens > 1 && kv_len > seg.num_tokens);
        #[cfg(not(feature = "metal"))]
        let use_sdpa = false;

        #[cfg(feature = "metal")]
        if !use_sdpa
            && self.attn_softcap.is_none()
            && !super::metal_ops::sdpa_available(&q, self.head_dim)
        {
            log_sdpa_fallback_once(self.head_dim, q.dtype());
        }

        if use_sdpa {
            #[cfg(feature = "metal")]
            for (i, seg) in segments.iter().enumerate() {
                let q_seg = q.narrow(2, q_offset, seg.num_tokens)?;
                let (k_seg, v_seg, _) = truncate_kv_window(
                    k_parts[i].clone(),
                    v_parts[i].clone(),
                    kv_lengths[i],
                    self.sliding_window,
                    seg.num_tokens,
                )?;

                // SDPA handles GQA natively — no repeat_kv needed.
                let q_c_owned;
                let q_c = if q_seg.is_contiguous() {
                    &q_seg
                } else {
                    q_c_owned = q_seg.contiguous()?;
                    &q_c_owned
                };
                let k_c_owned;
                let k_c = if k_seg.is_contiguous() {
                    &k_seg
                } else {
                    k_c_owned = k_seg.contiguous()?;
                    &k_c_owned
                };
                let v_c_owned;
                let v_c = if v_seg.is_contiguous() {
                    &v_seg
                } else {
                    v_c_owned = v_seg.contiguous()?;
                    &v_c_owned
                };

                let seg_out = super::metal_ops::sdpa(
                    q_c,
                    k_c,
                    v_c,
                    None,
                    seg.num_tokens > 1, // causal for prefill segments
                    self.scale as f32,
                    self.attn_softcap.map(|s| s as f32).unwrap_or(1.0),
                )?;
                let seg_out = seg_out.contiguous()?;
                out_buf.slice_set(&seg_out, 2, q_offset)?;
                q_offset += seg.num_tokens;
            }
        } else {
            // ── Fallback: standard attention ─────────────────────────
            for (i, seg) in segments.iter().enumerate() {
                let q_seg = q.narrow(2, q_offset, seg.num_tokens)?;
                let (k_seg, v_seg, kv_len) = truncate_kv_window(
                    k_parts[i].clone(),
                    v_parts[i].clone(),
                    kv_lengths[i],
                    self.sliding_window,
                    seg.num_tokens,
                )?;

                let k_seg = self.repeat_kv(k_seg)?;
                let v_seg = self.repeat_kv(v_seg)?;

                let mut scores = q_seg
                    .matmul(&k_seg.transpose(D::Minus1, D::Minus2)?)?
                    .affine(self.scale, 0.)?;

                if let Some(softcap) = self.attn_softcap {
                    scores = (scores / softcap)?.tanh()?.affine(softcap, 0.)?;
                }

                let scores = if let Some(m) = mask {
                    let seg_mask = m
                        .narrow(2, q_offset, seg.num_tokens)?
                        .narrow(3, 0, kv_len)?;
                    scores.broadcast_add(&seg_mask.to_dtype(scores.dtype())?)?
                } else if seg.num_tokens > 1 {
                    let cm = if kv_len > seg.num_tokens {
                        super::mask::causal_mask_prefixed_cached_dtype(
                            seg.num_tokens,
                            kv_len,
                            scores.dtype(),
                            device,
                        )?
                    } else {
                        super::mask::causal_mask_cached_dtype(
                            seg.num_tokens,
                            scores.dtype(),
                            device,
                        )?
                    };
                    scores.broadcast_add(&cm)?
                } else {
                    scores
                };

                let attn = softmax_last_dim(&scores)?;
                let seg_out = attn.matmul(&v_seg)?.contiguous()?;
                out_buf.slice_set(&seg_out, 2, q_offset)?;
                q_offset += seg.num_tokens;
            }
        }

        let out = out_buf
            .transpose(1, 2)?
            .reshape((b, total_seq, self.n_heads * hd))?;
        self.o_proj.forward(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config::{Activation, NormType};
    use candle_core::Device;

    fn test_block_cfg(norm_type: NormType, sliding_window: Option<usize>) -> BlockConfig {
        BlockConfig {
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-5,
            qk_norm: false,
            attention_scale: None,
            activation: Activation::SiLU,
            norm_type,
            attn_softcap: None,
            v_norm: false,
            has_ffn_norms: false,
            sliding_window,
        }
    }

    fn make_kv_tensors(seq_len: usize, head_dim: usize) -> Result<(Tensor, Tensor)> {
        let total = seq_len * head_dim;
        let k_data: Vec<f32> = (0..total).map(|v| v as f32).collect();
        let v_data: Vec<f32> = (0..total).map(|v| 1000.0 + v as f32).collect();
        let k = Tensor::from_vec(k_data, (1, 1, seq_len, head_dim), &Device::Cpu)?;
        let v = Tensor::from_vec(v_data, (1, 1, seq_len, head_dim), &Device::Cpu)?;
        Ok((k, v))
    }

    #[test]
    fn compute_sliding_window_uses_config_even_for_gemma_norm() {
        let cfg = test_block_cfg(NormType::Gemma, Some(128));
        assert_eq!(compute_sliding_window(&cfg), Some(128));
    }

    #[test]
    fn compute_sliding_window_keeps_standard_setting() {
        let cfg = test_block_cfg(NormType::Standard, Some(256));
        assert_eq!(compute_sliding_window(&cfg), Some(256));
    }

    #[test]
    fn truncate_kv_window_uses_tail_for_decode_window() -> Result<()> {
        let (k, v) = make_kv_tensors(10, 2)?;
        let (k2, v2, kv_len) = truncate_kv_window(k, v, 10, Some(4), 1)?;

        assert_eq!(kv_len, 4);
        assert_eq!(k2.dims4()?, (1, 1, 4, 2));
        assert_eq!(v2.dims4()?, (1, 1, 4, 2));

        let k_flat: Vec<f32> = k2.flatten_all()?.to_vec1()?;
        let v_flat: Vec<f32> = v2.flatten_all()?.to_vec1()?;
        assert_eq!(k_flat[0], 12.0);
        assert_eq!(k_flat[1], 13.0);
        assert_eq!(v_flat[0], 1012.0);
        assert_eq!(v_flat[1], 1013.0);
        Ok(())
    }

    #[test]
    fn truncate_kv_window_clamps_to_kv_len_when_cache_is_shorter() -> Result<()> {
        let (k, v) = make_kv_tensors(8, 2)?;
        let (k2, v2, kv_len) = truncate_kv_window(k, v, 5, None, 1)?;

        assert_eq!(kv_len, 5);
        assert_eq!(k2.dims4()?, (1, 1, 5, 2));
        assert_eq!(v2.dims4()?, (1, 1, 5, 2));
        Ok(())
    }

    #[test]
    fn truncate_kv_window_keeps_full_prefill_even_with_window() -> Result<()> {
        let (k, v) = make_kv_tensors(10, 2)?;
        let (k2, v2, kv_len) = truncate_kv_window(k, v, 10, Some(4), 3)?;

        assert_eq!(kv_len, 10);
        assert_eq!(k2.dims4()?, (1, 1, 10, 2));
        assert_eq!(v2.dims4()?, (1, 1, 10, 2));
        Ok(())
    }

    #[test]
    fn rms_norm_no_weight_normalizes_last_dimension() -> Result<()> {
        let x = Tensor::from_vec(vec![3f32, 4f32, 0f32, 5f32], (2, 2), &Device::Cpu)?;
        let y = rms_norm_no_weight(&x, 1e-6)?;
        assert_eq!(y.dtype(), x.dtype());

        let rows: Vec<Vec<f32>> = y.to_vec2()?;
        for row in rows {
            let rms = ((row[0] * row[0] + row[1] * row[1]) / 2.0).sqrt();
            assert!((rms - 1.0).abs() < 1e-3);
        }
        Ok(())
    }
}
