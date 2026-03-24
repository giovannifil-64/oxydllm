use candle_core::{Result, Tensor, D};
use std::cell::Cell;
use super::config::{BlockConfig, NormType};
use super::gguf_weights::GgufWeights;
use super::paged::PagedKvCache;
use super::linear::{softmax_last_dim, AnyLinear, Linear, QLinear};
use super::norm::RMSNorm;
use super::rope::RotaryEmbedding;
use super::weights::ModelWeights;

thread_local! {
    static SDPA_FALLBACK_LOGGED: Cell<bool> = const { Cell::new(false) };
}

fn log_sdpa_fallback_once(head_dim: usize, dtype: candle_core::DType) {
    SDPA_FALLBACK_LOGGED.with(|logged| {
        if !logged.get() {
            eprintln!(
                "[attention] Metal SDPA unavailable (head_dim={}, dtype={:?}) — using standard attention",
                head_dim, dtype
            );
            logged.set(true);
        }
    });
}

pub struct SegmentInfo<'a> {
    pub num_tokens: usize,
    pub cache: &'a mut PagedKvCache,
}

enum QkvProjection {
    Fused(Linear),
    Separate { q: AnyLinear, k: AnyLinear, v: AnyLinear },
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
}

fn truncate_kv_window(
    k: candle_core::Tensor,
    v: candle_core::Tensor,
    kv_len: usize,
    window: Option<usize>,
    num_tokens: usize,
) -> Result<(candle_core::Tensor, candle_core::Tensor, usize)> {
    if let Some(w) = window {
        if num_tokens == 1 && kv_len > w {
            return Ok((k.narrow(2, kv_len - w, w)?, v.narrow(2, kv_len - w, w)?, w));
        }
    }
    if kv_len < k.dim(2)? {
        Ok((k.narrow(2, 0, kv_len)?, v.narrow(2, 0, kv_len)?, kv_len))
    } else {
        Ok((k, v, kv_len))
    }
}

fn compute_sliding_window(cfg: &BlockConfig, layer_idx: usize) -> Option<usize> {
    if cfg.norm_type == NormType::Gemma && layer_idx % 2 == 1 {
        None
    } else {
        cfg.sliding_window
    }
}

impl Attention {
    pub fn load(cfg: &BlockConfig, layer_idx: usize, weights: &ModelWeights) -> Result<Self> {
        let p = format!("model.layers.{}.self_attn", layer_idx);
        let hd = cfg.head_dim;

        let q_w = weights.get(&format!("{}.q_proj.weight", p))?;
        let k_w = weights.get(&format!("{}.k_proj.weight", p))?;
        let v_w = weights.get(&format!("{}.v_proj.weight", p))?;
        let qkv_w = Tensor::cat(&[q_w, k_w, v_w], 0)?;
        let qkv_bias = match (
            weights.try_get(&format!("{}.q_proj.bias", p)),
            weights.try_get(&format!("{}.k_proj.bias", p)),
            weights.try_get(&format!("{}.v_proj.bias", p)),
        ) {
            (Some(qb), Some(kb), Some(vb)) => Some(Tensor::cat(&[qb, kb, vb], 0)?),
            _ => None,
        };
        let qkv_proj = Linear::new(qkv_w, qkv_bias);

        let o_proj = AnyLinear::Float(Linear::new(
            weights.get(&format!("{}.o_proj.weight", p))?.clone(),
            None,
        ));

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

        let q_dim = cfg.n_heads * hd;
        let kv_dim = cfg.n_kv_heads * hd;

        let actual_window = compute_sliding_window(cfg, layer_idx);

        Ok(Self {
            qkv: QkvProjection::Fused(qkv_proj),
            o_proj,
            q_norm, k_norm,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: hd,
            q_dim,
            kv_dim,
            scale: cfg.attention_scale.unwrap_or(1.0 / (hd as f64).sqrt()),
            attn_softcap: cfg.attn_softcap,
            sliding_window: actual_window,
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
        let q_proj = QLinear::from_arc(gguf.get(&format!("{prefix}.attn_q.weight"))?, dtype)?;
        let k_proj = QLinear::from_arc(gguf.get(&format!("{prefix}.attn_k.weight"))?, dtype)?;
        let v_proj = QLinear::from_arc(gguf.get(&format!("{prefix}.attn_v.weight"))?, dtype)?;
        let o_proj = QLinear::from_arc(gguf.get(&format!("{prefix}.attn_output.weight"))?, dtype)?;

        let q_norm = if cfg.qk_norm {
            let qt = gguf.get(&format!("{prefix}.attn_q_norm.weight"))?;
            Some(RMSNorm::from_qtensor(&qt, device, dtype, cfg.rms_norm_eps, cfg.norm_type)?)
        } else {
            None
        };
        let k_norm = if cfg.qk_norm {
            let qt = gguf.get(&format!("{prefix}.attn_k_norm.weight"))?;
            Some(RMSNorm::from_qtensor(&qt, device, dtype, cfg.rms_norm_eps, cfg.norm_type)?)
        } else {
            None
        };

        let q_dim = cfg.n_heads * hd;
        let kv_dim = cfg.n_kv_heads * hd;

        let actual_window = compute_sliding_window(cfg, layer_idx);

        Ok(Self {
            qkv: QkvProjection::Separate {
                q: AnyLinear::Quantized(q_proj),
                k: AnyLinear::Quantized(k_proj),
                v: AnyLinear::Quantized(v_proj),
            },
            o_proj: AnyLinear::Quantized(o_proj),
            q_norm, k_norm,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: hd,
            q_dim,
            kv_dim,
            scale: cfg.attention_scale.unwrap_or(1.0 / (hd as f64).sqrt()),
            attn_softcap: cfg.attn_softcap,
            sliding_window: actual_window,
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
        let q = q_raw.reshape((b, total_seq, self.n_heads, hd))?.transpose(1, 2)?;
        let k = k_raw.reshape((b, total_seq, self.n_kv_heads, hd))?.transpose(1, 2)?;
        let v = v_raw.reshape((b, total_seq, self.n_kv_heads, hd))?.transpose(1, 2)?;

        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };

        let (q, k) = if let Some((r, position_ids)) = rope {
            (r.apply_with_positions(&q, position_ids)?, r.apply_with_positions(&k, position_ids)?)
        } else {
            (q, k)
        };

        let mut k_parts: Vec<Tensor> = Vec::with_capacity(segments.len());
        let mut v_parts: Vec<Tensor> = Vec::with_capacity(segments.len());
        let mut kv_lengths: Vec<usize> = Vec::with_capacity(segments.len());
        let mut offset = 0usize;

        for seg in segments.iter_mut() {
            let seg_k = k.narrow(2, offset, seg.num_tokens)?;
            let seg_v = v.narrow(2, offset, seg.num_tokens)?;
            let (full_k, full_v) = seg.cache.append(&seg_k, &seg_v)?;
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
        let all_vector = segments.iter().all(|seg| seg.num_tokens <= 8);
        #[cfg(feature = "metal")]
        let use_sdpa = (self.attn_softcap.is_none() || all_vector)
            && super::metal_ops::sdpa_available(&q, self.head_dim)
            && !kv_lengths.iter().zip(segments.iter()).any(|(&kv_len, seg)| {
                seg.num_tokens > 1 && kv_len > seg.num_tokens
            });
        #[cfg(not(feature = "metal"))]
        let use_sdpa = false;

        #[cfg(feature = "metal")]
        if !use_sdpa && self.attn_softcap.is_none() && !super::metal_ops::sdpa_available(&q, self.head_dim) {
            log_sdpa_fallback_once(self.head_dim, q.dtype());
        }

        if use_sdpa {
            #[cfg(feature = "metal")]
            for (i, seg) in segments.iter().enumerate() {
                let q_seg = q.narrow(2, q_offset, seg.num_tokens)?;
                let (k_seg, v_seg, _) = truncate_kv_window(
                    k_parts[i].clone(), v_parts[i].clone(),
                    kv_lengths[i], self.sliding_window, seg.num_tokens,
                )?;

                // SDPA handles GQA natively — no repeat_kv needed.
                let q_c = q_seg.contiguous()?;
                let k_c = k_seg.contiguous()?;
                let v_c = v_seg.contiguous()?;

                let seg_out = super::metal_ops::sdpa(
                    &q_c,
                    &k_c,
                    &v_c,
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
                    k_parts[i].clone(), v_parts[i].clone(),
                    kv_lengths[i], self.sliding_window, seg.num_tokens,
                )?;
                let k_seg = self.repeat_kv(k_seg)?;
                let v_seg = self.repeat_kv(v_seg)?;

                let mut scores = q_seg.matmul(&k_seg.transpose(D::Minus1, D::Minus2)?)?.affine(self.scale, 0.)?;

                if let Some(softcap) = self.attn_softcap {
                    scores = (scores / softcap)?.tanh()?.affine(softcap, 0.)?;
                }

                let scores = if let Some(m) = mask {
                    let seg_mask = m.narrow(2, q_offset, seg.num_tokens)?.narrow(3, 0, kv_len)?;
                    scores.broadcast_add(&seg_mask.to_dtype(scores.dtype())?)?
                } else if seg.num_tokens > 1 {
                    let cm = super::mask::causal_mask_cached(seg.num_tokens, device)?;
                    if kv_len > seg.num_tokens {
                        let visible = Tensor::zeros(
                            (1, 1, seg.num_tokens, kv_len - seg.num_tokens),
                            candle_core::DType::F32,
                            device,
                        )?;
                        let full_mask = Tensor::cat(&[&visible, &cm], 3)?;
                        scores.broadcast_add(&full_mask.to_dtype(scores.dtype())?)?
                    } else {
                        scores.broadcast_add(&cm.to_dtype(scores.dtype())?)?
                    }
                } else {
                    scores
                };

                let attn = softmax_last_dim(&scores)?;
                let seg_out = attn.matmul(&v_seg)?.contiguous()?;
                out_buf.slice_set(&seg_out, 2, q_offset)?;
                q_offset += seg.num_tokens;
            }
        }

        let out = out_buf.transpose(1, 2)?.reshape((b, total_seq, self.n_heads * hd))?;
        self.o_proj.forward(&out)
    }
}
