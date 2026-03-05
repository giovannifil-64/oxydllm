use candle_core::{Result, Tensor, D};
use std::cell::Cell;
use super::config::BlockConfig;
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

pub struct Attention {
    q_proj: Option<AnyLinear>,
    k_proj: Option<AnyLinear>,
    v_proj: Option<AnyLinear>,
    o_proj: AnyLinear,
    qkv_proj: Option<Linear>,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn load(cfg: &BlockConfig, layer_idx: usize, weights: &ModelWeights) -> Result<Self> {
        let p = format!("model.layers.{}.self_attn", layer_idx);
        let hd = cfg.head_dim;

        let q_w = weights.get(&format!("{}.q_proj.weight", p))?;
        let k_w = weights.get(&format!("{}.k_proj.weight", p))?;
        let v_w = weights.get(&format!("{}.v_proj.weight", p))?;
        let qkv_w = Tensor::cat(&[q_w, k_w, v_w], 0)?;
        let qkv_proj = Some(Linear::new(qkv_w, None));

        let o_proj = AnyLinear::Float(Linear::new(
            weights.get(&format!("{}.o_proj.weight", p))?.clone(),
            None,
        ));

        let q_norm = cfg.qk_norm.then(|| {
            RMSNorm::new(
                weights.get(&format!("{}.q_norm.weight", p)).expect("q_norm.weight").clone(),
                cfg.rms_norm_eps,
            )
        });
        let k_norm = cfg.qk_norm.then(|| {
            RMSNorm::new(
                weights.get(&format!("{}.k_norm.weight", p)).expect("k_norm.weight").clone(),
                cfg.rms_norm_eps,
            )
        });

        let q_dim = cfg.n_heads * hd;
        let kv_dim = cfg.n_kv_heads * hd;

        Ok(Self {
            q_proj: None,
            k_proj: None,
            v_proj: None,
            o_proj,
            qkv_proj,
            q_norm, k_norm,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: hd,
            q_dim,
            kv_dim,
            scale: cfg.attention_scale.unwrap_or(1.0 / (hd as f64).sqrt()),
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
            Some(RMSNorm::from_qtensor(&qt, device, dtype, cfg.rms_norm_eps)?)
        } else {
            None
        };
        let k_norm = if cfg.qk_norm {
            let qt = gguf.get(&format!("{prefix}.attn_k_norm.weight"))?;
            Some(RMSNorm::from_qtensor(&qt, device, dtype, cfg.rms_norm_eps)?)
        } else {
            None
        };

        let q_dim = cfg.n_heads * hd;
        let kv_dim = cfg.n_kv_heads * hd;

        Ok(Self {
            q_proj: Some(AnyLinear::Quantized(q_proj)),
            k_proj: Some(AnyLinear::Quantized(k_proj)),
            v_proj: Some(AnyLinear::Quantized(v_proj)),
            o_proj: AnyLinear::Quantized(o_proj),
            qkv_proj: None,
            q_norm, k_norm,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: hd,
            q_dim,
            kv_dim,
            scale: cfg.attention_scale.unwrap_or(1.0 / (hd as f64).sqrt()),
        })
    }

    fn qkv_split(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        if let Some(ref qkv) = self.qkv_proj {
            let out = qkv.forward(x)?; // [..., q_dim + 2*kv_dim]
            let q = out.narrow(D::Minus1, 0, self.q_dim)?;
            let k = out.narrow(D::Minus1, self.q_dim, self.kv_dim)?;
            let v = out.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?;
            Ok((q, k, v))
        } else {
            let q = self.q_proj.as_ref().expect("q_proj missing").forward(x)?;
            let k = self.k_proj.as_ref().expect("k_proj missing").forward(x)?;
            let v = self.v_proj.as_ref().expect("v_proj missing").forward(x)?;
            Ok((q, k, v))
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

    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        start_pos: usize,
        mask: Option<&Tensor>,
        cache: &mut PagedKvCache,
    ) -> Result<Tensor> {
        self.forward_optional_rope(x, Some((rope, start_pos)), mask, cache)
    }

    pub fn forward_optional_rope(
        &self,
        x: &Tensor,
        rope: Option<(&RotaryEmbedding, usize)>,
        mask: Option<&Tensor>,
        cache: &mut PagedKvCache,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let hd = self.head_dim;

        let (q_raw, k_raw, v_raw) = self.qkv_split(x)?;
        let q = q_raw.reshape((b, seq, self.n_heads, hd))?.transpose(1, 2)?;
        let k = k_raw.reshape((b, seq, self.n_kv_heads, hd))?.transpose(1, 2)?;
        let v = v_raw.reshape((b, seq, self.n_kv_heads, hd))?.transpose(1, 2)?;

        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };

        let (q, k) = if let Some((r, start_pos)) = rope {
            (r.apply(&q, start_pos)?, r.apply(&k, start_pos)?)
        } else {
            (q, k)
        };

        let (k, v) = cache.append(&k, &v)?;

        // ── Metal SDPA path ─────────────────────────────────────────
        // Use candle's built-in fused SDPA Metal kernels (derived from MLX).
        // Handles GQA natively — no repeat_kv needed.
        #[cfg(feature = "metal")]
        {
            if super::metal_ops::sdpa_available(&q, self.head_dim) {
                let q_c = q.contiguous()?;
                let k_c = k.contiguous()?;
                let v_c = v.contiguous()?;

                let out = super::metal_ops::sdpa(
                    &q_c,
                    &k_c,
                    &v_c,
                    None,      // mask — SDPA supports causal flag directly
                    seq > 1,   // do_causal: true for prefill, false for decode
                    self.scale as f32,
                )?;

                let out = out
                    .transpose(1, 2)?
                    .reshape((b, seq, self.n_heads * hd))?;
                return self.o_proj.forward(&out);
            } else {
                log_sdpa_fallback_once(self.head_dim, q.dtype());
            }
        }

        // ── Fallback: standard attention (CPU / CUDA / unsupported head dims)
        let n_rep = self.n_heads / self.n_kv_heads;

        if n_rep > 1 && seq == 1 {
            let q_g = q.reshape((b, self.n_kv_heads, n_rep, hd))?.affine(self.scale, 0.0)?;
            let scores = q_g.matmul(&k.contiguous()?.transpose(D::Minus1, D::Minus2)?)?;

            let scores = match mask {
                Some(m) => scores.broadcast_add(&m.to_dtype(scores.dtype())?)?,
                None => scores,
            };

            let attn = softmax_last_dim(&scores)?;
            let out = attn.matmul(&v.contiguous()?)?;

            let out = out
                .reshape((b, self.n_heads, 1, hd))?
                .transpose(1, 2)?
                .reshape((b, 1, self.n_heads * hd))?;
            return self.o_proj.forward(&out);
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let scores = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?.affine(self.scale, 0.)?;

        let scores = match mask {
            Some(m) => scores.broadcast_add(&m.to_dtype(scores.dtype())?)?,
            None => scores,
        };

        let attn = softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, seq, self.n_heads * hd))?;
        self.o_proj.forward(&out)
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
        let mut max_kv = 0usize;
        let mut kv_lengths: Vec<usize> = Vec::with_capacity(segments.len());
        let mut offset = 0usize;

        for seg in segments.iter_mut() {
            let seg_k = k.narrow(2, offset, seg.num_tokens)?;
            let seg_v = v.narrow(2, offset, seg.num_tokens)?;
            let (full_k, full_v) = seg.cache.append(&seg_k, &seg_v)?;
            let kv_len = full_k.dim(2)?;
            max_kv = max_kv.max(kv_len);
            kv_lengths.push(kv_len);
            k_parts.push(full_k);
            v_parts.push(full_v);
            offset += seg.num_tokens;
        }

        let device = x.device();
        let dtype = x.dtype();
        let mut padded_k_parts: Vec<Tensor> = Vec::new();
        let mut padded_v_parts: Vec<Tensor> = Vec::new();

        for (i, (kp, vp)) in k_parts.iter().zip(v_parts.iter()).enumerate() {
            let kv_len = kv_lengths[i];
            if kv_len < max_kv {
                let pad_len = max_kv - kv_len;
                let pad = Tensor::zeros((1, self.n_kv_heads, pad_len, hd), dtype, device)?;
                padded_k_parts.push(Tensor::cat(&[kp, &pad], 2)?);
                padded_v_parts.push(Tensor::cat(&[vp, &pad], 2)?);
            } else {
                padded_k_parts.push(kp.clone());
                padded_v_parts.push(vp.clone());
            }
        }

        let k_cat = Tensor::cat(&padded_k_parts, 0)?;
        let v_cat = Tensor::cat(&padded_v_parts, 0)?;

        let mut out_parts: Vec<Tensor> = Vec::new();
        let mut q_offset = 0usize;

        // ── Metal SDPA path for batch ────────────────────────────────
        #[cfg(feature = "metal")]
        let use_sdpa = super::metal_ops::sdpa_available(&q, self.head_dim);
        #[cfg(not(feature = "metal"))]
        let use_sdpa = false;

        #[cfg(feature = "metal")]
        if !use_sdpa {
            log_sdpa_fallback_once(self.head_dim, q.dtype());
        }

        if use_sdpa {
            #[cfg(feature = "metal")]
            for (i, seg) in segments.iter().enumerate() {
                let q_seg = q.narrow(2, q_offset, seg.num_tokens)?;
                let k_seg = k_cat.narrow(0, i, 1)?;
                let v_seg = v_cat.narrow(0, i, 1)?;
                let kv_len = kv_lengths[i];
                let k_seg = k_seg.narrow(2, 0, kv_len)?;
                let v_seg = v_seg.narrow(2, 0, kv_len)?;

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
                )?;
                out_parts.push(seg_out);
                q_offset += seg.num_tokens;
            }
        } else {
            // ── Fallback: standard attention ─────────────────────────
            let k_cat = self.repeat_kv(k_cat)?;
            let v_cat = self.repeat_kv(v_cat)?;

            for (i, seg) in segments.iter().enumerate() {
                let q_seg = q.narrow(2, q_offset, seg.num_tokens)?;
                let k_seg = k_cat.narrow(0, i, 1)?;
                let v_seg = v_cat.narrow(0, i, 1)?;
                let kv_len = kv_lengths[i];
                let k_seg = k_seg.narrow(2, 0, kv_len)?;
                let v_seg = v_seg.narrow(2, 0, kv_len)?;

                let scores = q_seg.matmul(&k_seg.transpose(D::Minus1, D::Minus2)?)?.affine(self.scale, 0.)?;

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
                let seg_out = attn.matmul(&v_seg)?;
                out_parts.push(seg_out);
                q_offset += seg.num_tokens;
            }
        }

        let out = Tensor::cat(&out_parts, 2)?;
        let out = out.transpose(1, 2)?.reshape((b, total_seq, self.n_heads * hd))?;
        self.o_proj.forward(&out)
    }
}
