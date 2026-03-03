use candle_core::{Result, Tensor, D};
use super::config::BlockConfig;
use super::paged::PagedKvCache;
use super::linear::{softmax_last_dim, Linear};
use super::norm::RMSNorm;
use super::rope::RotaryEmbedding;
use super::weights::ModelWeights;

pub struct SegmentInfo<'a> {
    pub num_tokens: usize,
    pub cache: &'a mut PagedKvCache,
}

pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
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

        let q_proj = Linear::new(weights.get(&format!("{}.q_proj.weight", p))?.clone(), None);
        let k_proj = Linear::new(weights.get(&format!("{}.k_proj.weight", p))?.clone(), None);
        let v_proj = Linear::new(weights.get(&format!("{}.v_proj.weight", p))?.clone(), None);
        let o_proj = Linear::new(weights.get(&format!("{}.o_proj.weight", p))?.clone(), None);

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

        let qkv_w = Tensor::cat(&[
            q_proj.weight(),
            k_proj.weight(),
            v_proj.weight(),
        ], 0)?;
        let qkv_proj = Some(Linear::new(qkv_w, None));

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            qkv_proj,
            q_norm, k_norm,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: hd,
            q_dim,
            kv_dim,
            scale: 1.0 / (hd as f64).sqrt(),
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
            Ok((self.q_proj.forward(x)?, self.k_proj.forward(x)?, self.v_proj.forward(x)?))
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

        let q = rope.apply(&q, start_pos)?;
        let k = rope.apply(&k, start_pos)?;

        let (k, v) = cache.append(&k, &v)?;

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

        let q = rope.apply_with_positions(&q, position_ids)?;
        let k = rope.apply_with_positions(&k, position_ids)?;

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

        let k_cat = self.repeat_kv(k_cat)?;
        let v_cat = self.repeat_kv(v_cat)?;

        let mut out_parts: Vec<Tensor> = Vec::new();
        let mut q_offset = 0usize;

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
                let cm = super::mask::causal_mask(seg.num_tokens, device)?;
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

        let out = Tensor::cat(&out_parts, 2)?;
        let out = out.transpose(1, 2)?.reshape((b, total_seq, self.n_heads * hd))?;
        self.o_proj.forward(&out)
    }
}