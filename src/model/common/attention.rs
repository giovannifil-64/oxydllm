use candle_core::{Result, Tensor, D};
use super::config::BlockConfig;
use super::kv_cache::KvCache;
use super::linear::{softmax_last_dim, Linear};
use super::norm::RMSNorm;
use super::rope::RotaryEmbedding;
use super::weights::ModelWeights;

pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
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

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            q_norm, k_norm,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: hd,
            scale: 1.0 / (hd as f64).sqrt(),
        })
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
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let hd = self.head_dim;

        let q = self.q_proj.forward(x)?.reshape((b, seq, self.n_heads, hd))?.transpose(1, 2)?;
        let k = self.k_proj.forward(x)?.reshape((b, seq, self.n_kv_heads, hd))?.transpose(1, 2)?;
        let v = self.v_proj.forward(x)?.reshape((b, seq, self.n_kv_heads, hd))?.transpose(1, 2)?;

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

        // Append nuovi K,V alla cache e ottieni la sequenza completa.
        let (k, v) = cache.append(&k, &v)?;

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
}