use candle_core::Result;
use candle_core::Tensor;
use super::{
    attention::{Attention, SegmentInfo},
    config::BlockConfig,
    ffn::FeedForward,
    gguf_weights::GgufWeights,
    linear::{AnyLinear, Embedding},
    norm::RMSNorm,
    paged::PagedKvCache,
    rope::RotaryEmbedding,
    weights::ModelWeights,
};

pub struct TransformerBlock {
    input_norm: RMSNorm,
    attention: Attention,
    ffn_norm: RMSNorm,
    ffn: FeedForward,
    pre_ffn_norm: Option<RMSNorm>,
    post_ffn_norm: Option<RMSNorm>,
}

impl TransformerBlock {
    pub fn load(cfg: &BlockConfig, layer_idx: usize, weights: &ModelWeights) -> Result<Self> {
        let p = format!("model.layers.{}", layer_idx);
        let input_norm = RMSNorm::load(weights, &format!("{}.input_layernorm", p), cfg.rms_norm_eps, cfg.norm_type)?;
        let ffn_norm = RMSNorm::load(weights, &format!("{}.post_attention_layernorm", p), cfg.rms_norm_eps, cfg.norm_type)?;
        let attention = Attention::load(cfg, layer_idx, weights)?;
        let ffn = FeedForward::load(layer_idx, weights, cfg.activation)?;
        
        let mut pre_ffn_norm = None;
        let mut post_ffn_norm = None;
        if cfg.has_ffn_norms {
            pre_ffn_norm = Some(RMSNorm::load(weights, &format!("{}.pre_feedforward_layernorm", p), cfg.rms_norm_eps, cfg.norm_type)?);
            post_ffn_norm = Some(RMSNorm::load(weights, &format!("{}.post_feedforward_layernorm", p), cfg.rms_norm_eps, cfg.norm_type)?);
        }
        
        Ok(Self { input_norm, attention, ffn_norm, ffn, pre_ffn_norm, post_ffn_norm })
    }

    pub fn load_gguf(
        cfg: &BlockConfig,
        layer_idx: usize,
        gguf: &GgufWeights,
        device: &candle_core::Device,
        dtype: candle_core::DType,
        intermediate_size: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{}", layer_idx);

        let attn_norm_qt = gguf.get(&format!("{prefix}.attn_norm.weight"))?;
        let input_norm = RMSNorm::from_qtensor(&attn_norm_qt, device, dtype, cfg.rms_norm_eps, cfg.norm_type)?;

        let ffn_norm_qt = gguf.get(&format!("{prefix}.ffn_norm.weight"))?;
        let ffn_norm = RMSNorm::from_qtensor(&ffn_norm_qt, device, dtype, cfg.rms_norm_eps, cfg.norm_type)?;

        let attention = Attention::load_gguf(cfg, layer_idx, gguf, device, dtype)?;
        let ffn = FeedForward::load_gguf(layer_idx, gguf, intermediate_size, device, dtype, cfg.activation)?;

        Ok(Self { input_norm, attention, ffn_norm, ffn, pre_ffn_norm: None, post_ffn_norm: None })
    }

    pub fn forward_batch(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        position_ids: &Tensor,
        mask: Option<&Tensor>,
        segments: &mut [SegmentInfo],
    ) -> Result<Tensor> {
        let residual = x;
        let normed = self.input_norm.forward(x)?;
        let mut attn_out = self.attention.forward_batch(&normed, rope, position_ids, mask, segments)?;
        
        if self.pre_ffn_norm.is_some() {
            attn_out = self.ffn_norm.forward(&attn_out)?;
        }

        let mut x = (residual + attn_out)?;
        let residual = x.clone();

        let ffn_inp;
        if let Some(pre_norm) = &self.pre_ffn_norm {
            ffn_inp = pre_norm.forward(&x)?;
        } else {
            ffn_inp = self.ffn_norm.forward(&x)?;
        }
        
        let mut ffn_out = self.ffn.forward(&ffn_inp)?;
        if let Some(post_norm) = &self.post_ffn_norm {
            ffn_out = post_norm.forward(&ffn_out)?;
        }
        
        x = (residual + ffn_out)?;
        Ok(x)
    }
}

/// Static model components shared by standard transformer architectures (Llama, Qwen3, …).
pub struct TransformerComponents<'a> {
    pub embed_tokens: &'a Embedding,
    pub blocks: &'a [TransformerBlock],
    pub norm: &'a RMSNorm,
    pub lm_head: &'a AnyLinear,
    pub rope: &'a RotaryEmbedding,
    pub embed_scale: Option<f64>,
    pub logit_softcap: Option<f64>,
}

/// Shared batched forward pass for standard transformer models.
pub fn run_transformer_layers_batch(
    c: TransformerComponents<'_>,
    token_ids: &Tensor,
    position_ids: &Tensor,
    seq_caches: &mut [&mut [PagedKvCache]],
    token_counts: &[usize],
) -> Result<Tensor> {
    debug_assert_eq!(
        token_counts.len(), seq_caches.len(),
        "token_counts.len() must equal seq_caches.len()"
    );
    debug_assert_eq!(
        token_counts.iter().sum::<usize>(),
        token_ids.dim(candle_core::D::Minus1).unwrap_or(0),
        "sum(token_counts) must equal token_ids sequence length"
    );
    for (i, seq_cache) in seq_caches.iter().enumerate() {
        debug_assert_eq!(
            seq_cache.len(), c.blocks.len(),
            "seq_caches[{i}].len() must equal number of transformer blocks"
        );
    }

    let mut x = c.embed_tokens.forward(token_ids)?;
    if let Some(scale) = c.embed_scale {
        x = (x * scale)?;
    }

    for (layer_idx, block) in c.blocks.iter().enumerate() {
        let mut segments: Vec<SegmentInfo> = Vec::with_capacity(seq_caches.len());
        for (seq_idx, seq_cache) in seq_caches.iter_mut().enumerate() {
            segments.push(SegmentInfo {
                num_tokens: token_counts[seq_idx],
                cache: &mut seq_cache[layer_idx],
            });
        }
        x = block.forward_batch(&x, c.rope, position_ids, None, &mut segments)?;
    }

    let x = c.norm.forward(&x)?;
    let logits = c.lm_head.forward(&x)?;
    
    if let Some(cap) = c.logit_softcap {
        let cap_t = cap as f64;
        (logits / cap_t)?.tanh()?.affine(cap_t, 0.0)
    } else {
        Ok(logits)
    }
}
