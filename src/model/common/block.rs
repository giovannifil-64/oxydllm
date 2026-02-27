use candle_core::{Result, Tensor};
use super::{
    attention::Attention,
    config::BlockConfig,
    ffn::FeedForward,
    paged::PagedKvCache,
    norm::RMSNorm,
    rope::RotaryEmbedding,
    weights::ModelWeights,
};

pub struct TransformerBlock {
    input_norm: RMSNorm,
    attention: Attention,
    post_attn_norm: RMSNorm,
    ffn: FeedForward,
}

impl TransformerBlock {
    pub fn load(cfg: &BlockConfig, layer_idx: usize, weights: &ModelWeights) -> Result<Self> {
        let p = format!("model.layers.{}", layer_idx);
        let input_norm = RMSNorm::load(weights, &format!("{}.input_layernorm", p), cfg.rms_norm_eps)?;
        let post_attn_norm = RMSNorm::load(weights, &format!("{}.post_attention_layernorm", p), cfg.rms_norm_eps)?;
        let attention = Attention::load(cfg, layer_idx, weights)?;
        let ffn = FeedForward::load(cfg, layer_idx, weights)?;
        Ok(Self { input_norm, attention, post_attn_norm, ffn })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        start_pos: usize,
        mask: Option<&Tensor>,
        cache: &mut PagedKvCache,
    ) -> Result<Tensor> {
        let residual = x;
        let x = (residual + self.attention.forward(&self.input_norm.forward(x)?, rope, start_pos, mask, cache)?)?;
        let residual = &x;
        let x = (residual + self.ffn.forward(&self.post_attn_norm.forward(&x)?)?)?;
        Ok(x)
    }
}
