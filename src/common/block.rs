use candle_core::Result;
use candle_core::Tensor;
use super::{
    attention::{Attention, SegmentInfo},
    config::BlockConfig,
    ffn::FeedForward,
    gguf_weights::GgufWeights,
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
        let ffn = FeedForward::load(layer_idx, weights)?;
        Ok(Self { input_norm, attention, post_attn_norm, ffn })
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
        let input_norm = RMSNorm::from_qtensor(&attn_norm_qt, device, dtype, cfg.rms_norm_eps)?;

        let ffn_norm_qt = gguf.get(&format!("{prefix}.ffn_norm.weight"))?;
        let post_attn_norm = RMSNorm::from_qtensor(&ffn_norm_qt, device, dtype, cfg.rms_norm_eps)?;

        let attention = Attention::load_gguf(cfg, layer_idx, gguf, device, dtype)?;
        let ffn = FeedForward::load_gguf(layer_idx, gguf, intermediate_size, device, dtype)?;

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
        let attn_out = self.attention.forward_batch(&normed, rope, position_ids, mask, segments)?;
        let x = (residual + attn_out)?;
        let residual = &x;
        let x = (residual + self.ffn.forward(&self.post_attn_norm.forward(&x)?)?)?;
        Ok(x)
    }
}
