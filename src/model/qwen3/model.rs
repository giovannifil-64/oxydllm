use candle_core::{Device, Result, Tensor};
use crate::model::traits::Model;
use crate::model::common::{
    block::TransformerBlock,
    config::BlockConfig,
    ffn::Activation,
    linear::{Embedding, Linear},
    mask::causal_mask,
    norm::RMSNorm,
    rope::RotaryEmbedding,
    weights::ModelWeights,
};
use super::config::Qwen3Config;

pub struct Qwen3 {
    embed_tokens: Embedding,
    blocks: Vec<TransformerBlock>,
    norm: RMSNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    device: Device,
    eos_token_id: u32,
    vocab_size: usize,
    max_seq_len: usize,
}

impl Qwen3 {
    pub fn load(cfg: Qwen3Config, weights: &ModelWeights, device: &Device) -> Result<Self> {
        let embed_tokens = Embedding::new(weights.get("model.embed_tokens.weight")?.clone());

        let head_dim = cfg.head_dim();

        let block_cfg = BlockConfig {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            qk_norm: true,
            sliding_window: None,
            activation: Activation::Silu,
        };

        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| TransformerBlock::load(&block_cfg, i, weights))
            .collect::<Result<Vec<_>>>()?;

        let norm = RMSNorm::load(weights, "model.norm", cfg.rms_norm_eps)?;
        let lm_head = Linear::new(weights.get("lm_head.weight")?.clone(), None);

        let rope = RotaryEmbedding::new(head_dim, cfg.max_position_embeddings, cfg.rope_theta, device)?;

        Ok(Self {
            embed_tokens,
            blocks,
            norm,
            lm_head,
            rope,
            device: device.clone(),
            eos_token_id: 151645,
            vocab_size: cfg.vocab_size,
            max_seq_len: cfg.max_position_embeddings,
        })
    }
}

impl Model for Qwen3 {
    fn forward(&self, tokens: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_b, seq) = tokens.dims2()?;
        let mut x = self.embed_tokens.forward(tokens)?;

        let mask = if seq > 1 {
            Some(causal_mask(seq, tokens.device())?)
        } else {
            None
        };

        for block in &self.blocks {
            x = block.forward(&x, &self.rope, start_pos, mask.as_ref())?;
        }

        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    fn vocab_size(&self) -> usize { self.vocab_size }
    fn eos_token_id(&self) -> u32 { self.eos_token_id }
    fn max_seq_len(&self) -> usize { self.max_seq_len }
    fn device(&self) -> &Device { &self.device }
}
