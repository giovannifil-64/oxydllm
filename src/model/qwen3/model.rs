use candle_core::{DType, Device, Result, Tensor};
use std::cell::RefCell;
use std::rc::Rc;
use crate::model::traits::{Model, BatchModel};
use crate::model::common::{
    block::TransformerBlock,
    config::BlockConfig,
    ffn::Activation,
    linear::{Embedding, Linear},
    mask::causal_mask,
    norm::RMSNorm,
    paged::{BlockAllocator, PagedKvCache, SharedBlockAllocator, DEFAULT_BLOCK_SIZE},
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
    caches: Vec<PagedKvCache>,
    allocators: Vec<SharedBlockAllocator>,
    device: Device,
    eos_token_id: u32,
    vocab_size: usize,
    max_seq_len: usize,
    num_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
}

impl Qwen3 {
    pub fn load(cfg: Qwen3Config, weights: &ModelWeights, device: &Device, dtype: DType, kv_block_multiplier: usize) -> Result<Self> {
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

        // Paged KV cache — one BlockAllocator per layer, shared across sequences.
        let num_blocks = kv_block_multiplier * ((cfg.max_position_embeddings + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE);
        let mut allocators = Vec::with_capacity(cfg.num_hidden_layers);
        let mut caches = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            let allocator = Rc::new(RefCell::new(
                BlockAllocator::new(
                    num_blocks,
                    DEFAULT_BLOCK_SIZE,
                    cfg.num_key_value_heads,
                    head_dim,
                    dtype,
                    device,
                )?
            ));
            caches.push(PagedKvCache::new(Rc::clone(&allocator)));
            allocators.push(allocator);
        }

        Ok(Self {
            embed_tokens,
            blocks,
            norm,
            lm_head,
            rope,
            caches,
            allocators,
            device: device.clone(),
            eos_token_id: 151645,
            vocab_size: cfg.vocab_size,
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim,
            dtype,
        })
    }
}

impl Qwen3 {
    fn forward_impl(
        &self,
        tokens: &Tensor,
        start_pos: usize,
        caches: &mut [PagedKvCache],
    ) -> Result<Tensor> {
        let (_b, seq) = tokens.dims2()?;
        let mut x = self.embed_tokens.forward(tokens)?;

        let mask = if seq > 1 {
            Some(causal_mask(seq, tokens.device())?)
        } else {
            None
        };

        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            x = block.forward(&x, &self.rope, start_pos, mask.as_ref(), cache)?;
        }

        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }
}

impl Model for Qwen3 {
    fn forward(&mut self, tokens: &Tensor, start_pos: usize) -> Result<Tensor> {
        // Split borrow: take caches out, run forward, put them back via pointer.
        // Safe because forward_impl only reads &self fields other than caches.
        let mut caches = std::mem::take(&mut self.caches);
        let result = self.forward_impl(tokens, start_pos, &mut caches);
        self.caches = caches;
        result
    }

    fn clear_cache(&mut self) {
        for cache in &mut self.caches {
            cache.clear();
        }
    }

    fn vocab_size(&self) -> usize { self.vocab_size }
    fn eos_token_id(&self) -> u32 { self.eos_token_id }
    fn max_seq_len(&self) -> usize { self.max_seq_len }
    fn device(&self) -> &Device { &self.device }
}

impl BatchModel for Qwen3 {
    fn forward_with_cache(
        &self,
        tokens: &Tensor,
        start_pos: usize,
        caches: &mut [PagedKvCache],
    ) -> Result<Tensor> {
        self.forward_impl(tokens, start_pos, caches)
    }

    fn vocab_size(&self) -> usize { self.vocab_size }
    fn eos_token_id(&self) -> u32 { self.eos_token_id }
    fn max_seq_len(&self) -> usize { self.max_seq_len }
    fn device(&self) -> &Device { &self.device }
    fn num_layers(&self) -> usize { self.num_layers }
    fn n_kv_heads(&self) -> usize { self.n_kv_heads }
    fn head_dim(&self) -> usize { self.head_dim }
    fn dtype(&self) -> DType { self.dtype }
    fn allocators(&self) -> &[SharedBlockAllocator] { &self.allocators }
}
