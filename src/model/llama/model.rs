use candle_core::{DType, Device, Result, Tensor};
use std::cell::RefCell;
use std::rc::Rc;
use crate::model::traits::{BatchModel, Model};
use crate::model::common::{
    attention::SegmentInfo,
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
use super::config::LlamaConfig;

pub struct Llama {
    embed_tokens: Embedding,
    blocks: Vec<TransformerBlock>,
    norm: RMSNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    caches: Vec<PagedKvCache>,
    allocators: Vec<SharedBlockAllocator>,
    device: Device,
    eos_token_id: u32,
    stop_token_ids: Vec<u32>,
    vocab_size: usize,
    max_seq_len: usize,
    num_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
}

impl Llama {
    pub fn load(
        cfg: LlamaConfig,
        weights: &ModelWeights,
        device: &Device,
        dtype: DType,
        kv_block_multiplier: usize,
    ) -> Result<Self> {
        let mut stop_token_ids = cfg.eos_token_ids.clone();
        if !stop_token_ids.contains(&128009) {
            stop_token_ids.push(128009); // <|eot_id|>
        }
        if !stop_token_ids.contains(&128008) {
            stop_token_ids.push(128008); // <|eom_id|>
        }

        let head_dim = cfg.head_dim();

        // When tie_word_embeddings is true, lm_head reuses the embedding table.
        let embed_weight = weights.get("model.embed_tokens.weight")?.clone();
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_weight.clone(), None)
        } else {
            Linear::new(weights.get("lm_head.weight")?.clone(), None)
        };
        let embed_tokens = Embedding::new(embed_weight);

        let block_cfg = BlockConfig {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            // Llama does not apply QK-norm.
            qk_norm: false,
            sliding_window: None,
            activation: Activation::Silu,
        };

        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| TransformerBlock::load(&block_cfg, i, weights))
            .collect::<Result<Vec<_>>>()?;

        let norm = RMSNorm::load(weights, "model.norm", cfg.rms_norm_eps)?;

        let rope = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            device,
        )?;

        let num_blocks = kv_block_multiplier
            * ((cfg.max_position_embeddings + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE);
        let mut allocators = Vec::with_capacity(cfg.num_hidden_layers);
        let mut caches = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            let allocator = Rc::new(RefCell::new(BlockAllocator::new(
                num_blocks,
                DEFAULT_BLOCK_SIZE,
                cfg.num_key_value_heads,
                head_dim,
                dtype,
                device,
            )?));
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
            eos_token_id: cfg.primary_eos_token_id(),
            stop_token_ids,
            vocab_size: cfg.vocab_size,
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim,
            dtype,
        })
    }
}

impl Llama {
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

    fn forward_batch_impl(
        &self,
        token_ids: &Tensor,
        position_ids: &Tensor,
        seq_caches: &mut [&mut [PagedKvCache]],
        token_counts: &[usize],
    ) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(token_ids)?;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let mut segments: Vec<SegmentInfo> = Vec::with_capacity(seq_caches.len());
            for (seq_idx, seq_cache) in seq_caches.iter_mut().enumerate() {
                segments.push(SegmentInfo {
                    num_tokens: token_counts[seq_idx],
                    cache: &mut seq_cache[layer_idx],
                });
            }
            x = block.forward_batch(&x, &self.rope, position_ids, None, &mut segments)?;
        }

        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }
}

impl Model for Llama {
    fn forward(&mut self, tokens: &Tensor, start_pos: usize) -> Result<Tensor> {
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

impl BatchModel for Llama {
    fn forward_with_cache(
        &self,
        tokens: &Tensor,
        start_pos: usize,
        caches: &mut [PagedKvCache],
    ) -> Result<Tensor> {
        self.forward_impl(tokens, start_pos, caches)
    }

    fn forward_batch(
        &self,
        token_ids: &Tensor,
        position_ids: &Tensor,
        seq_caches: &mut [&mut [PagedKvCache]],
        token_counts: &[usize],
    ) -> Result<Tensor> {
        self.forward_batch_impl(token_ids, position_ids, seq_caches, token_counts)
    }

    fn vocab_size(&self) -> usize { self.vocab_size }
    fn eos_token_id(&self) -> u32 { self.eos_token_id }
    fn stop_token_ids(&self) -> &[u32] { &self.stop_token_ids }
    fn max_seq_len(&self) -> usize { self.max_seq_len }
    fn device(&self) -> &Device { &self.device }
    fn num_layers(&self) -> usize { self.num_layers }
    fn n_kv_heads(&self) -> usize { self.n_kv_heads }
    fn head_dim(&self) -> usize { self.head_dim }
    fn dtype(&self) -> DType { self.dtype }
    fn allocators(&self) -> &[SharedBlockAllocator] { &self.allocators }
}
