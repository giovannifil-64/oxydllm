// ─────────────────────────────────────────────────────────────────────────────
// gguf_model.rs — Architecture-agnostic GGUF model
// ─────────────────────────────────────────────────────────────────────────────
//
// A single generic transformer model that loads from *any* GGUF file.
// The GGUF format standardises tensor names (blk.{i}.attn_q.weight, etc.) and
// stores all configuration as typed metadata, so no per-architecture code is
// needed.
//
// When adding a new architecture to rllm, the safetensors path still needs a
// dedicated model file (because HF models use different tensor naming per
// architecture), but the GGUF path works out of the box.
// ─────────────────────────────────────────────────────────────────────────────

use candle_core::{DType, Device, Result, Tensor};
use std::sync::{Arc, Mutex};

use crate::common::{
    attention::SegmentInfo,
    block::TransformerBlock,
    config::BlockConfig,
    gguf_weights::GgufWeights,
    linear::{AnyLinear, Embedding, Linear, QLinear},
    mask::causal_mask_cached,
    norm::RMSNorm,
    paged::{BlockAllocator, PagedKvCache, SharedBlockAllocator, DEFAULT_BLOCK_SIZE},
    rope::RotaryEmbedding,
};
use crate::models::traits::BatchModel;

pub struct GgufModel {
    embed_tokens: Embedding,
    blocks: Vec<TransformerBlock>,
    norm: RMSNorm,
    lm_head: AnyLinear,
    rope: RotaryEmbedding,
    allocators: Vec<SharedBlockAllocator>,
    device: Device,
    stop_token_ids: Vec<u32>,
    vocab_size: usize,
    max_seq_len: usize,
    num_layers: usize,
}

impl GgufModel {
    pub fn load(
        gguf: &GgufWeights,
        device: &Device,
        dtype: DType,
        num_kv_blocks: usize,
    ) -> anyhow::Result<Self> {
        let arch = gguf.architecture()?;
        let prefix = &arch;

        let num_hidden_layers = gguf.metadata_u32(&format!("{prefix}.block_count"))? as usize;
        let num_attention_heads =
            gguf.metadata_u32(&format!("{prefix}.attention.head_count"))? as usize;
        let num_key_value_heads =
            gguf.metadata_u32(&format!("{prefix}.attention.head_count_kv"))? as usize;

        let head_dim = {
            let from_meta =
                gguf.metadata_u32_or(&format!("{prefix}.attention.key_length"), 0) as usize;
            if from_meta > 0 {
                from_meta
            } else {
                let q0 = gguf
                    .get("blk.0.attn_q.weight")
                    .map_err(|e| anyhow::anyhow!("Cannot determine head_dim: {e}"))?;
                q0.shape().dims()[0] / num_attention_heads
            }
        };

        let rms_norm_eps = gguf
            .metadata_f32_or(&format!("{prefix}.attention.layer_norm_rms_epsilon"), 1e-5)
            as f64;
        let rope_theta =
            gguf.metadata_f32_or(&format!("{prefix}.rope.freq_base"), 500_000.0) as f64;
        let max_position_embeddings =
            gguf.metadata_u32_or(&format!("{prefix}.context_length"), 131072) as usize;

        let intermediate_size = {
            let from_meta =
                gguf.metadata_u32_or(&format!("{prefix}.feed_forward_length"), 0) as usize;
            if from_meta > 0 {
                from_meta
            } else {
                let qt = gguf
                    .get("blk.0.ffn_gate.weight")
                    .map_err(|e| anyhow::anyhow!("Cannot determine intermediate_size: {e}"))?;
                qt.shape().dims()[0]
            }
        };

        let has_qk_norm = gguf.try_get("blk.0.attn_q_norm.weight").is_some();

        let embed_qt = gguf
            .get("token_embd.weight")
            .map_err(|e| anyhow::anyhow!("Missing token_embd.weight: {e}"))?;
        let vocab_size = embed_qt.shape().dims()[0];
        let embed_tokens = Embedding::from_qtensor(&embed_qt, device, dtype)?;

        let lm_head = match gguf.try_get("output.weight") {
            Some(qt) => AnyLinear::Quantized(
                QLinear::from_arc(qt, dtype)
                    .map_err(|e| anyhow::anyhow!("Failed to load output.weight: {e}"))?,
            ),
            None => {
                let w = embed_qt
                    .dequantize(device)
                    .map_err(|e| anyhow::anyhow!("dequantize embed for tie: {e}"))?
                    .to_dtype(dtype)
                    .map_err(|e| anyhow::anyhow!("dtype cast: {e}"))?;
                AnyLinear::Float(Linear::new(w, None))
            }
        };

        let block_cfg = BlockConfig {
            n_heads: num_attention_heads,
            n_kv_heads: num_key_value_heads,
            head_dim,
            rms_norm_eps,
            qk_norm: has_qk_norm,
            attention_scale: None,
        };

        let blocks = (0..num_hidden_layers)
            .map(|i| {
                TransformerBlock::load_gguf(&block_cfg, i, gguf, device, dtype, intermediate_size)
            })
            .collect::<Result<Vec<_>>>()
            .map_err(|e| anyhow::anyhow!("Failed to load transformer block: {e}"))?;

        let norm_qt = gguf
            .get("output_norm.weight")
            .map_err(|e| anyhow::anyhow!("Missing output_norm.weight: {e}"))?;
        let norm = RMSNorm::from_qtensor(&norm_qt, device, dtype, rms_norm_eps)
            .map_err(|e| anyhow::anyhow!("Failed to load output_norm: {e}"))?;

        let rope =
            RotaryEmbedding::new(head_dim, max_position_embeddings, rope_theta, device)
                .map_err(|e| anyhow::anyhow!("Failed to create RoPE: {e}"))?;

        let mut allocators = Vec::with_capacity(num_hidden_layers);
        for _ in 0..num_hidden_layers {
            let allocator = Arc::new(Mutex::new(
                BlockAllocator::new(
                    num_kv_blocks,
                    DEFAULT_BLOCK_SIZE,
                    num_key_value_heads,
                    head_dim,
                    dtype,
                    device,
                )
                .map_err(|e| anyhow::anyhow!("Failed to create block allocator: {e}"))?,
            ));
            allocators.push(allocator);
        }

        let stop_token_ids = gguf.eos_token_ids();

        Ok(Self {
            embed_tokens,
            blocks,
            norm,
            lm_head,
            rope,
            allocators,
            device: device.clone(),
            stop_token_ids,
            vocab_size,
            max_seq_len: max_position_embeddings,
            num_layers: num_hidden_layers,
        })
    }
}


impl GgufModel {
    fn forward_impl(
        &self,
        tokens: &Tensor,
        start_pos: usize,
        caches: &mut [PagedKvCache],
    ) -> Result<Tensor> {
        let (_b, seq) = tokens.dims2()?;
        let mut x = self.embed_tokens.forward(tokens)?;

        let mask = if seq > 1 {
            Some(causal_mask_cached(seq, tokens.device())?)
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


impl BatchModel for GgufModel {
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
    fn stop_token_ids(&self) -> &[u32] { &self.stop_token_ids }
    fn max_seq_len(&self) -> usize { self.max_seq_len }
    fn device(&self) -> &Device { &self.device }
    fn num_layers(&self) -> usize { self.num_layers }
    fn allocators(&self) -> &[SharedBlockAllocator] { &self.allocators }
}
