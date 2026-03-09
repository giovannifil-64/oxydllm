// ─────────────────────────────────────────────────────────────────────────────
// gguf_model.rs — StandardTransformer: unified model for all standard archs
// ─────────────────────────────────────────────────────────────────────────────
//
// `StandardTransformer` is the single concrete model struct used by every
// standard pre-norm transformer architecture (Llama, Qwen3, GGUF, and any
// future architecture that fits the same TransformerBlock pattern).
//
// • GGUF loading:       StandardTransformer::load_gguf(...)
// • Safetensors loading: loader::load_standard_safetensors(cfg, ...)  ← in loader.rs
//
// Adding a new standard architecture requires only:
//   1. Define its JSON config struct and implement From<Config> for StandardTransformerConfig.
//   2. Add one arm to the match in loader::load_batch_model.
// ─────────────────────────────────────────────────────────────────────────────

use candle_core::{DType, Device, Result, Tensor};
use std::sync::{Arc, Mutex};

use crate::common::{
    block::{TransformerBlock, TransformerComponents, run_transformer_layers_batch},
    config::BlockConfig,
    gguf_weights::GgufWeights,
    linear::{AnyLinear, Embedding, Linear, QLinear},
    norm::RMSNorm,
    paged::{BlockAllocator, PagedKvCache, SharedBlockAllocator, DEFAULT_BLOCK_SIZE},
    rope::RotaryEmbedding,
};
use crate::models::traits::BatchModel;

/// Single generic transformer model used by all standard architectures
/// (Llama, Qwen3, GGUF, and any future architecture that fits the standard
/// pre-norm TransformerBlock pattern).
pub struct StandardTransformer {
    pub(crate) embed_tokens: Embedding,
    pub(crate) blocks: Vec<TransformerBlock>,
    pub(crate) norm: RMSNorm,
    pub(crate) lm_head: AnyLinear,
    pub(crate) rope: RotaryEmbedding,
    pub(crate) allocators: Vec<SharedBlockAllocator>,
    pub(crate) device: Device,
    pub(crate) stop_token_ids: Vec<u32>,
    pub(crate) vocab_size: usize,
    pub(crate) max_seq_len: usize,
    pub(crate) embed_scale: Option<f64>,
    pub(crate) logit_softcap: Option<f64>,
}

pub(crate) struct GgufTopology {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub context_length: usize,
}

pub(crate) fn parse_gguf_topology(gguf: &GgufWeights) -> anyhow::Result<GgufTopology> {
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
    let context_length =
        gguf.metadata_u32_or(&format!("{prefix}.context_length"), 131072) as usize;
    Ok(GgufTopology { num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, context_length })
}

impl StandardTransformer {
    fn components(&self) -> TransformerComponents<'_> {
        TransformerComponents {
            embed_tokens: &self.embed_tokens,
            blocks: &self.blocks,
            norm: &self.norm,
            lm_head: &self.lm_head,
            rope: &self.rope,
            embed_scale: self.embed_scale,
            logit_softcap: self.logit_softcap,
        }
    }

    pub fn load_gguf(
        gguf: &GgufWeights,
        device: &Device,
        dtype: DType,
        num_kv_blocks: usize,
    ) -> anyhow::Result<Self> {
        let arch = gguf.architecture()?;
        let prefix = &arch;

        let arch_def = crate::models::arch_defaults::arch_defaults(&arch)
            .ok_or_else(|| anyhow::anyhow!("Architecture '{arch}' not supported"))?;
            
        let activation = arch_def.activation;
        let norm_type = arch_def.norm_type;
        let logit_softcap = arch_def.logit_softcap;
        let attn_softcap = arch_def.attn_softcap;
        let has_ffn_norms = arch_def.has_ffn_norms;
        let has_qk_norm = arch_def.qk_norm;
        
        let topo = parse_gguf_topology(gguf)?;
        let num_hidden_layers = topo.num_hidden_layers;
        let num_attention_heads = topo.num_attention_heads;
        let num_key_value_heads = topo.num_key_value_heads;
        let head_dim = topo.head_dim;

        let hidden_size = gguf.metadata_u32_or(&format!("{prefix}.embedding_length"), (head_dim * num_attention_heads) as u32) as usize;

        let mut embed_scale = None;
        if arch_def.embed_scale_from_hidden {
            embed_scale = Some((hidden_size as f64).sqrt());
        }

        let rms_norm_eps = gguf
            .metadata_f32_or(&format!("{prefix}.attention.layer_norm_rms_epsilon"), 1e-5)
            as f64;
        let rope_theta =
            gguf.metadata_f32_or(&format!("{prefix}.rope.freq_base"), arch_def.default_rope_theta as f32) as f64;
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

        // Read query_pre_attn_scalar from GGUF if present (e.g. Gemma2 27B uses 224, not head_dim)
        let attention_scale = {
            let scalar = gguf.metadata_f32_or(
                &format!("{prefix}.attention.query_pre_attn_scalar"), 0.0,
            ) as f64;
            if scalar > 0.0 { Some(1.0 / scalar.sqrt()) } else { None }
        };

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

        let sliding_window_meta = gguf.metadata_u32_or(&format!("{prefix}.attention.sliding_window"), 0) as usize;
        let sliding_window = if sliding_window_meta > 0 {
            Some(sliding_window_meta)
        } else {
            arch_def.default_sliding_window
        };

        let block_cfg = BlockConfig {
            n_heads: num_attention_heads,
            n_kv_heads: num_key_value_heads,
            head_dim,
            rms_norm_eps,
            qk_norm: has_qk_norm,
            attention_scale,
            activation,
            norm_type,
            attn_softcap,
            has_ffn_norms,
            sliding_window,
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
        let norm = RMSNorm::from_qtensor(&norm_qt, device, dtype, rms_norm_eps, norm_type)
            .map_err(|e| anyhow::anyhow!("Failed to load output_norm: {e}"))?;

        let rope =
            RotaryEmbedding::new(head_dim, max_position_embeddings, rope_theta, dtype, device)
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
            embed_scale,
            logit_softcap,
        })
    }
}

impl BatchModel for StandardTransformer {
    fn forward_batch(
        &self,
        token_ids: &Tensor,
        position_ids: &Tensor,
        seq_caches: &mut [&mut [PagedKvCache]],
        token_counts: &[usize],
    ) -> Result<Tensor> {
        run_transformer_layers_batch(self.components(), token_ids, position_ids, seq_caches, token_counts)
    }

    fn vocab_size(&self) -> usize { self.vocab_size }
    fn stop_token_ids(&self) -> &[u32] { &self.stop_token_ids }
    fn max_seq_len(&self) -> usize { self.max_seq_len }
    fn device(&self) -> &Device { &self.device }
    fn num_layers(&self) -> usize { self.blocks.len() }
    fn allocators(&self) -> &[SharedBlockAllocator] { &self.allocators }
}
