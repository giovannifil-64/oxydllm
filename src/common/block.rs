//! The transformer layer and the top-level batched forward pass.
//!
//! A [`TransformerBlock`] is one pre-norm decoder layer: a token mixer
//! (attention or Gated DeltaNet) and a feed-forward sub-layer (dense or MoE),
//! each wrapped in RMSNorm and a residual connection. The same struct covers
//! every supported architecture; which optional sub-components exist is decided
//! by the [`BlockConfig`] at load time.
//!
//! [`run_transformer_layers_batch`] is the architecture-agnostic forward pass:
//! given the static [`TransformerComponents`] and a batch of tokens, it runs
//! embeddings, every block in turn, the final norm and the lm-head, and returns
//! logits. It is the function the GGUF and safetensors runtimes both call.

use super::decode_profile;
use super::{
    attention::{Attention, SegmentInfo},
    config::{Activation, BlockConfig},
    ffn::FeedForward,
    gdn::GatedDeltaNet,
    gguf_weights::GgufWeights,
    linear::{AnyLinear, Embedding},
    moe::MoeFeedForward,
    norm::RMSNorm,
    paged::PagedKvCache,
    rope::RotaryEmbedding,
    weights::ModelWeights,
};
use candle_core::DType;
use candle_core::Result;
use candle_core::Tensor;

/// Dense or MoE FFN sub-layer. Both variants share the same `forward(x) -> x`
/// signature so the block can dispatch uniformly. `Moe` is only constructed
/// when [`BlockConfig::moe`] is `Some`.
enum FeedForwardLayer {
    Dense(FeedForward),
    Moe(MoeFeedForward),
}

impl FeedForwardLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(f) => f.forward(x),
            Self::Moe(f) => f.forward(x),
        }
    }
}

/// Token mixer of a layer: softmax attention or, on the linear-attention
/// layers of hybrid models (Qwen3.5), a Gated DeltaNet with per-sequence
/// recurrent state instead of a KV cache.
enum TokenMixer {
    Attention(Attention),
    Gdn(GatedDeltaNet),
}

impl TokenMixer {
    fn forward_batch(
        &self,
        normed: &Tensor,
        rope: &RotaryEmbedding,
        position_ids: &Tensor,
        mask: Option<&Tensor>,
        segments: &mut [SegmentInfo],
    ) -> Result<Tensor> {
        match self {
            Self::Attention(attn) => attn.forward_batch(normed, rope, position_ids, mask, segments),
            Self::Gdn(gdn) => {
                let token_counts: Vec<usize> = segments.iter().map(|s| s.num_tokens).collect();
                let mut states: Vec<&mut Option<crate::common::paged::RecurrentState>> = segments
                    .iter_mut()
                    .map(|s| s.cache.recurrent_mut())
                    .collect();
                gdn.forward_batch(normed, &token_counts, &mut states)
            }
        }
    }
}

/// One pre-norm transformer decoder layer.
///
/// The fixed backbone is two residual sub-layers:
///
/// 1. `x = x + mix(input_norm(x))`: the token mixer ([`TokenMixer`]:
///    attention or Gated DeltaNet).
/// 2. `x = x + ffn(ffn_norm(x))`: the feed-forward sub-layer
///    ([`FeedForwardLayer`]: dense or MoE).
///
/// Everything else is optional and driven by the [`BlockConfig`]:
///
/// - **Gemma "sandwich" norms** ([`BlockConfig::has_ffn_norms`]): when present,
///   `pre_ffn_norm` and `post_ffn_norm` wrap the FFN, and `ffn_norm` is reused
///   as a *post-attention* norm applied to the mixer output before the residual
///   add. So the same `ffn_norm` field sits in different places depending on
///   this flag: see [`forward_batch`](Self::forward_batch).
/// - **Per-layer input** (Gemma 3n): `per_layer_input_gate`,
///   `per_layer_projection` and `post_per_layer_input_norm`, when all present,
///   add a third gated residual that mixes in this layer's per-layer embedding.
/// - **`layer_scalar`**: a final scalar multiply on the block output.
///
/// Build one with [`load`](Self::load) (safetensors) or
/// [`load_gguf`](Self::load_gguf).
pub struct TransformerBlock {
    input_norm: RMSNorm,
    attention: TokenMixer,
    ffn_norm: RMSNorm,
    ffn: FeedForwardLayer,
    pre_ffn_norm: Option<RMSNorm>,
    post_ffn_norm: Option<RMSNorm>,
    per_layer_input_gate: Option<AnyLinear>,
    per_layer_projection: Option<AnyLinear>,
    post_per_layer_input_norm: Option<RMSNorm>,
    layer_scalar: Option<f64>,
    activation: Activation,
}

fn tensor_to_scalar_f64(t: &Tensor) -> Result<f64> {
    let t = if t.dtype() == candle_core::DType::F32 {
        t.clone()
    } else {
        t.to_dtype(candle_core::DType::F32)?
    };

    if t.rank() == 0 {
        return Ok(t.to_scalar::<f32>()? as f64);
    }
    let flat = t.flatten_all()?;
    let vals = flat.to_vec1::<f32>()?;
    vals.first()
        .copied()
        .map(|v| v as f64)
        .ok_or_else(|| candle_core::Error::Msg("layer_scalar tensor is empty".to_string()))
}

impl TransformerBlock {
    /// Loads layer `layer_idx` from safetensors weights (`model.layers.{i}.*`).
    ///
    /// The token mixer is a Gated DeltaNet when [`BlockConfig::linear_attn`] is
    /// set, otherwise attention; the FFN is MoE when [`BlockConfig::moe`] is set
    /// (with a GPT-OSS variant), otherwise dense. The optional sandwich norms
    /// and per-layer-input projections are loaded only when their tensors are
    /// present in `weights`.
    ///
    /// ## Errors
    /// Fails if a required tensor is missing, or if an FP8 weight is present
    /// without its companion `*_scale_inv`.
    pub fn load(cfg: &BlockConfig, layer_idx: usize, weights: &ModelWeights) -> Result<Self> {
        let p = format!("model.layers.{}", layer_idx);
        let input_norm = RMSNorm::load(
            weights,
            &format!("{}.input_layernorm", p),
            cfg.rms_norm_eps,
            cfg.norm_type,
        )?;
        let ffn_norm = RMSNorm::load(
            weights,
            &format!("{}.post_attention_layernorm", p),
            cfg.rms_norm_eps,
            cfg.norm_type,
        )?;
        let attention = if cfg.linear_attn.is_some() {
            TokenMixer::Gdn(GatedDeltaNet::load(cfg, layer_idx, weights)?)
        } else {
            TokenMixer::Attention(Attention::load(cfg, layer_idx, weights)?)
        };
        let ffn = match cfg.moe {
            Some(moe_cfg) if moe_cfg.gpt_oss => {
                FeedForwardLayer::Moe(MoeFeedForward::load_gpt_oss(
                    layer_idx,
                    weights,
                    moe_cfg.num_experts,
                    moe_cfg.num_experts_per_tok,
                    moe_cfg.swiglu_limit,
                )?)
            }
            Some(moe_cfg) => FeedForwardLayer::Moe(MoeFeedForward::load(
                layer_idx,
                weights,
                cfg.activation,
                moe_cfg.num_experts,
                moe_cfg.num_experts_per_tok,
                moe_cfg.norm_topk_prob,
            )?),
            None => FeedForwardLayer::Dense(FeedForward::load(layer_idx, weights, cfg.activation)?),
        };

        let mut pre_ffn_norm = None;
        let mut post_ffn_norm = None;
        if cfg.has_ffn_norms {
            pre_ffn_norm = Some(RMSNorm::load(
                weights,
                &format!("{}.pre_feedforward_layernorm", p),
                cfg.rms_norm_eps,
                cfg.norm_type,
            )?);
            post_ffn_norm = Some(RMSNorm::load(
                weights,
                &format!("{}.post_feedforward_layernorm", p),
                cfg.rms_norm_eps,
                cfg.norm_type,
            )?);
        }

        let per_layer_input_gate_name = format!("{}.per_layer_input_gate.weight", p);
        let per_layer_input_gate = if let Some(w) = weights.try_get(&per_layer_input_gate_name) {
            let w = w.clone();
            let scale_inv = weights
                .try_get_scale_inv(&per_layer_input_gate_name)
                .cloned();
            if w.dtype() == DType::F8E4M3 && scale_inv.is_none() {
                candle_core::bail!(
                    "missing '{}' required by FP8 tensor '{}'",
                    format!("{}_scale_inv", per_layer_input_gate_name),
                    per_layer_input_gate_name
                );
            }
            Some(AnyLinear::from_weight_with_scale_inv(w, scale_inv, None)?)
        } else {
            None
        };

        let per_layer_projection_name = format!("{}.per_layer_projection.weight", p);
        let per_layer_projection = if let Some(w) = weights.try_get(&per_layer_projection_name) {
            let w = w.clone();
            let scale_inv = weights
                .try_get_scale_inv(&per_layer_projection_name)
                .cloned();
            if w.dtype() == DType::F8E4M3 && scale_inv.is_none() {
                candle_core::bail!(
                    "missing '{}' required by FP8 tensor '{}'",
                    format!("{}_scale_inv", per_layer_projection_name),
                    per_layer_projection_name
                );
            }
            Some(AnyLinear::from_weight_with_scale_inv(w, scale_inv, None)?)
        } else {
            None
        };
        let post_per_layer_input_norm =
            if per_layer_input_gate.is_some() && per_layer_projection.is_some() {
                Some(RMSNorm::load(
                    weights,
                    &format!("{}.post_per_layer_input_norm", p),
                    cfg.rms_norm_eps,
                    cfg.norm_type,
                )?)
            } else {
                None
            };
        let layer_scalar = weights
            .try_get(&format!("{}.layer_scalar", p))
            .map(tensor_to_scalar_f64)
            .transpose()?;

        Ok(Self {
            input_norm,
            attention,
            ffn_norm,
            ffn,
            pre_ffn_norm,
            post_ffn_norm,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            layer_scalar,
            activation: cfg.activation,
        })
    }

    /// Loads layer `layer_idx` from GGUF weights (`blk.{i}.*`), dequantizing the
    /// norm tensors to `dtype` on `device`.
    ///
    /// The GGUF runtime is the dense, single-stream path: only attention/GDN
    /// mixers and a dense FFN are built here. The pre-FFN norm is read from
    /// either `ffn_norm` (standard archs) or `post_attention_norm` (Qwen3.5),
    /// matching llama.cpp's naming.
    ///
    /// ## Errors
    /// Fails if a required tensor is missing, or if `cfg.moe` is set: GGUF MoE
    /// models are not yet supported (different tensor naming and per-expert
    /// quantization layout).
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
        let input_norm = RMSNorm::from_qtensor(
            &attn_norm_qt,
            device,
            dtype,
            cfg.rms_norm_eps,
            cfg.norm_type,
        )?;

        let ffn_norm_qt = match gguf.try_get(&format!("{prefix}.ffn_norm.weight")) {
            Some(qt) => qt,
            None => gguf.get(&format!("{prefix}.post_attention_norm.weight"))?,
        };
        let ffn_norm =
            RMSNorm::from_qtensor(&ffn_norm_qt, device, dtype, cfg.rms_norm_eps, cfg.norm_type)?;

        let attention = if cfg.linear_attn.is_some() {
            TokenMixer::Gdn(GatedDeltaNet::load_gguf(
                cfg, layer_idx, gguf, device, dtype,
            )?)
        } else {
            TokenMixer::Attention(Attention::load_gguf(cfg, layer_idx, gguf, device, dtype)?)
        };
        if cfg.moe.is_some() {
            candle_core::bail!(
                "GGUF MoE models are not yet supported (layer {layer_idx} has cfg.moe = Some)"
            );
        }
        let ffn = FeedForwardLayer::Dense(FeedForward::load_gguf(
            layer_idx,
            gguf,
            intermediate_size,
            device,
            dtype,
            cfg.activation,
        )?);

        Ok(Self {
            input_norm,
            attention,
            ffn_norm,
            ffn,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            per_layer_input_gate: None,
            per_layer_projection: None,
            post_per_layer_input_norm: None,
            layer_scalar: None,
            activation: cfg.activation,
        })
    }

    /// Runs the layer over a packed batch of tokens, returning the new hidden
    /// state (same shape as `x`).
    ///
    /// `segments` carries one [`SegmentInfo`] per sequence in the batch: its
    /// token count and its KV cache (or recurrent state) for this layer, so a
    /// single call advances several sequences at once. `per_layer_input` is the
    /// Gemma-3n per-layer embedding for this layer, or `None`.
    ///
    /// The norm placement follows the [`TransformerBlock`] type docs: with
    /// sandwich norms, `ffn_norm` is applied to the attention output before the
    /// residual add and `pre_ffn_norm` feeds the FFN; without them, `ffn_norm`
    /// feeds the FFN directly.
    ///
    /// ## Errors
    /// Propagates any tensor-op failure from the sub-layers.
    pub fn forward_batch(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        position_ids: &Tensor,
        per_layer_input: Option<&Tensor>,
        mask: Option<&Tensor>,
        segments: &mut [SegmentInfo],
    ) -> Result<Tensor> {
        let dev = x.device().clone();
        let residual = x;
        let mut attn_out = decode_profile::phase(&dev, "attn", || {
            let normed = self.input_norm.forward(x)?;
            self.attention
                .forward_batch(&normed, rope, position_ids, mask, segments)
        })?;

        if self.pre_ffn_norm.is_some() {
            attn_out = self.ffn_norm.forward(&attn_out)?;
        }

        let mut x = (residual + attn_out)?;
        let residual = x.clone();

        let mut ffn_out = decode_profile::phase(&dev, "ffn", || {
            let ffn_inp = if let Some(pre_norm) = &self.pre_ffn_norm {
                pre_norm.forward(&x)?
            } else {
                self.ffn_norm.forward(&x)?
            };
            self.ffn.forward(&ffn_inp)
        })?;
        if let Some(post_norm) = &self.post_ffn_norm {
            ffn_out = post_norm.forward(&ffn_out)?;
        }

        x = (residual + ffn_out)?;

        if let (Some(gate), Some(proj), Some(post_norm), Some(per_layer_input)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let residual = x.clone();
            let mut gated = gate.forward(&x)?;
            gated = match self.activation {
                Activation::SiLU => gated.silu()?,
                Activation::GeLUTanh => gated.gelu()?,
            };
            let mixed = (gated * per_layer_input)?;
            let projected = proj.forward(&mixed)?;
            let projected = post_norm.forward(&projected)?;
            x = (residual + projected)?;
        }

        if let Some(layer_scalar) = self.layer_scalar {
            x = (x * layer_scalar)?;
        }

        Ok(x)
    }
}

/// A borrowed bundle of the static parts of a model, passed by value into
/// [`run_transformer_layers_batch`].
///
/// Everything here is owned by the model struct and lives for the duration of
/// the forward pass. `ropes` holds one [`RotaryEmbedding`] per layer (models
/// can use a different `rope_theta` per layer); the `per_layer_*` and softcap
/// fields are the optional features described on
/// [`super::config::StandardTransformerConfig`].
pub struct TransformerComponents<'a> {
    pub embed_tokens: &'a Embedding,
    pub blocks: &'a [TransformerBlock],
    pub norm: &'a RMSNorm,
    pub lm_head: &'a AnyLinear,
    pub ropes: &'a [RotaryEmbedding],
    pub embed_scale: Option<f64>,
    pub logit_softcap: Option<f64>,
    pub per_layer_input_embed: Option<&'a Embedding>,
    pub per_layer_input_embed_scale: Option<f64>,
    pub per_layer_model_projection: Option<&'a AnyLinear>,
    pub per_layer_model_projection_scale: Option<f64>,
    pub per_layer_projection_norm: Option<&'a RMSNorm>,
    pub per_layer_input_scale: Option<f64>,
    pub kv_shared_layer_map: Option<&'a [Option<usize>]>,
}

/// The architecture-agnostic forward pass: tokens in, logits out.
///
/// Several sequences are packed into one batch. `token_ids` is their tokens
/// concatenated end to end, and `token_counts[i]` is how many of those belong to
/// sequence `i` (typically many during prefill, exactly `1` during decode).
/// `seq_caches[i]` is that sequence's KV cache, one [`PagedKvCache`] per layer.
/// The result has one logit row per input token.
///
/// The pass runs embeddings (plus the optional per-layer-input embedding),
/// every [`TransformerBlock`] in order, the final norm, the lm-head, and the
/// optional logit soft-cap. Per-layer it assembles the [`SegmentInfo`] slice
/// each block needs, honouring [`TransformerComponents::kv_shared_layer_map`]
/// so layers that share a KV cache read the same one.
///
/// ## Panics
/// In debug builds, asserts the batch is internally consistent:
/// `token_counts.len() == seq_caches.len()`, `sum(token_counts)` equals the
/// `token_ids` sequence length, every `seq_caches[i]` has one entry per block,
/// and `token_ids`/`position_ids` live on the same device: a cross-device
/// mismatch would otherwise be silently miscomputed by downstream ops rather
/// than caught here. These are `debug_assert!`s, compiled out of release builds.
///
/// ## Errors
/// Propagates tensor-op failures, and fails if a per-layer-input embedding
/// dimension is not divisible by the layer count.
pub fn run_transformer_layers_batch(
    c: TransformerComponents<'_>,
    token_ids: &Tensor,
    position_ids: &Tensor,
    seq_caches: &mut [&mut [PagedKvCache]],
    token_counts: &[usize],
) -> Result<Tensor> {
    debug_assert_eq!(
        token_counts.len(),
        seq_caches.len(),
        "token_counts.len() must equal seq_caches.len()"
    );
    debug_assert_eq!(
        token_counts.iter().sum::<usize>(),
        token_ids.dim(candle_core::D::Minus1).unwrap_or(0),
        "sum(token_counts) must equal token_ids sequence length"
    );
    for (i, seq_cache) in seq_caches.iter().enumerate() {
        debug_assert_eq!(
            seq_cache.len(),
            c.blocks.len(),
            "seq_caches[{i}].len() must equal number of transformer blocks"
        );
    }
    debug_assert_eq!(
        token_ids.device().location(),
        position_ids.device().location(),
        "token_ids and position_ids must live on the same device",
    );

    decode_profile::set_active(token_counts.iter().all(|&t| t == 1));
    let dev = token_ids.device().clone();
    decode_profile::barrier(&dev);

    let mut x = decode_profile::phase(&dev, "embed", || c.embed_tokens.forward(token_ids))?;
    if let Some(scale) = c.embed_scale {
        x = (x * scale)?;
    }

    let per_layer_inputs = if let (
        Some(embed),
        Some(embed_scale),
        Some(model_proj),
        Some(model_proj_scale),
        Some(proj_norm),
        Some(input_scale),
    ) = (
        c.per_layer_input_embed,
        c.per_layer_input_embed_scale,
        c.per_layer_model_projection,
        c.per_layer_model_projection_scale,
        c.per_layer_projection_norm,
        c.per_layer_input_scale,
    ) {
        let mut per_token = embed.forward(token_ids)?;
        per_token = (per_token * embed_scale)?;
        let (b, s, flat_dim) = per_token.dims3()?;
        let n_layers = c.blocks.len();
        if flat_dim % n_layers != 0 {
            candle_core::bail!(
                "per_layer embed dim {flat_dim} is not divisible by n_layers {n_layers}"
            );
        }
        let per_layer_hidden = flat_dim / n_layers;
        let per_token = per_token.reshape((b, s, n_layers, per_layer_hidden))?;

        let projected = model_proj.forward(&x)?;
        let projected = (projected * model_proj_scale)?;
        let projected = projected.reshape((b, s, n_layers, per_layer_hidden))?;
        let projected = proj_norm.forward(&projected)?;

        Some(((projected + per_token)? * input_scale)?)
    } else {
        None
    };

    for (layer_idx, block) in c.blocks.iter().enumerate() {
        let mut segments: Vec<SegmentInfo> = Vec::with_capacity(seq_caches.len());
        for (seq_idx, seq_cache) in seq_caches.iter_mut().enumerate() {
            let shared_cache_idx = c
                .kv_shared_layer_map
                .and_then(|m| m.get(layer_idx).copied().flatten());
            let use_shared_cache = shared_cache_idx.is_some();
            let cache_idx = if use_shared_cache {
                shared_cache_idx.unwrap_or(layer_idx)
            } else {
                layer_idx
            };
            segments.push(SegmentInfo {
                num_tokens: token_counts[seq_idx],
                cache: &mut seq_cache[cache_idx],
                reuse_cache: use_shared_cache,
            });
        }

        let per_layer_input = if let Some(all_inputs) = &per_layer_inputs {
            Some(all_inputs.narrow(2, layer_idx, 1)?.squeeze(2)?)
        } else {
            None
        };

        x = block.forward_batch(
            &x,
            &c.ropes[layer_idx],
            position_ids,
            per_layer_input.as_ref(),
            None,
            &mut segments,
        )?;
    }

    let x = decode_profile::phase(&dev, "final_norm", || c.norm.forward(&x))?;
    let logits = decode_profile::phase(&dev, "lm_head", || c.lm_head.forward(&x))?;

    let out = if let Some(cap) = c.logit_softcap {
        #[cfg(feature = "metal")]
        {
            if logits.device().is_metal() {
                let l = logits.contiguous()?;
                super::metal_ops::softcap_fused(&l, cap as f32)
            } else {
                (logits / cap)?.tanh()?.affine(cap, 0.0)
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            (logits / cap)?.tanh()?.affine(cap, 0.0)
        }
    } else {
        Ok(logits)
    };
    decode_profile::mark_forward_end();
    out
}

/// Commits any buffered writes in every per-sequence, per-layer KV cache.
///
/// [`PagedKvCache`] may stage appends; call this after a batch to make them
/// visible before the caches are read again.
///
/// ## Errors
/// Propagates a flush failure from any underlying cache.
pub fn flush_caches(seq_caches: &mut [&mut [PagedKvCache]]) -> Result<()> {
    for seq_cache in seq_caches.iter_mut() {
        for cache in seq_cache.iter_mut() {
            cache.flush_pending()?;
        }
    }
    Ok(())
}
