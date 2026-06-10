use super::decode_profile;
use super::{
    attention::{Attention, SegmentInfo},
    config::{Activation, BlockConfig},
    ffn::FeedForward,
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

/// Dense or MoE FFN sub-layer. Both variants share the same `forward(x) → x`
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

pub struct TransformerBlock {
    input_norm: RMSNorm,
    attention: Attention,
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
        let attention = Attention::load(cfg, layer_idx, weights)?;
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

        let ffn_norm_qt = gguf.get(&format!("{prefix}.ffn_norm.weight"))?;
        let ffn_norm =
            RMSNorm::from_qtensor(&ffn_norm_qt, device, dtype, cfg.rms_norm_eps, cfg.norm_type)?;

        let attention = Attention::load_gguf(cfg, layer_idx, gguf, device, dtype)?;
        // GGUF runtime: dense FFN only — MoE GGUF support is future work
        // (different tensor naming and per-expert quantization layout).
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

/// Static model components shared by standard transformer architectures (Llama, Qwen3, …).
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

/// Shared batched forward pass for standard transformer models.
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
    // Cross-device tensors would be silently miscomputed by candle ops further
    // down; surface the misroute here so the panic names the offending input.
    // Single-device deployments are unaffected — the check is debug-only.
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

pub fn flush_caches(seq_caches: &mut [&mut [PagedKvCache]]) -> Result<()> {
    for seq_cache in seq_caches.iter_mut() {
        for cache in seq_cache.iter_mut() {
            cache.flush_pending()?;
        }
    }
    Ok(())
}
