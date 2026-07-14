//! Mixture-of-Experts (MoE) feed-forward layer.
//!
//! Drops in as a sibling of [`super::ffn::FeedForward`] for architectures with
//! sparse expert routing (Qwen3-MoE, OLMoE, GPT-OSS). The transformer block
//! ([`super::block::TransformerBlock`]) dispatches it through the same `forward`
//! signature as the dense FFN.
//!
//! ## Routing
//!
//! Standard top-k softmax routing:
//!   1. `logits = router(x)`           shape `[n_tokens, num_experts]`
//!   2. `probs = softmax(logits)`
//!   3. `(top_vals, top_idx) = top_k(probs)`  shape `[n_tokens, top_k]`
//!   4. (Optional) renormalise `top_vals` so each row sums to 1.
//!
//! ## Dispatch (hybrid)
//!
//! Two equivalent paths, chosen per call based on `n_tokens` vs `top_k`:
//!
//! * **Naive** (`n_tokens ≤ top_k`, decode): build a dense
//!   `[n_tokens, num_experts]` gate via `scatter_add`, then for each non-empty
//!   expert run its FFN on the full `x_flat` and accumulate
//!   `gate[:, e:e+1] * expert(x)`. Per-expert command-buffer overhead is
//!   minimal because there are only `top_k` non-empty experts at M=1.
//!
//! * **Sparse** (`n_tokens > top_k`, prefill): group token indices per expert
//!   on the CPU, then for each expert `index_select` its rows, run the FFN on
//!   the subset, and `index_add` the weighted result back. Per-expert compute
//!   drops from `n_tokens` to `~n_tokens × top_k / num_experts` (up to ~8× on
//!   OLMoE-1B-7B with 64 experts and top_k=8 for long prefill).
//!
//! Empirically on OLMoE-1B-7B (Metal, M5 base): the hybrid gives decode
//! ≈ 10 tok/s and TTFT ≈ 7.3 s on a 256-word prompt. Pure-naive matches
//! decode but is 20-30% slower on TTFT; pure-sparse matches TTFT but loses
//! ~25% on decode due to per-expert `index_select` / `index_add` overhead.
//!
//! ## Supported layouts
//!
//! Tensor naming follows the Hugging Face `Qwen3MoeForCausalLM` /
//! `OlmoeForCausalLM` convention:
//!   * Router: `model.layers.{layer}.mlp.gate.weight`
//!   * Per-expert: `model.layers.{layer}.mlp.experts.{e}.{gate,up,down}_proj.weight`
//!
//! Mixtral's `block_sparse_moe.experts.{e}.{w1,w2,w3}` naming is **not**
//! supported in this first cut.

use super::config::Activation;
use super::expert_stream::StreamedExperts;
use super::linear::{AnyLinear, gelu_tanh, silu, softmax_last_dim};
use super::mxfp4::Mxfp4Linear;
use super::weights::ModelWeights;
use candle_core::{D, DType, Result, Tensor};
use std::sync::Arc;

/// A single MoE expert, in one of two formats.
///
/// `Standard` is an ordinary SwiGLU FFN with separate gate/up/down projections
/// (Qwen3-MoE, OLMoE). `GptOss` is the GPT-OSS expert: MXFP4 weights with gate
/// and up interleaved in one projection (even columns gate, odd up) and a
/// clamped SwiGLU with alpha = 1.702 and a `+1` on the up branch, i.e.
/// `glu = min(gate, limit) * sigmoid(1.702 * min(gate, limit))` then
/// `out = down((clamp(up, ±limit) + 1) * glu)`.
pub(crate) enum MoeExpert {
    Standard {
        gate_proj: AnyLinear,
        up_proj: AnyLinear,
        down_proj: AnyLinear,
    },
    GptOss {
        gate_up: Mxfp4Linear,
        down: Mxfp4Linear,
        limit: f64,
    },
}

const GPT_OSS_SWIGLU_ALPHA: f64 = 1.702;

impl MoeExpert {
    /// Runs this expert on `x`: gated SwiGLU for `Standard`, the clamped
    /// interleaved SwiGLU for `GptOss`.
    pub(crate) fn forward(&self, x: &Tensor, activation: Activation) -> Result<Tensor> {
        match self {
            Self::Standard {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                let gate = gate_proj.forward(x)?;
                let up = up_proj.forward(x)?;
                let activated = match activation {
                    Activation::SiLU => silu(&gate)?,
                    Activation::GeLUTanh => gelu_tanh(&gate)?,
                };
                let gated = (activated * up)?;
                down_proj.forward(&gated)
            }
            Self::GptOss {
                gate_up,
                down,
                limit,
            } => {
                let gu = gate_up.forward(x)?;
                let mut dims = gu.dims().to_vec();
                let inter = dims.last().unwrap() / 2;
                *dims.last_mut().unwrap() = inter;
                dims.push(2);
                let gu = gu.reshape(dims)?;
                let gate = gu.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
                let up = gu.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

                let gate = gate.clamp(f64::NEG_INFINITY, *limit)?;
                let up = up.clamp(-*limit, *limit)?;
                let z = gate.affine(GPT_OSS_SWIGLU_ALPHA, 0.0)?;
                let sig = (z.neg()?.exp()?.affine(1.0, 1.0)?).recip()?;
                let glu = (gate * sig)?;
                let h = (up.affine(1.0, 1.0)? * glu)?;
                down.forward(&h)
            }
        }
    }
}

/// Where a layer's experts live.
///
/// `Resident` holds every expert on the device (the eager path). `Streamed`
/// resolves experts through the model-wide LRU pool
/// ([`StreamedExperts`]), fetching missing ones from the checkpoint mmap on
/// demand; the handles returned by [`resolve`](Self::resolve) keep fetched
/// experts alive for the duration of a dispatch even across evictions.
enum ExpertBank {
    Resident(Vec<MoeExpert>),
    Streamed {
        layer_idx: usize,
        num_experts: usize,
        pool: Arc<StreamedExperts>,
    },
}

/// A resolved expert handle, valid for one dispatch.
enum ExpertHandle<'a> {
    Borrowed(&'a MoeExpert),
    Shared(Arc<MoeExpert>),
}

impl std::ops::Deref for ExpertHandle<'_> {
    type Target = MoeExpert;
    fn deref(&self) -> &MoeExpert {
        match self {
            Self::Borrowed(e) => e,
            Self::Shared(e) => e,
        }
    }
}

impl ExpertBank {
    fn num_experts(&self) -> usize {
        match self {
            Self::Resident(v) => v.len(),
            Self::Streamed { num_experts, .. } => *num_experts,
        }
    }

    /// Resolves the given expert ids for a dispatch, fetching streamed misses.
    fn resolve(&self, ids: &[usize]) -> Result<Vec<(usize, ExpertHandle<'_>)>> {
        ids.iter()
            .map(|&e| {
                let handle = match self {
                    Self::Resident(v) => ExpertHandle::Borrowed(&v[e]),
                    Self::Streamed {
                        layer_idx, pool, ..
                    } => ExpertHandle::Shared(pool.fetch(*layer_idx, e)?),
                };
                Ok((e, handle))
            })
            .collect()
    }
}

/// A sparse Mixture-of-Experts feed-forward layer.
///
/// A linear `router` scores the experts per token; the top `top_k` are run and
/// their outputs combined (renormalised when `norm_topk`). See the module docs
/// for the routing math and the naive/sparse dispatch. Load it with
/// [`load`](Self::load) (standard experts) or
/// [`load_gpt_oss`](Self::load_gpt_oss) (MXFP4 GPT-OSS experts); both switch
/// to the streamed bank when the weights carry an expert pool.
pub struct MoeFeedForward {
    router: AnyLinear,
    experts: ExpertBank,
    top_k: usize,
    activation: Activation,
    norm_topk: bool,
}

impl MoeFeedForward {
    /// Loads a standard MoE layer for `layer_idx` (Qwen3-MoE / OLMoE).
    ///
    /// Reads the router (`mlp.gate.weight` or `mlp.router.weight`) and the
    /// `num_experts` per-expert gate/up/down projections; `norm_topk` controls
    /// whether the top-k gate weights are renormalised to sum to 1.
    ///
    /// ## Errors
    /// Fails if `top_k` is not in `(0, num_experts]`, or a required tensor is
    /// missing.
    pub fn load(
        layer_idx: usize,
        weights: &ModelWeights,
        activation: Activation,
        num_experts: usize,
        top_k: usize,
        norm_topk: bool,
    ) -> Result<Self> {
        if top_k == 0 || top_k > num_experts {
            candle_core::bail!("MoE top_k={top_k} must be in (0, {num_experts}] (num_experts)");
        }
        let p = format!("model.layers.{layer_idx}.mlp");

        let router_w = weights
            .try_get(&format!("{p}.gate.weight"))
            .or_else(|| weights.try_get(&format!("{p}.router.weight")))
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "MoE layer {layer_idx}: missing router weight (tried '{p}.gate.weight' and '{p}.router.weight')"
                ))
            })?
            .clone();
        let router = AnyLinear::from_weight(router_w, None)?;

        let experts = if let Some(pool) = weights.expert_pool() {
            ExpertBank::Streamed {
                layer_idx,
                num_experts,
                pool,
            }
        } else {
            let mut experts = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let prefix = format!("{p}.experts.{e}");
                let gate = weights.get(&format!("{prefix}.gate_proj.weight"))?.clone();
                let up = weights.get(&format!("{prefix}.up_proj.weight"))?.clone();
                let down = weights.get(&format!("{prefix}.down_proj.weight"))?.clone();
                experts.push(MoeExpert::Standard {
                    gate_proj: AnyLinear::from_weight(gate, None)?,
                    up_proj: AnyLinear::from_weight(up, None)?,
                    down_proj: AnyLinear::from_weight(down, None)?,
                });
            }
            ExpertBank::Resident(experts)
        };

        Ok(Self {
            router,
            experts,
            top_k,
            activation,
            norm_topk,
        })
    }

    /// GPT-OSS layout: stacked MXFP4 expert tensors
    /// (`mlp.experts.{gate_up,down}_proj_{blocks,scales,bias}`, expert dim 0)
    /// and a router with bias. Routing is softmax-over-top-k, which equals the
    /// standard path with `norm_topk = true`.
    ///
    /// ## Errors
    /// Fails if `top_k` is not in `(0, num_experts]`, a required tensor is
    /// missing, or a blocks tensor is not shaped `[E, out, K/32, 16]`.
    pub fn load_gpt_oss(
        layer_idx: usize,
        weights: &ModelWeights,
        num_experts: usize,
        top_k: usize,
        swiglu_limit: f64,
    ) -> Result<Self> {
        if top_k == 0 || top_k > num_experts {
            candle_core::bail!("MoE top_k={top_k} must be in (0, {num_experts}] (num_experts)");
        }
        let p = format!("model.layers.{layer_idx}.mlp");

        let router_w = weights.get(&format!("{p}.router.weight"))?.clone();
        let router_b = weights.try_get(&format!("{p}.router.bias")).cloned();
        let router = AnyLinear::from_weight(router_w, router_b)?;

        if let Some(pool) = weights.expert_pool() {
            return Ok(Self {
                router,
                experts: ExpertBank::Streamed {
                    layer_idx,
                    num_experts,
                    pool,
                },
                top_k,
                activation: Activation::SiLU,
                norm_topk: true,
            });
        }

        let slice_expert = |t: &Tensor, e: usize| -> Result<Tensor> {
            t.narrow(0, e, 1)?.squeeze(0)?.contiguous()
        };
        let gu_blocks = weights.get(&format!("{p}.experts.gate_up_proj_blocks"))?;
        let gu_scales = weights.get(&format!("{p}.experts.gate_up_proj_scales"))?;
        let gu_bias = weights.get(&format!("{p}.experts.gate_up_proj_bias"))?;
        let dn_blocks = weights.get(&format!("{p}.experts.down_proj_blocks"))?;
        let dn_scales = weights.get(&format!("{p}.experts.down_proj_scales"))?;
        let dn_bias = weights.get(&format!("{p}.experts.down_proj_bias"))?;

        let dims_of = |t: &Tensor, what: &str| -> Result<(usize, usize)> {
            let d = t.dims();
            if d.len() != 4 || d[3] != 16 {
                candle_core::bail!(
                    "GPT-OSS MoE layer {layer_idx}: {what} shape {d:?} != [E, out, K/32, 16]"
                );
            }
            Ok((d[1], d[2] * 32))
        };
        let (gu_out, gu_in) = dims_of(gu_blocks, "gate_up_proj_blocks")?;
        let (dn_out, dn_in) = dims_of(dn_blocks, "down_proj_blocks")?;

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let gate_up = Mxfp4Linear::new(
                slice_expert(gu_blocks, e)?,
                slice_expert(gu_scales, e)?,
                Some(slice_expert(gu_bias, e)?),
                gu_in,
                gu_out,
            )?;
            let down = Mxfp4Linear::new(
                slice_expert(dn_blocks, e)?,
                slice_expert(dn_scales, e)?,
                Some(slice_expert(dn_bias, e)?),
                dn_in,
                dn_out,
            )?;
            experts.push(MoeExpert::GptOss {
                gate_up,
                down,
                limit: swiglu_limit,
            });
        }

        Ok(Self {
            router,
            experts: ExpertBank::Resident(experts),
            top_k,
            activation: Activation::SiLU,
            norm_topk: true,
        })
    }

    #[cfg(test)]
    fn resident_expert(&self, e: usize) -> &MoeExpert {
        match &self.experts {
            ExpertBank::Resident(v) => &v[e],
            ExpertBank::Streamed { .. } => panic!("resident_expert on a streamed bank"),
        }
    }

    /// Routes `x` through the MoE: softmax router scores, top-k selection
    /// (optionally renormalised), then the naive or sparse dispatch. The output
    /// has the same shape as `x`.
    ///
    /// ## Errors
    /// Propagates router, routing, or expert tensor-op failures.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_shape = x.dims().to_vec();
        let hidden = *original_shape.last().unwrap();
        let n_tokens: usize = original_shape[..original_shape.len() - 1].iter().product();
        let x_flat = x.reshape((n_tokens, hidden))?.contiguous()?;

        let logits = self.router.forward(&x_flat)?;
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let probs = softmax_last_dim(&logits_f32)?;

        let sorted_idx = probs.arg_sort_last_dim(false)?;
        let top_idx = sorted_idx.narrow(D::Minus1, 0, self.top_k)?.contiguous()?;
        let top_vals = probs.gather(&top_idx, D::Minus1)?;

        let top_vals = if self.norm_topk {
            let denom = top_vals.sum_keepdim(D::Minus1)?;
            top_vals.broadcast_div(&denom)?
        } else {
            top_vals
        };

        let out = if n_tokens > self.top_k {
            self.dispatch_sparse(&x_flat, &top_idx, &top_vals, n_tokens, hidden)?
        } else {
            self.dispatch_naive(&x_flat, &top_idx, &top_vals, n_tokens, hidden)?
        };
        out.reshape(original_shape)
    }

    /// Decode-path dispatch (`n_tokens <= top_k`): runs only the chosen experts
    /// on the full input and accumulates their gate-weighted outputs.
    ///
    /// Per-expert token weights are built on the CPU to avoid a dense
    /// `[n_tokens, num_experts]` gate tensor; at M=1 only `top_k` experts have
    /// non-zero mass, so there are no wasted FFN calls.
    ///
    /// ## Errors
    /// Propagates tensor-op failures.
    fn dispatch_naive(
        &self,
        x_flat: &Tensor,
        top_idx: &Tensor,
        top_vals: &Tensor,
        n_tokens: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let num_experts = self.experts.num_experts();
        let device = x_flat.device();

        let top_idx_cpu: Vec<u32> = top_idx.flatten_all()?.to_vec1::<u32>()?;
        let top_vals_cpu: Vec<f32> = top_vals.flatten_all()?.to_vec1::<f32>()?;
        let mut per_expert_w: Vec<Option<Vec<f32>>> = vec![None; num_experts];
        for token in 0..n_tokens {
            for slot in 0..self.top_k {
                let flat = token * self.top_k + slot;
                let e = top_idx_cpu[flat] as usize;
                per_expert_w[e].get_or_insert_with(|| vec![0.0; n_tokens])[token] +=
                    top_vals_cpu[flat];
            }
        }

        let active: Vec<usize> = (0..num_experts)
            .filter(|&e| per_expert_w[e].is_some())
            .collect();
        let mut acc: Option<Tensor> = None;
        for (e, expert) in self.experts.resolve(&active)? {
            let w = per_expert_w[e].as_ref().expect("resolved id is active");
            let w_t =
                Tensor::from_vec(w.clone(), (n_tokens, 1), device)?.to_dtype(x_flat.dtype())?;
            let expert_out = expert.forward(x_flat, self.activation)?;
            let weighted = expert_out.broadcast_mul(&w_t)?;
            acc = Some(match acc {
                Some(a) => (a + weighted)?,
                None => weighted,
            });
        }
        Ok(match acc {
            Some(a) => a,
            None => Tensor::zeros((n_tokens, hidden), x_flat.dtype(), device)?,
        })
    }

    /// Prefill-path dispatch (`n_tokens > top_k`): groups tokens by expert, runs
    /// each expert on only its routed rows (`index_select`), and scatters the
    /// gate-weighted results back (`index_add`).
    ///
    /// Per-expert compute drops from `n_tokens` to roughly
    /// `n_tokens * top_k / num_experts`.
    ///
    /// ## Errors
    /// Propagates tensor-op failures.
    fn dispatch_sparse(
        &self,
        x_flat: &Tensor,
        top_idx: &Tensor,
        top_vals: &Tensor,
        n_tokens: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let num_experts = self.experts.num_experts();
        let device = x_flat.device();

        let top_idx_cpu: Vec<u32> = top_idx.flatten_all()?.to_vec1::<u32>()?;
        let top_vals_cpu: Vec<f32> = top_vals.flatten_all()?.to_vec1::<f32>()?;

        let mut per_expert_tokens: Vec<Vec<u32>> = vec![Vec::new(); num_experts];
        let mut per_expert_weights: Vec<Vec<f32>> = vec![Vec::new(); num_experts];
        for token in 0..n_tokens {
            let row = token * self.top_k;
            for slot in 0..self.top_k {
                let flat = row + slot;
                let e = top_idx_cpu[flat] as usize;
                per_expert_tokens[e].push(token as u32);
                per_expert_weights[e].push(top_vals_cpu[flat]);
            }
        }

        let active: Vec<usize> = (0..num_experts)
            .filter(|&e| !per_expert_tokens[e].is_empty())
            .collect();
        let mut output = Tensor::zeros((n_tokens, hidden), x_flat.dtype(), device)?;
        for (e, expert) in self.experts.resolve(&active)? {
            let token_idx = &per_expert_tokens[e];
            let n_selected = token_idx.len();
            let idx_t = Tensor::from_vec(token_idx.clone(), (n_selected,), device)?;
            let weights_t = Tensor::from_vec(per_expert_weights[e].clone(), (n_selected,), device)?
                .to_dtype(x_flat.dtype())?
                .reshape((n_selected, 1))?;

            let x_subset = x_flat.index_select(&idx_t, 0)?;
            let expert_out = expert.forward(&x_subset, self.activation)?;
            let weighted = expert_out.broadcast_mul(&weights_t)?;
            output = output.index_add(&idx_t, &weighted, 0)?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::linear::{AnyLinear, Linear};
    use candle_core::Device;

    fn build_synth_moe(
        device: &Device,
        hidden: usize,
        intermediate: usize,
        num_experts: usize,
        top_k: usize,
        salt: u64,
    ) -> Result<MoeFeedForward> {
        if top_k == 0 || top_k > num_experts {
            candle_core::bail!("MoE top_k={top_k} must be in (0, {num_experts}] (num_experts)");
        }
        let mk = |rows: usize, cols: usize, offset: u64| -> Result<Tensor> {
            let total = rows * cols;
            let data: Vec<f32> = (0..total)
                .map(|i| {
                    let raw = (i as u64)
                        .wrapping_mul(2654435761)
                        .wrapping_add(offset + salt);
                    ((raw & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.1
                })
                .collect();
            Tensor::from_vec(data, (rows, cols), device)
        };

        let router = AnyLinear::Float(Linear::new(mk(num_experts, hidden, 0)?, None)?);

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let off = 1000 + e as u64 * 17;
            experts.push(MoeExpert::Standard {
                gate_proj: AnyLinear::Float(Linear::new(mk(intermediate, hidden, off)?, None)?),
                up_proj: AnyLinear::Float(Linear::new(mk(intermediate, hidden, off + 1)?, None)?),
                down_proj: AnyLinear::Float(Linear::new(mk(hidden, intermediate, off + 2)?, None)?),
            });
        }

        Ok(MoeFeedForward {
            router,
            experts: ExpertBank::Resident(experts),
            top_k,
            activation: Activation::SiLU,
            norm_topk: true,
        })
    }

    #[test]
    fn moe_single_expert_topk1_matches_single_ffn() -> Result<()> {
        let device = Device::Cpu;
        let moe = build_synth_moe(&device, 16, 32, 1, 1, 0xc0ffee)?;

        let input_data: Vec<f32> = (0..2 * 4 * 16).map(|i| (i as f32 * 0.013).sin()).collect();
        let x = Tensor::from_vec(input_data, (2, 4, 16), &device)?;
        let y = moe.forward(&x)?;

        let x_flat = x.reshape((8, 16))?.contiguous()?;
        let ref_out = moe.resident_expert(0).forward(&x_flat, Activation::SiLU)?;
        let ref_y = ref_out.reshape((2, 4, 16))?;

        let diff = (y - ref_y)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "max_abs_diff = {diff}");
        Ok(())
    }

    #[test]
    fn moe_topk_routing_uses_only_topk_experts() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, intermediate, num_experts, top_k) = (8, 16, 4, 2);
        let moe = build_synth_moe(&device, hidden, intermediate, num_experts, top_k, 0xdead)?;

        let input_data: Vec<f32> = (0..3 * hidden).map(|i| (i as f32 * 0.027).cos()).collect();
        let x = Tensor::from_vec(input_data, (1, 3, hidden), &device)?;
        let y = moe.forward(&x)?;
        let y_norm = y.flatten_all()?.to_vec1::<f32>()?;
        assert!(y_norm.iter().all(|v| v.is_finite()));

        let x_flat = x.reshape((3, hidden))?.contiguous()?;
        let ref0 = moe.resident_expert(0).forward(&x_flat, Activation::SiLU)?;
        let ref_norm = ref0.flatten_all()?.to_vec1::<f32>()?;
        let y_mag: f32 = y_norm.iter().map(|v| v.abs()).sum::<f32>() / y_norm.len() as f32;
        let r_mag: f32 = ref_norm.iter().map(|v| v.abs()).sum::<f32>() / ref_norm.len() as f32;
        assert!(y_mag < 10.0 * r_mag && r_mag < 10.0 * y_mag);
        Ok(())
    }

    // Contract: the GPT-OSS expert math (interleaved gate/up split, clamping at
    // swiglu_limit, alpha=1.702 sigmoid gate, and the (up + 1) branch) matches a
    // scalar reference computed independently in f32.
    #[test]
    fn gpt_oss_expert_matches_scalar_reference() -> Result<()> {
        use crate::common::mxfp4::Mxfp4Linear;
        let device = Device::Cpu;
        let (hidden, inter) = (32usize, 32usize);
        let limit = 7.0f32;

        // MXFP4 weights with non-trivial codes: gate_up rows cycle FP4 codes,
        // scales cycle around 127 so magnitudes vary.
        let gu_out = 2 * inter;
        let nb = hidden / 32;
        let gu_blocks: Vec<u8> = (0..gu_out * nb * 16).map(|i| (i * 23 + 7) as u8).collect();
        let gu_scales: Vec<u8> = (0..gu_out * nb).map(|i| 125 + (i % 5) as u8).collect();
        let dn_blocks: Vec<u8> = (0..hidden * nb * 16).map(|i| (i * 41 + 3) as u8).collect();
        let dn_scales: Vec<u8> = (0..hidden * nb).map(|i| 126 + (i % 3) as u8).collect();
        let gu_bias: Vec<f32> = (0..gu_out).map(|i| (i as f32 * 0.7).sin() * 2.0).collect();
        let dn_bias: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.3).cos() * 0.5).collect();

        let expert = MoeExpert::GptOss {
            gate_up: Mxfp4Linear::new(
                Tensor::from_vec(gu_blocks.clone(), gu_out * nb * 16, &device)?,
                Tensor::from_vec(gu_scales.clone(), gu_out * nb, &device)?,
                Some(Tensor::from_vec(gu_bias.clone(), gu_out, &device)?),
                hidden,
                gu_out,
            )?,
            down: Mxfp4Linear::new(
                Tensor::from_vec(dn_blocks.clone(), hidden * nb * 16, &device)?,
                Tensor::from_vec(dn_scales.clone(), hidden * nb, &device)?,
                Some(Tensor::from_vec(dn_bias.clone(), hidden, &device)?),
                inter,
                hidden,
            )?,
            limit: limit as f64,
        };

        let x_data: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32) * 0.21).sin() * 3.0)
            .collect();
        let x = Tensor::from_vec(x_data.clone(), (1, hidden), &device)?;
        let y = expert
            .forward(&x, Activation::SiLU)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Independent scalar reference.
        let w_gu =
            crate::common::mxfp4::dequantize_mxfp4_f32(&gu_blocks, &gu_scales, gu_out, hidden)?;
        let w_dn =
            crate::common::mxfp4::dequantize_mxfp4_f32(&dn_blocks, &dn_scales, hidden, inter)?;
        let mut gate_up = vec![0f32; gu_out];
        for (o, gu) in gate_up.iter_mut().enumerate() {
            *gu = gu_bias[o]
                + x_data
                    .iter()
                    .enumerate()
                    .map(|(k, &xv)| xv * w_gu[o * hidden + k])
                    .sum::<f32>();
        }
        let mut h = vec![0f32; inter];
        for (k, hv) in h.iter_mut().enumerate() {
            let gate = gate_up[2 * k].min(limit);
            let up = gate_up[2 * k + 1].clamp(-limit, limit);
            let glu = gate * (1.0 / (1.0 + (-gate * 1.702f32).exp()));
            *hv = (up + 1.0) * glu;
        }
        let mut expected = vec![0f32; hidden];
        for (o, ev) in expected.iter_mut().enumerate() {
            *ev = dn_bias[o]
                + h.iter()
                    .enumerate()
                    .map(|(k, &hv)| hv * w_dn[o * inter + k])
                    .sum::<f32>();
        }

        let max_abs = expected.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let tol = 1e-3 * max_abs + 1e-3;
        for (i, (&got, &exp)) in y.iter().zip(&expected).enumerate() {
            assert!(
                (got - exp).abs() < tol,
                "out[{i}]: got {got}, expected {exp} (tol {tol})"
            );
        }
        Ok(())
    }

    // Contract: the naive decode dispatch must produce the same output as the
    // sparse dispatch for the same input (they are alternative evaluations of
    // the same routing).
    #[test]
    fn naive_and_sparse_dispatch_agree() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, intermediate, num_experts, top_k) = (8, 16, 4, 2);
        let moe = build_synth_moe(&device, hidden, intermediate, num_experts, top_k, 0xbeef)?;

        // 2 tokens with top_k=2: n_tokens <= top_k -> forward takes the naive
        // path; compute the sparse path directly on the same routing.
        let x_data: Vec<f32> = (0..2 * hidden).map(|i| (i as f32 * 0.05).sin()).collect();
        let x_flat = Tensor::from_vec(x_data, (2, hidden), &device)?;

        let logits = moe.router.forward(&x_flat)?;
        let probs = softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
        let sorted_idx = probs.arg_sort_last_dim(false)?;
        let top_idx = sorted_idx.narrow(D::Minus1, 0, top_k)?.contiguous()?;
        let top_vals = probs.gather(&top_idx, D::Minus1)?;
        let denom = top_vals.sum_keepdim(D::Minus1)?;
        let top_vals = top_vals.broadcast_div(&denom)?;

        let naive = moe.dispatch_naive(&x_flat, &top_idx, &top_vals, 2, hidden)?;
        let sparse = moe.dispatch_sparse(&x_flat, &top_idx, &top_vals, 2, hidden)?;
        let diff = (naive - sparse)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "naive vs sparse max_abs_diff = {diff}");
        Ok(())
    }

    #[test]
    fn moe_rejects_invalid_topk() -> Result<()> {
        let device = Device::Cpu;
        let moe = build_synth_moe(&device, 4, 8, 2, 3, 1);
        assert!(moe.is_err());
        Ok(())
    }
}
