//! Mixture-of-Experts (MoE) feed-forward layer.
//!
//! Drops in as a sibling of [`super::ffn::FeedForward`] for architectures with
//! sparse expert routing (Qwen3-MoE, OLMoE, Mixtral-style). The block-level
//! integration is in [`super::ffn::FeedForwardLayer`] which the transformer
//! block dispatches via the same `forward` signature.
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
//!   drops from `n_tokens` to `~n_tokens × top_k / num_experts` — up to ~8×
//!   on OLMoE-1B-7B (64 × top_k=8) for long prefill.
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
use super::linear::{AnyLinear, gelu_tanh, silu, softmax_last_dim};
use super::weights::ModelWeights;
use candle_core::{D, DType, Result, Tensor};

struct MoeExpert {
    gate_proj: AnyLinear,
    up_proj: AnyLinear,
    down_proj: AnyLinear,
}

impl MoeExpert {
    fn forward(&self, x: &Tensor, activation: Activation) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = match activation {
            Activation::SiLU => silu(&gate)?,
            Activation::GeLUTanh => gelu_tanh(&gate)?,
        };
        let gated = (activated * up)?;
        self.down_proj.forward(&gated)
    }
}

pub struct MoeFeedForward {
    router: AnyLinear,
    experts: Vec<MoeExpert>,
    top_k: usize,
    activation: Activation,
    norm_topk: bool,
}

impl MoeFeedForward {
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

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let prefix = format!("{p}.experts.{e}");
            let gate = weights.get(&format!("{prefix}.gate_proj.weight"))?.clone();
            let up = weights.get(&format!("{prefix}.up_proj.weight"))?.clone();
            let down = weights.get(&format!("{prefix}.down_proj.weight"))?.clone();
            experts.push(MoeExpert {
                gate_proj: AnyLinear::from_weight(gate, None)?,
                up_proj: AnyLinear::from_weight(up, None)?,
                down_proj: AnyLinear::from_weight(down, None)?,
            });
        }

        Ok(Self {
            router,
            experts,
            top_k,
            activation,
            norm_topk,
        })
    }

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

        // Per-expert `index_select` / `index_add` overhead beats the FFN saving
        // when `n_tokens ≤ top_k` (decode); cross over to the sparse path above.
        let out = if n_tokens > self.top_k {
            self.dispatch_sparse(&x_flat, &top_idx, &top_vals, n_tokens, hidden)?
        } else {
            self.dispatch_naive(&x_flat, &top_idx, &top_vals, n_tokens, hidden)?
        };
        out.reshape(original_shape)
    }

    fn dispatch_naive(
        &self,
        x_flat: &Tensor,
        top_idx: &Tensor,
        top_vals: &Tensor,
        n_tokens: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let num_experts = self.experts.len();
        let device = x_flat.device();
        let gate_f32 = Tensor::zeros((n_tokens, num_experts), DType::F32, device)?;
        let gate_f32 = gate_f32.scatter_add(top_idx, top_vals, D::Minus1)?;
        let gate = gate_f32.to_dtype(x_flat.dtype())?;

        let per_expert_mass: Vec<f32> = gate_f32.sum(0)?.flatten_all()?.to_vec1::<f32>()?;

        let mut acc: Option<Tensor> = None;
        for (e, expert) in self.experts.iter().enumerate() {
            if per_expert_mass[e] == 0.0 {
                continue;
            }
            let weight_e = gate.narrow(D::Minus1, e, 1)?;
            let expert_out = expert.forward(x_flat, self.activation)?;
            let weighted = expert_out.broadcast_mul(&weight_e)?;
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

    fn dispatch_sparse(
        &self,
        x_flat: &Tensor,
        top_idx: &Tensor,
        top_vals: &Tensor,
        n_tokens: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let num_experts = self.experts.len();
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

        let mut output = Tensor::zeros((n_tokens, hidden), x_flat.dtype(), device)?;
        for (e, expert) in self.experts.iter().enumerate() {
            let token_idx = &per_expert_tokens[e];
            if token_idx.is_empty() {
                continue;
            }
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
            experts.push(MoeExpert {
                gate_proj: AnyLinear::Float(Linear::new(mk(intermediate, hidden, off)?, None)?),
                up_proj: AnyLinear::Float(Linear::new(mk(intermediate, hidden, off + 1)?, None)?),
                down_proj: AnyLinear::Float(Linear::new(mk(hidden, intermediate, off + 2)?, None)?),
            });
        }

        Ok(MoeFeedForward {
            router,
            experts,
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
        let ref_out = moe.experts[0].forward(&x_flat, Activation::SiLU)?;
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

        let input_data: Vec<f32> = (0..1 * 3 * hidden)
            .map(|i| (i as f32 * 0.027).cos())
            .collect();
        let x = Tensor::from_vec(input_data, (1, 3, hidden), &device)?;
        let y = moe.forward(&x)?;
        let y_norm = y.flatten_all()?.to_vec1::<f32>()?;
        assert!(y_norm.iter().all(|v| v.is_finite()));

        let x_flat = x.reshape((3, hidden))?.contiguous()?;
        let ref0 = moe.experts[0].forward(&x_flat, Activation::SiLU)?;
        let ref_norm = ref0.flatten_all()?.to_vec1::<f32>()?;
        let y_mag: f32 = y_norm.iter().map(|v| v.abs()).sum::<f32>() / y_norm.len() as f32;
        let r_mag: f32 = ref_norm.iter().map(|v| v.abs()).sum::<f32>() / ref_norm.len() as f32;
        assert!(y_mag < 10.0 * r_mag && r_mag < 10.0 * y_mag);
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
