use super::awq::{AwqRawTensors, PackDim, concat_awq_along_out};
use super::config::Activation;
use super::gguf_weights::GgufWeights;
use super::linear::{AnyLinear, QLinear, gelu_tanh, silu};
use super::weights::ModelWeights;
use candle_core::DType;
use candle_core::{D, Result, Tensor};

enum GateUpProjection {
    Fused(AnyLinear),
    Separate { gate: AnyLinear, up: AnyLinear },
    Packed(AnyLinear),
    Simple(AnyLinear),
}

pub struct FeedForward {
    gate_up: GateUpProjection,
    down_proj: AnyLinear,
    intermediate_size: usize,
    activation: Activation,
}

impl FeedForward {
    pub fn load(layer_idx: usize, weights: &ModelWeights, activation: Activation) -> Result<Self> {
        let p = format!("model.layers.{}.mlp", layer_idx);

        if let Some(down_raw) = weights.try_get_quant(&format!("{p}.down_proj")) {
            return Self::load_awq(&p, down_raw, weights, activation, layer_idx);
        }

        let down_weight_name = format!("{}.down_proj.weight", p);
        let down_w = weights.get(&down_weight_name)?.clone();
        let down_scale_inv = weights.try_get_scale_inv(&down_weight_name).cloned();
        if down_w.dtype() == DType::F8E4M3 && down_scale_inv.is_none() {
            candle_core::bail!(
                "missing '{}' required by FP8 tensor '{}'",
                format!("{}_scale_inv", down_weight_name),
                down_weight_name
            );
        }
        let down_proj =
            AnyLinear::from_weight_with_scale_inv(down_w.clone(), down_scale_inv, None)?;
        let intermediate_size = down_w.dim(1)?;

        let gate_up = if let (Some(gate_w), Some(up_w)) = (
            weights.try_get(&format!("{}.gate_proj.weight", p)),
            weights.try_get(&format!("{}.up_proj.weight", p)),
        ) {
            let gate_w = gate_w.clone();
            let up_w = up_w.clone();
            let gate_out = gate_w.dim(0)?;
            let up_out = up_w.dim(0)?;
            if gate_out != intermediate_size || up_out != intermediate_size {
                candle_core::bail!(
                    "FFN shape mismatch at {}: gate/up dim0 must both be {}, got gate={} up={}",
                    p,
                    intermediate_size,
                    gate_out,
                    up_out
                );
            }
            // Fuse gate+up into one matrix; individual weight tensors are not retained.
            let gate_up_w = Tensor::cat(&[&gate_w, &up_w], 0)?;
            let gate_is_fp8 = gate_w.dtype() == DType::F8E4M3;
            let up_is_fp8 = up_w.dtype() == DType::F8E4M3;
            let gate_up_scale_inv = if gate_is_fp8 || up_is_fp8 {
                let gate_scale = weights
                    .try_get_scale_inv(&format!("{}.gate_proj.weight", p))
                    .cloned();
                let up_scale = weights
                    .try_get_scale_inv(&format!("{}.up_proj.weight", p))
                    .cloned();
                match (gate_scale, up_scale) {
                    (Some(gs), Some(us)) => Some(Tensor::cat(&[&gs, &us], 0)?),
                    _ => {
                        candle_core::bail!(
                            "missing gate/up *_scale_inv tensors required by FP8 FFN at {}",
                            p
                        )
                    }
                }
            } else {
                None
            };
            GateUpProjection::Fused(AnyLinear::from_weight_with_scale_inv(
                gate_up_w,
                gate_up_scale_inv,
                None,
            )?)
        } else if let Some(gate_up_w) = weights.try_get(&format!("{}.gate_up_proj.weight", p)) {
            let gate_up_w = gate_up_w.clone();
            let packed_out = gate_up_w.dim(0)?;
            if packed_out != 2 * intermediate_size {
                candle_core::bail!(
                    "FFN shape mismatch at {}: gate_up dim0 must be {}, got {}",
                    p,
                    2 * intermediate_size,
                    packed_out
                );
            }
            let gate_up_weight_name = format!("{}.gate_up_proj.weight", p);
            let gate_up_scale_inv = weights.try_get_scale_inv(&gate_up_weight_name).cloned();
            if gate_up_w.dtype() == DType::F8E4M3 && gate_up_scale_inv.is_none() {
                candle_core::bail!(
                    "missing '{}' required by FP8 tensor '{}'",
                    format!("{}_scale_inv", gate_up_weight_name),
                    gate_up_weight_name
                );
            }
            GateUpProjection::Fused(AnyLinear::from_weight_with_scale_inv(
                gate_up_w,
                gate_up_scale_inv,
                None,
            )?)
        } else if let Some(up_w) = weights.try_get(&format!("{}.up_proj.weight", p)) {
            let up_w = up_w.clone();
            let up_out = up_w.dim(0)?;
            if up_out != intermediate_size {
                candle_core::bail!(
                    "FFN shape mismatch at {}: up dim0 must be {}, got {}",
                    p,
                    intermediate_size,
                    up_out
                );
            }
            let up_weight_name = format!("{}.up_proj.weight", p);
            let up_scale_inv = weights.try_get_scale_inv(&up_weight_name).cloned();
            if up_w.dtype() == DType::F8E4M3 && up_scale_inv.is_none() {
                candle_core::bail!(
                    "missing '{}' required by FP8 tensor '{}'",
                    format!("{}_scale_inv", up_weight_name),
                    up_weight_name
                );
            }
            GateUpProjection::Simple(AnyLinear::from_weight_with_scale_inv(
                up_w,
                up_scale_inv,
                None,
            )?)
        } else {
            candle_core::bail!(
                "Unsupported FFN layout at {}: expected gate_proj+up_proj, gate_up_proj, or up_proj",
                p
            );
        };

        Ok(Self {
            gate_up,
            down_proj,
            intermediate_size,
            activation,
        })
    }

    fn load_awq(
        p: &str,
        down_raw: AwqRawTensors,
        weights: &ModelWeights,
        activation: Activation,
        layer_idx: usize,
    ) -> Result<Self> {
        let device = down_raw.scales.device().clone();
        let dtype = down_raw.scales.dtype();
        let is_awq = down_raw.pack_dim == PackDim::Out;

        // down_proj: out_features = hidden, in_features = intermediate.
        let intermediate_size = down_raw.in_features().map_err(|e| {
            candle_core::Error::Msg(format!("packed-quant down_proj in_features: {e}"))
        })?;
        let down_proj = AnyLinear::from_quant(&down_raw, None, &device, dtype)?;

        let gate_prefix = format!("{p}.gate_proj");
        let up_prefix = format!("{p}.up_proj");
        let gate_up_prefix = format!("{p}.gate_up_proj");

        let gate_up = if let (Some(gate_raw), Some(up_raw)) = (
            weights.try_get_quant(&gate_prefix),
            weights.try_get_quant(&up_prefix),
        ) {
            let gate_out = gate_raw.scales.dim(1)?;
            let up_out = up_raw.scales.dim(1)?;
            if gate_out != intermediate_size || up_out != intermediate_size {
                candle_core::bail!(
                    "Packed-quant FFN shape mismatch at {p}: gate/up out_features must both be {intermediate_size}, got gate={gate_out} up={up_out}"
                );
            }

            // GPTQ packs along in_features ⇒ out-dim concat (`concat_awq_along_out`)
            // doesn't apply. Take Separate; the dequant-at-load dominates anyway.
            let gate_up_fused = is_awq && intermediate_size.is_multiple_of(8);
            if layer_idx == 0 {
                tracing::info!(
                    intermediate_size,
                    gate_up_fused,
                    quant = if is_awq { "awq" } else { "gptq" },
                    "Packed-quant FFN loader engaged (separate gate/up tensors)"
                );
            }
            if gate_up_fused {
                let fused_raw = concat_awq_along_out(&[gate_raw, up_raw]).map_err(|e| {
                    candle_core::Error::Msg(format!("AWQ gate+up fuse failed: {e:#}"))
                })?;
                GateUpProjection::Fused(AnyLinear::from_quant(&fused_raw, None, &device, dtype)?)
            } else {
                GateUpProjection::Separate {
                    gate: AnyLinear::from_quant(&gate_raw, None, &device, dtype)?,
                    up: AnyLinear::from_quant(&up_raw, None, &device, dtype)?,
                }
            }
        } else if let Some(gate_up_raw) = weights.try_get_quant(&gate_up_prefix) {
            let packed_out = gate_up_raw.scales.dim(1)?;
            if packed_out != 2 * intermediate_size {
                candle_core::bail!(
                    "Packed-quant FFN gate_up out_features {packed_out} != 2*{intermediate_size}"
                );
            }
            if layer_idx == 0 {
                tracing::info!(
                    intermediate_size,
                    "Packed-quant FFN loader engaged (pre-fused gate_up_proj)"
                );
            }
            GateUpProjection::Fused(AnyLinear::from_quant(&gate_up_raw, None, &device, dtype)?)
        } else if let Some(up_raw) = weights.try_get_quant(&up_prefix) {
            let up_out = up_raw.scales.dim(1)?;
            if up_out != intermediate_size {
                candle_core::bail!(
                    "Packed-quant FFN up_proj out_features {up_out} != {intermediate_size}"
                );
            }
            if layer_idx == 0 {
                tracing::info!(
                    intermediate_size,
                    "Packed-quant FFN loader engaged (ungated up-only path)"
                );
            }
            GateUpProjection::Simple(AnyLinear::from_quant(&up_raw, None, &device, dtype)?)
        } else {
            candle_core::bail!(
                "Mixed quantization at {p}: down_proj is packed but no gate/up tensors are packed. \
                 Expected one of: ({{gate,up}}_proj.qweight) or (gate_up_proj.qweight) or (up_proj.qweight)."
            );
        };

        Ok(Self {
            gate_up,
            down_proj,
            intermediate_size,
            activation,
        })
    }

    pub fn load_gguf(
        layer_idx: usize,
        gguf: &GgufWeights,
        intermediate_size: usize,
        _device: &candle_core::Device,
        dtype: candle_core::DType,
        activation: Activation,
    ) -> Result<Self> {
        let prefix = format!("blk.{}", layer_idx);

        let up_qt = gguf.get(&format!("{prefix}.ffn_up.weight"))?;
        let up_out = up_qt.shape().dims()[0];

        let down_qt = gguf.get(&format!("{prefix}.ffn_down.weight"))?;
        let down_in = down_qt.shape().dims()[1];
        if down_in != intermediate_size {
            candle_core::bail!(
                "GGUF ffn_down shape mismatch at {prefix}: expected dim1={}, got {}",
                intermediate_size,
                down_in
            );
        }

        let gate_up = if let Some(gate_qt) = gguf.try_get(&format!("{prefix}.ffn_gate.weight")) {
            let gate_out = gate_qt.shape().dims()[0];
            if gate_out != intermediate_size {
                candle_core::bail!(
                    "GGUF ffn_gate shape mismatch at {prefix}: expected dim0={}, got {}",
                    intermediate_size,
                    gate_out
                );
            }
            if up_out != intermediate_size {
                candle_core::bail!(
                    "GGUF ffn_up shape mismatch at {prefix}: expected dim0={}, got {}",
                    intermediate_size,
                    up_out
                );
            }
            let gate = QLinear::from_arc(gate_qt, dtype)?;
            let up = QLinear::from_arc(up_qt, dtype)?;
            GateUpProjection::Separate {
                gate: AnyLinear::Quantized(gate),
                up: AnyLinear::Quantized(up),
            }
        } else if up_out == 2 * intermediate_size {
            // Some GGUF variants (e.g. Phi-3) pack gate+up into ffn_up.
            let packed = QLinear::from_arc(up_qt, dtype)?;
            GateUpProjection::Packed(AnyLinear::Quantized(packed))
        } else if up_out == intermediate_size {
            let up = QLinear::from_arc(up_qt, dtype)?;
            GateUpProjection::Simple(AnyLinear::Quantized(up))
        } else {
            candle_core::bail!(
                "Unsupported GGUF FFN up-proj shape at {prefix}: dim0={} (expected {} or {})",
                up_out,
                intermediate_size,
                2 * intermediate_size
            );
        };
        let down_proj = QLinear::from_arc(down_qt, dtype)?;

        Ok(Self {
            gate_up,
            down_proj: AnyLinear::Quantized(down_proj),
            intermediate_size,
            activation,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = match &self.gate_up {
            GateUpProjection::Fused(gu) => {
                let out = gu.forward(x)?;
                // On Metal: fuse the activation+multiply into one kernel, avoiding two
                // separate encoder creations and an intermediate buffer. Both SiLU
                // (Llama/Qwen/Mistral) and GeLU-tanh (Gemma family) have fused kernels.
                #[cfg(feature = "metal")]
                if out.device().is_metal() {
                    match self.activation {
                        Activation::SiLU => {
                            let act =
                                super::metal_ops::gated_silu_fused(&out, self.intermediate_size)?;
                            return self.down_proj.forward(&act);
                        }
                        Activation::GeLUTanh => {
                            let act = super::metal_ops::gated_gelu_tanh_fused(
                                &out,
                                self.intermediate_size,
                            )?;
                            return self.down_proj.forward(&act);
                        }
                    }
                }
                let gate = out.narrow(D::Minus1, 0, self.intermediate_size)?;
                let up = out.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
                let activated = match self.activation {
                    Activation::SiLU => silu(&gate)?,
                    Activation::GeLUTanh => gelu_tanh(&gate)?,
                };
                (activated * up)?
            }
            GateUpProjection::Separate { gate: gp, up: up_p } => {
                let gate = gp.forward(x)?;
                let up = up_p.forward(x)?;
                #[cfg(feature = "metal")]
                if gate.device().is_metal() {
                    match self.activation {
                        Activation::SiLU => {
                            let act = super::metal_ops::silu_mul_fused(&gate, &up)?;
                            return self.down_proj.forward(&act);
                        }
                        Activation::GeLUTanh => {
                            let act = super::metal_ops::gelu_tanh_mul_fused(&gate, &up)?;
                            return self.down_proj.forward(&act);
                        }
                    }
                }
                let activated = match self.activation {
                    Activation::SiLU => silu(&gate)?,
                    Activation::GeLUTanh => gelu_tanh(&gate)?,
                };
                (activated * up)?
            }
            GateUpProjection::Packed(gu) => {
                let out = gu.forward(x)?;
                #[cfg(feature = "metal")]
                if out.device().is_metal() {
                    match self.activation {
                        Activation::SiLU => {
                            let act =
                                super::metal_ops::gated_silu_fused(&out, self.intermediate_size)?;
                            return self.down_proj.forward(&act);
                        }
                        Activation::GeLUTanh => {
                            let act = super::metal_ops::gated_gelu_tanh_fused(
                                &out,
                                self.intermediate_size,
                            )?;
                            return self.down_proj.forward(&act);
                        }
                    }
                }
                let gate = out.narrow(D::Minus1, 0, self.intermediate_size)?;
                let up = out.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
                let activated = match self.activation {
                    Activation::SiLU => silu(&gate)?,
                    Activation::GeLUTanh => gelu_tanh(&gate)?,
                };
                (activated * up)?
            }
            GateUpProjection::Simple(up) => {
                let up = up.forward(x)?;
                match self.activation {
                    Activation::SiLU => silu(&up)?,
                    Activation::GeLUTanh => gelu_tanh(&up)?,
                }
            }
        };
        self.down_proj.forward(&gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

    #[test]
    fn fused_fp8_gate_up_with_rank1_scales_matches_reference() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mut tensors = FxHashMap::default();

        let gate_w_fp8 =
            Tensor::from_vec(vec![1.0f32, -0.5, 0.25, 2.0, -1.5, 0.75], (3, 2), &device)?
                .to_dtype(DType::F8E4M3)?;
        let up_w_fp8 =
            Tensor::from_vec(vec![0.5f32, 1.5, -2.0, 0.25, 1.25, -0.75], (3, 2), &device)?
                .to_dtype(DType::F8E4M3)?;
        let gate_scale_rank1 = Tensor::from_vec(vec![0.5f32, 1.25, 2.0], (3,), &device)?;
        let up_scale_rank1 = Tensor::from_vec(vec![1.0f32, 0.25, 1.5], (3,), &device)?;
        let down_w = Tensor::from_vec(vec![0.25f32, -0.5, 1.0, 0.75, 0.1, -0.2], (2, 3), &device)?;

        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            gate_w_fp8.clone(),
        );
        tensors.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            up_w_fp8.clone(),
        );
        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight_scale_inv".to_string(),
            gate_scale_rank1.clone(),
        );
        tensors.insert(
            "model.layers.0.mlp.up_proj.weight_scale_inv".to_string(),
            up_scale_rank1.clone(),
        );
        tensors.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            down_w.clone(),
        );

        let weights = ModelWeights::from_tensors(tensors);
        let ffn = FeedForward::load(0, &weights, Activation::SiLU)?;

        let x = Tensor::from_vec(vec![1.0f32, -1.0, 0.5, 2.0], (2, 2), &device)?;
        let out = ffn.forward(&x)?;

        let gate_scale = gate_scale_rank1.reshape((3, 1))?;
        let up_scale = up_scale_rank1.reshape((3, 1))?;
        let gate_w = gate_w_fp8
            .to_dtype(DType::F32)?
            .broadcast_mul(&gate_scale)?;
        let up_w = up_w_fp8.to_dtype(DType::F32)?.broadcast_mul(&up_scale)?;
        let gate_up_w = Tensor::cat(&[&gate_w, &up_w], 0)?;

        let gate_up_out = AnyLinear::from_weight(gate_up_w, None)?.forward(&x)?;
        let gate = gate_up_out.narrow(D::Minus1, 0, 3)?;
        let up = gate_up_out.narrow(D::Minus1, 3, 3)?;
        let gated = (silu(&gate)? * up)?;
        let expected = AnyLinear::from_weight(down_w, None)?.forward(&gated)?;

        let out_vals = out.to_vec2::<f32>()?;
        let expected_vals = expected.to_vec2::<f32>()?;

        for (out_row, exp_row) in out_vals.iter().zip(expected_vals.iter()) {
            for (o, e) in out_row.iter().zip(exp_row.iter()) {
                assert!((o - e).abs() < 1e-5, "o={o}, e={e}");
            }
        }

        Ok(())
    }
}
