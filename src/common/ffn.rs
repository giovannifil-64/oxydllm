use super::config::Activation;
use super::gguf_weights::GgufWeights;
use super::linear::{AnyLinear, Linear, QLinear, gelu_tanh, silu};
use super::weights::ModelWeights;
use candle_core::{D, Result, Tensor};

enum GateUpProjection {
    Fused(Linear),
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
        let down_w = weights.get(&format!("{}.down_proj.weight", p))?.clone();
        let down_proj = Linear::new(down_w.clone(), None);
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
            GateUpProjection::Fused(Linear::new(gate_up_w, None))
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
            GateUpProjection::Fused(Linear::new(gate_up_w, None))
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
            GateUpProjection::Simple(AnyLinear::Float(Linear::new(up_w, None)))
        } else {
            candle_core::bail!(
                "Unsupported FFN layout at {}: expected gate_proj+up_proj, gate_up_proj, or up_proj",
                p
            );
        };

        Ok(Self {
            gate_up,
            down_proj: AnyLinear::Float(down_proj),
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
                let activated = match self.activation {
                    Activation::SiLU => silu(&gate)?,
                    Activation::GeLUTanh => gelu_tanh(&gate)?,
                };
                (activated * up)?
            }
            GateUpProjection::Packed(gu) => {
                let out = gu.forward(x)?;
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
