use candle_core::{Result, Tensor, D};
use super::gguf_weights::GgufWeights;
use super::linear::{gelu_tanh, silu, AnyLinear, Linear, QLinear};
use super::weights::ModelWeights;
use super::config::Activation;

enum GateUpProjection {
    Fused(Linear),
    Separate { gate: AnyLinear, up: AnyLinear },
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
        let gate_w = weights.get(&format!("{}.gate_proj.weight", p))?.clone();
        let up_w   = weights.get(&format!("{}.up_proj.weight",   p))?.clone();
        let down_proj = Linear::new(weights.get(&format!("{}.down_proj.weight", p))?.clone(), None);

        let intermediate_size = gate_w.dim(0)?;
        // Fuse gate+up into one matrix; individual weight tensors are not retained.
        let gate_up_w = Tensor::cat(&[&gate_w, &up_w], 0)?;

        Ok(Self {
            gate_up: GateUpProjection::Fused(Linear::new(gate_up_w, None)),
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

        let gate = QLinear::from_arc(gguf.get(&format!("{prefix}.ffn_gate.weight"))?, dtype)?;
        let up   = QLinear::from_arc(gguf.get(&format!("{prefix}.ffn_up.weight"))?,   dtype)?;
        let down_proj = QLinear::from_arc(gguf.get(&format!("{prefix}.ffn_down.weight"))?, dtype)?;

        Ok(Self {
            gate_up: GateUpProjection::Separate {
                gate: AnyLinear::Quantized(gate),
                up:   AnyLinear::Quantized(up),
            },
            down_proj: AnyLinear::Quantized(down_proj),
            intermediate_size,
            activation,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (gate, up) = match &self.gate_up {
            GateUpProjection::Fused(gu) => {
                let out = gu.forward(x)?;
                let gate = out.narrow(D::Minus1, 0, self.intermediate_size)?;
                let up   = out.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
                (gate, up)
            }
            GateUpProjection::Separate { gate: gp, up: up_p } => {
                (gp.forward(x)?, up_p.forward(x)?)
            }
        };
        let activated = match self.activation {
            Activation::SiLU => silu(&gate)?,
            Activation::GeLUTanh => gelu_tanh(&gate)?,
        };
        self.down_proj.forward(&(activated * up)?)
    }
}