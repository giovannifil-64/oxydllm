use candle_core::{Result, Tensor, D};
use super::gguf_weights::GgufWeights;
use super::linear::{silu, AnyLinear, Linear, QLinear};
use super::weights::ModelWeights;

pub struct FeedForward {
    gate_proj: AnyLinear,
    up_proj: AnyLinear,
    down_proj: AnyLinear,
    gate_up_proj: Option<Linear>,
    intermediate_size: usize,
}

impl FeedForward {
    pub fn load(layer_idx: usize, weights: &ModelWeights) -> Result<Self> {
        let p = format!("model.layers.{}.mlp", layer_idx);
        let gate_proj = Linear::new(weights.get(&format!("{}.gate_proj.weight", p))?.clone(), None);
        let up_proj   = Linear::new(weights.get(&format!("{}.up_proj.weight",   p))?.clone(), None);
        let down_proj = Linear::new(weights.get(&format!("{}.down_proj.weight", p))?.clone(), None);
        let intermediate_size = gate_proj.weight().dim(0)?;
        let gate_up_w = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?;
        let gate_up_proj = Some(Linear::new(gate_up_w, None));

        Ok(Self {
            gate_proj: AnyLinear::Float(gate_proj),
            up_proj: AnyLinear::Float(up_proj),
            down_proj: AnyLinear::Float(down_proj),
            gate_up_proj,
            intermediate_size,
        })
    }

    pub fn load_gguf(
        layer_idx: usize,
        gguf: &GgufWeights,
        intermediate_size: usize,
        _device: &candle_core::Device,
        dtype: candle_core::DType,
    ) -> Result<Self> {
        let prefix = format!("blk.{}", layer_idx);

        let gate_proj = QLinear::from_arc(gguf.get(&format!("{prefix}.ffn_gate.weight"))?, dtype)?;
        let up_proj   = QLinear::from_arc(gguf.get(&format!("{prefix}.ffn_up.weight"))?,   dtype)?;
        let down_proj = QLinear::from_arc(gguf.get(&format!("{prefix}.ffn_down.weight"))?, dtype)?;

        Ok(Self {
            gate_proj: AnyLinear::Quantized(gate_proj),
            up_proj: AnyLinear::Quantized(up_proj),
            down_proj: AnyLinear::Quantized(down_proj),
            gate_up_proj: None,
            intermediate_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(ref gu) = self.gate_up_proj {
            let out = gu.forward(x)?; // [..., 2*intermediate_size]
            let gate = out.narrow(D::Minus1, 0, self.intermediate_size)?;
            let up   = out.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
            let gate = silu(&gate)?;
            return self.down_proj.forward(&(gate * up)?);
        }
        let gate = self.gate_proj.forward(x)?;
        let gate = silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}