use candle_core::{Result, Tensor, D};
use super::gguf_weights::GgufWeights;
use super::linear::{silu, AnyLinear, Linear, QLinear};
use super::weights::ModelWeights;

pub struct FeedForward {
    /// Fused gate+up projection (safetensors). When Some, gate_proj/up_proj are None.
    gate_up_proj: Option<Linear>,
    /// Separate projections (GGUF). When Some, gate_up_proj is None.
    gate_proj: Option<AnyLinear>,
    up_proj: Option<AnyLinear>,
    down_proj: AnyLinear,
    intermediate_size: usize,
}

impl FeedForward {
    pub fn load(layer_idx: usize, weights: &ModelWeights) -> Result<Self> {
        let p = format!("model.layers.{}.mlp", layer_idx);
        let gate_w = weights.get(&format!("{}.gate_proj.weight", p))?.clone();
        let up_w   = weights.get(&format!("{}.up_proj.weight",   p))?.clone();
        let down_proj = Linear::new(weights.get(&format!("{}.down_proj.weight", p))?.clone(), None);

        let intermediate_size = gate_w.dim(0)?;
        // Fuse gate+up into one matrix; individual weight tensors are not retained.
        let gate_up_w = Tensor::cat(&[&gate_w, &up_w], 0)?;

        Ok(Self {
            gate_up_proj: Some(Linear::new(gate_up_w, None)),
            gate_proj: None,
            up_proj: None,
            down_proj: AnyLinear::Float(down_proj),
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
            gate_up_proj: None,
            gate_proj: Some(AnyLinear::Quantized(gate_proj)),
            up_proj: Some(AnyLinear::Quantized(up_proj)),
            down_proj: AnyLinear::Quantized(down_proj),
            intermediate_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(ref gu) = self.gate_up_proj {
            let out = gu.forward(x)?;
            let gate = out.narrow(D::Minus1, 0, self.intermediate_size)?;
            let up   = out.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
            return self.down_proj.forward(&(silu(&gate)? * up)?);
        }
        let gate = silu(&self.gate_proj.as_ref().expect("gate_proj").forward(x)?)?;
        let up   = self.up_proj.as_ref().expect("up_proj").forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}