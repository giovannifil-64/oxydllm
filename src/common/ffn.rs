use candle_core::{Result, Tensor, D};
use super::config::BlockConfig;
use super::linear::{silu, Linear};
use super::weights::ModelWeights;

#[derive(Clone, Copy)]
pub enum Activation {
    Silu,
}

pub struct FeedForward {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    gate_up_proj: Option<Linear>,
    intermediate_size: usize,
    activation: Activation,
}

impl FeedForward {
    pub fn load(cfg: &BlockConfig, layer_idx: usize, weights: &ModelWeights) -> Result<Self> {
        let p = format!("model.layers.{}.mlp", layer_idx);
        let gate_proj = Linear::new(weights.get(&format!("{}.gate_proj.weight", p))?.clone(), None);
        let up_proj   = Linear::new(weights.get(&format!("{}.up_proj.weight",   p))?.clone(), None);
        let down_proj = Linear::new(weights.get(&format!("{}.down_proj.weight", p))?.clone(), None);
        let intermediate_size = gate_proj.weight().dim(0)?;
        let gate_up_w = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?;
        let gate_up_proj = Some(Linear::new(gate_up_w, None));

        Ok(Self { gate_proj, up_proj, down_proj, gate_up_proj, intermediate_size, activation: cfg.activation })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(ref gu) = self.gate_up_proj {
            let out = gu.forward(x)?; // [..., 2*intermediate_size]
            let gate = out.narrow(D::Minus1, 0, self.intermediate_size)?;
            let up   = out.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
            let gate = match self.activation {
                Activation::Silu => silu(&gate)?,
            };
            return self.down_proj.forward(&(gate * up)?);
        }
        let gate = self.gate_proj.forward(x)?;
        let gate = match self.activation {
            Activation::Silu => silu(&gate)?,
        };
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}