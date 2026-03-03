use candle_core::{Result, Tensor};
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
    activation: Activation,
}

impl FeedForward {
    pub fn load(cfg: &BlockConfig, layer_idx: usize, weights: &ModelWeights) -> Result<Self> {
        let p = format!("model.layers.{}.mlp", layer_idx);
        let gate_proj = Linear::new(weights.get(&format!("{}.gate_proj.weight", p))?.clone(), None);
        let up_proj   = Linear::new(weights.get(&format!("{}.up_proj.weight",   p))?.clone(), None);
        let down_proj = Linear::new(weights.get(&format!("{}.down_proj.weight", p))?.clone(), None);
        Ok(Self { gate_proj, up_proj, down_proj, activation: cfg.activation })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = match self.activation {
            Activation::Silu => silu(&gate)?,
        };
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}