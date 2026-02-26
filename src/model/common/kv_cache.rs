use candle_core::{Result, Tensor};

pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = match &self.k {
            Some(prev) => Tensor::cat(&[prev, new_k], 2)?,
            None => new_k.clone(),
        };
        let v = match &self.v {
            Some(prev) => Tensor::cat(&[prev, new_v], 2)?,
            None => new_v.clone(),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }
}
