use candle_core::{DType, Device, Result, Tensor};

pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len * seq_len)
        .map(|i| {
            let (row, col) = (i / seq_len, i % seq_len);
            if col > row { f32::NEG_INFINITY } else { 0.0f32 }
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?.to_dtype(DType::F32)
}
