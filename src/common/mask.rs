use std::cell::RefCell;
use std::num::NonZeroUsize;
use candle_core::{Device, Result, Tensor};
use lru::LruCache;

const MASK_CACHE_CAP: usize = 32;

pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let row: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let col: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let rows = Tensor::from_vec(row, (seq_len, 1), device)?;
    let cols = Tensor::from_vec(col, (1, seq_len), device)?;
    let diff  = cols.broadcast_sub(&rows)?;
    let step  = diff.affine(1000.0, -500.0)?.tanh()?;
    let upper = step.affine(0.5, 0.5)?;
    upper.affine(-1e30, 0.0)?.reshape((1, 1, seq_len, seq_len))
}

thread_local! {
    static MASK_CACHE: RefCell<LruCache<usize, Tensor>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(MASK_CACHE_CAP).unwrap()));
}

pub fn causal_mask_cached(seq_len: usize, device: &Device) -> Result<Tensor> {
    MASK_CACHE.with(|cache| {
        let mut map = cache.borrow_mut();
        if let Some(t) = map.get(&seq_len) {
            return Ok(t.clone());
        }
        let mask = causal_mask(seq_len, device)?;
        map.push(seq_len, mask.clone());
        Ok(mask)
    })
}
