use std::cell::RefCell;
use std::num::NonZeroUsize;
use candle_core::{DType, Device, Result, Tensor};
use lru::LruCache;

const MASK_CACHE_CAP: usize = 32;

pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len * seq_len)
        .map(|i| {
            let (row, col) = (i / seq_len, i % seq_len);
            if col > row { f32::NEG_INFINITY } else { 0.0f32 }
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?.to_dtype(DType::F32)
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
