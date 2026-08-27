//! Additive causal attention masks, with a small per-thread LRU cache.
//!
//! A mask is an additive bias added to the attention logits before softmax:
//! allowed positions get `0.0`, disallowed (future) positions get a large
//! negative value (`-1e30`). [`causal_mask`] builds a square mask;
//! [`causal_mask_prefixed`] builds the rectangular query-by-kv mask used when
//! part of the sequence is already in the KV cache. The `*_cached_dtype`
//! variants memoise masks by shape, device, and dtype to avoid rebuilding them
//! every step.

use candle_core::{DType, Device, Result, Tensor};
use lru::LruCache;
use std::cell::RefCell;
use std::num::NonZeroUsize;

const MASK_CACHE_CAP: usize = 32;

/// Builds a `[1, 1, seq_len, seq_len]` additive causal mask on `device`.
///
/// Entry `(i, j)` is `0.0` when key `j` is at or before query `i`, and `-1e30`
/// otherwise, so a query attends only to itself and earlier positions. The step
/// is produced with a steep tanh rather than a boolean compare to stay in plain
/// tensor ops.
///
/// ## Errors
/// Propagates tensor allocation failures.
pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let row: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let col: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let rows = Tensor::from_vec(row, (seq_len, 1), device)?;
    let cols = Tensor::from_vec(col, (1, seq_len), device)?;
    let diff = cols.broadcast_sub(&rows)?;
    let step = diff.affine(1000.0, -500.0)?.tanh()?;
    let upper = step.affine(0.5, 0.5)?;
    upper.affine(-1e30, 0.0)?.reshape((1, 1, seq_len, seq_len))
}

/// Builds a `[1, 1, query_len, kv_len]` additive causal mask where the first
/// `kv_len - query_len` keys are an already-cached prefix.
///
/// Query `i` sits at absolute position `prefix_len + i` and attends to every key
/// up to and including its own position; later keys get `-1e30`. Used for
/// prefill segments that extend a cached prefix.
///
/// ## Panics
/// Debug-asserts `kv_len >= query_len`.
///
/// ## Errors
/// Propagates tensor allocation failures.
pub fn causal_mask_prefixed(query_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
    debug_assert!(kv_len >= query_len);
    let prefix_len = kv_len.saturating_sub(query_len);

    let row: Vec<f32> = (0..query_len).map(|i| (prefix_len + i) as f32).collect();
    let col: Vec<f32> = (0..kv_len).map(|i| i as f32).collect();
    let rows = Tensor::from_vec(row, (query_len, 1), device)?;
    let cols = Tensor::from_vec(col, (1, kv_len), device)?;
    let diff = cols.broadcast_sub(&rows)?;
    let step = diff.affine(1000.0, -500.0)?.tanh()?;
    let upper = step.affine(0.5, 0.5)?;
    upper.affine(-1e30, 0.0)?.reshape((1, 1, query_len, kv_len))
}

/// Causal mask that also closes a sliding window: query `i` sits at absolute
/// position `prefix_len + i` and attends to itself and the `window - 1` keys
/// before it, nothing older and nothing later.
///
/// The paths that mask inside their own kernel do not need this; the generic
/// attention does, and without it a windowed layer read the whole prefix at
/// prefill, which is not what the checkpoint describes.
///
/// ## Panics
/// Debug-asserts `kv_len >= query_len` and a non-zero window.
///
/// ## Errors
/// Propagates tensor allocation failures.
pub fn sliding_window_mask_prefixed(
    query_len: usize,
    kv_len: usize,
    window: usize,
    device: &Device,
) -> Result<Tensor> {
    debug_assert!(kv_len >= query_len);
    debug_assert!(window > 0);
    let prefix_len = kv_len.saturating_sub(query_len);

    let row: Vec<f32> = (0..query_len).map(|i| (prefix_len + i) as f32).collect();
    let col: Vec<f32> = (0..kv_len).map(|i| i as f32).collect();
    let rows = Tensor::from_vec(row, (query_len, 1), device)?;
    let cols = Tensor::from_vec(col, (1, kv_len), device)?;
    let diff = cols.broadcast_sub(&rows)?;
    let later = diff.affine(1000.0, -500.0)?.tanh()?.affine(0.5, 0.5)?;
    let older = diff
        .affine(-1000.0, -1000.0 * window as f64 + 500.0)?
        .tanh()?
        .affine(0.5, 0.5)?;
    later
        .maximum(&older)?
        .affine(-1e30, 0.0)?
        .reshape((1, 1, query_len, kv_len))
}

fn device_discriminant(device: &Device) -> u8 {
    if device.is_cpu() {
        0
    } else if device.is_metal() {
        1
    } else {
        2
    }
}

fn dtype_discriminant(dtype: DType) -> u8 {
    match dtype {
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::F32 => 3,
        DType::F64 => 4,
        _ => 0,
    }
}

/// `(query_len, kv_len, window, device, dtype)`.
type WindowMaskKey = (usize, usize, usize, u8, u8);

thread_local! {
    static MASK_CACHE: RefCell<LruCache<(usize, u8, u8), Tensor>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(MASK_CACHE_CAP).unwrap()));
    static PREFIX_MASK_CACHE: RefCell<LruCache<(usize, usize, u8, u8), Tensor>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(MASK_CACHE_CAP).unwrap()));
    static WINDOW_MASK_CACHE: RefCell<LruCache<WindowMaskKey, Tensor>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(MASK_CACHE_CAP).unwrap()));
}

/// Returns a windowed causal mask in `dtype`, reusing a cached tensor when one
/// exists for this `(query_len, kv_len, window, device, dtype)`.
///
/// ## Errors
/// Propagates failures from building or dtype-casting the mask.
pub fn sliding_window_mask_cached_dtype(
    query_len: usize,
    kv_len: usize,
    window: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let key = (
        query_len,
        kv_len,
        window,
        device_discriminant(device),
        dtype_discriminant(dtype),
    );
    WINDOW_MASK_CACHE.with(|cache| {
        let mut map = cache.borrow_mut();
        if let Some(t) = map.get(&key) {
            return Ok(t.clone());
        }
        let mut mask = sliding_window_mask_prefixed(query_len, kv_len, window, device)?;
        if mask.dtype() != dtype {
            mask = mask.to_dtype(dtype)?;
        }
        map.push(key, mask.clone());
        Ok(mask)
    })
}

/// Returns a causal mask of `seq_len` in `dtype`, reusing a cached tensor when
/// one already exists for this `(seq_len, device, dtype)`.
///
/// Backed by a per-thread LRU cache; the first call for a key builds the mask
/// (see [`causal_mask`]) and caches a clone.
///
/// ## Errors
/// Propagates failures from building or dtype-casting the mask.
pub fn causal_mask_cached_dtype(seq_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let key = (
        seq_len,
        device_discriminant(device),
        dtype_discriminant(dtype),
    );
    MASK_CACHE.with(|cache| {
        let mut map = cache.borrow_mut();
        if let Some(t) = map.get(&key) {
            return Ok(t.clone());
        }
        let mut mask = causal_mask(seq_len, device)?;
        if mask.dtype() != dtype {
            mask = mask.to_dtype(dtype)?;
        }
        map.push(key, mask.clone());
        Ok(mask)
    })
}

/// Returns a prefixed causal mask in `dtype`, reusing a cached tensor when one
/// exists for this `(query_len, kv_len, device, dtype)`.
///
/// The per-thread LRU analogue of [`causal_mask_cached_dtype`] for the
/// query-by-kv prefix mask (see [`causal_mask_prefixed`]).
///
/// ## Errors
/// Propagates failures from building or dtype-casting the mask.
pub fn causal_mask_prefixed_cached_dtype(
    query_len: usize,
    kv_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let key = (
        query_len,
        kv_len,
        device_discriminant(device),
        dtype_discriminant(dtype),
    );
    PREFIX_MASK_CACHE.with(|cache| {
        let mut map = cache.borrow_mut();
        if let Some(t) = map.get(&key) {
            return Ok(t.clone());
        }
        let mut mask = causal_mask_prefixed(query_len, kv_len, device)?;
        if mask.dtype() != dtype {
            mask = mask.to_dtype(dtype)?;
        }
        map.push(key, mask.clone());
        Ok(mask)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Contract: the windowed mask opens exactly the keys a token may read and
    /// closes every other one, at both edges.
    ///
    /// A token sees itself and the `window - 1` keys before it. Off by one at
    /// the old edge is the mistake this catches, and it is invisible in an
    /// average: it drops or admits a single key per row.
    #[test]
    fn the_windowed_mask_opens_exactly_the_window() {
        let dev = Device::Cpu;
        for (query_len, kv_len, window) in
            [(4usize, 10usize, 3usize), (1, 5, 5), (6, 6, 2), (3, 9, 100)]
        {
            let m = sliding_window_mask_prefixed(query_len, kv_len, window, &dev).unwrap();
            let flat: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
            let prefix = kv_len - query_len;
            for i in 0..query_len {
                let q_pos = prefix + i;
                for j in 0..kv_len {
                    let open = j <= q_pos && q_pos < j + window;
                    let got = flat[i * kv_len + j];
                    assert_eq!(
                        got == 0.0,
                        open,
                        "query {query_len}x{kv_len} w={window}: key {j} for a query at {q_pos} \
                         should be {}, mask reads {got}",
                        if open { "open" } else { "closed" }
                    );
                }
            }
        }
    }

    /// Contract: with a window at least as wide as the keys, the windowed mask
    /// is the plain causal one.
    #[test]
    fn a_window_wider_than_the_keys_is_just_causal() {
        let dev = Device::Cpu;
        let (q, kv) = (5usize, 12usize);
        let windowed = sliding_window_mask_prefixed(q, kv, kv, &dev).unwrap();
        let causal = causal_mask_prefixed(q, kv, &dev).unwrap();
        let a: Vec<f32> = windowed.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = causal.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a, b);
    }
}
