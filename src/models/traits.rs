use crate::common::paged::{PagedKvCache, SharedBlockAllocator};
use candle_core::{Device, Result, Tensor};

pub trait BatchModel {
    fn forward_batch(
        &self,
        token_ids: &Tensor,
        position_ids: &Tensor,
        seq_caches: &mut [&mut [PagedKvCache]],
        token_counts: &[usize],
    ) -> Result<Tensor>;

    /// Logits for the last token of each sequence, one row per sequence.
    ///
    /// The engine samples exactly one row per sequence, so a model that can
    /// project only those rows should: running the head over a whole prompt
    /// allocates `tokens x vocab` values, which for a long prompt is gigabytes
    /// read once. The default projects everything and then selects, which is
    /// correct but keeps the allocation.
    ///
    /// ## Errors
    ///
    /// Propagates the forward's own failures.
    fn forward_batch_last(
        &self,
        token_ids: &Tensor,
        position_ids: &Tensor,
        seq_caches: &mut [&mut [PagedKvCache]],
        token_counts: &[usize],
    ) -> Result<Tensor> {
        let full = self.forward_batch(token_ids, position_ids, seq_caches, token_counts)?;
        let mut end = 0u32;
        let idx: Vec<u32> = token_counts
            .iter()
            .map(|&n| {
                end += n as u32;
                end - 1
            })
            .collect();
        let idx = Tensor::from_vec(idx, (token_counts.len(),), full.device())?;
        full.index_select(&idx, full.rank() - 2)
    }

    fn vocab_size(&self) -> usize;
    fn stop_token_ids(&self) -> &[u32];
    fn device(&self) -> &Device;
    fn num_layers(&self) -> usize;

    fn allocators(&self) -> &[SharedBlockAllocator];

    /// A number folded from one small resident weight, for checking that the
    /// weights a load produced are the weights it read.
    ///
    /// Weight uploads and the KV pool allocation are both queued work on the
    /// same device, and a failure in the first is not always reported: a load
    /// can report success while a buffer's writes never landed, after which the
    /// first forward reads whatever the buffer held before. Comparing this
    /// before and after the pool exists catches that, which is what lets the
    /// loader size the pool by trying rather than by a constant fitted on one
    /// machine. `None` when the model exposes no suitable tensor.
    fn weight_fingerprint(&self) -> Option<f64> {
        None
    }

    /// True for hybrid models whose linear-attention layers carry per-sequence
    /// recurrent state. Such state cannot skip tokens (prefix cache) or roll
    /// back (speculative decoding), so the engine disables both.
    fn has_recurrent_state(&self) -> bool {
        false
    }

    /// Returns the total bytes allocated for KV caches across all layers.
    /// Hybrid models alias one allocator across their linear layers; count
    /// each distinct pool once.
    fn kv_cache_bytes(&self) -> usize {
        let mut seen: Vec<*const std::sync::Mutex<crate::common::paged::BlockAllocator>> =
            Vec::new();
        self.allocators()
            .iter()
            .filter(|a| {
                let ptr = std::sync::Arc::as_ptr(a);
                if seen.contains(&ptr) {
                    false
                } else {
                    seen.push(ptr);
                    true
                }
            })
            .map(|a| a.lock().unwrap().pool_bytes())
            .sum()
    }
}
