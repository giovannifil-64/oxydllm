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

    fn vocab_size(&self) -> usize;
    fn stop_token_ids(&self) -> &[u32];
    fn max_seq_len(&self) -> usize;
    fn device(&self) -> &Device;
    fn num_layers(&self) -> usize;

    fn allocators(&self) -> &[SharedBlockAllocator];

    /// Returns the total bytes allocated for KV caches across all layers.
    fn kv_cache_bytes(&self) -> usize {
        self.allocators()
            .iter()
            .map(|a| {
                let alloc = a.lock().unwrap();
                alloc.pool_bytes()
            })
            .sum()
    }
}
