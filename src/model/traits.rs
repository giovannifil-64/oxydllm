use candle_core::{Device, Result, Tensor};
use super::common::paged::{PagedKvCache, SharedBlockAllocator};

pub trait BatchModel {
    fn forward_with_cache(
        &self,
        tokens: &Tensor,
        start_pos: usize,
        caches: &mut [PagedKvCache],
    ) -> Result<Tensor>;

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
}
