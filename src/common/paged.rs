use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result, Tensor};

pub const DEFAULT_BLOCK_SIZE: usize = 16;

pub struct BlockAllocator {
    pool_k: Tensor,
    pool_v: Tensor,
    free_list: Vec<usize>,
    ref_counts: Vec<u32>,
    num_blocks: usize,
    block_size: usize,
}

impl BlockAllocator {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let total_slots = num_blocks * block_size;
        let pool_k = Tensor::zeros((total_slots, n_kv_heads, head_dim), dtype, device)?;
        let pool_v = Tensor::zeros((total_slots, n_kv_heads, head_dim), dtype, device)?;
        let free_list = (0..num_blocks).rev().collect();
        let ref_counts = vec![0u32; num_blocks];
        Ok(Self { pool_k, pool_v, free_list, ref_counts, num_blocks, block_size })
    }

    pub fn allocate(&mut self) -> Result<usize> {
        let id = self.free_list.pop().ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "KV cache memory exhausted: all {} blocks allocated",
                self.num_blocks,
            ))
        })?;
        self.ref_counts[id] = 1;
        Ok(id)
    }

    pub fn share(&mut self, block_id: usize) {
        debug_assert!(block_id < self.num_blocks, "invalid block_id {block_id}");
        debug_assert!(self.ref_counts[block_id] > 0, "share on un-allocated block {block_id}");
        self.ref_counts[block_id] += 1;
    }

    pub fn free(&mut self, block_id: usize) {
        debug_assert!(block_id < self.num_blocks, "invalid block_id {block_id}");
        debug_assert!(self.ref_counts[block_id] > 0, "double-free of block {block_id}");
        self.ref_counts[block_id] -= 1;
        if self.ref_counts[block_id] == 0 {
            self.free_list.push(block_id);
        }
    }

    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn write(
        &self,
        block_id: usize,
        offset: usize,
        data_k: &Tensor,
        data_v: &Tensor,
    ) -> Result<()> {
        let start = block_id * self.block_size + offset;
        self.pool_k.slice_set(data_k, 0, start)?;
        self.pool_v.slice_set(data_v, 0, start)?;
        Ok(())
    }

    pub fn gather(&self, slot_indices: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = self.pool_k.index_select(slot_indices, 0)?;
        let v = self.pool_v.index_select(slot_indices, 0)?;
        let k = k.transpose(0, 1)?.unsqueeze(0)?;
        let v = v.transpose(0, 1)?.unsqueeze(0)?;
        Ok((k, v))
    }
}


pub type SharedBlockAllocator = Arc<Mutex<BlockAllocator>>;

pub struct BlockTable {
    pub block_ids: Vec<usize>,
    pub num_tokens: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self { block_ids: Vec::new(), num_tokens: 0 }
    }

    pub fn slot_indices(&self, block_size: usize) -> Vec<u32> {
        let mut indices = Vec::with_capacity(self.num_tokens);
        let full_blocks = self.num_tokens / block_size;
        let remainder = self.num_tokens % block_size;

        for i in 0..full_blocks {
            let base = self.block_ids[i] * block_size;
            for off in 0..block_size {
                indices.push((base + off) as u32);
            }
        }
        if remainder > 0 {
            let base = self.block_ids[full_blocks] * block_size;
            for off in 0..remainder {
                indices.push((base + off) as u32);
            }
        }
        indices
    }
}

pub struct PagedKvCache {
    allocator: SharedBlockAllocator,
    table: BlockTable,
}

impl PagedKvCache {
    pub fn new(allocator: SharedBlockAllocator) -> Self {
        Self { allocator, table: BlockTable::new() }
    }

    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, new_seq, _) = new_k.dims4()?;
        let k_flat = new_k.squeeze(0)?.transpose(0, 1)?;
        let v_flat = new_v.squeeze(0)?.transpose(0, 1)?;
        let block_size = self.allocator.lock().unwrap().block_size();

        let mut written = 0;
        
        while written < new_seq {
            let current_offset = self.table.num_tokens % block_size;

            if current_offset == 0 {
                let block_id = self.allocator.lock().unwrap().allocate()?;
                self.table.block_ids.push(block_id);
            }

            let space = block_size - current_offset;
            let n = (new_seq - written).min(space);
            let k_chunk = k_flat.narrow(0, written, n)?.contiguous()?;
            let v_chunk = v_flat.narrow(0, written, n)?.contiguous()?;
            let block_id = *self.table.block_ids.last().unwrap();

            self.allocator
                .lock().unwrap()
                .write(block_id, current_offset, &k_chunk, &v_chunk)?;

            self.table.num_tokens += n;
            written += n;
        }

        let slots = self.table.slot_indices(block_size);
        let idx = Tensor::from_vec(slots, (self.table.num_tokens,), new_k.device())?;

        self.allocator.lock().unwrap().gather(&idx)
    }

    pub fn clear(&mut self) {
        for &bid in &self.table.block_ids {
            self.allocator.lock().unwrap().free(bid);
        }
        self.table.block_ids.clear();
        self.table.num_tokens = 0;
    }

    pub fn prepopulate_block(&mut self, block_id: usize) {
        self.allocator.lock().unwrap().share(block_id);
        self.table.block_ids.push(block_id);
    }

    pub fn set_num_tokens(&mut self, n: usize) {
        self.table.num_tokens = n;
    }

    #[allow(dead_code)]
    pub fn num_tokens(&self) -> usize {
        self.table.num_tokens
    }

    pub fn block_id_at(&self, idx: usize) -> Option<usize> {
        self.table.block_ids.get(idx).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn make_allocator(num_blocks: usize, block_size: usize) -> SharedBlockAllocator {
        Arc::new(Mutex::new(
            BlockAllocator::new(num_blocks, block_size, 2, 4, DType::F32, &Device::Cpu)
                .unwrap(),
        ))
    }

    #[test]
    fn allocator_alloc_free() {
        let alloc = make_allocator(4, 2);
        assert_eq!(alloc.lock().unwrap().num_free(), 4);

        let b0 = alloc.lock().unwrap().allocate().unwrap();
        let b1 = alloc.lock().unwrap().allocate().unwrap();
        let b2 = alloc.lock().unwrap().allocate().unwrap();
        let b3 = alloc.lock().unwrap().allocate().unwrap();
        assert_eq!(alloc.lock().unwrap().num_free(), 0);
        assert!(alloc.lock().unwrap().allocate().is_err());

        alloc.lock().unwrap().free(b1);
        assert_eq!(alloc.lock().unwrap().num_free(), 1);
        let b1_again = alloc.lock().unwrap().allocate().unwrap();
        assert_eq!(b1_again, b1);

        alloc.lock().unwrap().free(b0);
        alloc.lock().unwrap().free(b1_again);
        alloc.lock().unwrap().free(b2);
        alloc.lock().unwrap().free(b3);
        assert_eq!(alloc.lock().unwrap().num_free(), 4);
    }

    #[test]
    fn paged_cache_matches_naive_cat() {
        // Use tiny dimensions: n_kv_heads=2, head_dim=4, block_size=2
        let alloc = make_allocator(8, 2);
        let mut cache = PagedKvCache::new(alloc);
        let dev = Device::Cpu;

        // Simulate prefill: 5 tokens
        let k1 = Tensor::randn(0f32, 1., (1, 2, 5, 4), &dev).unwrap();
        let v1 = Tensor::randn(0f32, 1., (1, 2, 5, 4), &dev).unwrap();
        let (k_out, v_out) = cache.append(&k1, &v1).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 5, 4]);
        assert_eq!(v_out.dims(), &[1, 2, 5, 4]);

        // Verify data matches: gather should reproduce the original
        let k1_gathered = k_out.squeeze(0).unwrap().transpose(0, 1).unwrap(); // (5,2,4)
        let k1_flat = k1.squeeze(0).unwrap().transpose(0, 1).unwrap(); // (5,2,4)
        let diff = (k1_gathered - k1_flat).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-6, "prefill K mismatch: diff={diff}");

        // Simulate decode: 3 single-token appends
        let mut naive_k = k1.clone();
        let mut naive_v = v1.clone();
        for _ in 0..3 {
            let k_new = Tensor::randn(0f32, 1., (1, 2, 1, 4), &dev).unwrap();
            let v_new = Tensor::randn(0f32, 1., (1, 2, 1, 4), &dev).unwrap();
            let (k_paged, v_paged) = cache.append(&k_new, &v_new).unwrap();

            naive_k = Tensor::cat(&[&naive_k, &k_new], 2).unwrap();
            naive_v = Tensor::cat(&[&naive_v, &v_new], 2).unwrap();

            let dk = (&k_paged - &naive_k).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            let dv = (&v_paged - &naive_v).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(dk < 1e-6, "decode K mismatch: diff={dk}");
            assert!(dv < 1e-6, "decode V mismatch: diff={dv}");
        }
        // After prefill(5) + decode(3) = 8 tokens, 4 blocks of size 2
        assert_eq!(k_out.device().location(), dev.location());
    }

    #[test]
    fn clear_returns_blocks() {
        let alloc = make_allocator(4, 2);
        let mut cache = PagedKvCache::new(Arc::clone(&alloc));
        let dev = Device::Cpu;

        // Fill 4 tokens → 2 blocks
        let k = Tensor::zeros((1, 2, 4, 4), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 2, 4, 4), DType::F32, &dev).unwrap();
        cache.append(&k, &v).unwrap();
        assert_eq!(alloc.lock().unwrap().num_free(), 2); // 4 total - 2 used

        cache.clear();
        assert_eq!(alloc.lock().unwrap().num_free(), 4); // all returned
    }

    #[test]
    fn exhaustion_error() {
        // Only 2 blocks of size 2 → max 4 tokens
        let alloc = make_allocator(2, 2);
        let mut cache = PagedKvCache::new(alloc);
        let dev = Device::Cpu;

        let k = Tensor::zeros((1, 2, 4, 4), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 2, 4, 4), DType::F32, &dev).unwrap();
        cache.append(&k, &v).unwrap(); // fills both blocks

        // Next token should fail
        let k1 = Tensor::zeros((1, 2, 1, 4), DType::F32, &dev).unwrap();
        let v1 = Tensor::zeros((1, 2, 1, 4), DType::F32, &dev).unwrap();
        assert!(cache.append(&k1, &v1).is_err());
    }
}
