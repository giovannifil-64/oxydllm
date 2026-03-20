use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::{DType, Device, Result, Tensor};

pub const DEFAULT_BLOCK_SIZE: usize = 16;

pub struct GlobalKvBudget {
    total_bytes: usize,
    allocated_bytes: AtomicUsize,
}

pub type SharedGlobalKvBudget = Arc<GlobalKvBudget>;

impl GlobalKvBudget {
    pub fn new(total_bytes: usize) -> Self {
        Self { total_bytes, allocated_bytes: AtomicUsize::new(0) }
    }

    pub fn acquire(&self, desired_bytes: usize) -> usize {
        loop {
            let current = self.allocated_bytes.load(Ordering::Relaxed);
            let available = self.total_bytes.saturating_sub(current);
            let granted = desired_bytes.min(available);
            match self.allocated_bytes.compare_exchange_weak(
                current,
                current + granted,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return granted,
                Err(_) => continue,
            }
        }
    }

    pub fn release(&self, bytes: usize) {
        self.allocated_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |cur| {
                Some(cur.saturating_sub(bytes))
            })
            .ok();
    }

    pub fn available_bytes(&self) -> usize {
        self.total_bytes
            .saturating_sub(self.allocated_bytes.load(Ordering::Relaxed))
    }
}

pub fn detect_system_kv_budget(memory_budget_bytes: Option<usize>, is_cpu: bool) -> usize {
    let base = if let Some(b) = memory_budget_bytes {
        let total = detect_system_memory_bytes().unwrap_or(usize::MAX);
        b.min(total)
    } else {
        detect_available_memory_bytes()
            .unwrap_or_else(|| detect_system_memory_bytes().unwrap_or(8 * 1024 * 1024 * 1024))
    };
    // Leave ~40-45% for model weights + OS + activations; KV gets the rest.
    let kv_fraction: f64 = if is_cpu { 0.65 } else { 0.55 };
    let headroom: usize = 512 * 1024 * 1024; // 512 MB
    ((base as f64 * kv_fraction) as usize).saturating_sub(headroom)
}

#[cfg(target_os = "macos")]
fn detect_system_memory_bytes() -> Option<usize> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    std::str::from_utf8(&output.stdout).ok()?.trim().parse().ok()
}

#[cfg(target_os = "macos")]
fn detect_available_memory_bytes() -> Option<usize> {
    let ps = std::process::Command::new("sysctl")
        .args(["-n", "hw.pagesize"])
        .output()
        .ok()?;
    let page_size: usize = std::str::from_utf8(&ps.stdout).ok()?.trim().parse().ok()?;

    let vm = std::process::Command::new("vm_stat").output().ok()?;
    let text = std::str::from_utf8(&vm.stdout).ok()?;

    let mut pages: usize = 0;
    for line in text.lines() {
        let reclaimable = line.starts_with("Pages free:")
            || line.starts_with("Pages inactive:")
            || line.starts_with("Pages speculative:");
        if reclaimable {
            // Lines look like: "Pages free:      174978."
            if let Some(n) = line
                .split_whitespace()
                .last()
                .and_then(|s| s.trim_end_matches('.').parse::<usize>().ok())
            {
                pages += n;
            }
        }
    }
    Some(pages * page_size)
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kb(key: &str) -> Option<usize> {
    std::fs::read_to_string("/proc/meminfo").ok()?
        .lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<usize>().ok())
        .map(|kb| kb * 1024)
}

#[cfg(target_os = "linux")]
fn detect_system_memory_bytes() -> Option<usize> {
    parse_meminfo_kb("MemTotal:")
}

#[cfg(target_os = "linux")]
fn detect_available_memory_bytes() -> Option<usize> {
    parse_meminfo_kb("MemAvailable:")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_system_memory_bytes() -> Option<usize> {
    None
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_available_memory_bytes() -> Option<usize> {
    None
}

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

    /// Total bytes occupied by the K and V pool tensors.
    pub fn pool_bytes(&self) -> usize {
        let k_bytes = self.pool_k.elem_count() * self.pool_k.dtype().size_in_bytes();
        let v_bytes = self.pool_v.elem_count() * self.pool_v.dtype().size_in_bytes();
        k_bytes + v_bytes
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
    cached_slots: Vec<u32>,
}

impl BlockTable {
    pub fn new() -> Self {
        Self { block_ids: Vec::new(), num_tokens: 0, cached_slots: Vec::new() }
    }
}

pub struct PagedKvCache {
    allocator: SharedBlockAllocator,
    table: BlockTable,
    block_size: usize,
    contig_k: Option<Tensor>,
    contig_v: Option<Tensor>,
}

impl PagedKvCache {
    pub fn new(allocator: SharedBlockAllocator) -> Self {
        let block_size = allocator.lock().unwrap().block_size();
        Self { allocator, table: BlockTable::new(), block_size, contig_k: None, contig_v: None }
    }

    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, new_seq, _) = new_k.dims4()?;
        let k_flat = new_k.squeeze(0)?.transpose(0, 1)?;
        let v_flat = new_v.squeeze(0)?.transpose(0, 1)?;
        let block_size = self.block_size;

        let prev_tokens = self.table.num_tokens;
        let mut written = 0;

        while written < new_seq {
            let current_offset = self.table.num_tokens % block_size;
            let n = (new_seq - written).min(block_size - current_offset);
            let k_chunk = k_flat.narrow(0, written, n)?.contiguous()?;
            let v_chunk = v_flat.narrow(0, written, n)?.contiguous()?;

            let block_id = {
                let mut alloc = self.allocator.lock().unwrap();
                if current_offset == 0 {
                    let id = alloc.allocate()?;
                    self.table.block_ids.push(id);
                }
                let id = *self.table.block_ids.last().unwrap();
                alloc.write(id, current_offset, &k_chunk, &v_chunk)?;
                id
            };

            let base = (block_id * block_size) as u32;
            for off in current_offset as u32..(current_offset + n) as u32 {
                self.table.cached_slots.push(base + off);
            }

            self.table.num_tokens += n;
            written += n;
        }

        let (ck, cv) = match (self.contig_k.take(), self.contig_v.take()) {
            (Some(k), Some(v)) => {
                // Fast decode path: cat only the new token(s) onto the existing buffer.
                (Tensor::cat(&[&k, new_k], 2)?, Tensor::cat(&[&v, new_v], 2)?)
            }
            (None, None) => {
                if prev_tokens > 0 {
                    let prefix_slots = &self.table.cached_slots[..prev_tokens];
                    let idx = Tensor::from_slice(prefix_slots, (prev_tokens,), new_k.device())?;
                    let (pk, pv) = self.allocator.lock().unwrap().gather(&idx)?;
                    (Tensor::cat(&[&pk, new_k], 2)?, Tensor::cat(&[&pv, new_v], 2)?)
                } else {
                    (new_k.clone(), new_v.clone())
                }
            }
            _ => unreachable!("contig_k and contig_v must always be in sync"),
        };
        self.contig_k = Some(ck);
        self.contig_v = Some(cv);

        Ok((
            self.contig_k.as_ref().unwrap().clone(),
            self.contig_v.as_ref().unwrap().clone(),
        ))
    }

    pub fn clear(&mut self) {
        if !self.table.block_ids.is_empty() {
            let mut alloc = self.allocator.lock().unwrap();
            for &bid in &self.table.block_ids {
                alloc.free(bid);
            }
        }
        self.table.block_ids.clear();
        self.table.num_tokens = 0;
        self.table.cached_slots.clear();
        self.contig_k = None;
        self.contig_v = None;
    }

    pub fn prepopulate_block(&mut self, block_id: usize) {
        self.allocator.lock().unwrap().share(block_id);
        self.table.block_ids.push(block_id);
        let base = (block_id * self.block_size) as u32;
        for off in 0..self.block_size as u32 {
            self.table.cached_slots.push(base + off);
        }
    }

    pub fn set_num_tokens(&mut self, n: usize) {
        self.table.cached_slots.truncate(n);
        self.table.num_tokens = n;
        if let Some(ck) = self.contig_k.take() {
            let len = ck.dim(2).unwrap_or(0);
            self.contig_k = if len > n { ck.narrow(2, 0, n).ok() } else { Some(ck) };
        }
        if let Some(cv) = self.contig_v.take() {
            let len = cv.dim(2).unwrap_or(0);
            self.contig_v = if len > n { cv.narrow(2, 0, n).ok() } else { Some(cv) };
        }
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
