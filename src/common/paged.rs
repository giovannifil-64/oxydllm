use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::{DType, Device, Result, Tensor};
use super::kv_quant::KvQuantizer;

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

enum KvPool {
    Full {
        pool_k: Tensor,
        pool_v: Tensor,
    },
    Quantized {
        packed_k: Vec<u8>,
        packed_v: Vec<u8>,
        norms_k: Vec<f32>,
        residual_norms_k: Option<Vec<f32>>,
        norms_v: Vec<f32>,
        quantizer: Arc<KvQuantizer>,
    },
}

pub struct BlockAllocator {
    pool: KvPool,
    free_list: Vec<usize>,
    ref_counts: Vec<u32>,
    num_blocks: usize,
    block_size: usize,
    n_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
    device: Device,
}

impl BlockAllocator {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
        quantizer: Option<Arc<KvQuantizer>>,
    ) -> Result<Self> {
        let total_slots = num_blocks * block_size;
        let free_list = (0..num_blocks).rev().collect();
        let ref_counts = vec![0u32; num_blocks];

        let pool = if let Some(q) = quantizer {
            let key_bph = q.key_packed_bytes();
            let value_bph = q.value_packed_bytes();
            KvPool::Quantized {
                packed_k: vec![0u8; total_slots * n_kv_heads * key_bph],
                packed_v: vec![0u8; total_slots * n_kv_heads * value_bph],
                norms_k: vec![0f32; total_slots * n_kv_heads],
                residual_norms_k: if q.qjl_quantization_enabled() {
                    Some(vec![0f32; total_slots * n_kv_heads])
                } else {
                    None
                },
                norms_v: vec![0f32; total_slots * n_kv_heads],
                quantizer: q,
            }
        } else {
            KvPool::Full {
                pool_k: Tensor::zeros((total_slots, n_kv_heads, head_dim), dtype, device)?,
                pool_v: Tensor::zeros((total_slots, n_kv_heads, head_dim), dtype, device)?,
            }
        };

        Ok(Self {
            pool, free_list, ref_counts, num_blocks, block_size,
            n_kv_heads, head_dim, dtype, device: device.clone(),
        })
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

    pub fn is_quantized(&self) -> bool {
        matches!(self.pool, KvPool::Quantized { .. })
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.n_kv_heads, self.head_dim)
    }

    /// Returns a clone of the quantizer Arc if the pool is quantized.
    pub fn get_quantizer(&self) -> Option<Arc<KvQuantizer>> {
        match &self.pool {
            KvPool::Quantized { quantizer, .. } => Some(Arc::clone(quantizer)),
            _ => None,
        }
    }

    /// Copy pre-quantized CPU-staged data into the block pool.
    /// All slices are indexed as flat `[n_tokens * n_kv_heads * bph]` (packed)
    /// or `[n_tokens * n_kv_heads]` (norms).
    /// Holds the lock for microseconds (pure memcpy — no quantization).
    pub fn write_staged(
        &mut self,
        block_id: usize,
        offset: usize,
        n_tokens: usize,
        pk_staged: &[u8],
        nk_staged: &[f32],
        rk_staged: &[f32],
        pv_staged: &[u8],
        nv_staged: &[f32],
    ) {
        let KvPool::Quantized {
            packed_k,
            norms_k,
            residual_norms_k,
            packed_v,
            norms_v,
            quantizer,
        } = &mut self.pool
        else {
            return;
        };
        let key_bph = quantizer.key_packed_bytes();
        let value_bph = quantizer.value_packed_bytes();
        let nkv = self.n_kv_heads;
        let start = block_id * self.block_size + offset;
        for t in 0..n_tokens {
            let slot = start + t;
            let sbk = t * nkv * key_bph;
            let sbv = t * nkv * value_bph;
            let sn = t * nkv;
            let dbk = slot * nkv * key_bph;
            let dbv = slot * nkv * value_bph;
            let dn = slot * nkv;
            packed_k[dbk..dbk + nkv * key_bph].copy_from_slice(&pk_staged[sbk..sbk + nkv * key_bph]);
            norms_k[dn..dn + nkv].copy_from_slice(&nk_staged[sn..sn + nkv]);
            if let Some(residual_norms_k) = residual_norms_k.as_mut() {
                residual_norms_k[dn..dn + nkv].copy_from_slice(&rk_staged[sn..sn + nkv]);
            }
            packed_v[dbv..dbv + nkv * value_bph].copy_from_slice(&pv_staged[sbv..sbv + nkv * value_bph]);
            norms_v[dn..dn + nkv].copy_from_slice(&nv_staged[sn..sn + nkv]);
        }
    }

    pub fn pool_bytes(&self) -> usize {
        match &self.pool {
            KvPool::Full { pool_k, pool_v } => {
                pool_k.elem_count() * pool_k.dtype().size_in_bytes()
                    + pool_v.elem_count() * pool_v.dtype().size_in_bytes()
            }
            KvPool::Quantized {
                packed_k,
                packed_v,
                norms_k,
                residual_norms_k,
                norms_v,
                ..
            } => {
                packed_k.len() + packed_v.len()
                    + (norms_k.len() + residual_norms_k.as_ref().map_or(0, Vec::len) + norms_v.len()) * std::mem::size_of::<f32>()
            }
        }
    }

    /// Write KV data for a chunk of tokens into the block pool.
    /// data_k, data_v shape: [n_tokens, n_kv_heads, head_dim]
    pub fn write(
        &mut self,
        block_id: usize,
        offset: usize,
        data_k: &Tensor,
        data_v: &Tensor,
    ) -> Result<()> {
        let start = block_id * self.block_size + offset;
        match &mut self.pool {
            KvPool::Full { pool_k, pool_v } => {
                pool_k.slice_set(data_k, 0, start)?;
                pool_v.slice_set(data_v, 0, start)?;
            }
            KvPool::Quantized {
                packed_k,
                packed_v,
                norms_k,
                residual_norms_k,
                norms_v,
                quantizer,
            } => {
                let n_tokens = data_k.dim(0)?;
                let k_f32 = data_k.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
                let v_f32 = data_v.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
                let k_vec: Vec<f32> = k_f32.flatten_all()?.to_vec1()?;
                let v_vec: Vec<f32> = v_f32.flatten_all()?.to_vec1()?;

                let key_bph = quantizer.key_packed_bytes();
                let value_bph = quantizer.value_packed_bytes();
                let hd = self.head_dim;
                let nkv = self.n_kv_heads;

                for t in 0..n_tokens {
                    let slot = start + t;
                    for h in 0..nkv {
                        let src = (t * nkv + h) * hd;
                        let key_byte_dst = slot * nkv * key_bph + h * key_bph;
                        let value_byte_dst = slot * nkv * value_bph + h * value_bph;
                        let norm_dst = slot * nkv + h;

                        let (pk, nk, rk) = quantizer.quantize_key(&k_vec[src..src + hd]);
                        packed_k[key_byte_dst..key_byte_dst + key_bph].copy_from_slice(&pk);
                        norms_k[norm_dst] = nk;
                        if let Some(residual_norms_k) = residual_norms_k.as_mut() {
                            residual_norms_k[norm_dst] = rk;
                        }

                        let (pv, nv) = quantizer.quantize(&v_vec[src..src + hd]);
                        packed_v[value_byte_dst..value_byte_dst + value_bph].copy_from_slice(&pv);
                        norms_v[norm_dst] = nv;
                    }
                }
            }
        }
        Ok(())
    }

    /// Gather KV data for given slot indices.
    /// Returns (K, V) with shape [1, n_kv_heads, num_tokens, head_dim].
    pub fn gather(&self, slot_indices: &Tensor) -> Result<(Tensor, Tensor)> {
        match &self.pool {
            KvPool::Full { pool_k, pool_v } => {
                let k = pool_k.index_select(slot_indices, 0)?;
                let v = pool_v.index_select(slot_indices, 0)?;
                let k = k.transpose(0, 1)?.unsqueeze(0)?;
                let v = v.transpose(0, 1)?.unsqueeze(0)?;
                Ok((k, v))
            }
            KvPool::Quantized {
                packed_k,
                packed_v,
                norms_k,
                residual_norms_k,
                norms_v,
                quantizer,
            } => {
                let indices: Vec<u32> = slot_indices
                    .to_device(&Device::Cpu)?
                    .to_vec1()?;
                let num_tokens = indices.len();
                let key_bph = quantizer.key_packed_bytes();
                let value_bph = quantizer.value_packed_bytes();
                let hd = self.head_dim;
                let nkv = self.n_kv_heads;

                let mut k_data = vec![0f32; num_tokens * nkv * hd];
                let mut v_data = vec![0f32; num_tokens * nkv * hd];

                for (t, &slot) in indices.iter().enumerate() {
                    let slot = slot as usize;
                    for h in 0..nkv {
                        let key_byte_src = slot * nkv * key_bph + h * key_bph;
                        let value_byte_src = slot * nkv * value_bph + h * value_bph;
                        let norm_src = slot * nkv + h;
                        let dst = (t * nkv + h) * hd;

                        let dk = quantizer.dequantize_key(
                            &packed_k[key_byte_src..key_byte_src + key_bph],
                            norms_k[norm_src],
                            residual_norms_k
                                .as_ref()
                                .map_or(0.0, |residual_norms_k| residual_norms_k[norm_src]),
                        );
                        k_data[dst..dst + hd].copy_from_slice(&dk);

                        let dv = quantizer.dequantize(
                            &packed_v[value_byte_src..value_byte_src + value_bph],
                            norms_v[norm_src],
                        );
                        v_data[dst..dst + hd].copy_from_slice(&dv);
                    }
                }

                let k = Tensor::from_vec(k_data, (num_tokens, nkv, hd), &Device::Cpu)?
                    .to_dtype(self.dtype)?
                    .to_device(&self.device)?
                    .transpose(0, 1)?
                    .unsqueeze(0)?;
                let v = Tensor::from_vec(v_data, (num_tokens, nkv, hd), &Device::Cpu)?
                    .to_dtype(self.dtype)?
                    .to_device(&self.device)?
                    .transpose(0, 1)?
                    .unsqueeze(0)?;
                Ok((k, v))
            }
        }
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

/// Deferred block pool write (avoids GPU→CPU sync during forward pass).
struct PendingWrite {
    block_id: usize,
    offset: usize,
    k_chunk: Tensor,
    v_chunk: Tensor,
}

/// Metadata for one staged block-pool write emitted during flush.
struct BgFlushItem {
    block_id: usize,
    offset: usize,
    n_tokens: usize,
}

pub struct PagedKvCache {
    allocator: SharedBlockAllocator,
    /// Cached quantizer reference — avoids locking the allocator in the hot path.
    quantizer: Option<Arc<KvQuantizer>>,
    table: BlockTable,
    block_size: usize,
    contig_k: Option<Tensor>,
    contig_v: Option<Tensor>,
    contig_len: usize,
    pending_writes: Vec<PendingWrite>,
}

impl PagedKvCache {
    pub fn new(allocator: SharedBlockAllocator) -> Self {
        let alloc = allocator.lock().unwrap();
        let block_size = alloc.block_size();
        let quantizer = alloc.get_quantizer();
        drop(alloc);
        Self {
            allocator,
            quantizer,
            table: BlockTable::new(),
            block_size,
            contig_k: None,
            contig_v: None,
            contig_len: 0,
            pending_writes: Vec::new(),
        }
    }

    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, new_seq, _) = new_k.dims4()?;
        let new_k = &new_k.contiguous()?;
        let new_v = &new_v.contiguous()?;
        let k_flat = new_k.squeeze(0)?.transpose(0, 1)?;
        let v_flat = new_v.squeeze(0)?.transpose(0, 1)?;
        let block_size = self.block_size;

        // Option 3: Skip block-pool writes for single-token decode steps.
        // The contiguous buffer is always present during decode and serves all
        // attention.  Pool writes are only needed for prefix-cache reuse, and
        // the prefix cache only uses full blocks filled during prefill.
        let skip_pool_write = self.contig_len > 0 && new_seq == 1;

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
                if !skip_pool_write {
                    if alloc.is_quantized() {
                        self.pending_writes.push(PendingWrite {
                            block_id: id,
                            offset: current_offset,
                            k_chunk: k_chunk.clone(),
                            v_chunk: v_chunk.clone(),
                        });
                    } else {
                        alloc.write(id, current_offset, &k_chunk, &v_chunk)?;
                    }
                }
                id
            };

            let base = (block_id * block_size) as u32;
            for off in current_offset as u32..(current_offset + n) as u32 {
                self.table.cached_slots.push(base + off);
            }

            self.table.num_tokens += n;
            written += n;
        }

        let total_needed = self.contig_len + new_seq;

        match (self.contig_k.take(), self.contig_v.take()) {
            (Some(k_buf), Some(v_buf)) => {
                let cap = k_buf.dim(2)?;
                if total_needed <= cap {
                    k_buf.slice_set(new_k, 2, self.contig_len)?;
                    v_buf.slice_set(new_v, 2, self.contig_len)?;
                    self.contig_k = Some(k_buf);
                    self.contig_v = Some(v_buf);
                } else {
                    let new_cap = (total_needed * 2).max(64);
                    let (_, n_kv, _, hd) = k_buf.dims4()?;
                    let device = k_buf.device().clone();
                    let dtype = k_buf.dtype();
                    let new_k_buf = Tensor::zeros((1, n_kv, new_cap, hd), dtype, &device)?;
                    let new_v_buf = Tensor::zeros((1, n_kv, new_cap, hd), dtype, &device)?;
                    if self.contig_len > 0 {
                        let old_k = k_buf.narrow(2, 0, self.contig_len)?.contiguous()?;
                        let old_v = v_buf.narrow(2, 0, self.contig_len)?.contiguous()?;
                        new_k_buf.slice_set(&old_k, 2, 0)?;
                        new_v_buf.slice_set(&old_v, 2, 0)?;
                    }
                    new_k_buf.slice_set(new_k, 2, self.contig_len)?;
                    new_v_buf.slice_set(new_v, 2, self.contig_len)?;
                    self.contig_k = Some(new_k_buf);
                    self.contig_v = Some(new_v_buf);
                }
            }
            (None, None) => {
                let init_cap = (total_needed * 2).max(64);
                let (_, n_kv, _, hd) = new_k.dims4()?;
                let device = new_k.device().clone();
                let dtype = new_k.dtype();
                let k_buf = Tensor::zeros((1, n_kv, init_cap, hd), dtype, &device)?;
                let v_buf = Tensor::zeros((1, n_kv, init_cap, hd), dtype, &device)?;

                if prev_tokens > 0 {
                    let prefix_slots = &self.table.cached_slots[..prev_tokens];
                    let idx = Tensor::from_slice(prefix_slots, (prev_tokens,), &device)?;
                    let (pk, pv) = self.allocator.lock().unwrap().gather(&idx)?;
                    k_buf.slice_set(&pk.contiguous()?, 2, 0)?;
                    v_buf.slice_set(&pv.contiguous()?, 2, 0)?;
                }
                k_buf.slice_set(new_k, 2, self.contig_len)?;
                v_buf.slice_set(new_v, 2, self.contig_len)?;
                self.contig_k = Some(k_buf);
                self.contig_v = Some(v_buf);
            }
            _ => unreachable!("contig_k and contig_v must always be in sync"),
        };
        self.contig_len = total_needed;

        Ok((
            self.contig_k.as_ref().unwrap().narrow(2, 0, self.contig_len)?,
            self.contig_v.as_ref().unwrap().narrow(2, 0, self.contig_len)?,
        ))
    }

    pub fn current(&self) -> Result<(Tensor, Tensor)> {
        match (&self.contig_k, &self.contig_v) {
            (Some(k), Some(v)) if self.contig_len > 0 => {
                Ok((k.narrow(2, 0, self.contig_len)?, v.narrow(2, 0, self.contig_len)?))
            }
            _ => Err(candle_core::Error::Msg("KV cache is empty".to_string())),
        }
    }

    /// Flush deferred quantized writes to the block pool.
    ///
    /// Two optimizations are applied here:
    /// - decode steps never populate `pending_writes`, so this is a no-op for
    ///   every decode token (see `append`).
    /// - all pending K chunks and all pending V chunks are concatenated into one
    ///   tensor each before the GPU→CPU transfer, reducing many small syncs to 2.
    ///
    /// The final quantize+pool-write is synchronous to guarantee that newly
    /// registered prefix-cache blocks always point to fully materialized data.
    pub fn flush_pending(&mut self) -> Result<()> {
        if self.pending_writes.is_empty() {
            return Ok(());
        }

        // Non-quantized path: write directly (full-precision pool).
        if self.quantizer.is_none() {
            let mut alloc = self.allocator.lock().unwrap();
            for pw in self.pending_writes.drain(..) {
                alloc.write(pw.block_id, pw.offset, &pw.k_chunk, &pw.v_chunk)?;
            }
            return Ok(());
        }

        // ── Option 1: single batched GPU→CPU transfer ─────────────────────────
        let pending_writes = std::mem::take(&mut self.pending_writes);

        let token_counts: Vec<usize> = pending_writes.iter()
            .map(|pw| pw.k_chunk.dim(0).unwrap_or(1))
            .collect();

        let k_cat = if pending_writes.len() == 1 {
            pending_writes[0].k_chunk.clone()
        } else {
            Tensor::cat(
                &pending_writes.iter().map(|pw| &pw.k_chunk).collect::<Vec<_>>(),
                0,
            )?
        };
        let v_cat = if pending_writes.len() == 1 {
            pending_writes[0].v_chunk.clone()
        } else {
            Tensor::cat(
                &pending_writes.iter().map(|pw| &pw.v_chunk).collect::<Vec<_>>(),
                0,
            )?
        };

        // Two GPU→CPU syncs total (one for K, one for V) regardless of layer count.
        let k_vec: Vec<f32> = k_cat
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?;
        let v_vec: Vec<f32> = v_cat
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?;

        let items: Vec<BgFlushItem> = pending_writes
            .iter()
            .zip(token_counts.iter())
            .map(|(pw, &n)| BgFlushItem {
                block_id: pw.block_id,
                offset: pw.offset,
                n_tokens: n,
            })
            .collect();

        let quantizer = Arc::clone(self.quantizer.as_ref().unwrap());
        let (nkv, hd) = self.allocator.lock().unwrap().dims();
        let qjl_enabled = quantizer.qjl_quantization_enabled();
        let key_bph = quantizer.key_packed_bytes();
        let value_bph = quantizer.value_packed_bytes();
        let total_tokens: usize = items.iter().map(|it| it.n_tokens).sum();

        let mut pk_staged = vec![0u8; total_tokens * nkv * key_bph];
        let mut nk_staged = vec![0f32; total_tokens * nkv];
        let mut rk_staged = if qjl_enabled {
            vec![0f32; total_tokens * nkv]
        } else {
            Vec::new()
        };
        let mut pv_staged = vec![0u8; total_tokens * nkv * value_bph];
        let mut nv_staged = vec![0f32; total_tokens * nkv];

        for t in 0..total_tokens {
            for h in 0..nkv {
                let src = (t * nkv + h) * hd;
                let dbk = (t * nkv + h) * key_bph;
                let dbv = (t * nkv + h) * value_bph;
                let dn = t * nkv + h;

                let (pk, nk, rk) = quantizer.quantize_key(&k_vec[src..src + hd]);
                pk_staged[dbk..dbk + key_bph].copy_from_slice(&pk);
                nk_staged[dn] = nk;
                if qjl_enabled {
                    rk_staged[dn] = rk;
                }

                let (pv, nv) = quantizer.quantize(&v_vec[src..src + hd]);
                pv_staged[dbv..dbv + value_bph].copy_from_slice(&pv);
                nv_staged[dn] = nv;
            }
        }

        let mut alloc = self.allocator.lock().unwrap();
        let mut t_off = 0usize;
        for item in &items {
            let t_end = t_off + item.n_tokens;
            let rk_slice: &[f32] = if qjl_enabled {
                &rk_staged[t_off * nkv..t_end * nkv]
            } else {
                &[]
            };
            alloc.write_staged(
                item.block_id,
                item.offset,
                item.n_tokens,
                &pk_staged[t_off * nkv * key_bph..t_end * nkv * key_bph],
                &nk_staged[t_off * nkv..t_end * nkv],
                rk_slice,
                &pv_staged[t_off * nkv * value_bph..t_end * nkv * value_bph],
                &nv_staged[t_off * nkv..t_end * nkv],
            );
            t_off += item.n_tokens;
        }

        Ok(())
    }

    pub fn clear(&mut self) {
        self.pending_writes.clear();
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
        self.contig_len = 0;
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
        if n < self.contig_len {
            self.contig_len = n;
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
            BlockAllocator::new(num_blocks, block_size, 2, 4, DType::F32, &Device::Cpu, None)
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
        let alloc = make_allocator(8, 2);
        let mut cache = PagedKvCache::new(alloc);
        let dev = Device::Cpu;

        let k1 = Tensor::randn(0f32, 1., (1, 2, 5, 4), &dev).unwrap();
        let v1 = Tensor::randn(0f32, 1., (1, 2, 5, 4), &dev).unwrap();
        let (k_out, v_out) = cache.append(&k1, &v1).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 5, 4]);
        assert_eq!(v_out.dims(), &[1, 2, 5, 4]);

        let k1_gathered = k_out.squeeze(0).unwrap().transpose(0, 1).unwrap();
        let k1_flat = k1.squeeze(0).unwrap().transpose(0, 1).unwrap();
        let diff = (k1_gathered - k1_flat).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-6, "prefill K mismatch: diff={diff}");

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
        assert_eq!(k_out.device().location(), dev.location());
    }

    #[test]
    fn clear_returns_blocks() {
        let alloc = make_allocator(4, 2);
        let mut cache = PagedKvCache::new(Arc::clone(&alloc));
        let dev = Device::Cpu;

        let k = Tensor::zeros((1, 2, 4, 4), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 2, 4, 4), DType::F32, &dev).unwrap();
        cache.append(&k, &v).unwrap();
        assert_eq!(alloc.lock().unwrap().num_free(), 2);

        cache.clear();
        assert_eq!(alloc.lock().unwrap().num_free(), 4);
    }

    #[test]
    fn exhaustion_error() {
        let alloc = make_allocator(2, 2);
        let mut cache = PagedKvCache::new(alloc);
        let dev = Device::Cpu;

        let k = Tensor::zeros((1, 2, 4, 4), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 2, 4, 4), DType::F32, &dev).unwrap();
        cache.append(&k, &v).unwrap();

        let k1 = Tensor::zeros((1, 2, 1, 4), DType::F32, &dev).unwrap();
        let v1 = Tensor::zeros((1, 2, 1, 4), DType::F32, &dev).unwrap();
        assert!(cache.append(&k1, &v1).is_err());
    }

    #[test]
    fn quantized_pool_reduces_memory() {
        let q = Arc::new(super::super::kv_quant::KvQuantizer::new_with_qjl(4, 64, true));
        let alloc_q = Arc::new(Mutex::new(
            BlockAllocator::new(4, 2, 2, 64, DType::F32, &Device::Cpu, Some(q)).unwrap(),
        ));
        let alloc_f = Arc::new(Mutex::new(
            BlockAllocator::new(4, 2, 2, 64, DType::F32, &Device::Cpu, None).unwrap(),
        ));
        let q_bytes = alloc_q.lock().unwrap().pool_bytes();
        let f_bytes = alloc_f.lock().unwrap().pool_bytes();
        assert!(q_bytes < f_bytes / 3, "quantized pool not smaller: q={q_bytes} f={f_bytes}");
    }
}
