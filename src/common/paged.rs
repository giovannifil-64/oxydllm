use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::kv_quant::KvQuantizer;
use candle_core::{DType, Device, Result, Tensor};

pub const DEFAULT_BLOCK_SIZE: usize = 16;

pub struct GlobalKvBudget {
    total_bytes: usize,
    allocated_bytes: AtomicUsize,
}

pub type SharedGlobalKvBudget = Arc<GlobalKvBudget>;

impl GlobalKvBudget {
    pub fn new(total_bytes: usize) -> Self {
        Self {
            total_bytes,
            allocated_bytes: AtomicUsize::new(0),
        }
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

/// Server-wide ceiling on KV bytes across every loaded model.
///
/// With an explicit `--memory-budget` the figure is that budget; otherwise it
/// is a fraction of *physical* memory. Sizing it from free memory instead,
/// as this used to, made the pool depend on whatever else the machine happened
/// to be doing at startup: measured on a 24 GB M5 the same
/// Phi-3-mini-4k-instruct-Q4 got anywhere from 4.6 GB to 10 GB of KV across
/// runs, and the 10 GB grant pushed the machine into swap, costing an eightfold
/// throughput drop (45.8 to 5.9 tok/s) and corrupt output. Physical memory does
/// not move, so the same machine now sizes the same way every time and
/// benchmarks are comparable across runs.
///
/// This is only the cross-model ceiling. A single model's share is additionally
/// capped against its own weights by [`safe_model_kv_ceiling`], since the
/// weights and the KV pool come out of the same memory.
pub fn detect_system_kv_budget(memory_budget_bytes: Option<usize>, is_cpu: bool) -> usize {
    if !is_cpu && let Some(budget) = device_working_set_bytes() {
        let room = match available_memory_bytes() {
            Some(free) => budget.min(free),
            None => budget,
        };
        return match memory_budget_bytes {
            Some(b) => b.min(room),
            None => room,
        };
    }
    let physical = detect_system_memory_bytes().unwrap_or(8 * 1024 * 1024 * 1024);
    let base = match memory_budget_bytes {
        Some(b) => b.min(physical),
        None => physical,
    };
    let kv_fraction: f64 = if is_cpu { 0.65 } else { 0.55 };
    let headroom: usize = 512 * 1024 * 1024;
    ((base as f64 * kv_fraction) as usize).saturating_sub(headroom)
}

/// Where a model's first attempt at a KV pool starts, as a share of physical
/// memory.
///
/// This is a starting point, not a guarantee. Correctness comes from the
/// loader checking that the weights survived the allocation and halving the
/// pool until they do, so a value that is too generous costs one extra load
/// attempt rather than the silent output corruption it used to. It is kept
/// near the measured-good band anyway, since paying for a retry on every
/// startup would be its own kind of wrong.
///
/// Largest share of physical memory one model's KV pool may occupy on its own.
///
/// Bounds the pool for models light enough that [`KV_TOTAL_FRACTION`] leaves
/// them room: on a 24 GB M5, Phi-3-mini-4k-instruct-Q4 decodes at 41 to
/// 46 tok/s with a 3 to 6 GB pool and degenerates to 5 tok/s with byte-ramp
/// output at 9 GB, so the pool alone needs a ceiling even when the weights are
/// small. The sequence count is not the variable: eight sequences over a 2048
/// context share the good 6 GB pool and stay correct.
///
/// Merely allocating an oversized pool is enough to corrupt a request that
/// never reads it, and nothing is logged. That is the same observable as the
/// streamed-expert oversubscription, and the same two upstream candidates
/// apply: candle's Metal allocator hands back buffers without asking the GPU
/// whether earlier work on them finished. Until that is settled upstream,
/// staying inside the measured-good band is the defence, not the explanation.
const KV_POOL_FRACTION: f64 = 0.25;

/// Where the first attempt starts for weights and pool together, as a share of
/// physical memory. A starting point like [`KV_POOL_FRACTION`], with the same
/// caveat: the loader's verification is what makes the result correct.
///
/// Largest share of physical memory weights and KV pool may occupy together.
///
/// Measured on a 24 GB M5. Phi-3-mini-4k-instruct-Q4 is correct at 8.2 GB
/// committed (2.2 weights, 6.0 pool) and corrupt at 11.2 GB (2.2 + 9.0);
/// Qwen3-4B-Instruct-2507-FP8 is correct at 10.5 GB (8.2 + 2.3) and corrupt at
/// 10.8 GB (8.2 + 2.6). Neither figure alone predicts the flip, their sum comes
/// closest, and the FP8 band is narrow enough (15% between the last good and
/// first bad pool) that this is a fitted safety margin, not an explanation:
/// see the roadmap for the unresolved mechanism.
const KV_TOTAL_FRACTION: f64 = 0.42;

/// Pool floor for models whose weights alone pass [`KV_TOTAL_FRACTION`].
///
/// gpt-oss-20b holds 12.8 GB of weights next to a 1.5 GB pool, over the total
/// share, and decodes correctly: a model that large cannot be served at all
/// without breaking the rule, so it keeps a workable pool instead of being
/// refused. Large-weight models also run few concurrent sequences in practice,
/// which is what this floor buys them.
const MIN_USEFUL_KV: usize = 1 << 30;

/// What the device itself says this process may hold.
///
/// Apple's driver answers the question directly through
/// `recommendedMaxWorkingSetSize`, which already accounts for the machine, its
/// memory and what the system needs to keep for itself: on a 24 GB M5 it
/// reports 17.76 GB. Asking it beats any share of physical memory we could
/// fit, because a fitted share is right for the machine it was fitted on and
/// wrong everywhere else.
///
/// `None` when there is no such device to ask, which is every backend but
/// Metal; callers then fall back to a share of physical memory.
#[cfg(feature = "metal")]
fn device_working_set_bytes() -> Option<usize> {
    use std::sync::OnceLock;
    static BUDGET: OnceLock<Option<usize>> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let device = Device::new_metal(0).ok()?;
        let Device::Metal(metal) = &device else {
            return None;
        };
        use objc2_metal::MTLDevice;
        let bytes = metal.device().as_ref().recommendedMaxWorkingSetSize() as usize;
        (bytes > 0).then_some(bytes)
    })
}

#[cfg(not(feature = "metal"))]
fn device_working_set_bytes() -> Option<usize> {
    None
}

/// Memory the system could hand out right now: free pages plus the ones it can
/// reclaim without touching disk.
///
/// The device's recommended working set is the ceiling for a machine with
/// nothing else on it; this is the other half of the answer, what is actually
/// going spare while a browser and an editor are open. `None` where the figure
/// cannot be read, which leaves the caller with the ceiling alone.
#[cfg(target_os = "macos")]
fn available_memory_bytes() -> Option<usize> {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<Option<usize>> = OnceLock::new();
    *AVAILABLE.get_or_init(read_available_memory_bytes)
}

/// Reads the figure once; [`available_memory_bytes`] caches it so a pool sized
/// twice in one run comes out the same size both times.
#[cfg(target_os = "macos")]
fn read_available_memory_bytes() -> Option<usize> {
    let out = std::process::Command::new("vm_stat").output().ok()?;
    let text = std::str::from_utf8(&out.stdout).ok()?;
    let mut page_size = 4096usize;
    let mut pages = 0usize;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics:")
            && let Some(p) = rest.split("page size of ").nth(1)
        {
            page_size = p.split_whitespace().next()?.parse().ok()?;
        }
        for label in [
            "Pages free:",
            "Pages inactive:",
            "Pages speculative:",
            "Pages purgeable:",
        ] {
            if let Some(rest) = line.strip_prefix(label) {
                pages += rest.trim().trim_end_matches('.').parse::<usize>().ok()?;
            }
        }
    }
    (pages > 0).then(|| pages * page_size)
}

#[cfg(not(target_os = "macos"))]
fn available_memory_bytes() -> Option<usize> {
    None
}

/// How much memory all loaded models together may occupy before the manager
/// starts evicting to make room.
///
/// This is the implicit default behind LRU eviction when no `--memory-budget`
/// is given. Without it eviction never ran unprompted: models accumulated
/// until their keep-alive expired, and a new one that did not fit was refused
/// while older idle models kept their memory. Making room is what an operator
/// expects, and it is the same share of physical memory the per-model ceiling
/// is measured against, so the two agree by construction.
pub fn safe_total_commitment_bytes() -> usize {
    if let Some(budget) = device_working_set_bytes() {
        return budget;
    }
    match detect_system_memory_bytes() {
        Some(p) => (p as f64 * KV_TOTAL_FRACTION) as usize,
        None => usize::MAX,
    }
}

/// The most KV a model with `weights_bytes` of weights should pool.
///
/// Whatever the device says it can hold, less the weights that will sit next to
/// the pool in the same memory. There is no safety fraction on top: the shares
/// this used to apply were fitted against forwards that allocated gigabytes of
/// transient buffers, and with those gone the loader's own check, that the
/// weights survived the allocation, is what makes an over-generous answer cost
/// a retry rather than corrupt an output.
pub fn safe_model_kv_ceiling(weights_bytes: usize) -> usize {
    if let Some(budget) = device_working_set_bytes() {
        let room = match available_memory_bytes() {
            Some(free) => budget.min(free),
            None => budget,
        };
        return room.saturating_sub(weights_bytes);
    }
    let physical = match detect_system_memory_bytes() {
        Some(p) => p,
        None => return usize::MAX,
    };
    let by_pool = (physical as f64 * KV_POOL_FRACTION) as usize;
    let by_total = ((physical as f64 * KV_TOTAL_FRACTION) as usize)
        .saturating_sub(weights_bytes)
        .max(MIN_USEFUL_KV);
    by_pool.min(by_total)
}

#[cfg(target_os = "macos")]
fn detect_system_memory_bytes() -> Option<usize> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    std::str::from_utf8(&output.stdout)
        .ok()?
        .trim()
        .parse()
        .ok()
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kb(key: &str) -> Option<usize> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()?
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

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_system_memory_bytes() -> Option<usize> {
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

struct ContigBuffer {
    k: Tensor,
    v: Tensor,
    cap: usize,
}

/// Invariant: `buffers` is sorted ascending by `cap` (smallest-fit via forward scan,
/// insertion via `partition_point`). On overflow the smallest is evicted because
/// large buffers cost more to rebuild.
struct ContigBufferPool {
    buffers: Vec<ContigBuffer>,
    max_buffers: usize,
}

const MAX_POOL_BUFFERS: usize = 4;

impl ContigBufferPool {
    fn new(max_buffers: usize) -> Self {
        Self {
            buffers: Vec::with_capacity(max_buffers),
            max_buffers,
        }
    }

    fn take(&mut self, needed: usize) -> Option<ContigBuffer> {
        let idx = self.buffers.iter().position(|b| b.cap >= needed)?;
        Some(self.buffers.remove(idx))
    }

    fn put(&mut self, buf: ContigBuffer) {
        if self.max_buffers == 0 {
            return;
        }
        let pos = self.buffers.partition_point(|b| b.cap < buf.cap);
        self.buffers.insert(pos, buf);
        if self.buffers.len() > self.max_buffers {
            self.buffers.remove(0);
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.buffers.len()
    }

    #[cfg(test)]
    fn capacities(&self) -> Vec<usize> {
        self.buffers.iter().map(|b| b.cap).collect()
    }
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
    contig_pool: ContigBufferPool,
}

pub struct StagedKvData<'a> {
    pub packed_k: &'a [u8],
    pub norms_k: &'a [f32],
    pub residual_norms_k: &'a [f32],
    pub packed_v: &'a [u8],
    pub norms_v: &'a [f32],
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
            pool,
            free_list,
            ref_counts,
            num_blocks,
            block_size,
            n_kv_heads,
            head_dim,
            dtype,
            device: device.clone(),
            contig_pool: ContigBufferPool::new(MAX_POOL_BUFFERS),
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
        debug_assert!(
            self.ref_counts[block_id] > 0,
            "share on un-allocated block {block_id}"
        );
        self.ref_counts[block_id] += 1;
    }

    pub fn free(&mut self, block_id: usize) {
        debug_assert!(block_id < self.num_blocks, "invalid block_id {block_id}");
        debug_assert!(
            self.ref_counts[block_id] > 0,
            "double-free of block {block_id}"
        );
        self.ref_counts[block_id] -= 1;
        if self.ref_counts[block_id] == 0 {
            self.free_list.push(block_id);
        }
    }

    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    pub fn num_total(&self) -> usize {
        self.num_blocks
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.n_kv_heads, self.head_dim)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn get_quantizer(&self) -> Option<Arc<KvQuantizer>> {
        match &self.pool {
            KvPool::Quantized { quantizer, .. } => Some(Arc::clone(quantizer)),
            _ => None,
        }
    }

    pub fn take_contig_buffer(&mut self, needed: usize) -> Option<(Tensor, Tensor, usize)> {
        self.contig_pool.take(needed).map(|b| (b.k, b.v, b.cap))
    }

    pub fn release_contig_buffer(&mut self, k: Tensor, v: Tensor, cap: usize) {
        self.contig_pool.put(ContigBuffer { k, v, cap });
    }

    #[cfg(test)]
    pub fn contig_pool_len(&self) -> usize {
        self.contig_pool.len()
    }

    #[cfg(test)]
    pub fn contig_pool_capacities(&self) -> Vec<usize> {
        self.contig_pool.capacities()
    }

    /// Pure-memcpy write of pre-quantized staged data; quantization happens at the caller.
    pub fn write_staged(
        &mut self,
        block_id: usize,
        offset: usize,
        n_tokens: usize,
        staged: StagedKvData<'_>,
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
            packed_k[dbk..dbk + nkv * key_bph]
                .copy_from_slice(&staged.packed_k[sbk..sbk + nkv * key_bph]);
            norms_k[dn..dn + nkv].copy_from_slice(&staged.norms_k[sn..sn + nkv]);
            if let Some(residual_norms_k) = residual_norms_k.as_mut() {
                residual_norms_k[dn..dn + nkv]
                    .copy_from_slice(&staged.residual_norms_k[sn..sn + nkv]);
            }
            packed_v[dbv..dbv + nkv * value_bph]
                .copy_from_slice(&staged.packed_v[sbv..sbv + nkv * value_bph]);
            norms_v[dn..dn + nkv].copy_from_slice(&staged.norms_v[sn..sn + nkv]);
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
                packed_k.len()
                    + packed_v.len()
                    + (norms_k.len()
                        + residual_norms_k.as_ref().map_or(0, Vec::len)
                        + norms_v.len())
                        * std::mem::size_of::<f32>()
            }
        }
    }

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
                let k_cpu = data_k.to_device(&Device::Cpu)?;
                let v_cpu = data_v.to_device(&Device::Cpu)?;
                let k_f32 = if k_cpu.dtype() == DType::F32 {
                    k_cpu
                } else {
                    k_cpu.to_dtype(DType::F32)?
                };
                let v_f32 = if v_cpu.dtype() == DType::F32 {
                    v_cpu
                } else {
                    v_cpu.to_dtype(DType::F32)?
                };
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
                let indices: Vec<u32> = slot_indices.to_device(&Device::Cpu)?.to_vec1()?;
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
        Self {
            block_ids: Vec::new(),
            num_tokens: 0,
            cached_slots: Vec::new(),
        }
    }
}

fn contig_buf_capacity(total_needed: usize) -> usize {
    let cap = if total_needed < 1024 {
        total_needed * 2
    } else {
        total_needed + (total_needed / 4).min(4096)
    };
    cap.max(64)
}

/// One contiguous run of a staged append: the `n_tokens` rows starting at
/// `src_offset` in the batch's source tensors belong at pool slot `dst_slot`.
struct PendingRun {
    dst_slot: usize,
    src_offset: usize,
    n_tokens: usize,
}

/// One append's deferred pool writes, keeping the source tensors whole.
///
/// Slicing the source per block would stage two GPU buffers for every sixteen
/// tokens of every layer, so a prefill of a few thousand tokens keeps tens of
/// thousands of buffers alive until the flush. The Metal allocator does not
/// survive that: it starts failing allocations without reporting an error and
/// the forward silently computes garbage, which measured as coherent output
/// below roughly twenty thousand staged buffers and nonsense above it.
struct PendingBatch {
    k_src: Tensor,
    v_src: Tensor,
    runs: Vec<PendingRun>,
}

struct BgFlushItem {
    block_id: usize,
    offset: usize,
    n_tokens: usize,
}

/// Per-sequence state of a recurrent (linear-attention) layer: the causal-conv
/// input window and the DeltaNet memory matrix. Lives in the same per-(seq,
/// layer) slot as paged KV so sequence lifecycle (retire / preempt / abort)
/// manages both uniformly.
pub struct RecurrentState {
    /// Last `conv_kernel - 1` raw conv inputs, shape [1, kernel-1, conv_dim].
    pub conv: Tensor,
    /// DeltaNet memory, shape [num_v_heads, head_k_dim, head_v_dim], F32.
    pub s: Tensor,
}

/// Per-sequence, per-layer KV cache backed by a shared paged block pool.
///
/// Keys and values live in fixed-size blocks allocated on demand from the shared
/// [`SharedBlockAllocator`], optionally KV-quantized (`quantizer`, cached at
/// construction so the hot path never relocks the allocator). Writes are staged
/// in `pending_writes` and flushed before reads. On the linear-attention layers
/// of hybrid models the layer instead keeps a [`RecurrentState`] in `recurrent`
/// and never touches the paged pool.
pub struct PagedKvCache {
    allocator: SharedBlockAllocator,
    quantizer: Option<Arc<KvQuantizer>>,
    table: BlockTable,
    block_size: usize,
    n_kv: usize,
    head_dim: usize,
    dtype: DType,
    device: Device,
    contig_k: Option<Tensor>,
    contig_v: Option<Tensor>,
    contig_len: usize,
    pending_writes: Vec<PendingBatch>,
    recurrent: Option<RecurrentState>,
}

impl PagedKvCache {
    pub fn new(allocator: SharedBlockAllocator) -> Self {
        let alloc = allocator.lock().unwrap();
        let block_size = alloc.block_size();
        let quantizer = alloc.get_quantizer();
        let (n_kv, head_dim) = alloc.dims();
        let dtype = alloc.dtype();
        let device = alloc.device().clone();
        drop(alloc);
        Self {
            allocator,
            quantizer,
            table: BlockTable::new(),
            block_size,
            n_kv,
            head_dim,
            dtype,
            device,
            contig_k: None,
            contig_v: None,
            contig_len: 0,
            pending_writes: Vec::new(),
            recurrent: None,
        }
    }

    /// Mutable access to the recurrent-state slot (linear-attention layers).
    pub fn recurrent_mut(&mut self) -> &mut Option<RecurrentState> {
        &mut self.recurrent
    }

    /// Lock is released before `Tensor::zeros` so GPU alloc doesn't serialize with other layers.
    fn acquire_contig(&self, needed: usize) -> Result<(Tensor, Tensor, usize)> {
        if let Some(t) = self.allocator.lock().unwrap().take_contig_buffer(needed) {
            debug_assert!(t.2 >= needed);
            return Ok(t);
        }
        let cap = contig_buf_capacity(needed);
        let dims = (1, self.n_kv, cap, self.head_dim);
        let k = Tensor::zeros(dims, self.dtype, &self.device)?;
        let v = Tensor::zeros(dims, self.dtype, &self.device)?;
        Ok((k, v, cap))
    }

    fn release_contig(&self, k: Tensor, v: Tensor, cap: usize) {
        self.allocator
            .lock()
            .unwrap()
            .release_contig_buffer(k, v, cap);
    }

    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, new_seq, _) = new_k.dims4()?;
        let new_k = &new_k.contiguous()?;
        let new_v = &new_v.contiguous()?;
        let k_flat = new_k.squeeze(0)?.transpose(0, 1)?;
        let v_flat = new_v.squeeze(0)?.transpose(0, 1)?;
        let block_size = self.block_size;

        // Decode-only: pool writes are only needed for prefix-cache reuse, which
        // only uses full blocks filled during prefill.
        let skip_pool_write = self.contig_len > 0 && new_seq == 1;

        let prev_tokens = self.table.num_tokens;
        debug_assert!(
            self.contig_len == 0 || self.contig_len == prev_tokens,
            "contig_len ({}) must match table tokens ({}) when buffer exists",
            self.contig_len,
            prev_tokens
        );
        let mut written = 0;

        let mut runs: Vec<PendingRun> = Vec::new();

        while written < new_seq {
            let current_offset = self.table.num_tokens % block_size;
            let n = (new_seq - written).min(block_size - current_offset);

            let block_id = {
                let mut alloc = self.allocator.lock().unwrap();
                if current_offset == 0 {
                    let id = alloc.allocate()?;
                    self.table.block_ids.push(id);
                }
                *self.table.block_ids.last().unwrap()
            };

            if !skip_pool_write {
                let dst_slot = block_id * block_size + current_offset;
                match runs.last_mut() {
                    Some(prev)
                        if prev.dst_slot + prev.n_tokens == dst_slot
                            && prev.src_offset + prev.n_tokens == written =>
                    {
                        prev.n_tokens += n;
                    }
                    _ => runs.push(PendingRun {
                        dst_slot,
                        src_offset: written,
                        n_tokens: n,
                    }),
                }
            }

            let base = u32::try_from(block_id * block_size)
                .expect("slot index overflow: block_id * block_size exceeds u32::MAX");
            for off in current_offset as u32..(current_offset + n) as u32 {
                self.table.cached_slots.push(base + off);
            }

            self.table.num_tokens += n;
            written += n;
        }

        if !runs.is_empty() {
            self.pending_writes.push(PendingBatch {
                k_src: k_flat.contiguous()?,
                v_src: v_flat.contiguous()?,
                runs,
            });
        }

        let total_needed = prev_tokens + new_seq;

        // Region past `contig_len` is never observed (all reads narrow to it), so
        // reusing dirty pooled memory is safe.
        match (self.contig_k.take(), self.contig_v.take()) {
            (Some(k_buf), Some(v_buf)) => {
                let cap = k_buf.dim(2)?;
                if total_needed <= cap {
                    k_buf.slice_set(new_k, 2, prev_tokens)?;
                    v_buf.slice_set(new_v, 2, prev_tokens)?;
                    self.contig_k = Some(k_buf);
                    self.contig_v = Some(v_buf);
                } else {
                    let (new_k_buf, new_v_buf, new_cap) = self.acquire_contig(total_needed)?;
                    if self.contig_len > 0 {
                        let old_k = k_buf.narrow(2, 0, self.contig_len)?.contiguous()?;
                        let old_v = v_buf.narrow(2, 0, self.contig_len)?.contiguous()?;
                        new_k_buf.slice_set(&old_k, 2, 0)?;
                        new_v_buf.slice_set(&old_v, 2, 0)?;
                    }
                    new_k_buf.slice_set(new_k, 2, prev_tokens)?;
                    new_v_buf.slice_set(new_v, 2, prev_tokens)?;
                    self.release_contig(k_buf, v_buf, cap);
                    self.contig_k = Some(new_k_buf);
                    self.contig_v = Some(new_v_buf);
                    let _ = new_cap;
                }
            }
            (None, None) => {
                let (k_buf, v_buf, _cap) = self.acquire_contig(total_needed)?;

                if prev_tokens > 0 {
                    let prefix_slots = &self.table.cached_slots[..prev_tokens];
                    let idx = Tensor::from_slice(prefix_slots, (prev_tokens,), &self.device)?;
                    let (pk, pv) = self.allocator.lock().unwrap().gather(&idx)?;
                    k_buf.slice_set(&pk.contiguous()?, 2, 0)?;
                    v_buf.slice_set(&pv.contiguous()?, 2, 0)?;
                }
                k_buf.slice_set(new_k, 2, prev_tokens)?;
                v_buf.slice_set(new_v, 2, prev_tokens)?;
                self.contig_k = Some(k_buf);
                self.contig_v = Some(v_buf);
            }
            _ => unreachable!("contig_k and contig_v must always be in sync"),
        };
        self.contig_len = total_needed;

        Ok((
            self.contig_k
                .as_ref()
                .unwrap()
                .narrow(2, 0, self.contig_len)?,
            self.contig_v
                .as_ref()
                .unwrap()
                .narrow(2, 0, self.contig_len)?,
        ))
    }

    pub fn current(&self) -> Result<(Tensor, Tensor)> {
        match (&self.contig_k, &self.contig_v) {
            (Some(k), Some(v)) if self.contig_len > 0 => Ok((
                k.narrow(2, 0, self.contig_len)?,
                v.narrow(2, 0, self.contig_len)?,
            )),
            _ => Err(candle_core::Error::Msg("KV cache is empty".to_string())),
        }
    }

    /// Synchronous on completion: prefix-cache blocks must point to fully
    /// materialized data before reuse.
    pub fn flush_pending(&mut self) -> Result<()> {
        if self.pending_writes.is_empty() {
            return Ok(());
        }

        let block_size = self.block_size;
        if self.quantizer.is_none() {
            let batches = std::mem::take(&mut self.pending_writes);
            let mut alloc = self.allocator.lock().unwrap();
            for batch in &batches {
                for run in &batch.runs {
                    alloc.write(
                        run.dst_slot / block_size,
                        run.dst_slot % block_size,
                        &batch.k_src.narrow(0, run.src_offset, run.n_tokens)?,
                        &batch.v_src.narrow(0, run.src_offset, run.n_tokens)?,
                    )?;
                }
            }
            return Ok(());
        }

        // Batch every staged run into one GPU-to-CPU transfer each.
        let batches = std::mem::take(&mut self.pending_writes);

        let mut k_parts: Vec<Tensor> = Vec::new();
        let mut v_parts: Vec<Tensor> = Vec::new();
        let mut items: Vec<BgFlushItem> = Vec::new();
        for batch in &batches {
            for run in &batch.runs {
                k_parts.push(batch.k_src.narrow(0, run.src_offset, run.n_tokens)?);
                v_parts.push(batch.v_src.narrow(0, run.src_offset, run.n_tokens)?);
                items.push(BgFlushItem {
                    block_id: run.dst_slot / block_size,
                    offset: run.dst_slot % block_size,
                    n_tokens: run.n_tokens,
                });
            }
        }

        let k_cat = if k_parts.len() == 1 {
            k_parts[0].clone()
        } else {
            Tensor::cat(&k_parts, 0)?
        };
        let v_cat = if v_parts.len() == 1 {
            v_parts[0].clone()
        } else {
            Tensor::cat(&v_parts, 0)?
        };

        let k_vec: Vec<f32> = k_cat
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let v_vec: Vec<f32> = v_cat
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;

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
                StagedKvData {
                    packed_k: &pk_staged[t_off * nkv * key_bph..t_end * nkv * key_bph],
                    norms_k: &nk_staged[t_off * nkv..t_end * nkv],
                    residual_norms_k: rk_slice,
                    packed_v: &pv_staged[t_off * nkv * value_bph..t_end * nkv * value_bph],
                    norms_v: &nv_staged[t_off * nkv..t_end * nkv],
                },
            );
            t_off += item.n_tokens;
        }

        Ok(())
    }

    pub fn clear(&mut self) {
        // Preempted sequences re-run prefill from position 0, so the recurrent
        // state must restart from zero alongside the freed KV blocks.
        self.recurrent = None;
        self.pending_writes.clear();
        let retired_contig = match (self.contig_k.take(), self.contig_v.take()) {
            (Some(k), Some(v)) => k.dim(2).ok().filter(|&c| c > 0).map(|cap| (k, v, cap)),
            _ => None,
        };
        if !self.table.block_ids.is_empty() || retired_contig.is_some() {
            let mut alloc = self.allocator.lock().unwrap();
            for &bid in &self.table.block_ids {
                alloc.free(bid);
            }
            if let Some((k, v, cap)) = retired_contig {
                alloc.release_contig_buffer(k, v, cap);
            }
        }
        self.table.block_ids.clear();
        self.table.num_tokens = 0;
        self.table.cached_slots.clear();
        self.contig_len = 0;
    }

    pub fn prepopulate_block(&mut self, block_id: usize) {
        self.allocator.lock().unwrap().share(block_id);
        self.table.block_ids.push(block_id);
        let base = u32::try_from(block_id * self.block_size)
            .expect("slot index overflow: block_id * block_size exceeds u32::MAX");
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

    /// Drop buffered pool writes without materializing them. Speculative verify
    /// forwards (M>1) queue pool writes for what are really decode tokens; those
    /// never belong in the block pool (the normal M=1 decode path skips them too),
    /// so discard them before rollback to avoid writing to soon-freed blocks.
    pub fn discard_pending(&mut self) {
        self.pending_writes.clear();
    }

    /// Roll the cache back to `n` tokens, freeing blocks that now hold only
    /// dropped tokens. Used to discard rejected speculative tokens. Pending
    /// writes are flushed first so kept blocks stay materialized; dropped tokens
    /// either land in freed blocks (overwritten on realloc) or in the tail of the
    /// last kept block (overwritten by confirmed tokens before it ever fills).
    pub fn truncate_to(&mut self, n: usize) -> Result<()> {
        if n >= self.table.num_tokens {
            return Ok(());
        }
        self.flush_pending()?;
        let blocks_needed = n.div_ceil(self.block_size);
        if blocks_needed < self.table.block_ids.len() {
            let mut alloc = self.allocator.lock().unwrap();
            for &bid in &self.table.block_ids[blocks_needed..] {
                alloc.free(bid);
            }
            drop(alloc);
            self.table.block_ids.truncate(blocks_needed);
        }
        self.set_num_tokens(n);
        Ok(())
    }

    pub fn block_id_at(&self, idx: usize) -> Option<usize> {
        self.table.block_ids.get(idx).copied()
    }

    /// Number of tokens currently cached (the sequence length this cache holds).
    pub fn num_tokens(&self) -> usize {
        self.table.num_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Contract: the KV budget depends only on the machine, never on what the
    /// machine happens to be doing. It used to be a fraction of free memory,
    /// which made the same model get 4.6 GB on one run and 10 GB on the next,
    /// the latter swapping and dropping throughput eightfold.
    #[test]
    fn kv_budget_is_deterministic_across_calls() {
        let a = detect_system_kv_budget(None, false);
        let b = detect_system_kv_budget(None, false);
        assert_eq!(a, b, "budget must not vary with free memory");
        assert!(a > 0, "a machine with memory must get a budget");
    }

    /// Contract: an explicit budget is honoured and never exceeds physical
    /// memory, so `--memory-budget` stays the escape hatch it claims to be.
    #[test]
    fn explicit_budget_is_honoured_and_bounded() {
        let small = detect_system_kv_budget(Some(4 << 30), false);
        let large = detect_system_kv_budget(Some(64 << 30), false);
        assert!(
            small < large || large == small,
            "larger budget cannot shrink KV"
        );
        let physical = detect_system_memory_bytes().unwrap_or(0);
        if physical > 0 {
            assert!(large <= physical, "budget cannot exceed physical memory");
        }
    }

    /// Contract: the pool ceiling bounds every model, the total ceiling takes
    /// over as weights grow, and a model too large for either still receives a
    /// Contract: a model is never told it may pool more than the device says it
    /// can hold, and heavier weights leave less room for the pool.
    ///
    /// This replaces a rule that fixed the pool at a quarter of physical memory
    /// and the weights-plus-pool total at 42% of it. Those shares were fitted
    /// against forwards that allocated gigabytes of transient buffers, on one
    /// 24 GB machine; with that allocation gone they only left capacity unused,
    /// and on any other machine they were a guess.
    #[test]
    fn the_ceiling_follows_the_device_and_the_weights() {
        let light = safe_model_kv_ceiling(1 << 30);
        let heavy = safe_model_kv_ceiling(12 << 30);
        assert!(heavy <= light, "light {light}, heavy {heavy}");

        if let Some(budget) = device_working_set_bytes() {
            assert!(
                light <= budget,
                "ceiling {light} exceeds what the device reported, {budget}"
            );
        }
    }

    /// Contract: sizing the same model twice in one run gives the same pool.
    ///
    /// The figure the ceiling rests on is read once and cached for exactly this
    /// reason: two loads of one model must not differ because a page freed
    /// between them.
    #[test]
    fn the_ceiling_is_stable_within_a_run() {
        let a = safe_model_kv_ceiling(2 << 30);
        let b = safe_model_kv_ceiling(2 << 30);
        assert_eq!(a, b);
    }

    fn make_allocator(num_blocks: usize, block_size: usize) -> SharedBlockAllocator {
        Arc::new(Mutex::new(
            BlockAllocator::new(num_blocks, block_size, 2, 4, DType::F32, &Device::Cpu, None)
                .unwrap(),
        ))
    }

    /// A pseudo-random source with the seed in hand, so a failure reproduces by
    /// rerunning the same seed.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
        }

        fn between(&mut self, lo: usize, hi: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + ((self.0 >> 33) as usize) % (hi - lo + 1)
        }
    }

    /// Contract: blocks are conserved. Whatever sequence of allocations, shares
    /// and frees it is put through, the allocator's free count plus the blocks
    /// callers hold equals its capacity, and a block is never handed out twice
    /// while someone still holds it.
    ///
    /// Both halves matter to correctness rather than tidiness: a leaked block
    /// shrinks the pool until the server stops admitting work, and a block
    /// handed to two sequences at once mixes one sequence's attention into
    /// another's, which reads as a model that has lost its mind rather than as
    /// a bug.
    #[test]
    fn fuzz_blocks_are_conserved() {
        for seed in 0u64..64 {
            let mut rng = Rng::new(seed);
            let capacity = rng.between(4, 64);
            let alloc = make_allocator(capacity, DEFAULT_BLOCK_SIZE);
            let mut a = alloc.lock().unwrap();

            // How many outstanding references each block id has.
            let mut held: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();

            for step in 0..256 {
                let outstanding: usize = held.len();
                assert_eq!(
                    a.num_free() + outstanding,
                    capacity,
                    "seed {seed} step {step}: {} free plus {outstanding} held is not {capacity}",
                    a.num_free()
                );

                match rng.between(0, 3) {
                    0 | 1 => {
                        if let Ok(id) = a.allocate() {
                            assert!(
                                !held.contains_key(&id),
                                "seed {seed} step {step}: block {id} handed out while still held"
                            );
                            held.insert(id, 1);
                        } else {
                            assert_eq!(
                                a.num_free(),
                                0,
                                "seed {seed} step {step}: allocation refused with blocks free"
                            );
                        }
                    }
                    2 => {
                        let ids: Vec<usize> = held.keys().copied().collect();
                        if let Some(&id) = ids.get(rng.between(0, ids.len().max(1) - 1)) {
                            a.share(id);
                            *held.get_mut(&id).unwrap() += 1;
                        }
                    }
                    _ => {
                        let ids: Vec<usize> = held.keys().copied().collect();
                        if let Some(&id) = ids.get(rng.between(0, ids.len().max(1) - 1)) {
                            a.free(id);
                            let refs = held.get_mut(&id).unwrap();
                            *refs -= 1;
                            if *refs == 0 {
                                held.remove(&id);
                            }
                        }
                    }
                }
            }

            for (id, refs) in held {
                for _ in 0..refs {
                    a.free(id);
                }
            }
            assert_eq!(
                a.num_free(),
                capacity,
                "seed {seed}: pool did not come back"
            );
        }
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
    fn truncate_to_frees_blocks_and_preserves_prefix() {
        let alloc = make_allocator(8, 2); // block_size = 2
        let mut cache = PagedKvCache::new(Arc::clone(&alloc));
        let dev = Device::Cpu;

        // 6 tokens => 3 blocks.
        let k = Tensor::randn(0f32, 1., (1, 2, 6, 4), &dev).unwrap();
        let v = Tensor::randn(0f32, 1., (1, 2, 6, 4), &dev).unwrap();
        let (k_full, _) = cache.append(&k, &v).unwrap();
        assert_eq!(cache.table.num_tokens, 6);
        let free_before = alloc.lock().unwrap().num_free();

        // Roll back to 3 tokens => needs 2 blocks => frees exactly 1.
        cache.truncate_to(3).unwrap();
        assert_eq!(cache.table.num_tokens, 3);
        assert_eq!(alloc.lock().unwrap().num_free(), free_before + 1);

        let (k_trunc, _) = cache.current().unwrap();
        assert_eq!(k_trunc.dim(2).unwrap(), 3);
        let orig3 = k_full.narrow(2, 0, 3).unwrap();
        let diff = (&k_trunc - &orig3)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-6,
            "kept K must match original prefix: diff={diff}"
        );

        // Re-append after truncation continues correctly.
        let k2 = Tensor::randn(0f32, 1., (1, 2, 2, 4), &dev).unwrap();
        let v2 = Tensor::randn(0f32, 1., (1, 2, 2, 4), &dev).unwrap();
        let (k5, _) = cache.append(&k2, &v2).unwrap();
        assert_eq!(k5.dim(2).unwrap(), 5);
        assert_eq!(cache.table.num_tokens, 5);

        // truncate_to a no-op when n >= current length.
        cache.truncate_to(99).unwrap();
        assert_eq!(cache.table.num_tokens, 5);
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
        let diff = (k1_gathered - k1_flat)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "prefill K mismatch: diff={diff}");

        let mut naive_k = k1.clone();
        let mut naive_v = v1.clone();
        for _ in 0..3 {
            let k_new = Tensor::randn(0f32, 1., (1, 2, 1, 4), &dev).unwrap();
            let v_new = Tensor::randn(0f32, 1., (1, 2, 1, 4), &dev).unwrap();
            let (k_paged, v_paged) = cache.append(&k_new, &v_new).unwrap();

            naive_k = Tensor::cat(&[&naive_k, &k_new], 2).unwrap();
            naive_v = Tensor::cat(&[&naive_v, &v_new], 2).unwrap();

            let dk = (&k_paged - &naive_k)
                .unwrap()
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let dv = (&v_paged - &naive_v)
                .unwrap()
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(dk < 1e-6, "decode K mismatch: diff={dk}");
            assert!(dv < 1e-6, "decode V mismatch: diff={dv}");
        }
        assert_eq!(k_out.device().location(), dev.location());
    }

    #[test]
    fn prepopulated_prefix_is_preserved_on_first_append() {
        let alloc = make_allocator(8, 2);
        let dev = Device::Cpu;

        let mut source = PagedKvCache::new(Arc::clone(&alloc));
        let prefix_k =
            Tensor::from_vec((0..16).map(|x| x as f32).collect(), (1, 2, 2, 4), &dev).unwrap();
        let prefix_v =
            Tensor::from_vec((100..116).map(|x| x as f32).collect(), (1, 2, 2, 4), &dev).unwrap();
        let _ = source.append(&prefix_k, &prefix_v).unwrap();
        source.flush_pending().unwrap();
        let prefix_block_id = source.block_id_at(0).unwrap();

        let mut cache = PagedKvCache::new(Arc::clone(&alloc));
        cache.prepopulate_block(prefix_block_id);
        cache.set_num_tokens(2);

        let new_k =
            Tensor::from_vec((200..208).map(|x| x as f32).collect(), (1, 2, 1, 4), &dev).unwrap();
        let new_v =
            Tensor::from_vec((300..308).map(|x| x as f32).collect(), (1, 2, 1, 4), &dev).unwrap();

        let (k_out, v_out) = cache.append(&new_k, &new_v).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 3, 4]);
        assert_eq!(v_out.dims(), &[1, 2, 3, 4]);

        let expected_k = Tensor::cat(&[&prefix_k, &new_k], 2).unwrap();
        let expected_v = Tensor::cat(&[&prefix_v, &new_v], 2).unwrap();

        let dk = (&k_out - &expected_k)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let dv = (&v_out - &expected_v)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(dk < 1e-4, "prefix K corrupted after append: {dk}");
        assert!(dv < 1e-4, "prefix V corrupted after append: {dv}");
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
    fn contig_buffer_is_recycled_across_sequences() {
        let alloc = make_allocator(8, 2);
        let dev = Device::Cpu;

        assert_eq!(alloc.lock().unwrap().contig_pool_len(), 0);

        let mut cache_a = PagedKvCache::new(Arc::clone(&alloc));
        let k = Tensor::zeros((1, 2, 4, 4), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 2, 4, 4), DType::F32, &dev).unwrap();
        cache_a.append(&k, &v).unwrap();
        assert_eq!(alloc.lock().unwrap().contig_pool_len(), 0);

        cache_a.clear();
        assert_eq!(alloc.lock().unwrap().contig_pool_len(), 1);
        let pooled_cap = alloc.lock().unwrap().contig_pool_capacities()[0];
        assert!(pooled_cap >= 4);

        let mut cache_b = PagedKvCache::new(Arc::clone(&alloc));
        let k = Tensor::zeros((1, 2, 3, 4), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 2, 3, 4), DType::F32, &dev).unwrap();
        cache_b.append(&k, &v).unwrap();
        assert_eq!(
            alloc.lock().unwrap().contig_pool_len(),
            0,
            "second sequence should have drained the pool"
        );
    }

    #[test]
    fn contig_pool_evicts_smallest_on_overflow() {
        let alloc = make_allocator(64, 2);
        {
            let mut a = alloc.lock().unwrap();
            for cap in [10usize, 50, 30, 100, 70] {
                let k = Tensor::zeros((1, 2, cap, 4), DType::F32, &Device::Cpu).unwrap();
                let v = Tensor::zeros((1, 2, cap, 4), DType::F32, &Device::Cpu).unwrap();
                a.release_contig_buffer(k, v, cap);
            }
        }
        // MAX_POOL_BUFFERS = 4: smallest (10) evicted, remainder sorted ascending.
        let caps = alloc.lock().unwrap().contig_pool_capacities();
        assert_eq!(caps, vec![30, 50, 70, 100]);
    }

    #[test]
    fn contig_pool_take_picks_smallest_fit() {
        let alloc = make_allocator(64, 2);
        {
            let mut a = alloc.lock().unwrap();
            for cap in [32usize, 64, 256] {
                let k = Tensor::zeros((1, 2, cap, 4), DType::F32, &Device::Cpu).unwrap();
                let v = Tensor::zeros((1, 2, cap, 4), DType::F32, &Device::Cpu).unwrap();
                a.release_contig_buffer(k, v, cap);
            }
        }
        let mut a = alloc.lock().unwrap();
        let (_, _, cap) = a.take_contig_buffer(50).expect("expected hit");
        assert_eq!(cap, 64);
        let (_, _, cap) = a.take_contig_buffer(200).expect("expected hit");
        assert_eq!(cap, 256);
        assert!(a.take_contig_buffer(100).is_none());
        let (_, _, cap) = a.take_contig_buffer(16).expect("expected hit");
        assert_eq!(cap, 32);
    }

    #[test]
    fn contig_buffer_growth_retires_old_to_pool() {
        let alloc = Arc::new(Mutex::new(
            BlockAllocator::new(256, 4, 2, 4, DType::F32, &Device::Cpu, None).unwrap(),
        ));
        let dev = Device::Cpu;
        let mut cache = PagedKvCache::new(Arc::clone(&alloc));

        let k = Tensor::zeros((1, 2, 32, 4), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 2, 32, 4), DType::F32, &dev).unwrap();
        cache.append(&k, &v).unwrap();
        let first_cap = contig_buf_capacity(32);
        assert_eq!(alloc.lock().unwrap().contig_pool_len(), 0);

        let big_k = Tensor::zeros((1, 2, 50, 4), DType::F32, &dev).unwrap();
        let big_v = Tensor::zeros((1, 2, 50, 4), DType::F32, &dev).unwrap();
        cache.append(&big_k, &big_v).unwrap();

        let caps = alloc.lock().unwrap().contig_pool_capacities();
        assert_eq!(caps.len(), 1, "growth path must retire the old buffer");
        assert_eq!(caps[0], first_cap);
    }

    #[test]
    fn quantized_pool_reduces_memory() {
        let q = Arc::new(super::super::kv_quant::KvQuantizer::new_with_qjl(
            4, 64, true,
        ));
        let alloc_q = Arc::new(Mutex::new(
            BlockAllocator::new(4, 2, 2, 64, DType::F32, &Device::Cpu, Some(q)).unwrap(),
        ));
        let alloc_f = Arc::new(Mutex::new(
            BlockAllocator::new(4, 2, 2, 64, DType::F32, &Device::Cpu, None).unwrap(),
        ));
        let q_bytes = alloc_q.lock().unwrap().pool_bytes();
        let f_bytes = alloc_f.lock().unwrap().pool_bytes();
        assert!(
            q_bytes < f_bytes / 3,
            "quantized pool not smaller: q={q_bytes} f={f_bytes}"
        );
    }
}
