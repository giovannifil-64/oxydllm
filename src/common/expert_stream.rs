//! On-demand streaming of MoE expert weights from the checkpoint files.
//!
//! For MoE checkpoints larger than device memory, only the router-selected
//! experts need to be resident at any moment. [`StreamedExperts`] parses the
//! safetensors headers once, serves experts through an LRU cache with a byte
//! budget, and fetches misses with positioned uncached reads
//! (`pread` + `F_NOCACHE`) straight into a reusable scratch buffer. Uncached
//! sequential reads measure 3 to 5 times faster here than faulting the same
//! bytes through a memory map, and re-reads of evicted experts do not churn
//! the page cache: the LRU is the only cache. Tensors that need no dtype
//! conversion upload from the scratch buffer to the device in a single copy.
//!
//! Byte-identity contract: a streamed expert must be bitwise identical to the
//! same expert loaded resident. Fetch therefore replicates the exact cast
//! chain of the resident loader (`weights.rs::load_tensor_with_dtype` +
//! `apply_weight_scale_inv`), including the FP8 double round-trip
//! (F8 to F32 to runtime dtype, then F32 scale fold, then back), and slices
//! stacked tensors on the expert dimension exactly as the resident path's
//! `narrow(0, e, 1)` does.

use super::linear::AnyLinear;
use super::moe::MoeExpert;
use super::mxfp4::Mxfp4Linear;
use super::weights::{apply_scale_inv, drain_metal, metal_alloc_lock};
use candle_core::{DType, Device, Result, Tensor};
use rustc_hash::FxHashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::{Arc, Mutex};

/// How the checkpoint stores each expert's tensors.
pub enum ExpertLayout {
    /// Per-expert `model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight`
    /// tensors (Qwen-MoE / OLMoE style), optionally FP8 with block-wise
    /// `*_scale_inv` companions.
    Standard,
    /// GPT-OSS stacked MXFP4 tensors
    /// (`mlp.experts.{gate_up,down}_proj_{blocks,scales,bias}`), expert on
    /// dim 0.
    GptOss { swiglu_limit: f64 },
}

/// Configuration handed from the model loader to [`ModelWeights::load`](crate::common::weights::ModelWeights::load)
/// (`src/common/weights.rs`) when expert streaming is requested.
pub struct ExpertStreamConfig {
    pub layout: ExpertLayout,
    pub cache_bytes: usize,
}

/// Canonicalizes a multimodal-nested tensor name: the loaders address
/// tensors as `model.layers.*`, while multimodal checkpoints (Qwen3.5 family)
/// store the text model under `model.language_model.*`.
fn canonical_name(name: &str) -> String {
    match name.strip_prefix("model.language_model.") {
        Some(rest) => format!("model.{rest}"),
        None => name.to_string(),
    }
}

/// Marks the tensors that are skipped at load and served by the streamer
/// instead. Routers (`mlp.gate` / `mlp.router`) and shared experts
/// (`mlp.shared_expert`) do not match and stay resident.
pub fn is_streamed_expert_tensor(name: &str) -> bool {
    name.contains(".mlp.experts.")
}

struct CacheEntry {
    expert: Arc<MoeExpert>,
    bytes: usize,
    last_use: u64,
}

struct LruState {
    cache: FxHashMap<(u32, u32), CacheEntry>,
    clock: u64,
    bytes: usize,
    hits: u64,
    misses: u64,
    graveyard: Vec<Arc<MoeExpert>>,
    graveyard_bytes: usize,
}

/// Deferred-reclamation threshold for evicted experts.
///
/// candle's Metal buffer pool reuses any buffer whose strong count drops to
/// one with no completion check, so an evicted expert's buffers must not be
/// released while queued commands may still read them. Instead of paying one
/// full-pipeline drain per eviction batch, evicted `Arc`s are parked in a
/// graveyard (the reference count keeps their buffers unreusable, which is
/// hazard-free without any synchronization) and released together behind a
/// single drain once this many bytes accumulate. The transient memory peak is
/// the cache budget plus this threshold, covered by the auto-sizer's
/// headroom.
const GRAVEYARD_DRAIN_BYTES: usize = 512 << 20;

/// One tensor as read from disk: typing plus its raw bytes.
struct RawRead {
    name: String,
    dtype: DType,
    dims: Vec<usize>,
    bytes: Vec<u8>,
}

/// Location and typing of one tensor inside the checkpoint files.
struct TensorMeta {
    file: u32,
    offset: u64,
    len: u64,
    dtype: DType,
    shape: Vec<usize>,
}

/// Shared expert pool: one per model, referenced by every MoE layer.
pub struct StreamedExperts {
    files: Vec<File>,
    index: FxHashMap<String, TensorMeta>,
    layout: ExpertLayout,
    device: Device,
    dtype: DType,
    cache_bytes: usize,
    state: Mutex<LruState>,
}

impl StreamedExperts {
    /// Builds the pool by parsing the safetensors headers of `paths` and
    /// indexing every streamed-expert tensor's file position; the files stay
    /// open with `F_NOCACHE` for uncached positioned reads.
    ///
    /// ## Errors
    /// Fails if a file cannot be opened or its header is not valid
    /// safetensors JSON, or if an indexed tensor has an unsupported dtype.
    pub fn new(
        paths: &[&str],
        cfg: ExpertStreamConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut files = Vec::with_capacity(paths.len());
        let mut index = FxHashMap::default();
        for (fi, path) in paths.iter().enumerate() {
            let file = File::open(path)
                .map_err(|e| candle_core::Error::Msg(format!("open {path}: {e}")))?;
            set_nocache(&file);
            let mut len_bytes = [0u8; 8];
            file.read_exact_at(&mut len_bytes, 0)
                .map_err(|e| candle_core::Error::Msg(format!("read header size {path}: {e}")))?;
            let header_len = u64::from_le_bytes(len_bytes);
            let mut header = vec![0u8; header_len as usize];
            file.read_exact_at(&mut header, 8)
                .map_err(|e| candle_core::Error::Msg(format!("read header {path}: {e}")))?;
            let header: serde_json::Value = serde_json::from_slice(&header)
                .map_err(|e| candle_core::Error::Msg(format!("parse header {path}: {e}")))?;
            let data_start = 8 + header_len;
            let entries = header
                .as_object()
                .ok_or_else(|| candle_core::Error::Msg(format!("{path}: header not an object")))?;
            for (name, meta) in entries {
                if name == "__metadata__"
                    || !is_streamed_expert_tensor(name)
                    || name.starts_with("model.visual.")
                    || name.starts_with("mtp.")
                {
                    continue;
                }
                let canonical = canonical_name(name);
                let get = |k: &str| {
                    meta.get(k).ok_or_else(|| {
                        candle_core::Error::Msg(format!("{path}: {name} missing {k}"))
                    })
                };
                let dtype_str = get("dtype")?.as_str().unwrap_or_default().to_string();
                let shape: Vec<usize> = get("shape")?
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize))
                    .ok_or_else(|| candle_core::Error::Msg(format!("{path}: {name} bad shape")))?
                    .collect();
                let offs = get("data_offsets")?
                    .as_array()
                    .and_then(|a| Some((a.first()?.as_u64()?, a.get(1)?.as_u64()?)))
                    .ok_or_else(|| {
                        candle_core::Error::Msg(format!("{path}: {name} bad data_offsets"))
                    })?;
                index.insert(
                    canonical,
                    TensorMeta {
                        file: fi as u32,
                        offset: data_start + offs.0,
                        len: offs.1 - offs.0,
                        dtype: st_dtype(&dtype_str, name)?,
                        shape,
                    },
                );
            }
            files.push(file);
        }
        Ok(Self {
            files,
            index,
            layout: cfg.layout,
            device,
            dtype,
            cache_bytes: cfg.cache_bytes,
            state: Mutex::new(LruState {
                cache: FxHashMap::default(),
                clock: 0,
                bytes: 0,
                hits: 0,
                misses: 0,
                graveyard: Vec::new(),
                graveyard_bytes: 0,
            }),
        })
    }

    /// Returns expert `e` of layer `layer_idx`, fetching it from the mmap on a
    /// cache miss. Test-only convenience over [`fetch_many`](Self::fetch_many),
    /// which production dispatch always uses.
    ///
    /// ## Errors
    /// As [`fetch_many`](Self::fetch_many).
    #[cfg(test)]
    fn fetch(&self, layer_idx: usize, e: usize) -> Result<Arc<MoeExpert>> {
        Ok(self.fetch_many(layer_idx, &[e])?.pop().expect("one id in"))
    }

    /// Returns the requested experts of `layer_idx` in the order of `ids`,
    /// fetching cache misses from the mmap. The returned `Arc`s keep the
    /// experts alive for the caller's dispatch even across evictions.
    ///
    /// Misses are read and built in parallel (reads one task per tensor,
    /// dtype conversion one task per expert): FP8 checkpoints pay a CPU
    /// dequantization per fetched expert that would otherwise serialize.
    /// Parallel device uploads are safe here because they are synchronous
    /// memcpys guarded by the allocation lock and the forward thread is
    /// blocked inside this call, so nothing encodes GPU work concurrently.
    /// Miss ids are sorted ascending, which is disk order for stacked expert
    /// tensors, so a multi-miss batch reads the files mostly sequentially
    /// through the uncached positioned-read path. Host-to-device
    /// uploads are synchronous memcpys into fresh buffers (candle's
    /// `new_buffer_with_data`) and need no synchronization; the one Metal
    /// drain per batch happens after evictions, because an evicted expert's
    /// buffers return to candle's pool and must not be reused while queued
    /// commands may still read them. Eviction runs after insertion, so the
    /// transient peak is the budget plus the batch being inserted.
    ///
    /// ## Errors
    /// Fails if a tensor is missing from the checkpoint, malformed, or a
    /// cast or device transfer fails.
    pub(crate) fn fetch_many(
        &self,
        layer_idx: usize,
        ids: &[usize],
    ) -> Result<Vec<Arc<MoeExpert>>> {
        let mut state = self.state.lock().unwrap();
        state.clock += 1;
        let clock = state.clock;

        let mut misses: Vec<usize> = Vec::new();
        for &e in ids {
            let key = (layer_idx as u32, e as u32);
            if let Some(entry) = state.cache.get_mut(&key) {
                entry.last_use = clock;
                state.hits += 1;
            } else if !misses.contains(&e) {
                misses.push(e);
            }
        }
        misses.sort_unstable();

        let reads = self.read_misses(layer_idx, &misses)?;
        let built: Vec<(MoeExpert, usize)> = {
            use rayon::prelude::*;
            misses
                .par_iter()
                .zip(reads.into_par_iter())
                .map(|(&e, tensors)| match self.layout {
                    ExpertLayout::Standard => self.build_standard(tensors),
                    ExpertLayout::GptOss { swiglu_limit } => {
                        self.build_gpt_oss(layer_idx, e, swiglu_limit, tensors)
                    }
                })
                .collect::<Result<_>>()?
        };
        for (&e, (expert, bytes)) in misses.iter().zip(built) {
            state.misses += 1;
            state.bytes += bytes;
            state.cache.insert(
                (layer_idx as u32, e as u32),
                CacheEntry {
                    expert: Arc::new(expert),
                    bytes,
                    last_use: clock,
                },
            );
        }

        while state.bytes > self.cache_bytes && state.cache.len() > ids.len() {
            let (&victim, _) = state
                .cache
                .iter()
                .filter(|(_, entry)| entry.last_use != clock)
                .min_by_key(|(_, entry)| entry.last_use)
                .expect("cache larger than current batch");
            let entry = state.cache.remove(&victim).expect("victim exists");
            state.bytes -= entry.bytes;
            state.graveyard_bytes += entry.bytes;
            state.graveyard.push(entry.expert);
        }
        if state.graveyard_bytes >= GRAVEYARD_DRAIN_BYTES {
            drain_metal(&self.device, "expert graveyard reclamation")
                .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?;
            state.graveyard.clear();
            state.graveyard_bytes = 0;
        }

        ids.iter()
            .map(|&e| {
                let key = (layer_idx as u32, e as u32);
                Ok(Arc::clone(
                    &state.cache.get(&key).expect("just inserted or hit").expert,
                ))
            })
            .collect()
    }

    /// `(hits, misses, resident_bytes)` since construction, for logging.
    pub fn stats(&self) -> (u64, u64, usize) {
        let state = self.state.lock().unwrap();
        (state.hits, state.misses, state.bytes)
    }
}

impl Drop for StreamedExperts {
    fn drop(&mut self) {
        let (hits, misses, bytes) = self.stats();
        let total = hits + misses;
        if total > 0 {
            tracing::info!(
                hits,
                misses,
                hit_rate = format!("{:.1}%", 100.0 * hits as f64 / total as f64),
                resident_mb = bytes >> 20,
                "expert stream pool retired"
            );
        }
    }
}

impl StreamedExperts {
    /// Reads every tensor of every missed expert, in parallel across the
    /// whole batch (rayon over one task per tensor), and returns them grouped
    /// per expert in `misses` order. Parallel positioned reads raise the
    /// effective SSD bandwidth over a single-stream read; each task owns its
    /// buffer, so no synchronization is needed beyond the join.
    fn read_misses(&self, layer_idx: usize, misses: &[usize]) -> Result<Vec<Vec<RawRead>>> {
        use rayon::prelude::*;
        let mut flat: Vec<(usize, String, Option<usize>)> = Vec::new();
        for (slot, &e) in misses.iter().enumerate() {
            match self.layout {
                ExpertLayout::Standard => {
                    let p = format!("model.layers.{layer_idx}.mlp.experts.{e}");
                    for proj in ["gate_proj", "up_proj", "down_proj"] {
                        flat.push((slot, format!("{p}.{proj}.weight"), None));
                        let scale = format!("{p}.{proj}.weight_scale_inv");
                        if self.index.contains_key(&scale) {
                            flat.push((slot, scale, None));
                        }
                    }
                }
                ExpertLayout::GptOss { .. } => {
                    let p = format!("model.layers.{layer_idx}.mlp.experts");
                    for t in [
                        "gate_up_proj_blocks",
                        "gate_up_proj_scales",
                        "gate_up_proj_bias",
                        "down_proj_blocks",
                        "down_proj_scales",
                        "down_proj_bias",
                    ] {
                        flat.push((slot, format!("{p}.{t}"), Some(e)));
                    }
                }
            }
        }
        let reads: Vec<(usize, RawRead)> = flat
            .par_iter()
            .map(|(slot, name, row)| Ok((*slot, self.read_one(name, *row)?)))
            .collect::<Result<_>>()?;
        let mut grouped: Vec<Vec<RawRead>> = (0..misses.len()).map(|_| Vec::new()).collect();
        for (slot, read) in reads {
            grouped[slot].push(read);
        }
        Ok(grouped)
    }

    /// Reads a tensor (or row `row` of its dim 0) with an uncached positioned
    /// read. Row slices on dim 0 are contiguous because safetensors data is
    /// row-major and unstrided.
    fn read_one(&self, name: &str, row: Option<usize>) -> Result<RawRead> {
        let meta = self.index.get(name).ok_or_else(|| {
            candle_core::Error::Msg(format!("streamed tensor {name} not indexed"))
        })?;
        let (offset, len, dims) = match row {
            None => (meta.offset, meta.len, meta.shape.clone()),
            Some(r) => {
                if meta.shape.is_empty() || r >= meta.shape[0] {
                    candle_core::bail!(
                        "expert row {r} out of bounds for {name} with shape {:?}",
                        meta.shape
                    );
                }
                let row_bytes = meta.len / meta.shape[0] as u64;
                (
                    meta.offset + r as u64 * row_bytes,
                    row_bytes,
                    meta.shape[1..].to_vec(),
                )
            }
        };
        let mut bytes = vec![0u8; len as usize];
        self.files[meta.file as usize]
            .read_exact_at(&mut bytes, offset)
            .map_err(|e| candle_core::Error::Msg(format!("read {name}: {e}")))?;
        Ok(RawRead {
            name: name.to_string(),
            dtype: meta.dtype,
            dims,
            bytes,
        })
    }

    /// Turns one read into a device tensor, replicating the resident loader's
    /// cast chain for byte identity: integer and already-runtime-dtype tensors
    /// upload straight from the read buffer in one copy; FP8 dequantizes
    /// through F32 on the CPU with its block `scale` folded in F32 after the
    /// runtime-dtype round-trip; other floats cast on the CPU.
    fn tensor_from_read(&self, r: &RawRead, scale: Option<&RawRead>) -> Result<Tensor> {
        let no_cast = matches!(
            r.dtype,
            DType::U8 | DType::U32 | DType::I16 | DType::I32 | DType::I64
        ) || r.dtype == self.dtype;
        if no_cast && scale.is_none() {
            let _guard = self
                .device
                .is_metal()
                .then(|| metal_alloc_lock().lock().unwrap());
            return Tensor::from_raw_buffer(&r.bytes, r.dtype, &r.dims, &self.device);
        }
        let cpu = Tensor::from_raw_buffer(&r.bytes, r.dtype, &r.dims, &Device::Cpu)?;
        let cpu = if r.dtype == DType::F8E4M3 {
            cpu.to_dtype(DType::F32)?.to_dtype(self.dtype)?
        } else {
            cpu.to_dtype(self.dtype)?
        };
        let cpu = match scale {
            None => cpu,
            Some(sc) => {
                let scale = Tensor::from_raw_buffer(&sc.bytes, sc.dtype, &sc.dims, &Device::Cpu)?
                    .to_dtype(DType::F32)?;
                apply_scale_inv(&cpu.to_dtype(DType::F32)?, &scale)?.to_dtype(self.dtype)?
            }
        };
        self.to_device(cpu, &r.name)
    }

    fn to_device(&self, cpu: Tensor, name: &str) -> Result<Tensor> {
        if self.device.is_cpu() {
            return Ok(cpu);
        }
        let _guard = self
            .device
            .is_metal()
            .then(|| metal_alloc_lock().lock().unwrap());
        let dev = cpu.to_device(&self.device)?;
        drain_metal(&self.device, name).map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?;
        Ok(dev)
    }

    /// Assembles a standard expert from its reads: each `*.weight` entry,
    /// optionally followed by its `*_scale_inv` companion, in gate/up/down
    /// order as produced by [`read_misses`](Self::read_misses).
    ///
    /// Block-scaled FP8 projections on a BF16 Metal target cache in their
    /// file encoding ([`MoeExpert::Fp8Resident`]): half the bytes per expert,
    /// so double the experts per cache budget, at the cost of an on-use
    /// device dequantization that replays the exact resident cast chain.
    fn build_standard(&self, reads: Vec<RawRead>) -> Result<(MoeExpert, usize)> {
        let mut pairs: Vec<(&RawRead, Option<&RawRead>)> = Vec::with_capacity(3);
        let mut i = 0;
        while i < reads.len() {
            let weight = &reads[i];
            let scale = reads
                .get(i + 1)
                .filter(|r| r.name.ends_with(".weight_scale_inv"));
            i += 1 + usize::from(scale.is_some());
            pairs.push((weight, scale));
        }

        #[cfg(feature = "metal")]
        if pairs.len() == 3
            && self.dtype == DType::BF16
            && self.device.is_metal()
            && pairs.iter().all(|(w, sc)| {
                w.dtype == DType::F8E4M3
                    && w.dims.len() == 2
                    && matches!(sc, Some(sc) if sc.dims.len() == 2
                        && sc.dims[0] > 0
                        && sc.dims[1] > 0
                        && w.dims[0].is_multiple_of(sc.dims[0])
                        && w.dims[1].is_multiple_of(sc.dims[1]))
            })
        {
            let mut cached = Vec::with_capacity(3);
            let mut bytes = 0usize;
            for (w, sc) in &pairs {
                let sc = sc.expect("checked above");
                let scale_f32 =
                    Tensor::from_raw_buffer(&sc.bytes, sc.dtype, &sc.dims, &Device::Cpu)?
                        .to_dtype(DType::F32)?;
                let _guard = metal_alloc_lock().lock().unwrap();
                let proj = crate::common::moe::Fp8Cached {
                    bytes: Tensor::from_raw_buffer(&w.bytes, DType::U8, &w.dims, &self.device)?,
                    scales: scale_f32.to_device(&self.device)?,
                };
                bytes += tensor_bytes(&proj.bytes) + tensor_bytes(&proj.scales);
                cached.push(proj);
            }
            let down_proj = cached.pop().expect("three projections");
            let up_proj = cached.pop().expect("three projections");
            let gate_proj = cached.pop().expect("three projections");
            return Ok((
                MoeExpert::Fp8Resident {
                    gate_proj,
                    up_proj,
                    down_proj,
                },
                bytes,
            ));
        }

        let mut projections = Vec::with_capacity(3);
        for (weight, scale) in pairs {
            projections.push(self.tensor_from_read(weight, scale)?);
        }
        let [gate, up, down]: [Tensor; 3] = projections.try_into().map_err(|_| {
            candle_core::Error::Msg("standard expert needs gate/up/down projections".to_string())
        })?;
        let bytes = tensor_bytes(&gate) + tensor_bytes(&up) + tensor_bytes(&down);
        Ok((
            MoeExpert::Standard {
                gate_proj: AnyLinear::from_weight(gate, None)?,
                up_proj: AnyLinear::from_weight(up, None)?,
                down_proj: AnyLinear::from_weight(down, None)?,
            },
            bytes,
        ))
    }

    /// Assembles a GPT-OSS expert from its six reads, in the fixed
    /// gate_up/down blocks/scales/bias order of [`read_misses`](Self::read_misses).
    fn build_gpt_oss(
        &self,
        layer_idx: usize,
        e: usize,
        swiglu_limit: f64,
        reads: Vec<RawRead>,
    ) -> Result<(MoeExpert, usize)> {
        if reads.len() != 6 {
            candle_core::bail!(
                "GPT-OSS expert ({layer_idx}, {e}): expected 6 tensors, got {}",
                reads.len()
            );
        }
        let mut tensors = reads
            .iter()
            .map(|r| self.tensor_from_read(r, None))
            .collect::<Result<Vec<_>>>()?;
        let dn_bias = tensors.pop().expect("six tensors");
        let dn_scales = tensors.pop().expect("six tensors");
        let dn_blocks = tensors.pop().expect("six tensors");
        let gu_bias = tensors.pop().expect("six tensors");
        let gu_scales = tensors.pop().expect("six tensors");
        let gu_blocks = tensors.pop().expect("six tensors");

        let dims_of = |t: &Tensor, what: &str| -> Result<(usize, usize)> {
            let d = t.dims();
            if d.len() != 3 || d[2] != 16 {
                candle_core::bail!(
                    "GPT-OSS streamed expert ({layer_idx}, {e}): {what} row shape {d:?} != [out, K/32, 16]"
                );
            }
            Ok((d[0], d[1] * 32))
        };
        let (gu_out, gu_in) = dims_of(&gu_blocks, "gate_up_proj_blocks")?;
        let (dn_out, dn_in) = dims_of(&dn_blocks, "down_proj_blocks")?;

        let bytes = [
            &gu_blocks, &gu_scales, &gu_bias, &dn_blocks, &dn_scales, &dn_bias,
        ]
        .iter()
        .map(|t| tensor_bytes(t))
        .sum();

        Ok((
            MoeExpert::GptOss {
                gate_up: Mxfp4Linear::new(gu_blocks, gu_scales, Some(gu_bias), gu_in, gu_out)?,
                down: Mxfp4Linear::new(dn_blocks, dn_scales, Some(dn_bias), dn_in, dn_out)?,
                limit: swiglu_limit,
            },
            bytes,
        ))
    }
}

fn tensor_bytes(t: &Tensor) -> usize {
    t.elem_count() * t.dtype().size_in_bytes()
}

/// Opts the file out of the page cache (`fcntl` + `F_NOCACHE`), so expert
/// re-reads do not keep a kernel-side copy of the pool competing with the
/// LRU for memory. Darwin-only: on other platforms reads simply go through
/// the page cache, which stays correct.
#[cfg(target_os = "macos")]
fn set_nocache(file: &File) {
    unsafe {
        libc::fcntl(
            std::os::unix::io::AsRawFd::as_raw_fd(file),
            libc::F_NOCACHE,
            1,
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn set_nocache(_file: &File) {}

/// Maps a safetensors header dtype string to candle's [`DType`].
fn st_dtype(dtype: &str, name: &str) -> Result<DType> {
    Ok(match dtype {
        "U8" => DType::U8,
        "U32" => DType::U32,
        "I64" => DType::I64,
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        "F32" => DType::F32,
        "F8_E4M3" => DType::F8E4M3,
        other => candle_core::bail!("streamed expert tensor {name}: unsupported dtype {other}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config::Activation;
    use candle_core::Device;

    /// Writes two layers x three experts of gate/up `[4, 8]` and down
    /// `[8, 4]` F32 weights (candle's CPU matmul has no BF16; the fetch
    /// pipeline under test is dtype-generic), plus one stacked u8 tensor
    /// `[3, 2, 2, 16]` mimicking the GPT-OSS layout on dim 0.
    fn write_synthetic_checkpoint(dir: &std::path::Path) -> std::path::PathBuf {
        let device = Device::Cpu;
        let mut tensors: Vec<(String, Tensor)> = Vec::new();
        for layer in 0..2 {
            for e in 0..3 {
                let p = format!("model.layers.{layer}.mlp.experts.{e}");
                for (proj, shape) in [
                    ("gate_proj", (4, 8)),
                    ("up_proj", (4, 8)),
                    ("down_proj", (8, 4)),
                ] {
                    let base = (layer * 100 + e * 10) as f32;
                    let data: Vec<f32> = (0..4 * 8).map(|i| base + i as f32 * 0.25).collect();
                    let t = Tensor::from_vec(data, shape, &device).unwrap();
                    tensors.push((format!("{p}.{proj}.weight"), t));
                }
            }
        }
        let stacked: Vec<u8> = (0..3 * 2 * 2 * 16).map(|i| (i % 251) as u8).collect();
        tensors.push((
            "model.layers.0.mlp.experts.stacked_u8".to_string(),
            Tensor::from_vec(stacked, (3, 2, 2, 16), &device).unwrap(),
        ));

        let path = dir.join("experts.safetensors");
        candle_core::safetensors::save(
            &tensors
                .into_iter()
                .collect::<std::collections::HashMap<_, _>>(),
            &path,
        )
        .unwrap();
        path
    }

    fn pool_over(path: &std::path::Path, cache_bytes: usize) -> StreamedExperts {
        StreamedExperts::new(
            &[path.to_str().unwrap()],
            ExpertStreamConfig {
                layout: ExpertLayout::Standard,
                cache_bytes,
            },
            Device::Cpu,
            DType::F32,
        )
        .unwrap()
    }

    /// Contract: a streamed expert is bitwise identical in behaviour to the
    /// same expert built from directly loaded tensors (identical ops on
    /// identical bytes must give identical outputs on CPU).
    #[test]
    fn streamed_expert_matches_direct_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_synthetic_checkpoint(dir.path());
        let pool = pool_over(&path, usize::MAX);

        let streamed = pool.fetch(1, 2).unwrap();

        let direct = candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
        let get = |proj: &str| {
            direct
                .get(&format!("model.layers.1.mlp.experts.2.{proj}.weight"))
                .unwrap()
                .clone()
        };
        let reference = MoeExpert::Standard {
            gate_proj: AnyLinear::from_weight(get("gate_proj"), None).unwrap(),
            up_proj: AnyLinear::from_weight(get("up_proj"), None).unwrap(),
            down_proj: AnyLinear::from_weight(get("down_proj"), None).unwrap(),
        };

        let x_data: Vec<f32> = (0..2 * 8).map(|i| (i as f32 * 0.17).sin()).collect();
        let x = Tensor::from_vec(x_data, (2, 8), &Device::Cpu).unwrap();
        let got = streamed
            .forward(&x, Activation::SiLU)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = reference
            .forward(&x, Activation::SiLU)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got, want);
    }

    /// Contract: a positioned row read on dim 0 returns exactly the bytes of
    /// `narrow(0, e, 1)` on the whole tensor, for every row.
    #[test]
    fn read_row_matches_narrow() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_synthetic_checkpoint(dir.path());
        let pool = pool_over(&path, usize::MAX);

        let direct = candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
        let whole = direct.get("model.layers.0.mlp.experts.stacked_u8").unwrap();
        for e in 0..3 {
            let read = pool
                .read_one("model.layers.0.mlp.experts.stacked_u8", Some(e))
                .unwrap();
            let row =
                Tensor::from_raw_buffer(&read.bytes, read.dtype, &read.dims, &Device::Cpu).unwrap();
            let want = whole
                .narrow(0, e, 1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u8>()
                .unwrap();
            let got = row.flatten_all().unwrap().to_vec1::<u8>().unwrap();
            assert_eq!(got, want, "row {e}");
        }
    }

    /// Contract: a mixed hit/miss batch returns experts in the order of
    /// `ids`, deduplicates repeated ids, and each returned expert behaves
    /// identically to one obtained through a sequential single fetch.
    #[test]
    fn fetch_many_preserves_order_and_matches_sequential() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_synthetic_checkpoint(dir.path());

        let batched = pool_over(&path, usize::MAX);
        let sequential = pool_over(&path, usize::MAX);

        sequential.fetch(0, 1).unwrap();
        batched.fetch(0, 1).unwrap();

        let many = batched.fetch_many(0, &[2, 1, 0, 2]).unwrap();
        assert_eq!(many.len(), 4);
        assert!(Arc::ptr_eq(&many[0], &many[3]), "duplicate id, same expert");
        let (hits, misses, _) = batched.stats();
        assert_eq!(
            (hits, misses),
            (1, 3),
            "one warm hit; the pre-warm miss plus two batch builds; a \
             duplicated miss id within one batch is a single build"
        );

        let x_data: Vec<f32> = (0..2 * 8).map(|i| (i as f32 * 0.29).cos()).collect();
        let x = Tensor::from_vec(x_data, (2, 8), &Device::Cpu).unwrap();
        for (slot, e) in [(0usize, 2usize), (1, 1), (2, 0)] {
            let seq = sequential.fetch(0, e).unwrap();
            let a = many[slot]
                .forward(&x, Activation::SiLU)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let b = seq
                .forward(&x, Activation::SiLU)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(a, b, "expert {e}");
        }
    }

    /// Contract: the LRU respects its byte budget (800 bytes here, fitting
    /// two 384-byte experts), evicts the least recently used expert first,
    /// and evicted experts re-fetch as misses while survivors stay hits.
    #[test]
    fn lru_evicts_least_recent_and_refetches() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_synthetic_checkpoint(dir.path());
        let pool = pool_over(&path, 800);

        pool.fetch(0, 0).unwrap();
        pool.fetch(0, 1).unwrap();
        pool.fetch(0, 0).unwrap(); // refresh expert 0
        pool.fetch(0, 2).unwrap(); // evicts expert 1 (LRU)

        let (hits, misses, bytes) = pool.stats();
        assert_eq!((hits, misses), (1, 3));
        assert!(bytes <= 800, "cache bytes {bytes} over budget");

        pool.fetch(0, 0).unwrap();
        pool.fetch(0, 1).unwrap();
        let (hits, misses, _) = pool.stats();
        assert_eq!((hits, misses), (2, 4));
    }
}
