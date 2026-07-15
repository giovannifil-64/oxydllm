//! On-demand streaming of MoE expert weights from the checkpoint mmap.
//!
//! For MoE checkpoints larger than device memory, only the router-selected
//! experts need to be resident at any moment. [`StreamedExperts`] retains the
//! safetensors mmap after load, serves experts through an LRU cache with a
//! byte budget, and fetches misses by slicing the expert's bytes straight out
//! of the mapped file (zero-copy on the CPU side, one host-to-device transfer
//! per tensor).
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
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Result, Tensor};
use rustc_hash::FxHashMap;
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

/// Configuration handed from the model loader to [`ModelWeights::load`]
/// (`src/common/weights.rs`) when expert streaming is requested.
pub struct ExpertStreamConfig {
    pub layout: ExpertLayout,
    pub cache_bytes: usize,
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
}

/// Shared expert pool: one per model, referenced by every MoE layer.
pub struct StreamedExperts {
    mmap: MmapedSafetensors,
    layout: ExpertLayout,
    device: Device,
    dtype: DType,
    cache_bytes: usize,
    state: Mutex<LruState>,
}

impl StreamedExperts {
    /// Builds the pool over the checkpoint `mmap` retained from load.
    pub fn new(
        mmap: MmapedSafetensors,
        cfg: ExpertStreamConfig,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self {
            mmap,
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
            }),
        }
    }

    /// Returns expert `e` of layer `layer_idx`, fetching it from the mmap on a
    /// cache miss. The returned `Arc` keeps the expert alive for the caller's
    /// dispatch even if it is evicted concurrently.
    ///
    /// Eviction runs before insertion so the byte budget bounds the peak, and
    /// each eviction batch is followed by a Metal drain: dropped buffers are
    /// only reclaimed at sync points (see the two-phase note in
    /// [`ModelWeights::load`](super::weights::ModelWeights::load)).
    ///
    /// ## Errors
    /// Fails if a tensor is missing from the checkpoint, malformed, or a cast
    /// or device transfer fails.
    pub(crate) fn fetch(&self, layer_idx: usize, e: usize) -> Result<Arc<MoeExpert>> {
        let key = (layer_idx as u32, e as u32);
        let mut state = self.state.lock().unwrap();
        state.clock += 1;
        let clock = state.clock;
        if let Some(entry) = state.cache.get_mut(&key) {
            entry.last_use = clock;
            let expert = Arc::clone(&entry.expert);
            state.hits += 1;
            return Ok(expert);
        }
        state.misses += 1;

        let (expert, bytes) = match self.layout {
            ExpertLayout::Standard => self.build_standard(layer_idx, e)?,
            ExpertLayout::GptOss { swiglu_limit } => {
                self.build_gpt_oss(layer_idx, e, swiglu_limit)?
            }
        };

        let mut evicted = false;
        while state.bytes + bytes > self.cache_bytes && !state.cache.is_empty() {
            let (&victim, _) = state
                .cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_use)
                .expect("non-empty cache");
            let entry = state.cache.remove(&victim).expect("victim exists");
            state.bytes -= entry.bytes;
            drop(entry);
            evicted = true;
        }
        if evicted {
            drain_metal(&self.device, "expert eviction")
                .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?;
        }

        let expert = Arc::new(expert);
        state.bytes += bytes;
        state.cache.insert(
            key,
            CacheEntry {
                expert: Arc::clone(&expert),
                bytes,
                last_use: clock,
            },
        );
        Ok(expert)
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
    /// Loads a whole tensor by name from the mmap onto the device, replicating
    /// the resident loader's cast chain for byte identity: integer tensors as
    /// stored, FP8 dequantized through F32 with its block `*_scale_inv` folded
    /// in F32 after the runtime-dtype round-trip, floats cast to the runtime
    /// dtype.
    fn load_named(&self, name: &str) -> Result<Tensor> {
        let view = self.mmap.get(name)?;
        let file_dtype = st_dtype(view.dtype(), name)?;
        let cpu = Tensor::from_raw_buffer(view.data(), file_dtype, view.shape(), &Device::Cpu)?;
        let cpu = self.cast_like_resident(cpu, name)?;
        self.to_device(cpu, name)
    }

    /// Loads row `e` (dim 0) of a stacked tensor by slicing its bytes out of
    /// the mmap. Rows of dim-0 slices are contiguous for every dtype here
    /// because safetensors data is row-major and unstrided.
    fn load_row(&self, name: &str, e: usize) -> Result<Tensor> {
        let view = self.mmap.get(name)?;
        let dims = view.shape().to_vec();
        if dims.is_empty() || e >= dims[0] {
            candle_core::bail!("expert row {e} out of bounds for {name} with shape {dims:?}");
        }
        let data = view.data();
        let row_bytes = data.len() / dims[0];
        let slice = &data[e * row_bytes..(e + 1) * row_bytes];
        let file_dtype = st_dtype(view.dtype(), name)?;
        let cpu = Tensor::from_raw_buffer(slice, file_dtype, &dims[1..], &Device::Cpu)?;
        let cpu = self.cast_like_resident(cpu, name)?;
        self.to_device(cpu, name)
    }

    /// The resident loader's dtype rules (`load_tensor_with_dtype` +
    /// `apply_weight_scale_inv`), applied on the CPU.
    fn cast_like_resident(&self, t: Tensor, name: &str) -> Result<Tensor> {
        if matches!(
            t.dtype(),
            DType::U8 | DType::U32 | DType::I16 | DType::I32 | DType::I64
        ) {
            return Ok(t);
        }
        if t.dtype() == DType::F8E4M3 {
            let scale_name = format!("{name}_scale_inv");
            let dequant = t.to_dtype(DType::F32)?.to_dtype(self.dtype)?;
            let Ok(scale_view) = self.mmap.get(&scale_name) else {
                return Ok(dequant);
            };
            let scale = Tensor::from_raw_buffer(
                scale_view.data(),
                st_dtype(scale_view.dtype(), &scale_name)?,
                scale_view.shape(),
                &Device::Cpu,
            )?
            .to_dtype(DType::F32)?;
            let folded = apply_scale_inv(&dequant.to_dtype(DType::F32)?, &scale)?;
            return folded.to_dtype(self.dtype);
        }
        if t.dtype() == self.dtype {
            return Ok(t);
        }
        t.to_dtype(self.dtype)
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

    fn build_standard(&self, layer_idx: usize, e: usize) -> Result<(MoeExpert, usize)> {
        let p = format!("model.layers.{layer_idx}.mlp.experts.{e}");
        let gate = self.load_named(&format!("{p}.gate_proj.weight"))?;
        let up = self.load_named(&format!("{p}.up_proj.weight"))?;
        let down = self.load_named(&format!("{p}.down_proj.weight"))?;
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

    fn build_gpt_oss(
        &self,
        layer_idx: usize,
        e: usize,
        swiglu_limit: f64,
    ) -> Result<(MoeExpert, usize)> {
        let p = format!("model.layers.{layer_idx}.mlp.experts");
        let gu_blocks = self.load_row(&format!("{p}.gate_up_proj_blocks"), e)?;
        let gu_scales = self.load_row(&format!("{p}.gate_up_proj_scales"), e)?;
        let gu_bias = self.load_row(&format!("{p}.gate_up_proj_bias"), e)?;
        let dn_blocks = self.load_row(&format!("{p}.down_proj_blocks"), e)?;
        let dn_scales = self.load_row(&format!("{p}.down_proj_scales"), e)?;
        let dn_bias = self.load_row(&format!("{p}.down_proj_bias"), e)?;

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

/// Maps the safetensors dtype to candle's via its Debug name, matching the
/// repo convention (weights.rs) of not depending on the safetensors crate
/// directly (candle pins its own version).
fn st_dtype(dtype: impl std::fmt::Debug, name: &str) -> Result<DType> {
    Ok(match format!("{dtype:?}").as_str() {
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
        // SAFETY: the file is private to the test and outlives the mmap.
        let mmap = unsafe { MmapedSafetensors::multi(&[path]).unwrap() };
        StreamedExperts::new(
            mmap,
            ExpertStreamConfig {
                layout: ExpertLayout::Standard,
                cache_bytes,
            },
            Device::Cpu,
            DType::F32,
        )
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

    /// Contract: row slicing on dim 0 returns exactly the bytes of
    /// `narrow(0, e, 1)` on the whole tensor, for every row.
    #[test]
    fn load_row_matches_narrow() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_synthetic_checkpoint(dir.path());
        let pool = pool_over(&path, usize::MAX);

        let direct = candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
        let whole = direct.get("model.layers.0.mlp.experts.stacked_u8").unwrap();
        for e in 0..3 {
            let row = pool
                .load_row("model.layers.0.mlp.experts.stacked_u8", e)
                .unwrap();
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
