//! Zero-copy loader and accessor for GGUF weight files.
//!
//! [`GgufWeights`] memory-maps one or more GGUF files, parses the header, and
//! builds an `Arc<QTensor>` per tensor whose data points directly into the
//! mapped pages (the mmaps are kept alive for the struct's lifetime). Tensor
//! materialisation is parallelised with rayon. Besides tensor access it exposes
//! typed getters over the GGUF `metadata` map.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use anyhow::Context;
use candle_core::Device;
use candle_core::quantized::gguf_file;
use candle_core::quantized::{GgmlDType, QTensor};
use memmap2::Mmap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// A loaded GGUF model: quantized tensors by name, the raw metadata map, and the
/// backing memory maps held alive so the tensor data stays valid.
pub struct GgufWeights {
    tensors: FxHashMap<String, Arc<QTensor>>,
    pub metadata: HashMap<String, gguf_file::Value>,
    /// Where each tensor's bytes live in the maps, so a caller that wants to
    /// decode them itself does not have to go through the device.
    raw: FxHashMap<String, RawTensor>,
    _mmaps: Vec<Mmap>,
}

/// One tensor's quantized bytes: which map holds them, where, and how to read
/// them.
struct RawTensor {
    shard: usize,
    start: usize,
    end: usize,
    dtype: GgmlDType,
    dims: Vec<usize>,
}

/// Reads any of GGUF's integer widths as a `u32`.
///
/// A converter is free to write a count as `int32` where another writes
/// `uint32`, and both mean the same thing; insisting on one of them rejects a
/// file for how it spelled a number.
fn any_integer(v: &gguf_file::Value) -> Option<u32> {
    v.to_u32()
        .ok()
        .or_else(|| v.to_i32().ok().and_then(|x| u32::try_from(x).ok()))
        .or_else(|| v.to_u64().ok().and_then(|x| u32::try_from(x).ok()))
        .or_else(|| v.to_i64().ok().and_then(|x| u32::try_from(x).ok()))
        .or_else(|| v.to_u16().ok().map(u32::from))
        .or_else(|| v.to_i16().ok().and_then(|x| u32::try_from(x).ok()))
        .or_else(|| v.to_u8().ok().map(u32::from))
        .or_else(|| v.to_i8().ok().and_then(|x| u32::try_from(x).ok()))
}

impl GgufWeights {
    /// Loads a single GGUF file: mmaps it, parses the header, and materialises
    /// every tensor onto `device`.
    ///
    /// ## Errors
    /// Fails if the file cannot be opened or mapped, the GGUF header is invalid,
    /// or a tensor cannot be built.
    pub fn load(path: &str, device: &Device) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file: {}", path))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap GGUF file: {}", path))?;

        let mut cursor = Cursor::new(&mmap[..]);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF header: {}", e))?;

        tracing::info!(
            tensors = content.tensor_infos.len(),
            metadata_entries = content.metadata.len(),
            file_bytes = mmap.len(),
            "GGUF mmap+header parsed"
        );

        prefault(&file, mmap.len());
        let tensors = parallelise_tensor_load(
            &mmap,
            content.tensor_data_offset,
            &content.tensor_infos,
            device,
        )?;

        let raw = index_raw(0, content.tensor_data_offset, &content.tensor_infos);
        Ok(Self {
            tensors,
            metadata: content.metadata,
            raw,
            _mmaps: vec![mmap],
        })
    }

    /// Returns the tensor named `name`.
    ///
    /// ## Errors
    /// Fails if no tensor with that name exists.
    pub fn get(&self, name: &str) -> candle_core::Result<Arc<QTensor>> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("GGUF tensor not found: {}", name)))
    }

    /// Returns the tensor named `name`, or `None` if it is absent.
    pub fn try_get(&self, name: &str) -> Option<Arc<QTensor>> {
        self.tensors.get(name).cloned()
    }

    /// The tensor's bytes as the file stores them, with its type and shape.
    ///
    /// Decoding them directly is worth it where the alternative goes through
    /// candle's dequantiser, which blits the tensor back to the host and walks
    /// it on one thread: five seconds for a vocabulary-sized embedding.
    pub fn raw_tensor(&self, name: &str) -> Option<(&[u8], GgmlDType, &[usize])> {
        let r = self.raw.get(name)?;
        let map = self._mmaps.get(r.shard)?;
        Some((&map[r.start..r.end], r.dtype, &r.dims))
    }

    /// Total on-device size of all loaded tensors, in bytes.
    pub fn total_size_bytes(&self) -> usize {
        self.tensors
            .values()
            .map(|qt| qt.storage_size_in_bytes())
            .sum()
    }

    /// Reads metadata `key` as a `u32`.
    ///
    /// ## Errors
    /// Fails if the key is missing or not a `u32`.
    pub fn metadata_u32(&self, key: &str) -> anyhow::Result<u32> {
        self.metadata
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("Missing GGUF metadata key: {}", key))
            .and_then(|v| {
                v.to_u32()
                    .map_err(|e| anyhow::anyhow!("Bad u32 for '{}': {}", key, e))
            })
    }

    /// Reads metadata `key` as an `f32`.
    ///
    /// ## Errors
    /// Fails if the key is missing or not an `f32`.
    pub fn metadata_f32(&self, key: &str) -> anyhow::Result<f32> {
        self.metadata
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("Missing GGUF metadata key: {}", key))
            .and_then(|v| {
                v.to_f32()
                    .map_err(|e| anyhow::anyhow!("Bad f32 for '{}': {}", key, e))
            })
    }

    /// Reads metadata `key` as a `String`.
    ///
    /// ## Errors
    /// Fails if the key is missing or not a string.
    pub fn metadata_string(&self, key: &str) -> anyhow::Result<String> {
        self.metadata
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("Missing GGUF metadata key: {}", key))
            .and_then(|v| {
                v.to_string()
                    .cloned()
                    .map_err(|e| anyhow::anyhow!("Bad string for '{}': {}", key, e))
            })
    }

    /// Reads metadata `key` as an array of `u32`.
    ///
    /// `None` when the key is missing or is not an array, which is how a caller
    /// distinguishes a file that publishes a value per layer from one that
    /// publishes a single value for all of them.
    pub fn metadata_u32_array(&self, key: &str) -> Option<Vec<u32>> {
        let items = self.metadata.get(key)?.to_vec().ok()?;
        items.iter().map(any_integer).collect()
    }

    /// Reads metadata `key` as an array of `bool`. `None` as for
    /// [`metadata_u32_array`](Self::metadata_u32_array).
    pub fn metadata_bool_array(&self, key: &str) -> Option<Vec<bool>> {
        let items = self.metadata.get(key)?.to_vec().ok()?;
        items.iter().map(|v| v.to_bool().ok()).collect()
    }

    /// Reads metadata `key` as a `u32`, falling back to `default` if missing or
    /// the wrong type.
    pub fn metadata_u32_or(&self, key: &str, default: u32) -> u32 {
        self.metadata_u32(key).unwrap_or(default)
    }

    /// Reads metadata `key` as an `f32`, falling back to `default` if missing or
    /// the wrong type.
    pub fn metadata_f32_or(&self, key: &str, default: f32) -> f32 {
        self.metadata_f32(key).unwrap_or(default)
    }

    /// Loads a sharded GGUF model, merging the tensors of every shard into one
    /// [`GgufWeights`]; metadata is taken from the first shard.
    ///
    /// ## Errors
    /// Fails if `paths` is empty, or if any shard cannot be opened, mapped,
    /// parsed, or loaded.
    pub fn load_shards(paths: &[&str], device: &Device) -> anyhow::Result<Self> {
        anyhow::ensure!(!paths.is_empty(), "load_shards: paths must be non-empty");
        if paths.len() == 1 {
            return Self::load(paths[0], device);
        }
        let mut tensors = FxHashMap::default();
        let mut metadata = HashMap::new();
        let mut mmaps = Vec::with_capacity(paths.len());
        let mut raw: FxHashMap<String, RawTensor> = FxHashMap::default();
        let total_shards = paths.len();
        let mut total_tensors = 0usize;
        for (shard_idx, path) in paths.iter().enumerate() {
            let file = std::fs::File::open(path)
                .with_context(|| format!("Failed to open GGUF shard: {}", path))?;
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("Failed to mmap GGUF shard: {}", path))?;
            let mut cursor = Cursor::new(&mmap[..]);
            let content = gguf_file::Content::read(&mut cursor)
                .map_err(|e| anyhow::anyhow!("Failed to parse GGUF shard '{}': {}", path, e))?;
            if shard_idx == 0 {
                metadata = content.metadata.clone();
                tracing::info!(
                    shard = shard_idx + 1,
                    total_shards,
                    tensors = content.tensor_infos.len(),
                    metadata_entries = content.metadata.len(),
                    "GGUF shard mmap+header parsed"
                );
            } else {
                tracing::info!(
                    shard = shard_idx + 1,
                    total_shards,
                    tensors = content.tensor_infos.len(),
                    "GGUF shard mmap+header parsed"
                );
            }
            total_tensors += content.tensor_infos.len();
            raw.extend(index_raw(
                shard_idx,
                content.tensor_data_offset,
                &content.tensor_infos,
            ));
            let shard_tensors = parallelise_tensor_load(
                &mmap,
                content.tensor_data_offset,
                &content.tensor_infos,
                device,
            )
            .with_context(|| format!("Failed to load tensors from shard '{}'", path))?;
            tensors.extend(shard_tensors);
            mmaps.push(mmap);
        }
        tracing::info!(
            total_tensors,
            total_shards,
            "GGUF tensors loaded from mmapped shards"
        );
        Ok(Self {
            tensors,
            metadata,
            raw,
            _mmaps: mmaps,
        })
    }

    /// Returns the `general.architecture` metadata string (e.g. `llama`,
    /// `qwen2`, `qwen35`).
    ///
    /// ## Errors
    /// Fails if the key is absent.
    pub fn architecture(&self) -> anyhow::Result<String> {
        self.metadata_string("general.architecture")
    }

    /// Collects the end-of-sequence token ids from metadata: the single
    /// `eos_token_id` plus any in the `eos_token_ids` array, de-duplicated.
    pub fn eos_token_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        if let Ok(eos) = self.metadata_u32("tokenizer.ggml.eos_token_id") {
            ids.push(eos);
        }
        if let Some(gguf_file::Value::Array(arr)) =
            self.metadata.get("tokenizer.ggml.eos_token_ids")
        {
            for v in arr {
                if let gguf_file::Value::U32(id) = v
                    && !ids.contains(id)
                {
                    ids.push(*id);
                }
            }
        }
        ids
    }
}

/// Undoes the per-head row interleave that `convert_hf_to_gguf.py` applies to
/// the q/k projections of Llama-family architectures (llama, mistral, granite).
///
/// llama.cpp rotates consecutive dimension pairs in RoPE, so its converter
/// reorders each head's output rows from the HF layout `[first_half |
/// second_half]` to the interleaved `[0, h/2, 1, h/2+1, ...]`. Our
/// [`super::rope::RotaryEmbedding`] is NeoX/HF split-half and needs the
/// original layout, so those tensors must be de-interleaved at load. Each row
/// is quantized independently (blocks run along the input dimension), which
/// makes this a pure row-wise byte shuffle valid for every GGML dtype.
///
/// ## Errors
/// Fails if the tensor is not 2-D, its row count is not `n_heads * head_dim`
/// with an even `head_dim`, or the rebuilt tensor cannot be constructed.
pub fn depermute_qk_rows(
    qt: &QTensor,
    n_heads: usize,
    head_dim: usize,
    device: &Device,
) -> candle_core::Result<QTensor> {
    let dims = qt.shape().dims().to_vec();
    if dims.len() != 2 || dims[0] != n_heads * head_dim || !head_dim.is_multiple_of(2) {
        candle_core::bail!(
            "depermute_qk_rows: shape {dims:?} incompatible with {n_heads} heads x {head_dim} dims"
        );
    }
    let k = dims[1];
    let dtype = qt.dtype();
    if !k.is_multiple_of(dtype.block_size()) {
        candle_core::bail!(
            "depermute_qk_rows: row length {k} not divisible by {:?} block size {}",
            dtype,
            dtype.block_size()
        );
    }
    let row_bytes = k / dtype.block_size() * dtype.type_size();
    let data = qt.data()?;
    let mut out = vec![0u8; data.len()];
    let half = head_dim / 2;
    for h in 0..n_heads {
        for r in 0..head_dim {
            // HF row `r` of this head sits at interleaved row `2*(r%half) + r/half`.
            let src = h * head_dim + 2 * (r % half) + r / half;
            let dst = h * head_dim + r;
            out[dst * row_bytes..(dst + 1) * row_bytes]
                .copy_from_slice(&data[src * row_bytes..(src + 1) * row_bytes]);
        }
    }
    candle_core::quantized::ggml_file::qtensor_from_ggml(dtype, &out, dims, device)
}

/// Builds every tensor from the mmap in parallel (rayon), keyed by name.
/// Brings the whole file into the page cache before any tensor is built.
///
/// Materialising a tensor reads its slice, and building them one at a time
/// therefore reads the file one tensor at a time, through a lock, at a queue
/// depth of one: measured at 7.4 s for a 7 GB checkpoint that a plain
/// sequential read brings in in 1.05. Reading it here in parallel, in units the
/// device likes, hands that loop a warm map and costs a second.
///
/// The reads go into a small scratch buffer that is thrown away: what is wanted
/// is the side effect on the page cache, and the map is what the tensors are
/// built from. Touching one byte per page instead does work, but at a third of
/// the rate a read does.
fn prefault(file: &std::fs::File, len: usize) {
    use std::os::unix::fs::FileExt;
    const SLICE: usize = 64 * 1024 * 1024;
    let slices: Vec<usize> = (0..len).step_by(SLICE).collect();
    slices.par_iter().for_each(|&start| {
        let mut scratch = vec![0u8; SLICE.min(len - start)];
        let _ = file.read_at(&mut scratch, start as u64);
        std::hint::black_box(scratch.len());
    });
}

/// Byte range of one tensor inside the map that holds it.
fn raw_range(data_offset: u64, info: &gguf_file::TensorInfo) -> (usize, usize) {
    let elems = info.shape.elem_count();
    let size = elems / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
    let start = (data_offset + info.offset) as usize;
    (start, start + size)
}

/// Decodes a tensor's quantized bytes into `dtype`, in parallel.
///
/// candle's own dequantiser blits the tensor back to the host and walks it on
/// one thread through an `f32` buffer: for a 262144 by 3840 embedding that is
/// five seconds and four gigabytes of scratch, which is most of what loading a
/// 12B checkpoint costs. This walks row blocks across the rayon pool and writes
/// the target type straight out, keeping the scratch to one small buffer per
/// worker. Types it does not know fall back to the caller's slow path.
pub fn dequantize_rows_parallel(
    bytes: &[u8],
    dtype: GgmlDType,
    dims: &[usize],
    out_dtype: candle_core::DType,
    device: &Device,
) -> Option<candle_core::Result<candle_core::Tensor>> {
    use candle_core::quantized::k_quants::GgmlType;

    let elems: usize = dims.iter().product();
    let block = dtype.block_size();
    if elems == 0 || !elems.is_multiple_of(block) {
        return None;
    }
    // Rows keep the decode aligned to whole blocks.
    let row = *dims.last()?;
    if !row.is_multiple_of(block) || row == 0 {
        return None;
    }
    let rows = elems / row;
    let bytes_per_row = row / block * dtype.type_size();
    if bytes.len() < rows * bytes_per_row {
        return None;
    }

    macro_rules! decodifica {
        ($blocco:ty, $conversione:expr) => {{
            let mut out = vec![$conversione(0.0f32); elems];
            out.par_chunks_mut(row * ROWS_PER_TASK)
                .enumerate()
                .for_each(|(i, dst)| {
                    let first = i * ROWS_PER_TASK;
                    let n = dst.len() / row;
                    let src = &bytes[first * bytes_per_row..(first + n) * bytes_per_row];
                    let blocks: &[$blocco] = unsafe {
                        std::slice::from_raw_parts(
                            src.as_ptr() as *const $blocco,
                            src.len() / std::mem::size_of::<$blocco>(),
                        )
                    };
                    let mut scratch = vec![0f32; dst.len()];
                    <$blocco as GgmlType>::to_float(blocks, &mut scratch);
                    for (o, v) in dst.iter_mut().zip(scratch.iter()) {
                        *o = $conversione(*v);
                    }
                });
            Some(candle_core::Tensor::from_vec(out, dims, device))
        }};
    }

    const ROWS_PER_TASK: usize = 64;
    use candle_core::quantized::k_quants as kq;
    match (dtype, out_dtype) {
        (GgmlDType::Q4K, candle_core::DType::BF16) => {
            decodifica!(kq::BlockQ4K, half::bf16::from_f32)
        }
        (GgmlDType::Q6K, candle_core::DType::BF16) => {
            decodifica!(kq::BlockQ6K, half::bf16::from_f32)
        }
        (GgmlDType::Q5K, candle_core::DType::BF16) => {
            decodifica!(kq::BlockQ5K, half::bf16::from_f32)
        }
        (GgmlDType::Q8_0, candle_core::DType::BF16) => {
            decodifica!(kq::BlockQ8_0, half::bf16::from_f32)
        }
        (GgmlDType::Q4_0, candle_core::DType::BF16) => {
            decodifica!(kq::BlockQ4_0, half::bf16::from_f32)
        }
        _ => None,
    }
}

fn index_raw(
    shard: usize,
    data_offset: u64,
    infos: &HashMap<String, gguf_file::TensorInfo>,
) -> FxHashMap<String, RawTensor> {
    infos
        .iter()
        .map(|(name, info)| {
            let (start, end) = raw_range(data_offset, info);
            (
                name.clone(),
                RawTensor {
                    shard,
                    start,
                    end,
                    dtype: info.ggml_dtype,
                    dims: info.shape.dims().to_vec(),
                },
            )
        })
        .collect()
}

fn parallelise_tensor_load(
    mmap: &Mmap,
    data_offset: u64,
    tensor_infos: &HashMap<String, gguf_file::TensorInfo>,
    device: &Device,
) -> anyhow::Result<FxHashMap<String, Arc<QTensor>>> {
    let infos: Vec<(&String, &gguf_file::TensorInfo)> = tensor_infos.iter().collect();
    let pairs: anyhow::Result<Vec<(String, Arc<QTensor>)>> = infos
        .par_iter()
        .map(|(name, info)| {
            let qt = build_qtensor_from_mmap(mmap, data_offset, info, device)
                .with_context(|| format!("Failed to load GGUF tensor '{}'", name))?;
            Ok(((*name).clone(), Arc::new(qt)))
        })
        .collect();
    let pairs = pairs?;
    let mut tensors = FxHashMap::with_capacity_and_hasher(pairs.len(), Default::default());
    tensors.extend(pairs);
    Ok(tensors)
}

/// Builds one `QTensor` from its slice of the memory map, validating the element
/// count against the block size and that the slice lies within bounds.
fn build_qtensor_from_mmap(
    mmap: &Mmap,
    data_offset: u64,
    info: &gguf_file::TensorInfo,
    device: &Device,
) -> anyhow::Result<QTensor> {
    let tensor_elems = info.shape.elem_count();
    let block_size = info.ggml_dtype.block_size();
    if !tensor_elems.is_multiple_of(block_size) {
        anyhow::bail!(
            "tensor elements {} not divisible by block size {}",
            tensor_elems,
            block_size
        );
    }
    let size_in_bytes = tensor_elems / block_size * info.ggml_dtype.type_size();
    let start = (data_offset + info.offset) as usize;
    let end = start
        .checked_add(size_in_bytes)
        .ok_or_else(|| anyhow::anyhow!("tensor offset overflow"))?;
    if end > mmap.len() {
        anyhow::bail!(
            "tensor slice ({}..{}) out of mmap bounds ({})",
            start,
            end,
            mmap.len()
        );
    }
    let slice = &mmap[start..end];
    // Serialize Metal storage creation across the rayon workers: candle
    // 0.11's residency-set registration is not thread-safe (see
    // `weights::metal_alloc_lock`).
    let _guard = device
        .is_metal()
        .then(|| crate::common::weights::metal_alloc_lock().lock().unwrap());
    candle_core::quantized::ggml_file::qtensor_from_ggml(
        info.ggml_dtype,
        slice,
        info.shape.dims().to_vec(),
        device,
    )
    .map_err(|e| anyhow::anyhow!("qtensor_from_ggml failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use candle_core::quantized::GgmlDType;

    /// Contract: decoding a tensor's bytes in parallel gives exactly what
    /// candle's own dequantiser gives.
    ///
    /// The fast path exists because candle's blits the tensor back to the host
    /// and walks it on one thread: five of the six seconds a 12B checkpoint
    /// takes to build. Faster is only worth having if it is the same, so this
    /// compares the two element by element on every type the fast path claims.
    #[test]
    fn decoding_in_parallel_matches_candle() {
        use candle_core::quantized::GgmlDType;
        let dev = Device::Cpu;
        for gtype in [
            GgmlDType::Q4K,
            GgmlDType::Q6K,
            GgmlDType::Q5K,
            GgmlDType::Q8_0,
            GgmlDType::Q4_0,
        ] {
            // Rows of 512 so every supported block size divides them.
            let (rows, row) = (200usize, 512usize);
            let valori: Vec<f32> = (0..rows * row)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.021)
                .collect();
            let denso = Tensor::from_vec(valori, (rows, row), &dev).unwrap();
            let qt = QTensor::quantize(&denso, gtype).unwrap();
            let atteso: Vec<f32> = qt
                .dequantize(&dev)
                .unwrap()
                .to_dtype(candle_core::DType::BF16)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<half::bf16>()
                .unwrap()
                .into_iter()
                .map(f32::from)
                .collect();

            let bytes = qt.data().unwrap();
            let ottenuto: Vec<f32> = dequantize_rows_parallel(
                &bytes,
                gtype,
                &[rows, row],
                candle_core::DType::BF16,
                &dev,
            )
            .unwrap_or_else(|| panic!("{gtype:?} should take the fast path"))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<half::bf16>()
            .unwrap()
            .into_iter()
            .map(f32::from)
            .collect();

            assert_eq!(ottenuto.len(), atteso.len(), "{gtype:?}: length");
            assert_eq!(ottenuto, atteso, "{gtype:?}: values differ from candle");
        }
    }

    /// Contract: `depermute_qk_rows` is the exact inverse of the
    /// `convert_hf_to_gguf.py` Llama-family permute
    /// (`reshape(n_head, 2, hd/2, k).swapaxes(1, 2)`), so a GGUF q/k tensor
    /// comes back in the HF row layout our split-half RoPE expects.
    #[test]
    fn depermute_qk_rows_inverts_converter_permute() {
        let dev = Device::Cpu;
        let (n_heads, head_dim, k) = (2usize, 8usize, 32usize);
        let n = n_heads * head_dim;

        let hf: Vec<f32> = (0..n * k).map(|i| i as f32).collect();
        let hf_t = Tensor::from_vec(hf.clone(), (n, k), &dev).unwrap();

        // Apply the converter's permute in HF row space.
        let half = head_dim / 2;
        let mut permuted = vec![0f32; n * k];
        for h in 0..n_heads {
            for j in 0..head_dim {
                let src = h * head_dim + (j % 2) * half + j / 2;
                let dst = h * head_dim + j;
                permuted[dst * k..(dst + 1) * k].copy_from_slice(&hf[src * k..(src + 1) * k]);
            }
        }
        let permuted_t = Tensor::from_vec(permuted, (n, k), &dev).unwrap();
        let qt = QTensor::quantize(&permuted_t, GgmlDType::F32).unwrap();

        let restored = depermute_qk_rows(&qt, n_heads, head_dim, &dev).unwrap();
        let restored_t = restored.dequantize(&dev).unwrap();

        let a = restored_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = hf_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b, "depermute must restore the HF row layout exactly");
    }

    #[test]
    fn depermute_qk_rows_rejects_bad_geometry() {
        let dev = Device::Cpu;
        let t = Tensor::zeros((16, 32), candle_core::DType::F32, &dev).unwrap();
        let qt = QTensor::quantize(&t, GgmlDType::F32).unwrap();
        assert!(depermute_qk_rows(&qt, 3, 8, &dev).is_err());
    }
}
