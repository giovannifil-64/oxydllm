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
use candle_core::quantized::QTensor;
use candle_core::quantized::gguf_file;
use memmap2::Mmap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// A loaded GGUF model: quantized tensors by name, the raw metadata map, and the
/// backing memory maps held alive so the tensor data stays valid.
pub struct GgufWeights {
    tensors: FxHashMap<String, Arc<QTensor>>,
    pub metadata: HashMap<String, gguf_file::Value>,
    _mmaps: Vec<Mmap>,
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

        let tensors = parallelise_tensor_load(
            &mmap,
            content.tensor_data_offset,
            &content.tensor_infos,
            device,
        )?;

        Ok(Self {
            tensors,
            metadata: content.metadata,
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
