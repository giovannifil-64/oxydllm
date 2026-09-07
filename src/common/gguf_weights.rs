//! Zero-copy loader and accessor for GGUF weight files.
//!
//! [`GgufWeights`] memory-maps one or more GGUF files, parses the header, and
//! builds an `Arc<QTensor>` per tensor, plus, where a kernel of this crate can
//! read a weight in the buffer candle keeps it in, the [`StagedWeight`] handle
//! it reads through. Tensor materialisation is parallelised with rayon.
//! Besides tensor access it exposes typed getters over the GGUF `metadata` map.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::common::gguf_header::{GgufHeader, TensorEntry, candle_dtype, type_name};
use crate::common::linear::StagedWeight;
use anyhow::Context;
use candle_core::Device;
use candle_core::quantized::gguf_file;
use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use memmap2::Mmap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// A loaded GGUF model: quantized tensors by name, the handles the staged
/// GEMM reads some of them through, the raw metadata map, and the backing
/// memory maps held alive so the tensor data stays valid.
pub struct GgufWeights {
    tensors: FxHashMap<String, Arc<QTensor>>,
    staged: FxHashMap<String, StagedWeight>,
    owned: FxHashMap<String, StagedWeight>,
    pub metadata: HashMap<String, gguf_file::Value>,
    _mmaps: Vec<Mmap>,
}

/// A linear weight as the loader holds it.
#[derive(Clone, Debug)]
pub enum LinearWeight {
    /// Candle's tensor, with the handle the staged GEMM reads it through where
    /// a kernel of this crate reads that layout.
    Candle {
        tensor: Arc<QTensor>,
        staged: Option<StagedWeight>,
    },
    /// A weight in a layout candle has no name for, decoded by this crate's
    /// kernels alone.
    Owned(StagedWeight),
}

impl LinearWeight {
    /// Rows of the weight: the output features.
    pub fn out_features(&self) -> usize {
        self.dims().0
    }

    /// Columns of the weight: the input features.
    pub fn in_features(&self) -> usize {
        self.dims().1
    }

    fn dims(&self) -> (usize, usize) {
        match self {
            Self::Candle { tensor, .. } => {
                let d = tensor.shape().dims();
                (d[0], d.get(1).copied().unwrap_or(1))
            }
            Self::Owned(w) => owned_dims(w),
        }
    }

    /// Bytes an owned weight holds on the device.
    fn owned_bytes(w: &StagedWeight) -> usize {
        #[cfg(feature = "metal")]
        {
            let (n, k) = w.dims();
            n * k / w.kind().block_elems() * w.kind().block_bytes()
        }
        #[cfg(not(feature = "metal"))]
        match *w {}
    }
}

/// The kind of an owned weight, named.
fn owned_kind(w: &StagedWeight) -> String {
    #[cfg(feature = "metal")]
    {
        format!("{:?}", w.kind())
    }
    #[cfg(not(feature = "metal"))]
    match *w {}
}

fn owned_dims(w: &StagedWeight) -> (usize, usize) {
    #[cfg(feature = "metal")]
    {
        w.dims()
    }
    #[cfg(not(feature = "metal"))]
    match *w {}
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

        let header = GgufHeader::parse(&mmap[..])
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF header: {e}"))?;

        tracing::info!(
            tensors = header.tensors.len(),
            metadata_entries = header.content.metadata.len(),
            file_bytes = mmap.len(),
            "GGUF mmap+header parsed"
        );

        let (tensors, staged, owned) = parallelise_tensor_load(
            &file,
            mmap.len(),
            header.content.tensor_data_offset,
            &header.tensors,
            device,
        )?;

        Ok(Self {
            tensors,
            staged,
            owned,
            metadata: header.content.metadata,
            _mmaps: vec![mmap],
        })
    }

    /// Returns the tensor named `name`.
    ///
    /// ## Errors
    /// Fails if no tensor with that name exists.
    pub fn get(&self, name: &str) -> candle_core::Result<Arc<QTensor>> {
        self.tensors.get(name).cloned().ok_or_else(|| {
            candle_core::Error::Msg(match self.owned.get(name) {
                Some(w) => format!(
                    "GGUF tensor {name} is {}, which this engine decodes only as a linear weight",
                    owned_kind(w)
                ),
                None => format!("GGUF tensor not found: {}", name),
            })
        })
    }

    /// Returns the tensor named `name` as a linear weight, in whichever form
    /// the loader holds it.
    ///
    /// ## Errors
    /// Fails if no tensor with that name exists.
    pub fn linear_weight(&self, name: &str) -> candle_core::Result<LinearWeight> {
        self.try_linear_weight(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("GGUF tensor not found: {}", name)))
    }

    /// Returns the tensor named `name` as a linear weight, or `None` if it is
    /// absent.
    pub fn try_linear_weight(&self, name: &str) -> Option<LinearWeight> {
        if let Some(tensor) = self.tensors.get(name) {
            return Some(LinearWeight::Candle {
                tensor: tensor.clone(),
                staged: self.staged.get(name).cloned(),
            });
        }
        self.owned.get(name).cloned().map(LinearWeight::Owned)
    }

    /// Returns the tensor named `name`, or `None` if it is absent.
    pub fn try_get(&self, name: &str) -> Option<Arc<QTensor>> {
        self.tensors.get(name).cloned()
    }

    /// Total on-device size of all loaded tensors, in bytes.
    pub fn total_size_bytes(&self) -> usize {
        let candle: usize = self
            .tensors
            .values()
            .map(|qt| qt.storage_size_in_bytes())
            .sum();
        candle
            + self
                .owned
                .values()
                .map(LinearWeight::owned_bytes)
                .sum::<usize>()
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
        let mut staged = FxHashMap::default();
        let mut owned = FxHashMap::default();
        let mut metadata = HashMap::new();
        let mut mmaps = Vec::with_capacity(paths.len());
        let total_shards = paths.len();
        let mut total_tensors = 0usize;
        for (shard_idx, path) in paths.iter().enumerate() {
            let file = std::fs::File::open(path)
                .with_context(|| format!("Failed to open GGUF shard: {}", path))?;
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("Failed to mmap GGUF shard: {}", path))?;
            let header = GgufHeader::parse(&mmap[..])
                .map_err(|e| anyhow::anyhow!("Failed to parse GGUF shard '{}': {e}", path))?;
            if shard_idx == 0 {
                metadata = header.content.metadata.clone();
                tracing::info!(
                    shard = shard_idx + 1,
                    total_shards,
                    tensors = header.tensors.len(),
                    metadata_entries = header.content.metadata.len(),
                    "GGUF shard mmap+header parsed"
                );
            } else {
                tracing::info!(
                    shard = shard_idx + 1,
                    total_shards,
                    tensors = header.tensors.len(),
                    "GGUF shard mmap+header parsed"
                );
            }
            total_tensors += header.tensors.len();
            let (shard_tensors, shard_staged, shard_owned) = parallelise_tensor_load(
                &file,
                mmap.len(),
                header.content.tensor_data_offset,
                &header.tensors,
                device,
            )
            .with_context(|| format!("Failed to load tensors from shard '{}'", path))?;
            tensors.extend(shard_tensors);
            staged.extend(shard_staged);
            owned.extend(shard_owned);
            mmaps.push(mmap);
        }
        tracing::info!(
            total_tensors,
            total_shards,
            "GGUF tensors loaded from mmapped shards"
        );
        Ok(Self {
            tensors,
            staged,
            owned,
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
    weight: &LinearWeight,
    n_heads: usize,
    head_dim: usize,
    device: &Device,
) -> candle_core::Result<LinearWeight> {
    let (rows, k) = match weight {
        LinearWeight::Candle { tensor, .. } => {
            let dims = tensor.shape().dims();
            if dims.len() != 2 {
                candle_core::bail!("depermute_qk_rows: shape {dims:?} is not a matrix");
            }
            (dims[0], dims[1])
        }
        LinearWeight::Owned(w) => owned_dims(w),
    };
    if rows != n_heads * head_dim || !head_dim.is_multiple_of(2) {
        candle_core::bail!(
            "depermute_qk_rows: shape [{rows}, {k}] incompatible with {n_heads} heads x {head_dim} dims"
        );
    }
    let (data, row_bytes) = match weight {
        LinearWeight::Candle { tensor, .. } => {
            let dtype = tensor.dtype();
            if !k.is_multiple_of(dtype.block_size()) {
                candle_core::bail!(
                    "depermute_qk_rows: row length {k} not divisible by {:?} block size {}",
                    dtype,
                    dtype.block_size()
                );
            }
            (
                tensor.data()?.into_owned(),
                k / dtype.block_size() * dtype.type_size(),
            )
        }
        LinearWeight::Owned(w) => owned_rows(w, k)?,
    };
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
    match weight {
        LinearWeight::Candle { tensor, .. } => {
            let (qt, staged) = qtensor_with_staged(tensor.dtype(), &out, vec![rows, k], device)?;
            Ok(LinearWeight::Candle {
                tensor: Arc::new(qt),
                staged,
            })
        }
        LinearWeight::Owned(w) => owned_from_rows(w, &out, rows, k, device),
    }
}

/// The bytes of an owned weight and the length of one of its rows.
fn owned_rows(w: &StagedWeight, k: usize) -> candle_core::Result<(Vec<u8>, usize)> {
    #[cfg(feature = "metal")]
    {
        Ok((
            w.bytes(),
            k / w.kind().block_elems() * w.kind().block_bytes(),
        ))
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = k;
        match *w {}
    }
}

/// An owned weight of the same kind as `like` over `bytes`.
fn owned_from_rows(
    like: &StagedWeight,
    bytes: &[u8],
    rows: usize,
    k: usize,
    device: &Device,
) -> candle_core::Result<LinearWeight> {
    #[cfg(feature = "metal")]
    {
        let Device::Metal(md) = device else {
            candle_core::bail!("an owned weight lives on Metal");
        };
        let buffer = md.new_buffer_with_data(bytes)?;
        StagedWeight::of_kind((*buffer).clone(), like.kind(), &[rows, k])
            .map(LinearWeight::Owned)
            .ok_or_else(|| candle_core::Error::Msg("owned weight rows are not whole blocks".into()))
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = (bytes, rows, k, device);
        match *like {}
    }
}

/// Builds a `QTensor` from GGUF block bytes and, where a kernel of this crate
/// reads that layout in place, the handle it reads it through.
///
/// Candle's own loader does the first half and keeps the buffer to itself.
/// Building the storage here, before the `QTensor` takes it, is the one moment
/// the buffer is visible, and cloning it is a retain on the same allocation,
/// not a copy. Off Metal there is no such kernel and the handle is `None`.
///
/// ## Errors
/// Fails when `bytes` is not exactly `dims` worth of `dtype` blocks.
pub fn qtensor_with_staged(
    dtype: GgmlDType,
    bytes: &[u8],
    dims: Vec<usize>,
    device: &Device,
) -> candle_core::Result<(QTensor, Option<StagedWeight>)> {
    let elems: usize = dims.iter().product();
    let block = dtype.block_size();
    if !elems.is_multiple_of(block) {
        candle_core::bail!("tensor elements {elems} not divisible by block size {block}");
    }
    let expected = elems / block * dtype.type_size();
    if bytes.len() != expected {
        candle_core::bail!(
            "tensor {dims:?} of {dtype:?} needs {expected} bytes, got {}",
            bytes.len()
        );
    }
    let storage = QStorage::from_data(Cow::Borrowed(bytes), device, dtype)?;
    #[cfg(feature = "metal")]
    let staged = match &storage {
        QStorage::Metal(ms) => StagedWeight::new(ms.buffer().clone(), dtype, &dims),
        _ => None,
    };
    #[cfg(not(feature = "metal"))]
    let staged = None;
    Ok((QTensor::new(storage, dims)?, staged))
}

/// Builds every tensor from the file in parallel (rayon), keyed by name: the
/// tensors candle holds, beside them the staged handles of those a kernel of
/// this crate reads in place, and apart the weights only this crate decodes.
#[allow(clippy::type_complexity)]
fn parallelise_tensor_load(
    file: &std::fs::File,
    file_len: usize,
    data_offset: u64,
    entries: &[TensorEntry],
    device: &Device,
) -> anyhow::Result<(
    FxHashMap<String, Arc<QTensor>>,
    FxHashMap<String, StagedWeight>,
    FxHashMap<String, StagedWeight>,
)> {
    let mut infos: Vec<&TensorEntry> = entries.iter().collect();
    infos.sort_unstable_by_key(|e| e.offset);
    let built: anyhow::Result<Vec<(String, Built)>> = infos
        .par_iter()
        .map(|entry| {
            let built = build_tensor(file, file_len, data_offset, entry, device)
                .with_context(|| format!("Failed to load GGUF tensor '{}'", entry.name))?;
            Ok((entry.name.clone(), built))
        })
        .collect();
    let built = built?;
    let mut tensors = FxHashMap::with_capacity_and_hasher(built.len(), Default::default());
    let mut staged = FxHashMap::default();
    let mut owned = FxHashMap::default();
    for (name, b) in built {
        match b {
            Built::Candle(qt, handle) => {
                if let Some(handle) = handle {
                    staged.insert(name.clone(), handle);
                }
                tensors.insert(name, Arc::new(qt));
            }
            Built::Owned(w) => {
                owned.insert(name, w);
            }
        }
    }
    Ok((tensors, staged, owned))
}

/// One tensor as the loader built it.
enum Built {
    Candle(QTensor, Option<StagedWeight>),
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    Owned(StagedWeight),
}

/// Reads `size_in_bytes` of one tensor from the file, checking the range lies
/// within it.
///
/// The bytes are read rather than taken from the map on purpose. Reading is what
/// the device does well: a plain sequential read of a 7 GB checkpoint takes a
/// second, where faulting the same pages in one tensor at a time, through the
/// lock the Metal allocation needs, took seven. Reading also keeps the page
/// cache out of it, which matters because materialising the tensors allocates
/// as much memory again and would evict whatever the file had warmed.
fn read_tensor_bytes(
    file: &std::fs::File,
    file_len: usize,
    data_offset: u64,
    entry: &TensorEntry,
    size_in_bytes: usize,
) -> anyhow::Result<Vec<u8>> {
    let start = (data_offset + entry.offset) as usize;
    let end = start
        .checked_add(size_in_bytes)
        .ok_or_else(|| anyhow::anyhow!("tensor offset overflow"))?;
    if end > file_len {
        anyhow::bail!(
            "tensor slice ({}..{}) out of file bounds ({})",
            start,
            end,
            file_len
        );
    }
    let mut bytes = vec![0u8; size_in_bytes];
    use std::os::unix::fs::FileExt;
    file.read_exact_at(&mut bytes, start as u64)
        .with_context(|| format!("reading tensor bytes at {start}"))?;
    Ok(bytes)
}

/// Builds one tensor: candle's `QTensor` for a type candle names, validating
/// the element count against the block size, or an owned weight for a type
/// only this crate decodes.
fn build_tensor(
    file: &std::fs::File,
    file_len: usize,
    data_offset: u64,
    entry: &TensorEntry,
    device: &Device,
) -> anyhow::Result<Built> {
    let tensor_elems: usize = entry.dims.iter().product();
    if let Some(dtype) = candle_dtype(entry.type_id) {
        let block_size = dtype.block_size();
        if !tensor_elems.is_multiple_of(block_size) {
            anyhow::bail!(
                "tensor elements {} not divisible by block size {}",
                tensor_elems,
                block_size
            );
        }
        let size_in_bytes = tensor_elems / block_size * dtype.type_size();
        let bytes = read_tensor_bytes(file, file_len, data_offset, entry, size_in_bytes)?;
        // Serialize Metal storage creation across the rayon workers: candle
        // 0.11's residency-set registration is not thread-safe (see
        // `weights::metal_alloc_lock`).
        let _guard = device
            .is_metal()
            .then(|| crate::common::weights::metal_alloc_lock().lock().unwrap());
        let (qt, staged) = qtensor_with_staged(dtype, &bytes, entry.dims.clone(), device)
            .map_err(|e| anyhow::anyhow!("building the tensor failed: {}", e))?;
        return Ok(Built::Candle(qt, staged));
    }
    build_owned(file, file_len, data_offset, entry, device)
}

/// Builds a weight in a type candle has no name for, which this crate's
/// kernels decode on Metal as a two-dimensional linear weight.
#[cfg(feature = "metal")]
fn build_owned(
    file: &std::fs::File,
    file_len: usize,
    data_offset: u64,
    entry: &TensorEntry,
    device: &Device,
) -> anyhow::Result<Built> {
    use crate::common::metal_ops::WeightKind;
    let name = type_name(entry.type_id);
    let Some(kind) = WeightKind::from_ggml_id_iq(entry.type_id) else {
        anyhow::bail!("tensor type {name} is one this engine does not decode");
    };
    let Device::Metal(md) = device else {
        anyhow::bail!("{name} weights are decoded on Metal only");
    };
    if !crate::common::metal_ops::mpp_owned_kernels_available(md.device()) {
        anyhow::bail!("this GPU does not build the kernels that decode {name} weights");
    }
    let [n, k] = entry.dims[..] else {
        anyhow::bail!(
            "tensor is {name} with shape {:?}; this engine decodes {name} only as a matrix",
            entry.dims
        );
    };
    if !k.is_multiple_of(kind.block_elems()) {
        anyhow::bail!(
            "row length {k} not divisible by {name} block size {}",
            kind.block_elems()
        );
    }
    let size_in_bytes = n * k / kind.block_elems() * kind.block_bytes();
    let bytes = read_tensor_bytes(file, file_len, data_offset, entry, size_in_bytes)?;
    let _guard = crate::common::weights::metal_alloc_lock().lock().unwrap();
    let buffer = md.new_buffer_with_data(&bytes)?;
    StagedWeight::of_kind((*buffer).clone(), kind, &[n, k])
        .map(Built::Owned)
        .ok_or_else(|| anyhow::anyhow!("{name} weight rows are not whole blocks"))
}

#[cfg(not(feature = "metal"))]
fn build_owned(
    _file: &std::fs::File,
    _file_len: usize,
    _data_offset: u64,
    entry: &TensorEntry,
    _device: &Device,
) -> anyhow::Result<Built> {
    anyhow::bail!(
        "tensor type {} is decoded on Metal only",
        type_name(entry.type_id)
    )
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

        let interleaved = LinearWeight::Candle {
            tensor: Arc::new(qt),
            staged: None,
        };
        let LinearWeight::Candle {
            tensor: restored, ..
        } = depermute_qk_rows(&interleaved, n_heads, head_dim, &dev).unwrap()
        else {
            panic!("a candle weight comes back as one");
        };
        let restored_t = restored.dequantize(&dev).unwrap();

        let a = restored_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = hf_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b, "depermute must restore the HF row layout exactly");
    }

    /// Contract: a handle exists exactly for the layouts the staged kernel
    /// reads, and never for the others.
    ///
    /// A handle for a layout the kernel does not understand would route a
    /// matmul into garbage; a missing handle for one it does would only cost
    /// speed. Both directions are pinned here on Metal, where the handle can
    /// exist at all.
    #[cfg(feature = "metal")]
    #[test]
    fn a_staged_handle_exists_only_for_layouts_the_kernel_reads() {
        let Ok(dev) = Device::new_metal(0) else {
            return;
        };
        let quantize = |dims: (usize, usize), dtype: GgmlDType| {
            let t = Tensor::zeros(dims, candle_core::DType::F32, &dev).unwrap();
            QTensor::quantize(&t, dtype)
                .unwrap()
                .data()
                .unwrap()
                .into_owned()
        };
        let stageable = qtensor_with_staged(
            GgmlDType::Q4K,
            &quantize((64, 512), GgmlDType::Q4K),
            vec![64, 512],
            &dev,
        )
        .unwrap()
        .1;
        assert!(stageable.is_some(), "Q4_K with whole-block rows");

        let six_bit = qtensor_with_staged(
            GgmlDType::Q6K,
            &quantize((64, 512), GgmlDType::Q6K),
            vec![64, 512],
            &dev,
        )
        .unwrap()
        .1;
        assert!(six_bit.is_some(), "Q6_K with whole-block rows");

        let dense = qtensor_with_staged(
            GgmlDType::F16,
            &quantize((64, 512), GgmlDType::F16),
            vec![64, 512],
            &dev,
        )
        .unwrap()
        .1;
        assert!(dense.is_none(), "a dense weight is not staged");

        let one_dimensional = qtensor_with_staged(
            GgmlDType::Q4K,
            &quantize((1, 512), GgmlDType::Q4K),
            vec![512],
            &dev,
        )
        .unwrap()
        .1;
        assert!(one_dimensional.is_none(), "a vector is not a matmul weight");

        assert!(
            qtensor_with_staged(GgmlDType::Q4K, &[0u8; 10], vec![64, 512], &dev).is_err(),
            "the wrong number of bytes is refused"
        );
    }

    #[test]
    fn depermute_qk_rows_rejects_bad_geometry() {
        let dev = Device::Cpu;
        let t = Tensor::zeros((16, 32), candle_core::DType::F32, &dev).unwrap();
        let qt = LinearWeight::Candle {
            tensor: Arc::new(QTensor::quantize(&t, GgmlDType::F32).unwrap()),
            staged: None,
        };
        assert!(depermute_qk_rows(&qt, 3, 8, &dev).is_err());
    }

    /// Contract: a file holding a weight in a type candle cannot name loads on
    /// Metal, hands that weight out as a linear weight of the stated shape and
    /// refuses it as a tensor by naming its kind; off Metal it says so.
    #[test]
    fn a_weight_candle_cannot_name_loads_as_a_linear_weight() {
        let (n, k) = (64usize, 512usize);
        let dims: [u64; 2] = [k as u64, n as u64];
        let tensors: [(&str, &[u64], u32); 2] = [
            ("blk.0.ffn_down.weight", &dims, 23),
            ("output_norm.weight", &[8], 0),
        ];
        let kv = [(
            "general.architecture",
            gguf_file::Value::String("qwen3".into()),
        )];
        let weight_bytes = n * k / 256 * 136;
        let mut data: Vec<u8> = (0..weight_bytes).map(|i| (i * 7 % 251) as u8).collect();
        data.extend_from_slice(&[0u8; 32]);
        let file = crate::common::gguf_header::tests::gguf_bytes(&kv, &tensors, &data);
        let path = std::env::temp_dir().join(format!("oxydllm-iq-{}.gguf", std::process::id()));
        std::fs::write(&path, &file).unwrap();
        let path_str = path.to_string_lossy().into_owned();

        let cpu = GgufWeights::load(&path_str, &Device::Cpu);
        let msg = match cpu {
            Ok(_) => String::from("loaded"),
            Err(e) => format!("{e:#}"),
        };
        assert!(msg.contains("Metal only"), "{msg}");

        #[cfg(feature = "metal")]
        if let Ok(dev) = Device::new_metal(0) {
            let Device::Metal(md) = &dev else {
                unreachable!()
            };
            if !crate::common::metal_ops::mpp_owned_kernels_available(md.device()) {
                let err = match GgufWeights::load(&path_str, &dev) {
                    Ok(_) => String::from("loaded"),
                    Err(e) => format!("{e:#}"),
                };
                assert!(err.contains("does not build the kernels"), "{err}");
                std::fs::remove_file(&path).ok();
                return;
            }
            let w = GgufWeights::load(&path_str, &dev).unwrap();
            let weight = w.linear_weight("blk.0.ffn_down.weight").unwrap();
            assert!(matches!(weight, LinearWeight::Owned(_)), "{weight:?}");
            assert_eq!((weight.out_features(), weight.in_features()), (n, k));
            let err = w.get("blk.0.ffn_down.weight").unwrap_err().to_string();
            assert!(err.contains("Iq4Xs"), "{err}");
            assert_eq!(w.total_size_bytes(), weight_bytes + 32);
            assert!(w.get("output_norm.weight").is_ok());
        }
        std::fs::remove_file(&path).ok();
    }
}
