//! Model discovery, path resolution, and checkpoint loading.
//!
//! The entry point is [`load_batch_model`], which routes a model directory to
//! one of two paths:
//!
//! 1. Safetensors: [`crate::models::parsers::hf_parser::parse`] turns
//!    config.json into a [`StandardTransformerConfig`], then
//!    `load_standard_safetensors` builds the [`StandardTransformer`],
//!    deciding MoE expert streaming automatically from the memory envelope.
//! 2. GGUF: `load_batch_model_gguf` selects the right variant/shards in the
//!    directory and defers to [`StandardTransformer::load_gguf`].
//!
//! Both paths size their KV pool through `compute_kv_blocks` against the
//! server-wide [`GlobalKvBudget`]. Around that core live the filesystem
//! helpers: [`discover_models`] scans the models dir (config.json dirs and
//! GGUF dirs, one level of namespacing), [`resolve_model_path`] maps a
//! user-supplied model id to a directory, and [`select_device_at`] picks the
//! CUDA/Metal/CPU device.

use crate::common::{
    block::TransformerBlock,
    config::StandardTransformerConfig,
    gguf_weights::GgufWeights,
    kv_quant::{self, KvQuantMode, KvQuantizer},
    linear::{AnyLinear, Embedding},
    norm::RMSNorm,
    paged::{
        BlockAllocator, DEFAULT_BLOCK_SIZE, GlobalKvBudget, SharedBlockAllocator,
        SharedGlobalKvBudget,
    },
    rope::RotaryEmbedding,
    weights::ModelWeights,
};
use crate::models::gguf_model::StandardTransformer;
use crate::models::parsers::hf_parser;
use crate::models::traits::BatchModel;
use candle_core::{DType, Device};
use std::sync::{Arc, Mutex};

use std::path::{Path, PathBuf};

/// Listing entry produced by [`discover_models`] for the CLI and the
/// server's model registry.
///
/// `id` is what the user passes to load the model (GGUF variants get their
/// canonicalized stem, namespaced models a `namespace/name` prefix);
/// `architecture` is the HF class name or `"{arch} (GGUF)"`; `size_bytes`
/// sums the weight files and `created_at` is the directory mtime in Unix
/// seconds.
#[derive(Debug, Clone)]
pub struct DiscoveredModel {
    pub id: String,
    pub architecture: String,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub size_bytes: usize,
    pub created_at: u64,
}

/// Scans `models_dir` for loadable models, sorted case-insensitively by id.
///
/// Each direct subdirectory is scanned via [`scan_model_entry`]; directories
/// that are neither a model nor a GGUF dir are treated as namespaces and
/// scanned one level deeper (`namespace/name` ids). Unreadable directories
/// are silently skipped.
pub fn discover_models(models_dir: &Path) -> Vec<DiscoveredModel> {
    let mut models = Vec::new();
    let entries = match std::fs::read_dir(models_dir) {
        Ok(e) => e,
        Err(_) => return models,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        if scan_model_entry(&name, None, &path, &mut models) {
            continue;
        }

        if let Ok(children) = std::fs::read_dir(&path) {
            for child in children.flatten() {
                let child_path = child.path();
                if !child_path.is_dir() {
                    continue;
                }
                let child_name = child_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let full_id = format!("{}/{}", name, child_name);
                scan_model_entry(&full_id, Some(&name), &child_path, &mut models);
            }
        }
    }
    models.sort_by_key(|m| m.id.to_ascii_lowercase());
    models
}

/// Appends the models found in one directory to `models`, returning whether
/// the directory was recognised (so the caller stops descending into it).
///
/// A config.json directory yields exactly one entry under `id`; a GGUF
/// directory yields one entry per quant variant, grouping split shards by
/// stem and skipping ids already discovered.
fn scan_model_entry(
    id: &str,
    namespace: Option<&str>,
    path: &Path,
    models: &mut Vec<DiscoveredModel>,
) -> bool {
    let config_path = path.join("config.json");
    if config_path.exists() {
        let raw = match std::fs::read_to_string(&config_path) {
            Ok(r) => r,
            Err(_) => return false,
        };
        let value: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let architecture = value["architectures"][0]
            .as_str()
            .unwrap_or("Unknown")
            .to_string();
        let vocab_size = value["vocab_size"].as_u64().unwrap_or(0) as usize;
        let num_layers = value["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
        let size_bytes = std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| x == "safetensors")
                    .unwrap_or(false)
            })
            .filter_map(|e| e.metadata().ok().map(|m| m.len() as usize))
            .sum();
        let created_at = path
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        models.push(DiscoveredModel {
            id: id.to_string(),
            architecture,
            vocab_size,
            num_layers,
            size_bytes,
            created_at,
        });
        return true;
    }

    if let Some(gguf_paths) = find_gguf_files(path) {
        let mut stem_groups: std::collections::HashMap<String, Vec<std::path::PathBuf>> =
            std::collections::HashMap::new();
        for gguf_path in &gguf_paths {
            let raw_stem = gguf_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let stem = strip_gguf_split_suffix(&raw_stem).to_string();
            let local_id = if stem.is_empty() {
                id.to_string()
            } else {
                canonicalize_gguf_id(&stem, path)
            };
            let effective_id = match namespace {
                Some(ns) => format!("{}/{}", ns, local_id),
                None => local_id,
            };
            stem_groups
                .entry(effective_id)
                .or_default()
                .push(gguf_path.clone());
        }

        for (effective_id, paths) in stem_groups {
            if models
                .iter()
                .any(|m: &DiscoveredModel| m.id == effective_id)
            {
                continue;
            }
            let size_bytes = paths
                .iter()
                .filter_map(|p| p.metadata().ok().map(|m| m.len() as usize))
                .sum();
            if let Some(mut info) = discover_gguf_model(&effective_id, &paths[0]) {
                info.size_bytes = size_bytes;
                models.push(info);
            }
        }
        return true;
    }

    false
}

/// Normalizes a GGUF file stem into a user-facing model id by aligning its
/// casing with the parent directory name.
///
/// When the stem starts with the directory name minus a `-GGUF`/`_GGUF`
/// suffix (case-insensitively), the id becomes the directory's casing plus
/// the stem's remainder uppercased, e.g. `qwen3-4b-q4_k_m` in `Qwen3-4B-GGUF`
/// resolves to `Qwen3-4B-Q4_K_M`; otherwise the stem is returned unchanged.
fn canonicalize_gguf_id(stem: &str, parent_dir: &Path) -> String {
    let dir_name = parent_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    let dir_base = {
        let lower = dir_name.to_ascii_lowercase();
        if lower.ends_with("-gguf") || lower.ends_with("_gguf") {
            &dir_name[..dir_name.len() - 5]
        } else {
            dir_name
        }
    };

    let dir_base_lower = dir_base.to_ascii_lowercase();
    let stem_lower = stem.to_ascii_lowercase();

    if !dir_base_lower.is_empty() && stem_lower.starts_with(&dir_base_lower) {
        let suffix = &stem[dir_base_lower.len()..];
        format!("{}{}", dir_base, suffix.to_ascii_uppercase())
    } else {
        stem.to_string()
    }
}

/// Strips a sharded-GGUF suffix like `-00001-of-00003` from a file stem, so
/// all shards of one variant share a stem; non-split stems pass through.
fn strip_gguf_split_suffix(stem: &str) -> &str {
    let parts: Vec<&str> = stem.split('-').collect();
    let n = parts.len();
    if n >= 3
        && parts[n - 2] == "of"
        && parts[n - 1].chars().all(|c| c.is_ascii_digit())
        && parts[n - 3].chars().all(|c| c.is_ascii_digit())
    {
        let trim = 1 + parts[n - 1].len() + 1 + parts[n - 2].len() + 1 + parts[n - 3].len();
        &stem[..stem.len() - trim]
    } else {
        stem
    }
}

/// Ids under which a GGUF stem can be requested: the
/// [canonicalized](canonicalize_gguf_id) id plus the raw stem when they
/// differ.
fn gguf_match_keys(stem: &str, parent_dir: &Path) -> Vec<String> {
    let canonical = canonicalize_gguf_id(stem, parent_dir);
    if canonical == stem {
        vec![stem.to_string()]
    } else {
        vec![canonical, stem.to_string()]
    }
}

/// Forms of a requested model id to match against: the full id and, for
/// `namespace/name` ids, the bare local name.
fn gguf_model_id_keys(model_id: &str) -> Vec<&str> {
    match model_id.split_once('/') {
        Some((_, local_id)) => vec![model_id, local_id],
        None => vec![model_id],
    }
}

/// Picks the shards of the GGUF variant matching `model_id` from a
/// directory's .gguf files, grouped case-insensitively by
/// [split-stripped](strip_gguf_split_suffix) stem.
///
/// A directory with a single variant matches any id, so `oxydllm run <dir>`
/// works without spelling out the quant suffix.
///
/// ## Errors
///
/// Fails when the directory holds several variants and none matches,
/// listing the available ids.
fn select_gguf_paths(
    dir: &Path,
    model_id: &str,
    all_gguf_paths: Vec<PathBuf>,
) -> anyhow::Result<Vec<PathBuf>> {
    let requested = gguf_model_id_keys(model_id);
    let mut groups: Vec<(String, Vec<PathBuf>, Vec<String>)> = Vec::new();

    for path in all_gguf_paths {
        let raw_stem = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let stem = strip_gguf_split_suffix(&raw_stem).to_string();
        let keys = gguf_match_keys(&stem, dir);

        if let Some((_, paths, _)) = groups
            .iter_mut()
            .find(|(group_stem, _, _)| group_stem.eq_ignore_ascii_case(&stem))
        {
            paths.push(path);
        } else {
            groups.push((stem, vec![path], keys));
        }
    }

    for (_, paths, keys) in &groups {
        let is_match = requested
            .iter()
            .any(|needle| keys.iter().any(|key| key.eq_ignore_ascii_case(needle)));
        if is_match {
            return Ok(paths.clone());
        }
    }

    if groups.len() == 1 {
        return Ok(groups.remove(0).1);
    }

    let mut available: Vec<String> = groups
        .iter()
        .filter_map(|(_, _, keys)| keys.first().cloned())
        .collect();
    available.sort_by_key(|s| s.to_ascii_lowercase());
    available.dedup_by(|a, b| a.eq_ignore_ascii_case(b));

    anyhow::bail!(
        "GGUF variant '{}' was not found in {}. Available variants: {}",
        model_id,
        dir.display(),
        available.join(", ")
    )
}

/// Maps a user-supplied model id to its directory under `models_dir`, or
/// `None` when nothing matches.
///
/// Candidates are tried in order of specificity:
///
/// 1. The id as a literal path (`PathBuf::join` resolves `/` as a subdir, so
///    both flat `ModelName` and nested `user/ModelName` forms work on all
///    platforms), accepted if it holds a config.json or a .gguf file.
/// 2. For namespaced ids, directories inside the namespace whose GGUF stems
///    (raw or [canonicalized](canonicalize_gguf_id)) match the local name.
/// 3. GGUF stems anywhere in `models_dir` matching the id.
/// 4. A case-insensitive directory-name prefix match in either direction,
///    preferring the shortest name, so `Ministral-3B` resolves to
///    `Ministral-3B-Instruct-2512`.
pub fn resolve_model_path(models_dir: &Path, model_id: &str) -> Option<PathBuf> {
    let direct = models_dir.join(model_id);
    if direct.is_dir() {
        let ok = direct.join("config.json").exists() || find_gguf_file(&direct).is_some();
        if ok {
            return Some(direct);
        }
    }

    if let Some((namespace, local_id)) = model_id.split_once('/') {
        let ns_dir = models_dir.join(namespace);
        if !ns_dir.is_dir() {
            return None;
        }
        let needle = local_id.to_ascii_lowercase();
        let entries: Vec<_> = std::fs::read_dir(&ns_dir)
            .ok()?
            .flatten()
            .filter(|e| e.path().is_dir())
            .collect();

        for entry in &entries {
            let path = entry.path();
            if let Some(gguf_paths) = find_gguf_files(&path) {
                for gguf_path in &gguf_paths {
                    let raw_stem = gguf_path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    let stem = strip_gguf_split_suffix(&raw_stem);
                    let local_effective = canonicalize_gguf_id(stem, &path);
                    if local_effective.eq_ignore_ascii_case(local_id)
                        || stem.eq_ignore_ascii_case(local_id)
                    {
                        return Some(path);
                    }
                }
            }
        }

        let mut prefix_match: Option<PathBuf> = None;
        for entry in &entries {
            let path = entry.path();
            let dir_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            if dir_name.starts_with(&needle) || needle.starts_with(&dir_name) {
                let valid = path.join("config.json").exists() || find_gguf_file(&path).is_some();
                if valid {
                    let better = prefix_match
                        .as_ref()
                        .map(|p| {
                            p.file_name()
                                .and_then(|n| n.to_str())
                                .map(|s| s.len())
                                .unwrap_or(usize::MAX)
                                > dir_name.len()
                        })
                        .unwrap_or(true);
                    if better {
                        prefix_match = Some(path);
                    }
                }
            }
        }
        return prefix_match;
    }

    let needle = model_id.to_ascii_lowercase();
    let entries: Vec<_> = std::fs::read_dir(models_dir)
        .ok()?
        .flatten()
        .filter(|e| e.path().is_dir())
        .collect();

    for entry in &entries {
        let path = entry.path();
        if let Some(gguf_paths) = find_gguf_files(&path) {
            for gguf_path in &gguf_paths {
                let raw_stem = gguf_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let stem = strip_gguf_split_suffix(&raw_stem);
                if stem.eq_ignore_ascii_case(model_id) {
                    return Some(path);
                }
            }
        }
    }

    let mut prefix_match: Option<PathBuf> = None;
    for entry in &entries {
        let path = entry.path();
        let dir_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        if dir_name.starts_with(&needle) || needle.starts_with(&dir_name) {
            let valid = path.join("config.json").exists() || find_gguf_file(&path).is_some();
            if valid {
                let better = prefix_match
                    .as_ref()
                    .map(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|s| s.len())
                            .unwrap_or(usize::MAX)
                            > dir_name.len()
                    })
                    .unwrap_or(true);
                if better {
                    prefix_match = Some(path);
                }
            }
        }
    }
    prefix_match
}

/// Lists the safetensors files of a checkpoint: the deduplicated, sorted
/// `weight_map` targets when `model.safetensors.index.json` exists, otherwise
/// the single-file layout. Some repos (e.g. Mistral-7B-Instruct-v0.3) ship
/// `consolidated.safetensors` instead of `model.safetensors`; both names are
/// probed.
///
/// ## Errors
///
/// Fails when the index file exists but cannot be read or lacks a
/// well-formed `weight_map`.
fn resolve_weight_paths(model_dir: &str) -> anyhow::Result<Vec<String>> {
    let index_path = format!("{}/model.safetensors.index.json", model_dir);

    if std::path::Path::new(&index_path).exists() {
        let raw = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&raw)?;

        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Missing weight_map in {}", index_path))?;

        let mut seen = std::collections::HashSet::new();
        let mut files: Vec<String> = Vec::new();
        for filename in weight_map.values() {
            let name = filename.as_str().ok_or_else(|| {
                anyhow::anyhow!("Expected string value in weight_map, got {:?}", filename)
            })?;
            if seen.insert(name.to_string()) {
                files.push(format!("{}/{}", model_dir, name));
            }
        }
        files.sort();
        tracing::info!(
            shared_weight_files = files.len(),
            "resolved shared weight files"
        );
        Ok(files)
    } else {
        for name in &["model.safetensors", "consolidated.safetensors"] {
            let path = format!("{}/{}", model_dir, name);
            if std::path::Path::new(&path).exists() {
                return Ok(vec![path]);
            }
        }
        Ok(vec![format!("{}/model.safetensors", model_dir)])
    }
}

/// All .gguf files in `dir`, or `None` when there are none.
///
/// A `gguf.index` file (one filename per line, `#` comments) takes
/// precedence; otherwise every .gguf in the directory is returned sorted, so
/// multi-variant dirs (e.g. Q4_K_M + f16 side by side) resolve by stem
/// instead of whichever file lists first.
pub fn find_gguf_files(dir: &Path) -> Option<Vec<std::path::PathBuf>> {
    let index_path = dir.join("gguf.index");
    if index_path.exists()
        && let Ok(content) = std::fs::read_to_string(&index_path)
    {
        let files: Vec<std::path::PathBuf> = content
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| dir.join(l))
            .collect();
        if !files.is_empty() {
            return Some(files);
        }
    }
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .collect();
    if files.is_empty() {
        None
    } else {
        files.sort();
        Some(files)
    }
}

/// First .gguf file in `dir` (honouring `gguf.index` like
/// [`find_gguf_files`]); used as a cheap "is this a GGUF dir" probe.
pub fn find_gguf_file(dir: &Path) -> Option<std::path::PathBuf> {
    let index_path = dir.join("gguf.index");
    if index_path.exists()
        && let Ok(content) = std::fs::read_to_string(&index_path)
        && let Some(first) = content
            .lines()
            .map(|l| l.trim())
            .find(|l| !l.is_empty() && !l.starts_with('#'))
    {
        return Some(dir.join(first));
    }
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Some(p);
        }
    }
    None
}

/// Builds a [`DiscoveredModel`] from one GGUF file's metadata (architecture,
/// layer count, vocab from the embedding shape); `size_bytes` is left 0 for
/// the caller to fill with the whole shard group. Returns `None` when the
/// file cannot be read as GGUF.
fn discover_gguf_model(id: &str, gguf_path: &Path) -> Option<DiscoveredModel> {
    use candle_core::quantized::gguf_file;
    let mut file = std::fs::File::open(gguf_path).ok()?;
    let content = gguf_file::Content::read(&mut file).ok()?;

    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());

    let prefix = &arch;
    let num_layers = content
        .metadata
        .get(&format!("{prefix}.block_count"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;
    let _hidden_size = content
        .metadata
        .get(&format!("{prefix}.embedding_length"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;

    let vocab_size = content
        .tensor_infos
        .get("token_embd.weight")
        .map(|info| info.shape.dims()[0])
        .unwrap_or(0);

    let arch_display = format!("{} (GGUF)", arch);

    let created_at = gguf_path
        .metadata()
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);

    Some(DiscoveredModel {
        id: id.to_string(),
        architecture: arch_display,
        vocab_size,
        num_layers,
        size_bytes: 0,
        created_at,
    })
}

/// True when the directory is loaded through the GGUF path: it has a .gguf
/// file and no config.json (config.json wins when both exist).
pub fn is_gguf_model(model_dir: &str) -> bool {
    let dir = Path::new(model_dir);
    if dir.join("config.json").exists() {
        return false;
    }
    find_gguf_file(dir).is_some()
}

/// Logs whether the CUDA device meets the minimum supported compute
/// capability (8.9, Ada Lovelace) and warns when the binary was compiled for
/// an older `sm_` than the hardware; never fails the load.
#[cfg(feature = "cuda")]
fn check_cuda_compute_capability(device: &Device, ordinal: usize) {
    let Ok(cuda_dev) = device.as_cuda_device() else {
        return;
    };
    match cuda_dev.cuda_stream().context().compute_capability() {
        Ok((major, minor)) => {
            let cap = major * 10 + minor;
            if cap < 89 {
                tracing::warn!(
                    compute_capability = format!("{major}.{minor}"),
                    cuda_device = ordinal,
                    "GPU compute capability {major}.{minor} is below the minimum supported (8.9 / Ada \
                     Lovelace). Inference may fail or produce incorrect results. Supported: \
                     8.9 (Ada Lovelace / RTX 40xx), 9.0 (Hopper / H100), 10.0 (Blackwell / B200), \
                     10.3 (Blackwell Ultra / GB300), 11.0 (Jetson GB), \
                     12.0 (Blackwell Desktop / RTX 50xx), 12.1 (DGX Spark / GB10)"
                );
            } else {
                if let Some(compiled_cap) =
                    option_env!("OXYDLLM_COMPILED_CAP").and_then(|s| s.parse::<i32>().ok())
                {
                    if compiled_cap < cap {
                        tracing::warn!(
                            compiled_cap,
                            hardware_cap = cap,
                            cuda_device = ordinal,
                            "Binary compiled for sm_{compiled_cap} but hardware is sm_{cap}. \
                             Recompile with CUDA_COMPUTE_CAP={cap} for optimal performance \
                             (sm_{cap}-specific features such as native flash attention are unavailable)."
                        );
                    }
                }
                tracing::info!(
                    compute_capability = format!("{major}.{minor}"),
                    cuda_device = ordinal,
                    "CUDA compute capability ok"
                );
            }
        }
        Err(e) => tracing::debug!(error = %e, "could not query CUDA compute capability"),
    }
}

/// Selects the inference [`Device`]: CUDA at `_cuda_idx` when compiled in
/// and available, then Metal, then CPU.
///
/// ## Errors
///
/// Fails when no GPU is available and `require_gpu` is set (the default;
/// `--allow-cpu` clears it and falls back to CPU with a warning).
pub fn select_device_at(_cuda_idx: usize, require_gpu: bool) -> anyhow::Result<Device> {
    #[cfg(feature = "cuda")]
    match Device::new_cuda(_cuda_idx) {
        Ok(d) => {
            tracing::info!(device = %format!("CUDA:{}", _cuda_idx), "device selected");
            check_cuda_compute_capability(&d, _cuda_idx);
            return Ok(d);
        }
        Err(e) => tracing::warn!(cuda_device = _cuda_idx, error = %e, "CUDA device not available"),
    }

    #[cfg(feature = "metal")]
    match Device::new_metal(0) {
        Ok(d) => {
            tracing::info!(device = "Metal", "device selected");
            return Ok(d);
        }
        Err(e) => tracing::warn!(error = %e, "Metal device not available"),
    }

    if require_gpu {
        return Err(anyhow::anyhow!(
            "[FATAL] No GPU device available. Pass --allow-cpu (or set OXYDLLM_ALLOW_CPU=1) \
             to run on CPU; expect severely degraded inference performance."
        ));
    }

    tracing::warn!(
        "GPU not available; --allow-cpu set, falling back to CPU. Inference performance may be severely degraded"
    );
    tracing::info!(device = "CPU", "device selected");
    Ok(Device::Cpu)
}

/// Sizing and quantization options for [`load_batch_model`].
///
/// `max_context_len` and `max_num_sequences` size the desired KV pool, drawn
/// from the shared `kv_budget`; `kv_quant` and `qjl_quantization` select KV
/// cache quantization. `expert_stream_mb`, when set, forces MoE expert
/// weights to stream from the checkpoint mmap through an LRU pool of that
/// many megabytes instead of loading resident. `memory_budget_bytes` is the
/// operator's `--memory-budget`, used as the memory envelope for the
/// automatic expert-streaming decision; free memory at load time when unset.
#[derive(Clone, Copy)]
pub struct LoadBatchOptions<'a> {
    pub max_context_len: usize,
    pub max_num_sequences: usize,
    pub kv_budget: &'a SharedGlobalKvBudget,
    pub kv_quant: KvQuantMode,
    pub qjl_quantization: bool,
    pub expert_stream_mb: Option<usize>,
    pub memory_budget_bytes: Option<usize>,
}

/// Loads the model in `model_dir` as a [`BatchModel`], returning it together
/// with its weight footprint in bytes (used by the manager's memory
/// accounting).
///
/// GGUF directories (per [`is_gguf_model`]) go through the GGUF path, where
/// `model_id` selects the quant variant; everything else parses config.json
/// and loads safetensors. Runtime dtype is BF16 on GPU, F32 on CPU.
///
/// ## Errors
///
/// Fails when the architecture is unsupported, weight files are missing or
/// malformed, the KV budget cannot grant a minimum pool, or a MoE model
/// cannot fit even with expert streaming.
pub fn load_batch_model(
    model_dir: &str,
    model_id: &str,
    device: &Device,
    opts: LoadBatchOptions<'_>,
) -> anyhow::Result<(Box<dyn BatchModel>, usize)> {
    if is_gguf_model(model_dir) {
        return load_batch_model_gguf(model_dir, model_id, device, opts);
    }

    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };
    let cfg = hf_parser::parse(&format!("{}/config.json", model_dir))?;
    load_standard_safetensors(cfg, model_dir, device, dtype, opts)
}

/// Sums a checkpoint's tensor bytes as they will exist at runtime, split into
/// streamable experts and everything else. Reads only the safetensors
/// headers; float tensors are scaled by `runtime dtype size / file dtype
/// size` (an FP8 file doubles into BF16), integer tensors keep their size.
///
/// ## Errors
///
/// Fails when the weight files cannot be memory-mapped or parsed as
/// safetensors.
fn expert_split_runtime_bytes(paths: &[&str], dtype: DType) -> anyhow::Result<(usize, usize)> {
    // SAFETY: same exclusive-ownership argument as ModelWeights::load.
    let mmap = unsafe {
        candle_core::safetensors::MmapedSafetensors::multi(paths)
            .map_err(|e| anyhow::anyhow!("Failed to memory-map weight files: {e:#}"))?
    };
    let (mut expert, mut rest) = (0usize, 0usize);
    for (name, view) in mmap.tensors() {
        if name.starts_with("model.visual.") || name.starts_with("mtp.") {
            continue;
        }
        let file_elem = match format!("{:?}", view.dtype()).as_str() {
            "F8_E4M3" => Some(1),
            "F16" | "BF16" => Some(2),
            "F32" => Some(4),
            _ => None,
        };
        let bytes = view.data().len();
        let runtime_bytes = match file_elem {
            Some(fe) => bytes * dtype.size_in_bytes() / fe,
            None => bytes,
        };
        if crate::common::expert_stream::is_streamed_expert_tensor(&name) {
            expert += runtime_bytes;
        } else {
            rest += runtime_bytes;
        }
    }
    Ok((expert, rest))
}

/// Headroom reserved next to the weights for KV cache, activations, and the
/// OS when deciding whether a MoE checkpoint fits resident.
const STREAM_HEADROOM: usize = 2 << 30;
/// Below this expert-cache size streaming would thrash; the model is rejected.
const MIN_EXPERT_CACHE: usize = 1 << 30;

/// Decides expert streaming for a MoE checkpoint: `None` when everything fits
/// resident (fastest), `Some(cache_bytes)` when streaming makes it fit. The
/// returned cache is always smaller than the expert bytes: a remainder that
/// covered every expert would have taken the resident branch.
///
/// ## Errors
///
/// Fails when not even the non-expert weights plus [`MIN_EXPERT_CACHE`] and
/// [`STREAM_HEADROOM`] fit in `available_bytes`: below that cache size
/// streaming would thrash, so the model is rejected with a clear message.
fn auto_expert_stream_budget(
    expert_bytes: usize,
    non_expert_bytes: usize,
    available_bytes: usize,
) -> anyhow::Result<Option<usize>> {
    if expert_bytes + non_expert_bytes + STREAM_HEADROOM <= available_bytes {
        return Ok(None);
    }
    let cache = available_bytes
        .saturating_sub(non_expert_bytes)
        .saturating_sub(STREAM_HEADROOM);
    if cache < MIN_EXPERT_CACHE {
        anyhow::bail!(
            "model needs {:.1} GB resident plus a streamed-expert cache, but only \
             {:.1} GB of memory is available; free memory or use a smaller quant",
            (non_expert_bytes + MIN_EXPERT_CACHE + STREAM_HEADROOM) as f64 / 1_073_741_824.0,
            available_bytes as f64 / 1_073_741_824.0,
        );
    }
    Ok(Some(cache))
}

/// Loads a safetensors checkpoint into a [`StandardTransformer`].
///
/// For MoE checkpoints, expert streaming is decided automatically from the
/// tensor sizes and the memory envelope ([`auto_expert_stream_budget`]); the
/// CLI flags only override that decision. The returned weight size counts a
/// streamed model as its resident tensors plus the expert-cache budget, so
/// the manager's memory accounting holds either way.
///
/// ## Errors
///
/// Fails on missing or misshapen tensors, `--stream-experts` on a dense
/// model, a MoE model too large even for streaming, an exhausted KV budget,
/// or a hybrid config whose `layer_types` length disagrees with the layer
/// count (or lists no full-attention layer). Any KV bytes acquired are
/// released before the error propagates.
fn load_standard_safetensors(
    cfg: StandardTransformerConfig,
    model_dir: &str,
    device: &Device,
    dtype: DType,
    opts: LoadBatchOptions<'_>,
) -> anyhow::Result<(Box<dyn BatchModel>, usize)> {
    let weight_paths = resolve_weight_paths(model_dir)?;
    let weight_path_refs: Vec<&str> = weight_paths.iter().map(|s| s.as_str()).collect();

    let cache_bytes: Option<usize> = match (opts.expert_stream_mb, cfg.moe_num_experts) {
        (Some(_), None) => anyhow::bail!(
            "--stream-experts requires a MoE model; this checkpoint has no experts \
             (dense streaming is bandwidth-bound and not supported)"
        ),
        (Some(mb), Some(_)) => Some(mb << 20),
        (None, Some(_)) => {
            let (expert_b, rest_b) = expert_split_runtime_bytes(&weight_path_refs, dtype)?;
            let available = opts
                .memory_budget_bytes
                .or_else(crate::common::paged::detect_reclaimable_memory_bytes)
                .unwrap_or(usize::MAX);
            let decision = auto_expert_stream_budget(expert_b, rest_b, available)?;
            if let Some(cache) = decision {
                tracing::info!(
                    expert_gb = expert_b >> 30,
                    resident_gb = rest_b >> 30,
                    available_gb = available >> 30,
                    cache_gb = cache >> 30,
                    "model exceeds available memory; streaming experts from disk automatically"
                );
            }
            decision
        }
        (None, None) => None,
    };
    let expert_stream = cache_bytes.map(|cache| {
        let layout = if cfg.moe_gpt_oss {
            crate::common::expert_stream::ExpertLayout::GptOss {
                swiglu_limit: cfg.moe_swiglu_limit.unwrap_or(7.0),
            }
        } else {
            crate::common::expert_stream::ExpertLayout::Standard
        };
        tracing::info!(
            num_experts = cfg.moe_num_experts.unwrap_or(0),
            cache_mb = cache >> 20,
            "expert streaming enabled; experts load on demand from the checkpoint mmap"
        );
        crate::common::expert_stream::ExpertStreamConfig {
            layout,
            cache_bytes: cache,
        }
    });
    let weights = ModelWeights::load(&weight_path_refs, device, dtype, expert_stream)?
        .with_quant_scheme(cfg.quant_scheme);
    let weights_size = weights.runtime_size_bytes() + cache_bytes.unwrap_or(0);
    #[cfg(feature = "metal")]
    let has_packed_quantized_weights = weights.has_packed_quantized_weights();

    let num_layers = cfg.num_hidden_layers;
    let per_layer_head_dims = cfg
        .per_layer_head_dims
        .clone()
        .filter(|v| v.len() == num_layers)
        .unwrap_or_else(|| vec![cfg.head_dim; num_layers]);
    let per_layer_kv_heads = cfg
        .per_layer_num_key_value_heads
        .clone()
        .filter(|v| v.len() == num_layers)
        .unwrap_or_else(|| vec![cfg.num_key_value_heads; num_layers]);
    let per_layer_sliding_windows = cfg
        .per_layer_sliding_windows
        .clone()
        .filter(|v| v.len() == num_layers)
        .unwrap_or_else(|| vec![cfg.sliding_window; num_layers]);
    let per_layer_rope_thetas = cfg
        .per_layer_rope_thetas
        .clone()
        .filter(|v| v.len() == num_layers)
        .unwrap_or_else(|| vec![cfg.rope_theta; num_layers]);

    // Hybrid models: linear-attention layers keep recurrent state instead of
    // KV blocks, so they contribute neither to the KV budget nor real pools.
    let layer_is_linear: Vec<bool> = match (&cfg.layer_types, cfg.linear_attn) {
        (Some(types), Some(_)) => {
            if types.len() != num_layers {
                anyhow::bail!(
                    "layer_types length {} != num_hidden_layers {num_layers}",
                    types.len()
                );
            }
            types
                .iter()
                .map(|t| *t == crate::common::config::LayerType::LinearAttention)
                .collect()
        }
        _ => vec![false; num_layers],
    };

    let layer_kv_specs: Vec<(usize, usize)> = per_layer_kv_heads
        .iter()
        .copied()
        .zip(per_layer_head_dims.iter().copied())
        .zip(layer_is_linear.iter().copied())
        .filter(|&(_, is_linear)| !is_linear)
        .map(|(spec, _)| spec)
        .collect();

    let ctx = opts.max_context_len.min(cfg.max_position_embeddings);
    let (num_blocks, acquired_kv_bytes) = compute_kv_blocks(
        &KvBlockParams {
            layer_kv_specs: layer_kv_specs.clone(),
            max_context_len: ctx,
            max_num_sequences: opts.max_num_sequences,
            dtype,
            kv_quant: opts.kv_quant,
            qjl_quantization: opts.qjl_quantization,
        },
        opts.kv_budget,
    )?;

    let layer_quantizers: Vec<Option<Arc<KvQuantizer>>> = match opts.kv_quant {
        KvQuantMode::Off => vec![None; num_layers],
        mode => per_layer_head_dims
            .iter()
            .map(|&hd| {
                Some(Arc::new(KvQuantizer::new_with_qjl(
                    mode.bit_width(),
                    hd,
                    opts.qjl_quantization,
                )))
            })
            .collect(),
    };

    let result = (|| -> anyhow::Result<(Box<dyn BatchModel>, usize)> {
        let embed_weight_name = "model.embed_tokens.weight";
        let embed_weight = weights
            .get(embed_weight_name)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .clone();

        let lm_head_weight_name = "lm_head.weight";
        let lm_head_scale_inv = if cfg.tie_word_embeddings {
            weights.try_get_scale_inv(embed_weight_name).cloned()
        } else {
            weights.try_get_scale_inv(lm_head_weight_name).cloned()
        };

        let (lm_head, lm_head_extra_bytes): (AnyLinear, usize) = if cfg.tie_word_embeddings {
            // Catch "loads cleanly, outputs garbage": config says tie but the
            // file ships its own lm_head, which would be silently ignored.
            if weights.try_get(lm_head_weight_name).is_some() {
                tracing::warn!(
                    model_dir,
                    "config has tie_word_embeddings=true but file also contains explicit \
                     `lm_head.weight`; the file's lm_head will be ignored. If the model \
                     produces wrong output, set `tie_word_embeddings: false` in config.json."
                );
            }

            // 4-bit tied lm_head (via RTN) applies to the 4-bit packed schemes
            // (AWQ-4bit and compressed-tensors INT4, which runs on the same
            // resident W4A16 path); 8-bit AWQ takes the plain-tied path. On a
            // 4-bit model a BF16 tied lm_head would otherwise dominate decode:
            // the 248k-vocab Qwen3.5 reads 1.27 GB/token through it (~25% of
            // the per-token budget measured on the INT4 checkpoint).
            #[cfg(feature = "metal")]
            if has_packed_quantized_weights
                && device.is_metal()
                && matches!(dtype, DType::F16 | DType::BF16)
                && matches!(
                    weights.quant_scheme(),
                    Some(crate::common::weights::QuantScheme::Awq { bits: 4 })
                        | Some(crate::common::weights::QuantScheme::CompressedTensors4)
                )
            {
                let raw = crate::common::awq::rtn_quantize_awq(&embed_weight, 128)?;
                let extra = raw.runtime_size_bytes();
                (
                    AnyLinear::from_awq(&raw, None, device, dtype)
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                    extra,
                )
            } else {
                (
                    AnyLinear::from_weight_with_scale_inv(
                        embed_weight.clone(),
                        lm_head_scale_inv,
                        None,
                    )
                    .map_err(|e| anyhow::anyhow!("{e}"))?,
                    0usize,
                )
            }
            #[cfg(not(feature = "metal"))]
            (
                AnyLinear::from_weight_with_scale_inv(
                    embed_weight.clone(),
                    lm_head_scale_inv,
                    None,
                )
                .map_err(|e| anyhow::anyhow!("{e}"))?,
                0usize,
            )
        } else if let Some(lm_head_quant) = weights.try_get_quant("lm_head") {
            (
                AnyLinear::from_quant(&lm_head_quant, None, device, dtype)
                    .map_err(|e| anyhow::anyhow!("{e}"))?,
                0usize,
            )
        } else {
            (
                AnyLinear::from_weight_with_scale_inv(
                    weights
                        .get(lm_head_weight_name)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                        .clone(),
                    lm_head_scale_inv,
                    None,
                )
                .map_err(|e| anyhow::anyhow!("{e}"))?,
                0usize,
            )
        };
        let weights_size = weights_size + lm_head_extra_bytes;
        let embed_tokens = Embedding::new(embed_weight);

        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| {
                let mut block_cfg = cfg.block_config();
                block_cfg.head_dim = per_layer_head_dims[i];
                block_cfg.n_kv_heads = per_layer_kv_heads[i];
                block_cfg.sliding_window = per_layer_sliding_windows[i];
                if layer_is_linear[i] {
                    block_cfg.linear_attn = cfg.linear_attn;
                }
                TransformerBlock::load(&block_cfg, i, &weights)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let norm = RMSNorm::load(&weights, "model.norm", cfg.rms_norm_eps, cfg.norm_type)?;

        let ropes = (0..cfg.num_hidden_layers)
            .map(|i| {
                RotaryEmbedding::new_with_scaling(
                    cfg.rotary_dim.unwrap_or(per_layer_head_dims[i]),
                    ctx,
                    per_layer_rope_thetas[i],
                    cfg.rope_scaling.clone(),
                    dtype,
                    device,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        // Linear-attention layers never allocate KV blocks; they alias the
        // first full-attention layer's allocator so the scheduler's
        // free-block accounting (allocators[0]) tracks a real pool.
        let allocators: Vec<SharedBlockAllocator> = {
            let mut real: Vec<Option<SharedBlockAllocator>> = vec![None; cfg.num_hidden_layers];
            for i in 0..cfg.num_hidden_layers {
                if !layer_is_linear[i] {
                    real[i] = Some(Arc::new(Mutex::new(BlockAllocator::new(
                        num_blocks,
                        DEFAULT_BLOCK_SIZE,
                        per_layer_kv_heads[i],
                        per_layer_head_dims[i],
                        dtype,
                        device,
                        layer_quantizers[i].clone(),
                    )?)));
                }
            }
            let first_full = real.iter().flatten().next().cloned().ok_or_else(|| {
                anyhow::anyhow!("hybrid model needs at least one full_attention layer")
            })?;
            real.into_iter()
                .map(|a| a.unwrap_or_else(|| Arc::clone(&first_full)))
                .collect()
        };

        let has_per_layer_stream = cfg.per_layer_input_hidden_size.is_some()
            && cfg.per_layer_input_vocab_size.is_some()
            && weights
                .try_get("model.embed_tokens_per_layer.weight")
                .is_some()
            && weights
                .try_get("model.per_layer_model_projection.weight")
                .is_some()
            && weights
                .try_get("model.per_layer_projection_norm.weight")
                .is_some();

        let per_layer_input_embed = if has_per_layer_stream {
            Some(Embedding::new(
                weights
                    .get("model.embed_tokens_per_layer.weight")
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .clone(),
            ))
        } else {
            None
        };
        let per_layer_model_projection = if has_per_layer_stream {
            let per_layer_proj_name = "model.per_layer_model_projection.weight";
            Some(
                AnyLinear::from_weight_with_scale_inv(
                    weights
                        .get(per_layer_proj_name)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                        .clone(),
                    weights.try_get_scale_inv(per_layer_proj_name).cloned(),
                    None,
                )
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            )
        } else {
            None
        };
        let per_layer_projection_norm = if has_per_layer_stream {
            Some(RMSNorm::load(
                &weights,
                "model.per_layer_projection_norm",
                cfg.rms_norm_eps,
                cfg.norm_type,
            )?)
        } else {
            None
        };

        Ok((
            Box::new(StandardTransformer {
                embed_tokens,
                blocks,
                norm,
                lm_head,
                ropes,
                allocators,
                device: device.clone(),
                stop_token_ids: cfg.eos_token_ids,
                vocab_size: cfg.vocab_size,
                max_seq_len: ctx,
                embed_scale: cfg.embed_scale,
                logit_softcap: cfg.logit_softcap,
                logits_scaling: cfg.logits_scaling,
                per_layer_input_embed,
                per_layer_input_embed_scale: cfg.per_layer_input_embed_scale,
                per_layer_model_projection,
                per_layer_model_projection_scale: cfg.per_layer_model_projection_scale,
                per_layer_projection_norm,
                per_layer_input_scale: cfg.per_layer_input_scale,
                kv_shared_layer_map: cfg.kv_shared_layer_map.clone(),
                has_recurrent_state: layer_is_linear.iter().any(|&b| b),
            }),
            weights_size,
        ))
    })();

    if result.is_err() {
        opts.kv_budget.release(acquired_kv_bytes);
    }
    result
}

/// GGUF branch of [`load_batch_model`]: selects the variant's shards
/// ([`select_gguf_paths`]), sizes the KV pool, and defers to
/// [`StandardTransformer::load_gguf`].
///
/// ## Errors
///
/// Fails when no .gguf file exists, the requested variant is not in the
/// directory, the KV budget cannot grant a minimum pool, or the GGUF loader
/// itself fails; acquired KV bytes are released on failure.
fn load_batch_model_gguf(
    model_dir: &str,
    model_id: &str,
    device: &Device,
    opts: LoadBatchOptions<'_>,
) -> anyhow::Result<(Box<dyn BatchModel>, usize)> {
    let dir = Path::new(model_dir);
    let all_gguf_paths = find_gguf_files(dir)
        .ok_or_else(|| anyhow::anyhow!("No .gguf file found in {}", model_dir))?;

    let gguf_paths = select_gguf_paths(dir, model_id, all_gguf_paths)?;

    if gguf_paths.len() == 1 {
        tracing::info!(path = %gguf_paths[0].display(), "loading GGUF model");
    } else {
        tracing::info!(
            first_shard = %gguf_paths[0].display(),
            shards = gguf_paths.len(),
            "loading sharded GGUF model"
        );
    }
    let gguf_path_strs: Vec<&str> = gguf_paths
        .iter()
        .map(|p| {
            p.to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF8 GGUF path: {}", p.display()))
        })
        .collect::<anyhow::Result<Vec<&str>>>()?;
    let gguf = GgufWeights::load_shards(&gguf_path_strs, device)?;

    let arch = gguf.architecture()?;
    tracing::info!(architecture = %arch, "GGUF architecture detected");

    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };

    let weights_size = gguf.total_size_bytes();

    let topo = crate::models::gguf_model::parse_gguf_topology(&gguf)?;
    let ctx = opts.max_context_len.min(topo.context_length);
    // Hybrid models budget KV only for their full-attention layers; the
    // linear layers keep O(1) recurrent state instead.
    let layer_kv_specs: Vec<(usize, usize)> = (0..topo.num_hidden_layers)
        .filter(|&i| !topo.layer_is_linear(i))
        .map(|_| (topo.num_key_value_heads, topo.head_dim))
        .collect();
    let (num_blocks, acquired_kv_bytes) = compute_kv_blocks(
        &KvBlockParams {
            layer_kv_specs,
            max_context_len: ctx,
            max_num_sequences: opts.max_num_sequences,
            dtype,
            kv_quant: opts.kv_quant,
            qjl_quantization: opts.qjl_quantization,
        },
        opts.kv_budget,
    )?;

    let quantizer: Option<Arc<KvQuantizer>> = match opts.kv_quant {
        KvQuantMode::Off => None,
        mode => Some(Arc::new(KvQuantizer::new_with_qjl(
            mode.bit_width(),
            topo.head_dim,
            opts.qjl_quantization,
        ))),
    };

    let model = match StandardTransformer::load_gguf(&gguf, device, dtype, num_blocks, quantizer) {
        Ok(m) => m,
        Err(e) => {
            opts.kv_budget.release(acquired_kv_bytes);
            return Err(e);
        }
    };
    Ok((Box::new(model), weights_size))
}

/// Inputs to [`compute_kv_blocks`]: `layer_kv_specs` lists
/// `(n_kv_heads, head_dim)` for each KV-carrying layer (linear-attention
/// layers excluded), the rest describe the desired capacity and KV
/// quantization mode.
struct KvBlockParams {
    layer_kv_specs: Vec<(usize, usize)>,
    max_context_len: usize,
    max_num_sequences: usize,
    dtype: DType,
    kv_quant: KvQuantMode,
    qjl_quantization: bool,
}

/// Sizes the model's KV pool against the global budget, returning
/// `(num_blocks, acquired_bytes)`; the caller must release the bytes if the
/// load fails afterwards.
///
/// The desired capacity is `max_num_sequences × max_context_len` slots; the
/// grant is capped by what [`GlobalKvBudget::acquire`] returns (a warning is
/// logged when capped) and floored at 256 blocks.
///
/// ## Errors
///
/// Fails when the budget cannot grant even the 256-block minimum; the
/// partial grant is released first.
fn compute_kv_blocks(
    p: &KvBlockParams,
    kv_budget: &GlobalKvBudget,
) -> anyhow::Result<(usize, usize)> {
    let total_slots = p.max_num_sequences * p.max_context_len;
    let desired_blocks = total_slots.div_ceil(DEFAULT_BLOCK_SIZE);
    let min_blocks: usize = 256;

    let per_block_bytes = match p.kv_quant {
        KvQuantMode::Off => p
            .layer_kv_specs
            .iter()
            .map(|(n_kv_heads, head_dim)| {
                DEFAULT_BLOCK_SIZE * (*n_kv_heads) * (*head_dim) * p.dtype.size_in_bytes() * 2
            })
            .sum::<usize>(),
        mode => p
            .layer_kv_specs
            .iter()
            .map(|(n_kv_heads, head_dim)| {
                let key_bph = kv_quant::quantized_key_bytes_per_head_with_qjl(
                    *head_dim,
                    mode.bit_width(),
                    p.qjl_quantization,
                );
                let value_bph =
                    kv_quant::quantized_value_bytes_per_head(*head_dim, mode.bit_width());
                DEFAULT_BLOCK_SIZE * (*n_kv_heads) * (key_bph + value_bph)
            })
            .sum::<usize>(),
    };

    if per_block_bytes == 0 {
        return Ok((desired_blocks, 0));
    }

    let desired_bytes = desired_blocks.max(min_blocks) * per_block_bytes;
    let granted_bytes = kv_budget.acquire(desired_bytes);
    let granted_blocks = granted_bytes / per_block_bytes;

    if granted_blocks < min_blocks {
        kv_budget.release(granted_bytes);
        anyhow::bail!(
            "KV cache budget exhausted: requested {} blocks ({:.2} GB minimum) \
             but only {} blocks ({:.2} GB) available. Unload other models, or \
             start the server with --memory-budget <MB> to size the global KV \
             pool independently of the free memory at startup.",
            min_blocks,
            min_blocks as f64 * per_block_bytes as f64 / 1_073_741_824.0,
            granted_blocks,
            granted_blocks as f64 * per_block_bytes as f64 / 1_073_741_824.0,
        );
    }

    if granted_blocks < desired_blocks {
        let desired_gb =
            ((desired_blocks as f64 * per_block_bytes as f64 / 1_073_741_824.0) * 100.0).round()
                / 100.0;
        let granted_gb =
            ((granted_blocks as f64 * per_block_bytes as f64 / 1_073_741_824.0) * 100.0).round()
                / 100.0;
        let remaining_pool_gb =
            ((kv_budget.available_bytes() as f64 / 1_073_741_824.0) * 100.0).round() / 100.0;

        tracing::warn!(
            desired_blocks,
            granted_blocks,
            desired_gb,
            granted_gb,
            remaining_pool_gb,
            "KV cache capped by global budget"
        );
    }

    Ok((granted_blocks, granted_bytes))
}

#[cfg(test)]
mod expert_stream_decision_tests {
    use super::*;

    const GB: usize = 1 << 30;

    /// Contract: a model that fits resident (with headroom) is never streamed.
    #[test]
    fn fitting_model_stays_resident() {
        let d = auto_expert_stream_budget(11 * GB, 2 * GB, 20 * GB).unwrap();
        assert!(d.is_none());
    }

    /// Contract: a model over the envelope streams, with the cache sized to
    /// what remains after resident weights and headroom.
    #[test]
    fn oversized_model_streams_with_remaining_cache() {
        let d = auto_expert_stream_budget(64 * GB, 5 * GB, 20 * GB).unwrap();
        assert_eq!(d, Some(13 * GB));
    }

    /// Contract: when the non-expert weights plus a minimum cache do not fit,
    /// loading fails with a clear error instead of thrashing.
    #[test]
    fn hopeless_model_is_rejected() {
        let err = auto_expert_stream_budget(64 * GB, 10 * GB, 12 * GB).unwrap_err();
        assert!(err.to_string().contains("available"), "{err}");
    }
}
