use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc as tokio_mpsc, oneshot};

use crate::common::kv_quant::KvQuantMode;
use crate::common::paged::{
    GlobalKvBudget, PagedKvCache, SharedGlobalKvBudget, detect_system_kv_budget,
};
use crate::engine::Engine;
use crate::models::estimate::{
    awq_qweight_expansion, read_quantization_config, sum_safetensors_bytes,
};
use crate::models::loader;
use crate::models::traits::BatchModel;
use crate::scheduler::SchedulerConfig;
use crate::server::{IncomingRequest, engine_loop};
use crate::tokenizer::Tokenizer;

#[derive(Clone)]
pub struct ReadyHandle {
    pub request_tx: tokio_mpsc::Sender<IncomingRequest>,
    pub tokenizer: Arc<Tokenizer>,
    pub max_seq_len: usize,
}

enum SlotState {
    Loading {
        waiters: Vec<oneshot::Sender<Result<ReadyHandle, String>>>,
    },
    Ready {
        request_tx: tokio_mpsc::Sender<IncomingRequest>,
        tokenizer: Arc<Tokenizer>,
        max_seq_len: usize,
        architecture: String,
        vocab_size: usize,
        num_layers: usize,
        last_used: Instant,
        effective_keep_alive: Duration,
        weights_size_bytes: usize,
        kv_cache_bytes: usize,
        shutdown: Arc<AtomicBool>,
    },
}

pub struct RunningModelInfo {
    pub id: String,
    pub architecture: String,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub idle_seconds: u64,
    pub weights_size_bytes: usize,
    pub kv_cache_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegistryEntry {
    pub architecture: String,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub size_bytes: usize,
    #[serde(default)]
    pub kv_cache_bytes: usize,
    pub last_used_secs: u64,
}

pub struct ModelManager {
    models_dir: PathBuf,
    slots: HashMap<String, SlotState>,
    keep_alive: Duration,
    memory_budget_bytes: Option<usize>,
    registry: BTreeMap<String, RegistryEntry>,
    cuda_devices: Vec<usize>,
    max_context_len: usize,
    kv_budget: SharedGlobalKvBudget,
    kv_quant: KvQuantMode,
    qjl_quantization: bool,
    require_gpu: bool,
    max_num_seqs: Option<usize>,
    max_queued_requests: usize,
    discovery_cache: Option<DiscoveryCache>,
}

struct DiscoveryCache {
    discovered: Vec<loader::DiscoveredModel>,
    refreshed_at: Instant,
}

const DISCOVERY_CACHE_TTL: Duration = Duration::from_secs(5);

pub type SharedModelManager = Arc<tokio::sync::Mutex<ModelManager>>;

pub enum GetResult {
    Ready(ReadyHandle),
    Wait(oneshot::Receiver<Result<ReadyHandle, String>>),
}

pub fn registry_path(models_dir: &Path) -> PathBuf {
    models_dir.join(".oxydllm_registry.json")
}

pub fn load_registry(models_dir: &Path) -> BTreeMap<String, RegistryEntry> {
    let path = registry_path(models_dir);
    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(_) => return BTreeMap::new(),
    };
    match serde_json::from_str(&raw) {
        Ok(reg) => reg,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "registry JSON is malformed, starting with empty registry");
            BTreeMap::new()
        }
    }
}

pub fn save_registry(models_dir: &Path, registry: &BTreeMap<String, RegistryEntry>) {
    let path = registry_path(models_dir);
    let Ok(json) = serde_json::to_string_pretty(registry) else {
        return;
    };
    let tmp_path = path.with_extension("json.tmp");
    if let Err(e) = std::fs::write(&tmp_path, &json) {
        tracing::error!(path = %tmp_path.display(), error = %e, "failed to write registry tmp file");
        return;
    }
    if let Err(e) = std::fs::rename(&tmp_path, &path) {
        tracing::error!(
            from = %tmp_path.display(),
            to = %path.display(),
            error = %e,
            "failed to rename registry tmp file"
        );
        let _ = std::fs::remove_file(&tmp_path);
    }
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

const BYTES_PER_GB: f64 = 1_073_741_824.0;

fn gb(bytes: usize) -> f64 {
    bytes as f64 / BYTES_PER_GB
}

fn round_2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn round_1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

pub fn estimate_model_size(model_dir: &Path) -> usize {
    std::fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "safetensors" || x == "gguf")
                .unwrap_or(false)
        })
        .filter_map(|e| std::fs::metadata(e.path()).ok())
        .map(|m| m.len() as usize)
        .sum()
}

pub struct ModelManagerConfig {
    pub models_dir: PathBuf,
    pub keep_alive: Duration,
    pub memory_budget_bytes: Option<usize>,
    pub cuda_devices: Vec<usize>,
    pub max_context_len: usize,
    pub kv_quant: KvQuantMode,
    pub qjl_quantization: bool,
    pub require_gpu: bool,
    pub max_num_seqs: Option<usize>,
    pub max_queued_requests: usize,
}

impl ModelManager {
    pub fn new(config: ModelManagerConfig) -> Self {
        let ModelManagerConfig {
            models_dir,
            keep_alive,
            memory_budget_bytes,
            cuda_devices,
            max_context_len,
            kv_quant,
            qjl_quantization,
            require_gpu,
            max_num_seqs,
            max_queued_requests,
        } = config;
        let mut registry = load_registry(&models_dir);
        let valid_ids: std::collections::HashSet<String> = loader::discover_models(&models_dir)
            .into_iter()
            .map(|m| m.id)
            .collect();
        let before = registry.len();
        registry.retain(|id, _| valid_ids.contains(id));
        if registry.len() < before {
            tracing::info!(
                removed_entries = before - registry.len(),
                "removed stale registry entries"
            );
            save_registry(&models_dir, &registry);
        }
        let is_cpu = cuda_devices.is_empty();
        let kv_total = detect_system_kv_budget(memory_budget_bytes, is_cpu);
        tracing::info!(
            kv_cache_budget_gb = round_2(gb(kv_total)),
            "global KV cache budget configured"
        );
        let kv_budget = Arc::new(GlobalKvBudget::new(kv_total));
        if kv_quant != KvQuantMode::Off {
            tracing::info!(mode = %kv_quant.label(), "KV cache quantization enabled");
            tracing::info!(qjl_quantization, "QJL key quantization configuration");
            tracing::warn!(
                "KV quantization currently runs on CPU: each write performs a \
                 GPU -> CPU transfer + F32 cast. Expect throughput regression vs \
                 --kv-quant off (most pronounced on discrete CUDA GPUs)."
            );
        }

        Self {
            models_dir,
            slots: HashMap::new(),
            keep_alive,
            memory_budget_bytes,
            registry,
            cuda_devices,
            max_context_len,
            kv_budget,
            kv_quant,
            qjl_quantization,
            require_gpu,
            max_num_seqs,
            max_queued_requests,
            discovery_cache: None,
        }
    }

    pub fn memory_budget_bytes(&self) -> Option<usize> {
        self.memory_budget_bytes
    }

    pub fn total_loaded_bytes(&self) -> usize {
        self.slots
            .values()
            .filter_map(|s| match s {
                SlotState::Ready {
                    weights_size_bytes,
                    kv_cache_bytes,
                    ..
                } => Some(*weights_size_bytes + *kv_cache_bytes),
                _ => None,
            })
            .sum()
    }

    pub fn list_running(&self) -> Vec<RunningModelInfo> {
        let now = Instant::now();
        self.slots
            .iter()
            .filter_map(|(id, slot)| match slot {
                SlotState::Ready {
                    architecture,
                    vocab_size,
                    num_layers,
                    last_used,
                    weights_size_bytes,
                    kv_cache_bytes,
                    ..
                } => Some(RunningModelInfo {
                    id: id.clone(),
                    architecture: architecture.clone(),
                    vocab_size: *vocab_size,
                    num_layers: *num_layers,
                    idle_seconds: now.duration_since(*last_used).as_secs(),
                    weights_size_bytes: *weights_size_bytes,
                    kv_cache_bytes: *kv_cache_bytes,
                }),
                _ => None,
            })
            .collect()
    }

    /// Atomic `(discovered, registry)` snapshot — both reads happen under the
    /// same manager lock acquisition, with discovery cached for
    /// `DISCOVERY_CACHE_TTL` to avoid re-scanning the filesystem on each call.
    pub fn discovered_with_registry(
        &mut self,
    ) -> (
        Vec<loader::DiscoveredModel>,
        BTreeMap<String, RegistryEntry>,
    ) {
        let fresh = self
            .discovery_cache
            .as_ref()
            .is_some_and(|c| c.refreshed_at.elapsed() < DISCOVERY_CACHE_TTL);
        if !fresh {
            self.discovery_cache = Some(DiscoveryCache {
                discovered: loader::discover_models(&self.models_dir),
                refreshed_at: Instant::now(),
            });
        }
        let discovered = self.discovery_cache.as_ref().unwrap().discovered.clone();
        (discovered, self.registry.clone())
    }

    #[cfg(test)]
    pub fn discovery_cache_age(&self) -> Option<Duration> {
        self.discovery_cache
            .as_ref()
            .map(|c| c.refreshed_at.elapsed())
    }

    #[cfg(test)]
    pub fn invalidate_discovery_cache(&mut self) {
        self.discovery_cache = None;
    }

    /// Best pre-load size estimate (weights + KV) for eviction decisions.
    /// Uses registry data when available; otherwise estimates from disk with
    /// dtype correction (CPU=F32 doubles BF16, AWQ keeps packed 4-bit).
    fn projected_size_bytes(&self, model_id: &str, model_path: &Path) -> usize {
        if let Some(entry) = self.registry.get(model_id)
            && entry.size_bytes > 0
        {
            return entry.size_bytes + entry.kv_cache_bytes;
        }

        let disk_bytes = estimate_model_size(model_path);
        let is_cpu = self.cuda_devices.is_empty();

        let config_path = model_path.join("config.json");
        let gpu_bytes = read_quantization_config(&config_path)
            .and_then(|qi| awq_qweight_expansion(qi.bits.unwrap_or(4)))
            .and_then(|expansion| {
                let (total, qweight) = sum_safetensors_bytes(model_path).ok()?;
                if qweight == 0 {
                    return None;
                }
                let other = total.saturating_sub(qweight);
                Some(other + (qweight as f64 * expansion) as usize)
            })
            .unwrap_or(disk_bytes);

        let (corrected, cpu_expansion) = if is_cpu {
            if let Some(gguf_path) = crate::models::loader::find_gguf_file(model_path) {
                let factor = crate::models::estimate::gguf_cpu_expansion(&gguf_path);
                ((gpu_bytes as f64 * factor) as usize, factor)
            } else {
                (gpu_bytes * 2, 2.0)
            }
        } else {
            (gpu_bytes, 1.0)
        };

        tracing::info!(
            model_id,
            projected_gb = round_2(gb(corrected)),
            estimate_source = "disk",
            cpu_expansion_factor = if is_cpu { cpu_expansion } else { 1.0 },
            "model not in registry, using projected memory size"
        );
        corrected
    }

    pub fn evict_lru_for_bytes(&mut self, needed_bytes: usize) -> usize {
        let budget = match self.memory_budget_bytes {
            Some(b) => b,
            None => return 0,
        };
        let mut evicted = 0;
        loop {
            let used = self.total_loaded_bytes();
            if used + needed_bytes <= budget {
                break;
            }
            let lru_id = self
                .slots
                .iter()
                .filter_map(|(id, s)| match s {
                    SlotState::Ready { last_used, .. } => Some((id.clone(), *last_used)),
                    _ => None,
                })
                .min_by_key(|(_, lu)| *lu)
                .map(|(id, _)| id);
            match lru_id {
                Some(id) => {
                    let (freed, kv_bytes, shutdown) = match self.slots.get(&id) {
                        Some(SlotState::Ready {
                            weights_size_bytes,
                            kv_cache_bytes,
                            shutdown,
                            ..
                        }) => (
                            *weights_size_bytes + *kv_cache_bytes,
                            *kv_cache_bytes,
                            Some(Arc::clone(shutdown)),
                        ),
                        _ => (0, 0, None),
                    };
                    tracing::warn!(
                        model_id = %id,
                        evicted_gb = round_2(gb(freed)),
                        needed_gb = round_2(gb(needed_bytes)),
                        budget_gb = round_2(gb(budget)),
                        used_gb = round_2(gb(used)),
                        "evicting LRU model due to memory pressure"
                    );
                    self.kv_budget.release(kv_bytes);
                    self.slots.remove(&id);
                    if let Some(s) = shutdown {
                        s.store(true, Ordering::Release);
                    }
                    evicted += 1;
                }
                None => break,
            }
        }
        evicted
    }

    pub fn get_or_load(
        &mut self,
        model_id: &str,
        manager_handle: SharedModelManager,
        keep_alive_override: Option<Duration>,
    ) -> GetResult {
        let model_path = match crate::models::loader::resolve_model_path(&self.models_dir, model_id)
        {
            Some(p) => p,
            None => {
                let (tx, rx) = oneshot::channel();
                let _ = tx.send(Err(format!(
                    "model '{}' not found in models directory",
                    model_id
                )));
                return GetResult::Wait(rx);
            }
        };

        if let Some(slot) = self.slots.get_mut(model_id) {
            match slot {
                SlotState::Ready {
                    request_tx,
                    tokenizer,
                    max_seq_len,
                    last_used,
                    effective_keep_alive,
                    ..
                } => {
                    *last_used = Instant::now();
                    if let Some(override_ka) = keep_alive_override {
                        *effective_keep_alive = override_ka;
                    }
                    return GetResult::Ready(ReadyHandle {
                        request_tx: request_tx.clone(),
                        tokenizer: Arc::clone(tokenizer),
                        max_seq_len: *max_seq_len,
                    });
                }
                SlotState::Loading { waiters } => {
                    let (tx, rx) = oneshot::channel();
                    waiters.push(tx);
                    return GetResult::Wait(rx);
                }
            }
        }

        let needed_bytes = self.projected_size_bytes(model_id, &model_path);
        self.evict_lru_for_bytes(needed_bytes);

        let effective_keep_alive = keep_alive_override.unwrap_or(self.keep_alive);

        let (tx, rx) = oneshot::channel();
        self.slots.insert(
            model_id.to_string(),
            SlotState::Loading { waiters: vec![tx] },
        );

        spawn_load(SpawnLoadParams {
            model_id: model_id.to_string(),
            model_path,
            manager: manager_handle,
            effective_keep_alive,
            cuda_devices: self.cuda_devices.clone(),
            max_context_len: self.max_context_len,
            kv_budget: Arc::clone(&self.kv_budget),
            kv_quant: self.kv_quant,
            qjl_quantization: self.qjl_quantization,
            require_gpu: self.require_gpu,
            max_num_seqs: self.max_num_seqs,
            max_queued_requests: self.max_queued_requests,
        });

        GetResult::Wait(rx)
    }

    pub fn evict_expired(&mut self) {
        let now = Instant::now();
        let expired: Vec<String> = self
            .slots
            .iter()
            .filter_map(|(id, slot)| match slot {
                SlotState::Ready {
                    last_used,
                    effective_keep_alive,
                    ..
                } if now.duration_since(*last_used) > *effective_keep_alive => Some(id.clone()),
                _ => None,
            })
            .collect();

        for id in &expired {
            let (kv_bytes, shutdown) = match self.slots.get(id) {
                Some(SlotState::Ready {
                    kv_cache_bytes,
                    shutdown,
                    ..
                }) => (*kv_cache_bytes, Some(Arc::clone(shutdown))),
                _ => (0, None),
            };
            tracing::info!(model_id = %id, "evicting idle model (keep-alive expired)");
            self.kv_budget.release(kv_bytes);
            self.slots.remove(id);
            if let Some(s) = shutdown {
                s.store(true, Ordering::Release);
            }
        }
    }

    #[cfg(test)]
    pub fn insert_ready_with_size_for_tests(
        &mut self,
        model_id: &str,
        handle: ReadyHandle,
        weights_bytes: usize,
        kv_bytes: usize,
    ) {
        self.slots.insert(
            model_id.to_string(),
            SlotState::Ready {
                request_tx: handle.request_tx,
                tokenizer: handle.tokenizer,
                max_seq_len: handle.max_seq_len,
                architecture: "TestArchitecture".to_string(),
                vocab_size: 0,
                num_layers: 0,
                last_used: Instant::now(),
                effective_keep_alive: self.keep_alive,
                weights_size_bytes: weights_bytes,
                kv_cache_bytes: kv_bytes,
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        );
    }

    #[cfg(test)]
    pub fn insert_aged_slot_for_tests(
        &mut self,
        model_id: &str,
        handle: ReadyHandle,
        age: Duration,
    ) {
        self.slots.insert(
            model_id.to_string(),
            SlotState::Ready {
                request_tx: handle.request_tx,
                tokenizer: handle.tokenizer,
                max_seq_len: handle.max_seq_len,
                architecture: "TestArchitecture".to_string(),
                vocab_size: 0,
                num_layers: 0,
                last_used: Instant::now() - age,
                effective_keep_alive: self.keep_alive,
                weights_size_bytes: 0,
                kv_cache_bytes: 0,
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        );
    }

    #[cfg(test)]
    pub fn insert_ready_for_tests(&mut self, model_id: &str, handle: ReadyHandle) {
        self.slots.insert(
            model_id.to_string(),
            SlotState::Ready {
                request_tx: handle.request_tx,
                tokenizer: handle.tokenizer,
                max_seq_len: handle.max_seq_len,
                architecture: "TestArchitecture".to_string(),
                vocab_size: 0,
                num_layers: 0,
                last_used: Instant::now(),
                effective_keep_alive: self.keep_alive,
                weights_size_bytes: 0,
                kv_cache_bytes: 0,
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        );
    }

    pub fn update_registry(
        &mut self,
        model_id: &str,
        architecture: &str,
        vocab_size: usize,
        num_layers: usize,
        size_bytes: usize,
        kv_cache_bytes: usize,
    ) {
        let entry = self.registry.entry(model_id.to_string()).or_default();
        entry.architecture = architecture.to_string();
        entry.vocab_size = vocab_size;
        entry.num_layers = num_layers;
        entry.size_bytes = size_bytes;
        entry.kv_cache_bytes = kv_cache_bytes;
        entry.last_used_secs = now_unix_secs();
        save_registry(&self.models_dir, &self.registry);
    }
}

struct LoadResult {
    request_tx: tokio_mpsc::Sender<IncomingRequest>,
    tokenizer: Arc<Tokenizer>,
    max_seq_len: usize,
    architecture: String,
    vocab_size: usize,
    num_layers: usize,
    weights_size_bytes: usize,
    kv_cache_bytes: usize,
    shutdown: Arc<AtomicBool>,
}

struct SpawnLoadParams {
    model_id: String,
    model_path: PathBuf,
    manager: SharedModelManager,
    effective_keep_alive: Duration,
    cuda_devices: Vec<usize>,
    max_context_len: usize,
    kv_budget: SharedGlobalKvBudget,
    kv_quant: KvQuantMode,
    qjl_quantization: bool,
    require_gpu: bool,
    max_num_seqs: Option<usize>,
    max_queued_requests: usize,
}

/// JIT-compile all SDPA kernels before marking the model Ready, otherwise the
/// first real request pays multi-second compilation latency. Phase 3 (full
/// prefill) MUST be last: a Metal candle bug emits NaN logits on the first
/// `call_sdpa_full` after a `call_sdpa_vector` call without an intervening full.
fn warm_up_model(model: &dyn BatchModel) {
    let device = model.device();
    let allocators = model.allocators();

    let lock = crate::gpu_lock::gpu_lock_for(device);
    let _gpu = lock.acquire();

    let mut caches: Vec<PagedKvCache> = allocators
        .iter()
        .map(|a| PagedKvCache::new(std::sync::Arc::clone(a)))
        .collect();

    {
        let dummy_tokens: Vec<u32> = (1..=8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let input = match Tensor::from_vec(dummy_tokens, (1, 8), device) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "warmup failed to create prefill tensor");
                return;
            }
        };
        let position_ids = match Tensor::from_vec(positions, (8,), device) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "warmup failed to create prefill position ids");
                return;
            }
        };

        let mut cache_slices: Vec<&mut [PagedKvCache]> = vec![caches.as_mut_slice()];
        if let Err(e) = model.forward_batch(&input, &position_ids, &mut cache_slices, &[8]) {
            tracing::warn!(error = %e, "warmup prefill forward failed (non-fatal)");
        } else if let Err(e) = crate::common::block::flush_caches(&mut cache_slices) {
            tracing::warn!(error = %e, "warmup prefill flush failed (non-fatal)");
        }
    }

    {
        let input = match Tensor::from_vec(vec![1u32], (1, 1), device) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "warmup failed to create decode tensor");
                return;
            }
        };
        let position_ids = match Tensor::from_vec(vec![8u32], (1,), device) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "warmup failed to create decode position ids");
                return;
            }
        };

        let mut cache_slices: Vec<&mut [PagedKvCache]> = vec![caches.as_mut_slice()];
        if let Err(e) = model.forward_batch(&input, &position_ids, &mut cache_slices, &[1]) {
            tracing::warn!(error = %e, "warmup decode forward failed (non-fatal)");
        }
    }

    for cache in &mut caches {
        cache.clear();
    }

    const FULL_WARMUP_TOKENS: usize = 16;
    {
        let dummy_tokens: Vec<u32> = (1..=FULL_WARMUP_TOKENS as u32).collect();
        let positions: Vec<u32> = (0..FULL_WARMUP_TOKENS as u32).collect();
        let input = match Tensor::from_vec(dummy_tokens, (1, FULL_WARMUP_TOKENS), device) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "warmup failed to create full-path prefill tensor");
                return;
            }
        };
        let position_ids = match Tensor::from_vec(positions, (FULL_WARMUP_TOKENS,), device) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "warmup failed to create full-path position ids");
                return;
            }
        };

        let mut cache_slices: Vec<&mut [PagedKvCache]> = vec![caches.as_mut_slice()];
        if let Err(e) = model.forward_batch(
            &input,
            &position_ids,
            &mut cache_slices,
            &[FULL_WARMUP_TOKENS],
        ) {
            tracing::warn!(error = %e, "warmup full-path prefill forward failed (non-fatal)");
        } else if let Err(e) = crate::common::block::flush_caches(&mut cache_slices) {
            tracing::warn!(error = %e, "warmup full-path prefill flush failed (non-fatal)");
        }
    }

    for cache in &mut caches {
        cache.clear();
    }

    #[cfg(feature = "metal")]
    if let candle_core::Device::Metal(dev) = device {
        dev.wait_until_completed().ok();
    }
}

fn spawn_load(params: SpawnLoadParams) {
    let SpawnLoadParams {
        model_id,
        model_path,
        manager,
        effective_keep_alive,
        cuda_devices,
        max_context_len,
        kv_budget,
        kv_quant,
        qjl_quantization,
        require_gpu,
        max_num_seqs,
        max_queued_requests,
    } = params;

    let (result_tx, result_rx) = oneshot::channel::<Result<LoadResult, String>>();

    let model_id_thread = model_id.clone();
    let model_path_thread = model_path.clone();
    let cuda_devices_thread = cuda_devices;
    let require_gpu_thread = require_gpu;
    std::thread::spawn(move || {
        let model_dir = model_path_thread.to_string_lossy().to_string();

        let tokenizer = match Tokenizer::from_dir(&model_dir) {
            Ok(t) => Arc::new(t),
            Err(e) => {
                let _ = result_tx.send(Err(format!("Failed to load tokenizer: {e}")));
                return;
            }
        };

        let cuda_idx = cuda_devices_thread.first().copied().unwrap_or(0);
        let device = match loader::select_device_at(cuda_idx, require_gpu_thread) {
            Ok(d) => d,
            Err(e) => {
                let _ = result_tx.send(Err(format!("Failed to select device: {e}")));
                return;
            }
        };

        tracing::info!(model_id = %model_id_thread, "loading model");
        let load_max_num_sequences: usize = max_num_seqs.unwrap_or(8);
        let (batch_model, weights_size_bytes) = match loader::load_batch_model(
            &model_dir,
            &model_id_thread,
            &device,
            loader::LoadBatchOptions {
                max_context_len,
                max_num_sequences: load_max_num_sequences,
                kv_budget: &kv_budget,
                kv_quant,
                qjl_quantization,
            },
        ) {
            Ok(m) => m,
            Err(e) => {
                let _ = result_tx.send(Err(format!("Failed to load model: {e}")));
                return;
            }
        };

        let max_seq_len = batch_model.max_seq_len();
        let vocab_size = batch_model.vocab_size();
        let num_layers = batch_model.num_layers();
        let kv_cache_bytes = batch_model.kv_cache_bytes();

        let config_path = model_path_thread.join("config.json");
        let architecture = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .and_then(|v| v["architectures"][0].as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| {
                if loader::is_gguf_model(&model_dir) {
                    "GGUF".to_string()
                } else {
                    "Unknown".to_string()
                }
            });

        let total_bytes = weights_size_bytes + kv_cache_bytes;
        tracing::info!(
            model_id = %model_id_thread,
            vocab_size,
            max_seq_len,
            num_layers,
            weights_gb = round_2(gb(weights_size_bytes)),
            kv_cache_gb = round_2(gb(kv_cache_bytes)),
            total_gb = round_2(gb(total_bytes)),
            precision_profile = if matches!(device, candle_core::Device::Cpu) {
                "F32/CPU"
            } else {
                "BF16/GPU"
            },
            "model loaded"
        );

        tracing::info!(model_id = %model_id_thread, "warming up model");
        let t_warmup = std::time::Instant::now();
        warm_up_model(batch_model.as_ref());
        tracing::info!(
            model_id = %model_id_thread,
            warmup_s = round_1(t_warmup.elapsed().as_secs_f64()),
            "model ready"
        );

        let block_size = if batch_model.allocators().is_empty() {
            crate::common::paged::DEFAULT_BLOCK_SIZE
        } else {
            batch_model.allocators()[0].lock().unwrap().block_size()
        };
        let total_blocks = if batch_model.allocators().is_empty() {
            0
        } else {
            batch_model.allocators()[0].lock().unwrap().num_total()
        };
        let blocks_per_seq = max_context_len.div_ceil(block_size).max(1);
        let dynamic_max = (total_blocks / blocks_per_seq).clamp(1, 256);
        let scheduler_max_seqs = match max_num_seqs {
            Some(n) => {
                let capped = n.min(dynamic_max);
                if capped < n {
                    tracing::warn!(
                        model_id = %model_id_thread,
                        requested = n,
                        capacity = dynamic_max,
                        "max-num-seqs capped by available KV blocks"
                    );
                }
                capped
            }
            None => {
                tracing::info!(
                    model_id = %model_id_thread,
                    max_num_seqs = dynamic_max,
                    "max concurrent sequences computed from KV block capacity"
                );
                dynamic_max
            }
        };

        let (request_tx, request_rx) = tokio_mpsc::channel(max_queued_requests);
        let shutdown = Arc::new(AtomicBool::new(false));

        let result = LoadResult {
            request_tx,
            tokenizer: Arc::clone(&tokenizer),
            max_seq_len,
            architecture,
            vocab_size,
            num_layers,
            weights_size_bytes,
            kv_cache_bytes,
            shutdown: Arc::clone(&shutdown),
        };
        let _ = result_tx.send(Ok(result));

        let config = SchedulerConfig {
            max_num_sequences: scheduler_max_seqs,
            max_tokens_per_step: 4096,
        };
        let extra_stop_ids = tokenizer.stop_token_ids();
        let extra_stop_sequences = tokenizer.stop_token_sequences();
        let engine = Engine::new_with_stop_controls(
            batch_model,
            config,
            &extra_stop_ids,
            &extra_stop_sequences,
        );
        engine_loop(engine, tokenizer, request_rx, shutdown);
    });

    tokio::spawn(async move {
        match result_rx.await {
            Ok(Ok(result)) => {
                let mut mgr = manager.lock().await;
                let slot = mgr.slots.remove(&model_id);
                let waiters = match slot {
                    Some(SlotState::Loading { waiters }) => waiters,
                    _ => Vec::new(),
                };

                let handle = ReadyHandle {
                    request_tx: result.request_tx.clone(),
                    tokenizer: Arc::clone(&result.tokenizer),
                    max_seq_len: result.max_seq_len,
                };

                mgr.update_registry(
                    &model_id,
                    &result.architecture,
                    result.vocab_size,
                    result.num_layers,
                    result.weights_size_bytes,
                    result.kv_cache_bytes,
                );

                mgr.slots.insert(
                    model_id.clone(),
                    SlotState::Ready {
                        request_tx: result.request_tx,
                        tokenizer: result.tokenizer,
                        max_seq_len: result.max_seq_len,
                        architecture: result.architecture,
                        vocab_size: result.vocab_size,
                        num_layers: result.num_layers,
                        last_used: Instant::now(),
                        effective_keep_alive,
                        weights_size_bytes: result.weights_size_bytes,
                        kv_cache_bytes: result.kv_cache_bytes,
                        shutdown: result.shutdown,
                    },
                );

                if let Some(budget) = mgr.memory_budget_bytes {
                    let total = mgr.total_loaded_bytes();
                    if total > budget {
                        let model_bytes = result.weights_size_bytes + result.kv_cache_bytes;
                        tracing::warn!(
                            model_id = %model_id,
                            used_gb = round_2(gb(total)),
                            budget_gb = round_2(gb(budget)),
                            model_gb = round_2(gb(model_bytes)),
                            "post-load memory budget exceeded, unloading model"
                        );
                        let (kv_bytes, shutdown) = match mgr.slots.get(&model_id) {
                            Some(SlotState::Ready {
                                kv_cache_bytes,
                                shutdown,
                                ..
                            }) => (*kv_cache_bytes, Some(Arc::clone(shutdown))),
                            _ => (0, None),
                        };
                        mgr.kv_budget.release(kv_bytes);
                        mgr.slots.remove(&model_id);
                        drop(mgr);
                        if let Some(s) = shutdown {
                            s.store(true, Ordering::Release);
                        }
                        let err = format!(
                            "model '{}' ({:.2} GB) exceeds memory budget ({:.2} GB) — \
                             use --memory-budget to increase the limit or --max-context-len \
                             to reduce KV cache size",
                            model_id,
                            model_bytes as f64 / 1_073_741_824.0,
                            budget as f64 / 1_073_741_824.0,
                        );
                        tracing::error!(model_id = %model_id, error = %err, "loaded model exceeds memory budget");
                        for waiter in waiters {
                            let _ = waiter.send(Err(err.clone()));
                        }
                        return;
                    }
                }

                drop(mgr);

                for waiter in waiters {
                    let _ = waiter.send(Ok(handle.clone()));
                }
            }
            Ok(Err(err)) => {
                let mut mgr = manager.lock().await;
                let slot = mgr.slots.remove(&model_id);
                let waiters = match slot {
                    Some(SlotState::Loading { waiters }) => waiters,
                    _ => Vec::new(),
                };
                drop(mgr);

                tracing::error!(model_id = %model_id, error = %err, "failed to load model");
                for waiter in waiters {
                    let _ = waiter.send(Err(err.clone()));
                }
            }
            Err(_) => {
                let mut mgr = manager.lock().await;
                let slot = mgr.slots.remove(&model_id);
                let waiters = match slot {
                    Some(SlotState::Loading { waiters }) => waiters,
                    _ => Vec::new(),
                };
                drop(mgr);

                let err = format!("Model '{}' loader thread panicked", model_id);
                tracing::error!(model_id = %model_id, error = %err, "model loader thread panicked");
                for waiter in waiters {
                    let _ = waiter.send(Err(err.clone()));
                }
            }
        }
    });
}

pub fn spawn_eviction_task(manager: SharedModelManager) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            let mut mgr = manager.lock().await;
            mgr.evict_expired();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn build_test_manager(
        tmp: &tempfile::TempDir,
        keep_alive: Duration,
        budget: Option<usize>,
    ) -> ModelManager {
        ModelManager::new(ModelManagerConfig {
            models_dir: tmp.path().to_path_buf(),
            keep_alive,
            memory_budget_bytes: budget,
            cuda_devices: vec![],
            max_context_len: 512,
            kv_quant: crate::common::kv_quant::KvQuantMode::Off,
            qjl_quantization: false,
            require_gpu: false,
            max_num_seqs: None,
            max_queued_requests: 200,
        })
    }

    fn build_dummy_handle() -> ReadyHandle {
        let tmp = tempfile::tempdir().unwrap();
        let model = WordLevel::builder()
            .vocab([("[UNK]".to_string(), 0u32)].into_iter().collect())
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut inner = tokenizers::Tokenizer::new(model);
        inner.with_pre_tokenizer(Some(Whitespace {}));
        inner
            .save(tmp.path().join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(tmp.path().join("tokenizer_config.json"), "{}").unwrap();
        let tok =
            Arc::new(crate::tokenizer::Tokenizer::from_dir(tmp.path().to_str().unwrap()).unwrap());
        let (tx, _rx) = tokio_mpsc::channel(200);
        ReadyHandle {
            request_tx: tx,
            tokenizer: tok,
            max_seq_len: 1024,
        }
    }

    #[test]
    fn registry_save_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let mut reg = BTreeMap::new();
        reg.insert(
            "model-a".to_string(),
            RegistryEntry {
                architecture: "TestArch".to_string(),
                vocab_size: 32000,
                num_layers: 32,
                size_bytes: 100,
                kv_cache_bytes: 50,
                last_used_secs: 12345,
            },
        );
        save_registry(tmp.path(), &reg);
        let loaded = load_registry(tmp.path());
        assert_eq!(loaded["model-a"].architecture, "TestArch");
        assert_eq!(loaded["model-a"].vocab_size, 32000);
        assert_eq!(loaded["model-a"].size_bytes, 100);
    }

    #[test]
    fn registry_keys_are_serialized_in_alphabetical_order() {
        let tmp = tempfile::tempdir().unwrap();
        let mut reg = BTreeMap::new();
        reg.insert("z-model".to_string(), RegistryEntry::default());
        reg.insert("a-model".to_string(), RegistryEntry::default());
        reg.insert("m-model".to_string(), RegistryEntry::default());
        save_registry(tmp.path(), &reg);
        let content = std::fs::read_to_string(registry_path(tmp.path())).unwrap();
        let a = content.find("a-model").unwrap();
        let m = content.find("m-model").unwrap();
        let z = content.find("z-model").unwrap();
        assert!(
            a < m && m < z,
            "registry keys must appear in alphabetical order"
        );
    }

    #[test]
    fn new_prunes_stale_registry_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let mut reg = BTreeMap::new();
        reg.insert("ghost-model".to_string(), RegistryEntry::default());
        save_registry(tmp.path(), &reg);

        let mgr = build_test_manager(&tmp, Duration::from_secs(60), None);
        assert!(
            mgr.registry.get("ghost-model").is_none(),
            "stale registry entry should be pruned on startup"
        );
    }

    #[test]
    fn update_registry_upserts_and_persists_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let mut mgr = build_test_manager(&tmp, Duration::from_secs(60), None);

        mgr.update_registry(
            "my-model",
            "LlamaForCausalLM",
            32000,
            32,
            1_000_000,
            500_000,
        );

        let entry = mgr.registry.get("my-model").expect("entry must exist");
        assert_eq!(entry.architecture, "LlamaForCausalLM");
        assert_eq!(entry.vocab_size, 32000);
        assert_eq!(entry.size_bytes, 1_000_000);

        let on_disk = load_registry(tmp.path());
        assert!(
            on_disk.contains_key("my-model"),
            "entry must be written to disk"
        );
    }

    #[test]
    fn discovery_cache_serves_repeated_reads_without_rescan() {
        let tmp = tempfile::tempdir().unwrap();
        let mut mgr = build_test_manager(&tmp, Duration::from_secs(60), None);

        let (_, _) = mgr.discovered_with_registry();
        let age_after_first = mgr.discovery_cache_age().expect("cache must be populated");

        std::thread::sleep(Duration::from_millis(20));
        let (_, _) = mgr.discovered_with_registry();
        let age_after_second = mgr.discovery_cache_age().expect("cache still populated");
        assert!(
            age_after_second >= age_after_first,
            "cached entry should not be refreshed within TTL: first={age_after_first:?} second={age_after_second:?}"
        );

        mgr.invalidate_discovery_cache();
        assert!(mgr.discovery_cache_age().is_none());
        let (_, _) = mgr.discovered_with_registry();
        assert!(
            mgr.discovery_cache_age().is_some(),
            "cache must be re-populated after invalidation"
        );
    }

    #[test]
    fn discovery_cache_snapshot_is_consistent_with_registry() {
        let tmp = tempfile::tempdir().unwrap();
        let mut mgr = build_test_manager(&tmp, Duration::from_secs(60), None);

        mgr.update_registry("registered", "LlamaForCausalLM", 32000, 32, 1_000, 500);
        let (_, registry) = mgr.discovered_with_registry();
        assert!(registry.contains_key("registered"));
        assert_eq!(registry["registered"].size_bytes, 1_000);
    }

    #[test]
    fn evict_lru_returns_zero_when_no_budget() {
        let tmp = tempfile::tempdir().unwrap();
        let mut mgr = build_test_manager(&tmp, Duration::from_secs(60), None);
        assert_eq!(mgr.evict_lru_for_bytes(1_000_000), 0);
    }

    #[test]
    fn evict_lru_removes_model_when_over_budget() {
        let tmp = tempfile::tempdir().unwrap();
        let budget: usize = 100;
        let mut mgr = build_test_manager(&tmp, Duration::from_secs(60), Some(budget));

        let h = build_dummy_handle();
        mgr.insert_ready_with_size_for_tests("big-model", h, 80, 0);

        let evicted = mgr.evict_lru_for_bytes(50);
        assert_eq!(evicted, 1);
        assert_eq!(
            mgr.list_running().len(),
            0,
            "evicted model must be removed from running"
        );
    }

    #[test]
    fn evict_expired_removes_idle_slot() {
        let tmp = tempfile::tempdir().unwrap();
        let keep_alive = Duration::from_secs(10);
        let mut mgr = build_test_manager(&tmp, keep_alive, None);

        let h = build_dummy_handle();
        mgr.insert_aged_slot_for_tests("old-model", h, Duration::from_secs(60));

        assert_eq!(mgr.list_running().len(), 1);
        mgr.evict_expired();
        assert_eq!(
            mgr.list_running().len(),
            0,
            "slot past keep-alive must be evicted"
        );
    }
}
