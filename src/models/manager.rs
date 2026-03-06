use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};

use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc as tokio_mpsc, oneshot};

use crate::common::paged::{detect_system_kv_budget, GlobalKvBudget, PagedKvCache, SharedGlobalKvBudget};
use crate::engine::Engine;
use crate::models::loader;
use crate::models::traits::BatchModel;
use crate::scheduler::SchedulerConfig;
use crate::server::{engine_loop, IncomingRequest};
use crate::tokenizer::Tokenizer;

#[derive(Clone)]
pub struct ReadyHandle {
    pub request_tx: tokio_mpsc::UnboundedSender<IncomingRequest>,
    pub tokenizer: Arc<Tokenizer>,
    pub max_seq_len: usize,
}

enum SlotState {
    Loading {
        waiters: Vec<oneshot::Sender<Result<ReadyHandle, String>>>,
    },
    Ready {
        request_tx: tokio_mpsc::UnboundedSender<IncomingRequest>,
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
    registry: HashMap<String, RegistryEntry>,
    cuda_devices: Vec<usize>,
    max_context_len: usize,
    kv_budget: SharedGlobalKvBudget,
}

pub type SharedModelManager = Arc<tokio::sync::Mutex<ModelManager>>;

pub enum GetResult {
    Ready(ReadyHandle),
    Wait(oneshot::Receiver<Result<ReadyHandle, String>>),
}


fn registry_path(models_dir: &Path) -> PathBuf {
    models_dir.join(".rllm_registry.json")
}

fn load_registry(models_dir: &Path) -> HashMap<String, RegistryEntry> {
    let path = registry_path(models_dir);
    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(_) => return HashMap::new(),
    };
    serde_json::from_str(&raw).unwrap_or_default()
}

fn save_registry(models_dir: &Path, registry: &HashMap<String, RegistryEntry>) {
    let path = registry_path(models_dir);
    if let Ok(json) = serde_json::to_string_pretty(registry) {
        let _ = std::fs::write(path, json);
    }
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Returns the best available size estimate for `model_id` before it is loaded.
/// Priority:
///   1. Registry entry — real in-memory footprint from a previous load (accurate).
///   2. Disk safetensors size — corrected by dtype: ×2 on CPU (F32), ×1 on GPU (BF16).
pub fn estimate_model_size(model_dir: &Path) -> usize {
    std::fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "safetensors")
                .unwrap_or(false)
        })
        .filter_map(|e| std::fs::metadata(e.path()).ok())
        .map(|m| m.len() as usize)
        .sum()
}


impl ModelManager {
    pub fn new(
        models_dir: PathBuf,
        keep_alive: Duration,
        memory_budget_bytes: Option<usize>,
        cuda_devices: Vec<usize>,
        max_context_len: usize,
    ) -> Self {
        let registry = load_registry(&models_dir);
        let is_cpu = cuda_devices.is_empty();
        let kv_total = detect_system_kv_budget(memory_budget_bytes, is_cpu);
        println!(
            "[kv-pool] Global KV cache budget: {:.2} GB",
            kv_total as f64 / 1_073_741_824.0,
        );
        let kv_budget = Arc::new(GlobalKvBudget::new(kv_total));
        Self {
            models_dir,
            slots: HashMap::new(),
            keep_alive,
            memory_budget_bytes,
            registry,
            cuda_devices,
            max_context_len,
            kv_budget,
        }
    }

    pub fn models_dir(&self) -> &PathBuf {
        &self.models_dir
    }

    pub fn memory_budget_bytes(&self) -> Option<usize> {
        self.memory_budget_bytes
    }

    pub fn total_loaded_bytes(&self) -> usize {
        self.slots
            .values()
            .filter_map(|s| match s {
                SlotState::Ready { weights_size_bytes, kv_cache_bytes, .. } => {
                    Some(*weights_size_bytes + *kv_cache_bytes)
                }
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

    pub fn list_registry(&self) -> &HashMap<String, RegistryEntry> {
        &self.registry
    }

    /// Returns the best pre-load size estimate for eviction decisions.
    /// Uses real measured registry data when available; falls back to a
    /// corrected disk estimate otherwise.
    /// The estimate includes both weights and KV cache.
    fn projected_size_bytes(&self, model_id: &str, model_path: &Path) -> usize {
        // Case 1: previously loaded — use the real in-memory footprint (weights + kv).
        if let Some(entry) = self.registry.get(model_id)
            && entry.size_bytes > 0 {
                return entry.size_bytes + entry.kv_cache_bytes;
            }

        // Case 2: first-ever load — estimate from disk.
        // On GPU we load BF16 (≈ same size as on-disk BF16 safetensors).
        // On CPU we load F32 (2× larger than BF16 on-disk files).
        let disk_bytes = estimate_model_size(model_path);
        let is_cpu = self.cuda_devices.is_empty();
        let corrected = if is_cpu { disk_bytes * 2 } else { disk_bytes };

        println!(
            "[memory] '{}' not in registry — disk estimate: {:.2} GB{}",
            model_id,
            corrected as f64 / 1_073_741_824.0,
            if is_cpu { " (×2 for F32/CPU)" } else { " (BF16/GPU)" },
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
                        Some(SlotState::Ready { weights_size_bytes, kv_cache_bytes, shutdown, .. }) => {
                            (*weights_size_bytes + *kv_cache_bytes, *kv_cache_bytes, Some(Arc::clone(shutdown)))
                        }
                        _ => (0, 0, None),
                    };
                    println!(
                        "[memory pressure] Evicting LRU model '{}' ({:.2} GB) — need {:.2} GB, budget {:.2} GB, used {:.2} GB",
                        id,
                        freed as f64 / 1_073_741_824.0,
                        needed_bytes as f64 / 1_073_741_824.0,
                        budget as f64 / 1_073_741_824.0,
                        used as f64 / 1_073_741_824.0,
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
        let model_path = match crate::models::loader::resolve_model_path(
            &self.models_dir,
            model_id,
        ) {
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
            SlotState::Loading {
                waiters: vec![tx],
            },
        );

        spawn_load(
            model_id.to_string(),
            model_path,
            manager_handle,
            effective_keep_alive,
            self.cuda_devices.clone(),
            self.max_context_len,
            Arc::clone(&self.kv_budget),
        );

        GetResult::Wait(rx)
    }

    pub fn evict_expired(&mut self) {
        let now = Instant::now();
        let expired: Vec<String> = self
            .slots
            .iter()
            .filter_map(|(id, slot)| match slot {
                SlotState::Ready { last_used, effective_keep_alive, .. }
                    if now.duration_since(*last_used) > *effective_keep_alive =>
                {
                    Some(id.clone())
                }
                _ => None,
            })
            .collect();

        for id in &expired {
            let (kv_bytes, shutdown) = match self.slots.get(id) {
                Some(SlotState::Ready { kv_cache_bytes, shutdown, .. }) => {
                    (*kv_cache_bytes, Some(Arc::clone(shutdown)))
                }
                _ => (0, None),
            };
            println!("Evicting idle model '{}' (keep-alive expired)", id);
            self.kv_budget.release(kv_bytes);
            self.slots.remove(id);
            if let Some(s) = shutdown {
                s.store(true, Ordering::Release);
            }
        }
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
    request_tx: tokio_mpsc::UnboundedSender<IncomingRequest>,
    tokenizer: Arc<Tokenizer>,
    max_seq_len: usize,
    architecture: String,
    vocab_size: usize,
    num_layers: usize,
    weights_size_bytes: usize,
    kv_cache_bytes: usize,
    shutdown: Arc<AtomicBool>,
}

/// Runs dummy forward passes to force the device (Metal/CUDA) to JIT-compile
/// GPU kernels for **both** prefill (multi-token) and decode (single-token) paths
/// before the model is marked as Ready.  Without this, the very first real
/// request pays the compilation cost, causing multi-second TTFT spikes.
///
/// Uses `forward_batch` — the same code path the engine uses at runtime — so
/// Metal specialises kernels for the exact shapes encountered during inference.
fn warm_up_model(model: &dyn BatchModel) {
    let device = model.device();
    let allocators = model.allocators();

    // Acquire the global GPU lock so warmup doesn't contend with any
    // model that is already running inference on the same device.
    let lock = crate::gpu_lock::gpu_lock();
    let _gpu = lock.acquire();

    // --- Phase 1: prefill-shaped batch (seq_len > 1, causal mask) ---
    {
        let mut caches: Vec<PagedKvCache> = allocators
            .iter()
            .map(|a| PagedKvCache::new(std::sync::Arc::clone(a)))
            .collect();

        let dummy_tokens: Vec<u32> = (1..=8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let input = match Tensor::from_vec(dummy_tokens, (1, 8), device) {
            Ok(t) => t,
            Err(e) => { eprintln!("[warmup] failed to create prefill tensor: {e}"); return; }
        };
        let position_ids = match Tensor::from_vec(positions, (8,), device) {
            Ok(t) => t,
            Err(e) => { eprintln!("[warmup] failed to create position ids: {e}"); return; }
        };

        let mut cache_slices: Vec<&mut [PagedKvCache]> = vec![caches.as_mut_slice()];
        if let Err(e) = model.forward_batch(&input, &position_ids, &mut cache_slices, &[8]) {
            eprintln!("[warmup] prefill forward failed (non-fatal): {e}");
        }

        for cache in &mut caches {
            cache.clear();
        }
    }

    // --- Phase 2: decode-shaped batch (seq_len == 1, no causal mask) ---
    {
        let mut caches: Vec<PagedKvCache> = allocators
            .iter()
            .map(|a| PagedKvCache::new(std::sync::Arc::clone(a)))
            .collect();

        let input = match Tensor::from_vec(vec![1u32], (1, 1), device) {
            Ok(t) => t,
            Err(e) => { eprintln!("[warmup] failed to create decode tensor: {e}"); return; }
        };
        let position_ids = match Tensor::from_vec(vec![8u32], (1,), device) {
            Ok(t) => t,
            Err(e) => { eprintln!("[warmup] failed to create decode position ids: {e}"); return; }
        };

        let mut cache_slices: Vec<&mut [PagedKvCache]> = vec![caches.as_mut_slice()];
        if let Err(e) = model.forward_batch(&input, &position_ids, &mut cache_slices, &[1]) {
            eprintln!("[warmup] decode forward failed (non-fatal): {e}");
        }

        for cache in &mut caches {
            cache.clear();
        }
    }
}

fn spawn_load(
    model_id: String,
    model_path: PathBuf,
    manager: SharedModelManager,
    effective_keep_alive: Duration,
    cuda_devices: Vec<usize>,
    max_context_len: usize,
    kv_budget: SharedGlobalKvBudget,
) {
    let (result_tx, result_rx) = oneshot::channel::<Result<LoadResult, String>>();

    let model_id_thread = model_id.clone();
    let model_path_thread = model_path.clone();
    let cuda_devices_thread = cuda_devices;
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
        let device = match loader::select_device_at(cuda_idx) {
            Ok(d) => d,
            Err(e) => {
                let _ = result_tx.send(Err(format!("Failed to select device: {e}")));
                return;
            }
        };

        println!("\nLoading model '{}'...", model_id_thread);
        let max_num_sequences: usize = 8;
        let (batch_model, weights_size_bytes) = match loader::load_batch_model(&model_dir, &model_id_thread, &device, max_context_len, max_num_sequences, &kv_budget) {
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
                // Try GGUF metadata for architecture
                if loader::is_gguf_model(&model_dir) {
                    "GGUF".to_string()
                } else {
                    "Unknown".to_string()
                }
            });

        // weights_size_bytes is the real in-memory footprint (post dtype-conversion).
        // On GPU: BF16 = 2 bytes/param → roughly half the on-disk F32 safetensors size.
        // On CPU: F32 = 4 bytes/param → matches or slightly exceeds on-disk size.
        let total_bytes = weights_size_bytes + kv_cache_bytes;
        println!(
            "Model '{}' loaded. vocab_size={}, max_seq_len={}, num_layers={}, size={:.2} GB (weights) + {:.2} GB (KV cache) = {:.2} GB total ({})",
            model_id_thread,
            vocab_size,
            max_seq_len,
            num_layers,
            weights_size_bytes as f64 / 1_073_741_824.0,
            kv_cache_bytes as f64 / 1_073_741_824.0,
            total_bytes as f64 / 1_073_741_824.0,
            if matches!(device, candle_core::Device::Cpu) { "F32/CPU" } else { "BF16/GPU" },
        );

        println!("Warming up model '{}'...", model_id_thread);
        let t_warmup = std::time::Instant::now();
        warm_up_model(batch_model.as_ref());
        println!("Model '{}' ready ({:.1}s warmup).", model_id_thread, t_warmup.elapsed().as_secs_f32());

        let (request_tx, request_rx) = tokio_mpsc::unbounded_channel();
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
            max_num_sequences: 8,
            max_tokens_per_step: 4096,
        };
        let extra_stop_ids = tokenizer.stop_token_ids();
        let engine = Engine::new_with_stop_tokens(batch_model, config, &extra_stop_ids);
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
                        println!(
                            "[memory] Post-load budget exceeded: {:.2} GB used / {:.2} GB budget \
                             — unloading '{}' ({:.2} GB)",
                            total as f64 / 1_073_741_824.0,
                            budget as f64 / 1_073_741_824.0,
                            model_id,
                            model_bytes as f64 / 1_073_741_824.0,
                        );
                        let (kv_bytes, shutdown) = match mgr.slots.get(&model_id) {
                            Some(SlotState::Ready { kv_cache_bytes, shutdown, .. }) => {
                                (*kv_cache_bytes, Some(Arc::clone(shutdown)))
                            }
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
                        eprintln!("{}", err);
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

                eprintln!("Failed to load model '{}': {}", model_id, err);
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
                eprintln!("{}", err);
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
