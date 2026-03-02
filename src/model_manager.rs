use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc as tokio_mpsc, oneshot};

use crate::engine::Engine;
use crate::model;
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
    },
}

pub struct RunningModelInfo {
    pub id: String,
    pub architecture: String,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub idle_seconds: u64,
    pub weights_size_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegistryEntry {
    pub architecture: String,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub size_bytes: usize,
    pub last_used_secs: u64,
}

pub struct ModelManager {
    models_dir: PathBuf,
    slots: HashMap<String, SlotState>,
    keep_alive: Duration,
    memory_budget_bytes: Option<usize>,
    registry: HashMap<String, RegistryEntry>,
    cuda_devices: Vec<usize>,
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
    ) -> Self {
        let registry = load_registry(&models_dir);
        Self {
            models_dir,
            slots: HashMap::new(),
            keep_alive,
            memory_budget_bytes,
            registry,
            cuda_devices,
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
                SlotState::Ready { weights_size_bytes, .. } => Some(*weights_size_bytes),
                _ => None,
            })
            .sum()
    }

    pub fn list_available(&self) -> Vec<model::DiscoveredModel> {
        model::discover_models(&self.models_dir)
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
                    ..
                } => Some(RunningModelInfo {
                    id: id.clone(),
                    architecture: architecture.clone(),
                    vocab_size: *vocab_size,
                    num_layers: *num_layers,
                    idle_seconds: now.duration_since(*last_used).as_secs(),
                    weights_size_bytes: *weights_size_bytes,
                }),
                _ => None,
            })
            .collect()
    }

    pub fn list_registry(&self) -> &HashMap<String, RegistryEntry> {
        &self.registry
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
                    println!("[memory pressure] Evicting LRU model '{}'", id);
                    self.slots.remove(&id);
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
        let model_path = self.models_dir.join(model_id);
        if !model_path.join("config.json").exists() {
            let (tx, rx) = oneshot::channel();
            let _ = tx.send(Err(format!("model '{}' not found in models directory", model_id)));
            return GetResult::Wait(rx);
        }

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

        let estimated_size = estimate_model_size(&model_path);
        self.evict_lru_for_bytes(estimated_size);

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
            println!("Evicting idle model '{}' (keep-alive expired)", id);
            self.slots.remove(id);
        }
    }

    pub fn update_registry(
        &mut self,
        model_id: &str,
        architecture: &str,
        vocab_size: usize,
        num_layers: usize,
        size_bytes: usize,
    ) {
        let entry = self.registry.entry(model_id.to_string()).or_default();
        entry.architecture = architecture.to_string();
        entry.vocab_size = vocab_size;
        entry.num_layers = num_layers;
        entry.size_bytes = size_bytes;
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
}

fn spawn_load(
    model_id: String,
    model_path: PathBuf,
    manager: SharedModelManager,
    effective_keep_alive: Duration,
    cuda_devices: Vec<usize>,
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
        let device = match model::select_device_at(cuda_idx) {
            Ok(d) => d,
            Err(e) => {
                let _ = result_tx.send(Err(format!("Failed to select device: {e}")));
                return;
            }
        };

        println!("\nLoading model '{}'...", model_id_thread);
        let batch_model = match model::load_batch_model(&model_dir, &device, 2) {
            Ok(m) => m,
            Err(e) => {
                let _ = result_tx.send(Err(format!("Failed to load model: {e}")));
                return;
            }
        };

        let max_seq_len = batch_model.max_seq_len();
        let vocab_size = batch_model.vocab_size();
        let num_layers = batch_model.num_layers();

        let config_path = model_path_thread.join("config.json");
        let architecture = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .and_then(|v| v["architectures"][0].as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "Unknown".to_string());

        let weights_size_bytes = estimate_model_size(&model_path_thread);

        println!(
            "Model '{}' loaded. vocab_size={}, max_seq_len={}, num_layers={}, size={:.2} GB",
            model_id_thread,
            vocab_size,
            max_seq_len,
            num_layers,
            weights_size_bytes as f64 / 1_073_741_824.0,
        );

        let (request_tx, request_rx) = tokio_mpsc::unbounded_channel();

        let result = LoadResult {
            request_tx,
            tokenizer: Arc::clone(&tokenizer),
            max_seq_len,
            architecture,
            vocab_size,
            num_layers,
            weights_size_bytes,
        };
        let _ = result_tx.send(Ok(result));

        let config = SchedulerConfig {
            max_num_sequences: 8,
            max_tokens_per_step: 4096,
        };
        let engine = Engine::new(batch_model, config);
        engine_loop(engine, tokenizer, request_rx);
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
                );

                mgr.slots.insert(
                    model_id,
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
                    },
                );
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
