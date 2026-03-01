use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    },
}

pub struct RunningModelInfo {
    pub id: String,
    pub architecture: String,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub idle_seconds: u64,
}

pub struct ModelManager {
    models_dir: PathBuf,
    slots: HashMap<String, SlotState>,
    keep_alive: Duration,
}

pub type SharedModelManager = Arc<tokio::sync::Mutex<ModelManager>>;

pub enum GetResult {
    Ready(ReadyHandle),
    Wait(oneshot::Receiver<Result<ReadyHandle, String>>),
}

impl ModelManager {
    pub fn new(models_dir: PathBuf, keep_alive: Duration) -> Self {
        Self {
            models_dir,
            slots: HashMap::new(),
            keep_alive,
        }
    }

    pub fn models_dir(&self) -> &PathBuf {
        &self.models_dir
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
                    ..
                } => Some(RunningModelInfo {
                    id: id.clone(),
                    architecture: architecture.clone(),
                    vocab_size: *vocab_size,
                    num_layers: *num_layers,
                    idle_seconds: now.duration_since(*last_used).as_secs(),
                }),
                _ => None,
            })
            .collect()
    }

    pub fn get_or_load(
        &mut self,
        model_id: &str,
        manager_handle: SharedModelManager,
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
                    ..
                } => {
                    *last_used = Instant::now();
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
        );

        GetResult::Wait(rx)
    }

    pub fn evict_expired(&mut self) {
        let now = Instant::now();
        let keep_alive = self.keep_alive;
        let expired: Vec<String> = self
            .slots
            .iter()
            .filter_map(|(id, slot)| match slot {
                SlotState::Ready { last_used, .. }
                    if now.duration_since(*last_used) > keep_alive =>
                {
                    Some(id.clone())
                }
                _ => None,
            })
            .collect();

        for id in &expired {
            println!("Evicting idle model: {}", id);
            self.slots.remove(id);
        }
    }
}

struct LoadResult {
    request_tx: tokio_mpsc::UnboundedSender<IncomingRequest>,
    tokenizer: Arc<Tokenizer>,
    max_seq_len: usize,
    architecture: String,
    vocab_size: usize,
    num_layers: usize,
}

fn spawn_load(
    model_id: String,
    model_path: PathBuf,
    manager: SharedModelManager,
) {
    let (result_tx, result_rx) = oneshot::channel::<Result<LoadResult, String>>();

    let model_id_thread = model_id.clone();
    std::thread::spawn(move || {
        let model_dir = model_path.to_string_lossy().to_string();

        let tokenizer = match Tokenizer::from_dir(&model_dir) {
            Ok(t) => Arc::new(t),
            Err(e) => {
                let _ = result_tx.send(Err(format!("Failed to load tokenizer: {e}")));
                return;
            }
        };

        let device = match model::select_device() {
            Ok(d) => d,
            Err(e) => {
                let _ = result_tx.send(Err(format!("Failed to select device: {e}")));
                return;
            }
        };

        println!("Loading model '{}'...", model_id_thread);
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

        let config_path = model_path.join("config.json");
        let architecture = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .and_then(|v| v["architectures"][0].as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "Unknown".to_string());

        println!(
            "Model '{}' loaded. vocab_size={}, max_seq_len={}, num_layers={}",
            model_id_thread, vocab_size, max_seq_len, num_layers
        );

        let (request_tx, request_rx) = tokio_mpsc::unbounded_channel();

        let result = LoadResult {
            request_tx,
            tokenizer: Arc::clone(&tokenizer),
            max_seq_len,
            architecture,
            vocab_size,
            num_layers,
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
