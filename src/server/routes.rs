use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc as tokio_mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::chat_template;
use crate::engine::Engine;
use crate::models::manager::{self, GetResult, ModelManager, SharedModelManager};
use crate::sampling::SamplingParams;
use crate::scheduler::sequence::SequenceId;
use crate::tokenizer::Tokenizer;

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub keep_alive: Option<u64>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct CompletionTokensDetails {
    reasoning_tokens: usize,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

pub struct IncomingRequest {
    pub prompt_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub max_tokens: usize,
    pub response_tx: tokio_mpsc::UnboundedSender<EngineEvent>,
    pub model_id: String,
    pub enqueued_at: std::time::Instant,
    pub enable_thinking: bool,
}

pub enum EngineEvent {
    Token(String),
    ReasoningToken(String),
    Finish { finish_reason: String, completion_tokens: usize },
    StreamEnd,
    Error(String),
}

struct AppState {
    manager: SharedModelManager,
}

pub fn apply_chat_template(tokenizer: &Tokenizer, messages: &[ChatMessage], enable_thinking: bool) -> String {
    let Some(template) = tokenizer.chat_template() else {
        return chat_template::format_plain_chat(messages);
    };

    let try_render = |msgs: &[ChatMessage]| {
        chat_template::apply_chat_template(
            template,
            msgs,
            tokenizer.bos_token(),
            tokenizer.eos_token(),
            true,
            enable_thinking,
        )
    };

    match try_render(messages) {
        Ok(prompt) => prompt,
        Err(e) => {
            let without_system: Vec<&ChatMessage> =
                messages.iter().filter(|m| m.role != "system").collect();

            if without_system.len() < messages.len() {
                let msgs_ref: Vec<ChatMessage> = without_system.into_iter().cloned().collect();
                if let Ok(prompt) = try_render(&msgs_ref) {
                    eprintln!("Warning: system role not supported by this model's template — retrying without system message.");
                    return prompt;
                }
            }

            eprintln!("Warning: chat template rendering failed: {e:#}. Falling back to plain text.");
            chat_template::format_plain_chat(messages)
        }
    }
}

struct SeqTracker {
    tx: tokio_mpsc::UnboundedSender<EngineEvent>,
    model_id: String,
    enqueued_at: std::time::Instant,
    first_token_at: Option<std::time::Instant>,
    token_count: usize,
    in_thinking: bool,
    output_ids: Vec<u32>,
    thinking_ids: Vec<u32>,
    decoded_len: usize,
    thinking_decoded_len: usize,
}

fn enqueue_request(
    req: IncomingRequest,
    engine: &mut Engine,
    trackers: &mut HashMap<SequenceId, SeqTracker>,
) {
    let model_id = req.model_id.clone();
    let enqueued_at = req.enqueued_at;
    let seq_id = engine.add_request(req.prompt_tokens, req.sampling_params, req.max_tokens);
    eprintln!("[req] {} seq={} enqueued", model_id, seq_id);
    trackers.insert(seq_id, SeqTracker {
        tx: req.response_tx,
        model_id,
        enqueued_at,
        first_token_at: None,
        token_count: 0,
        in_thinking: req.enable_thinking,
        output_ids: Vec::new(),
        thinking_ids: Vec::new(),
        decoded_len: 0,
        thinking_decoded_len: 0,
    });
}

fn prefix_decode_token(
    tokenizer: &Tokenizer,
    all_ids: &[u32],
    decoded_len: &mut usize,
    token: u32,
) -> Option<String> {
    let single = tokenizer.decode(&[token]).unwrap_or_default();
    if !single.is_empty() && !single.contains('\u{FFFD}') {
        *decoded_len += single.len();
        return Some(single);
    }
    let full = tokenizer.decode(all_ids).unwrap_or_default();
    let new_text = &full[*decoded_len..];
    let emit = new_text.trim_end_matches('\u{FFFD}');
    if !emit.is_empty() {
        *decoded_len += emit.len();
        Some(emit.to_string())
    } else {
        None
    }
}

pub fn engine_loop(
    mut engine: Engine,
    tokenizer: Arc<Tokenizer>,
    mut request_rx: tokio_mpsc::UnboundedReceiver<IncomingRequest>,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let mut trackers: HashMap<SequenceId, SeqTracker> = HashMap::new();

    let think_start_id = tokenizer.special_token_id("<think>");
    let think_end_id = tokenizer.special_token_id("</think>");

    loop {
        if shutdown.load(std::sync::atomic::Ordering::Acquire) {
            break;
        }

        if engine.has_pending_work() {
            while let Ok(req) = request_rx.try_recv() {
                enqueue_request(req, &mut engine, &mut trackers);
            }
        } else {
            match request_rx.blocking_recv() {
                Some(req) => enqueue_request(req, &mut engine, &mut trackers),
                None => break,
            }
        }

        if engine.has_pending_work() {
            // Acquire the global GPU lock so that only one model
            // does a forward pass at a time (prevents Metal/CUDA contention
            // when multiple models are loaded simultaneously).
            let step_result = {
                let lock = crate::gpu_lock::gpu_lock();
                let _gpu = lock.acquire();
                engine.step()
            };
            match step_result {
                Ok(step) => {
                    for tok in &step.new_tokens {
                        if let Some(tracker) = trackers.get_mut(&tok.seq_id) {
                            if tracker.first_token_at.is_none() {
                                let ttft_ms = tracker.enqueued_at.elapsed().as_secs_f64() * 1000.0;
                                eprintln!("[timing] {} seq={} TTFT: {:.1}ms", tracker.model_id, tok.seq_id, ttft_ms);
                                tracker.first_token_at = Some(std::time::Instant::now());
                            }
                            tracker.token_count += 1;

                            // Detect </think> token to transition from reasoning to content.
                            if tracker.in_thinking {
                                let raw = tokenizer
                                    .decode_with_special(&[tok.token])
                                    .unwrap_or_default();

                                let is_think_start = think_start_id == Some(tok.token)
                                    || raw.contains("<think>");
                                if is_think_start {
                                    continue;
                                }

                                let is_think_end = think_end_id == Some(tok.token)
                                    || raw.contains("</think>");
                                if is_think_end {
                                    tracker.in_thinking = false;
                                    continue;
                                }
                            }

                            if tracker.in_thinking {
                                tracker.thinking_ids.push(tok.token);
                                if let Some(text) = prefix_decode_token(
                                    &tokenizer, &tracker.thinking_ids, &mut tracker.thinking_decoded_len, tok.token,
                                ) {
                                    let _ = tracker.tx.send(EngineEvent::ReasoningToken(text));
                                }
                            } else {
                                tracker.output_ids.push(tok.token);
                                if let Some(text) = prefix_decode_token(
                                    &tokenizer, &tracker.output_ids, &mut tracker.decoded_len, tok.token,
                                ) {
                                    let _ = tracker.tx.send(EngineEvent::Token(text));
                                }
                            }
                        }
                    }
                    for completed in &step.completed {
                        if let Some(mut tracker) = trackers.remove(&completed.id) {
                            if !tracker.output_ids.is_empty() {
                                let full = tokenizer.decode(&tracker.output_ids).unwrap_or_default();
                                if tracker.decoded_len < full.len() {
                                    let rest = &full[tracker.decoded_len..];
                                    if !rest.is_empty() {
                                        tracker.decoded_len = full.len();
                                        let _ = tracker.tx.send(EngineEvent::Token(rest.to_string()));
                                    }
                                }
                            }
                            if !tracker.thinking_ids.is_empty() {
                                let full = tokenizer.decode(&tracker.thinking_ids).unwrap_or_default();
                                if tracker.thinking_decoded_len < full.len() {
                                    let rest = &full[tracker.thinking_decoded_len..];
                                    if !rest.is_empty() {
                                        let _ = tracker.tx.send(EngineEvent::ReasoningToken(rest.to_string()));
                                    }
                                }
                            }
                            let total_ms = tracker.enqueued_at.elapsed().as_secs_f64() * 1000.0;
                            let decode_s = tracker.first_token_at
                                .map(|t| t.elapsed().as_secs_f64())
                                .unwrap_or(0.001);
                            let tps = tracker.token_count as f64 / decode_s.max(0.001);
                            eprintln!(
                                "[timing] {} seq={} done: {} tokens, total={:.1}ms, decode={:.1}ms ({:.1} tok/s)",
                                tracker.model_id, completed.id,
                                tracker.token_count,
                                total_ms,
                                decode_s * 1000.0,
                                tps,
                            );
                            let _ = tracker.tx.send(EngineEvent::Finish {
                                finish_reason: completed.finish_reason
                                    .as_deref()
                                    .unwrap_or("stop")
                                    .to_string(),
                                completion_tokens: tracker.token_count,
                            });
                            let _ = tracker.tx.send(EngineEvent::StreamEnd);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Engine error: {e}");
                    let aborted_ids = engine.abort_running();
                    for id in aborted_ids {
                        if let Some(tracker) = trackers.remove(&id) {
                            let _ = tracker.tx.send(EngineEvent::Error(e.to_string()));
                            let _ = tracker.tx.send(EngineEvent::StreamEnd);
                        }
                    }
                    // Do not break, the engine remains alive for subsequent requests.
                }
            }
            std::thread::yield_now();
        }
    }

    for (_, tracker) in trackers.drain() {
        let _ = tracker.tx.send(EngineEvent::Error("Model unloaded".to_string()));
        let _ = tracker.tx.send(EngineEvent::StreamEnd);
    }
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mgr = state.manager.lock().await;
    let models_dir = mgr.models_dir().clone();
    let registry = mgr.list_registry().clone();
    drop(mgr);

    let discovered = crate::models::loader::discover_models(&models_dir);

    let data: Vec<serde_json::Value> = discovered
        .iter()
        .map(|m| {
            let size_bytes = registry
                .get(&m.id)
                .map(|e| e.size_bytes)
                .unwrap_or(0);
            let last_used_secs = registry
                .get(&m.id)
                .map(|e| e.last_used_secs)
                .unwrap_or(0);
            serde_json::json!({
                "id": m.id,
                "object": "model",
                "architecture": m.architecture,
                "vocab_size": m.vocab_size,
                "num_layers": m.num_layers,
                "size_bytes": size_bytes,
                "size_gb": (size_bytes as f64 / 1_073_741_824.0 * 100.0).round() / 100.0,
                "last_used_secs": last_used_secs,
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": data
    }))
}

async fn list_running_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mgr = state.manager.lock().await;
    let running = mgr.list_running();
    let budget_bytes = mgr.memory_budget_bytes();
    let total_loaded = mgr.total_loaded_bytes();
    drop(mgr);

    let data: Vec<serde_json::Value> = running
        .iter()
        .map(|m| {
            {
                let total = m.weights_size_bytes + m.kv_cache_bytes;
                serde_json::json!({
                    "id": m.id,
                    "object": "model",
                    "architecture": m.architecture,
                    "vocab_size": m.vocab_size,
                    "num_layers": m.num_layers,
                    "idle_seconds": m.idle_seconds,
                    "weights_size_bytes": m.weights_size_bytes,
                    "kv_cache_bytes": m.kv_cache_bytes,
                    "total_size_bytes": total,
                    "total_size_gb": (total as f64 / 1_073_741_824.0 * 100.0).round() / 100.0,
                })
            }
        })
        .collect();

    let mut resp = serde_json::json!({
        "object": "list",
        "data": data,
        "total_loaded_bytes": total_loaded,
        "total_loaded_gb": (total_loaded as f64 / 1_073_741_824.0 * 100.0).round() / 100.0,
    });

    if let Some(budget) = budget_bytes {
        resp["memory_budget_bytes"] = budget.into();
        resp["memory_budget_gb"] = ((budget as f64 / 1_073_741_824.0 * 100.0).round() / 100.0).into();
        resp["memory_free_bytes"] = budget.saturating_sub(total_loaded).into();
    }

    Json(resp)
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<serde_json::Value>)> {
    if body.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": {"message": "messages must not be empty"}})),
        ));
    }

    let model_id = body.model.as_deref().unwrap_or("").to_string();
    if model_id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": {"message": "model field is required"}})),
        ));
    }

    let t_request = std::time::Instant::now();

    let get_result = {
        let mut mgr = state.manager.lock().await;
        let keep_alive_override = body.keep_alive.map(Duration::from_secs);
        mgr.get_or_load(&model_id, Arc::clone(&state.manager), keep_alive_override)
    };

    let t_after_lock = t_request.elapsed();

    let handle = match get_result {
        GetResult::Ready(h) => {
            eprintln!("[timing] {} manager lock+ready: {:.1}ms", model_id, t_after_lock.as_secs_f64() * 1000.0);
            h
        }
        GetResult::Wait(rx) => {
            let load_result = rx.await.map_err(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": "Model loader dropped"}})),
                )
            })?;
            let h = load_result.map_err(|e| {
                let status = if e.contains("not found") {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                (status, Json(serde_json::json!({"error": {"message": e}})))
            })?;
            eprintln!("[timing] {} load completed: {:.1}ms", model_id, t_request.elapsed().as_secs_f64() * 1000.0);
            h
        }
    };

    let t_template = std::time::Instant::now();
    let enable_thinking = body.enable_thinking.unwrap_or(false)
        && handle.tokenizer.has_thinking_support();
    let prompt = apply_chat_template(&handle.tokenizer, &body.messages, enable_thinking);
    let template_ms = t_template.elapsed().as_secs_f64() * 1000.0;

    let t_encode = std::time::Instant::now();
    let prompt_tokens = handle.tokenizer.encode(&prompt).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string()}})),
        )
    })?;
    let encode_ms = t_encode.elapsed().as_secs_f64() * 1000.0;
    let prompt_len = prompt_tokens.len();

    eprintln!(
        "[timing] {} template: {:.1}ms, encode: {:.1}ms ({} tokens), total pre-engine: {:.1}ms",
        model_id, template_ms, encode_ms, prompt_len, t_request.elapsed().as_secs_f64() * 1000.0
    );

    let sampling_params = SamplingParams {
        temperature: body.temperature.unwrap_or(0.7),
        top_k: body.top_k.unwrap_or(0),
        top_p: body.top_p.unwrap_or(1.0),
        min_p: body.min_p.unwrap_or(0.0),
        repetition_penalty: body.repetition_penalty.unwrap_or(1.0),
        seed: body.seed,
    };

    let remaining = handle.max_seq_len.saturating_sub(prompt_len);
    let max_tokens = body
        .max_tokens
        .unwrap_or(remaining)
        .min(remaining);

    let (response_tx, response_rx) = tokio_mpsc::unbounded_channel();

    handle
        .request_tx
        .send(IncomingRequest {
            prompt_tokens,
            sampling_params,
            max_tokens,
            response_tx,
            model_id: model_id.clone(),
            enqueued_at: std::time::Instant::now(),
            enable_thinking,
        })
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": {"message": "Engine unavailable"}})),
            )
        })?;

    let chat_id = make_chat_id();
    let stream = body.stream.unwrap_or(false);

    if stream {
        let model_id_clone = model_id.clone();

        let stream = UnboundedReceiverStream::new(response_rx).map(move |event| {
            let sse_event = match event {
                EngineEvent::Token(text) => {
                    let chunk = ChatCompletionChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        model: model_id_clone.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(text),
                                reasoning_content: None,
                            },
                            finish_reason: None,
                        }],
                    };
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                }
                EngineEvent::ReasoningToken(text) => {
                    let chunk = ChatCompletionChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        model: model_id_clone.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: None,
                                reasoning_content: Some(text),
                            },
                            finish_reason: None,
                        }],
                    };
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                }
                EngineEvent::Finish { finish_reason, .. } => {
                    let chunk = ChatCompletionChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        model: model_id_clone.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: None,
                                reasoning_content: None,
                            },
                            finish_reason: Some(finish_reason),
                        }],
                    };
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                }
                EngineEvent::StreamEnd => Event::default().data("[DONE]"),
                EngineEvent::Error(msg) => {
                    Event::default().data(format!(r#"{{"error":"{}"}}"#, msg))
                }
            };
            Ok::<_, std::convert::Infallible>(sse_event)
        });

        Ok(Sse::new(stream).into_response())
    } else {
        let mut rx = response_rx;
        let mut content = String::new();
        let mut reasoning_content = String::new();
        let mut finish_reason = "stop".to_string();
        let mut completion_tokens: usize = 0;
        let mut reasoning_tokens: usize = 0;

        while let Some(event) = rx.recv().await {
            match event {
                EngineEvent::Token(text) => {
                    content.push_str(&text);
                }
                EngineEvent::ReasoningToken(text) => {
                    reasoning_content.push_str(&text);
                    reasoning_tokens += 1;
                }
                EngineEvent::Finish { finish_reason: fr, completion_tokens: ct } => {
                    finish_reason = fr;
                    completion_tokens = ct;
                }
                EngineEvent::StreamEnd => break,
                EngineEvent::Error(msg) => {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": msg}})),
                    ));
                }
            }
        }

        let reasoning_opt = if reasoning_content.is_empty() {
            None
        } else {
            Some(reasoning_content.trim().to_string())
        };

        let completion_tokens_details = if reasoning_tokens > 0 {
            Some(CompletionTokensDetails { reasoning_tokens })
        } else {
            None
        };

        let response = ChatCompletionResponse {
            id: chat_id,
            object: "chat.completion".to_string(),
            model: model_id,
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: content.trim().to_string(),
                    reasoning_content: reasoning_opt,
                },
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens,
                total_tokens: prompt_len + completion_tokens,
                completion_tokens_details,
            },
        };

        Ok(Json(response).into_response())
    }
}

fn make_chat_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("chatcmpl-{:x}{:x}-{}", t.as_secs(), t.subsec_nanos(), seq)
}

pub fn start_server(
    models_dir: PathBuf,
    port: u16,
    keep_alive: Duration,
    memory_budget_bytes: Option<usize>,
    cuda_devices: Vec<usize>,
    max_context_len: usize,
) -> anyhow::Result<()> {
    if !models_dir.exists() {
        std::fs::create_dir_all(&models_dir)?;
        println!("Created models directory: {}", models_dir.display());
    }
    let available = crate::models::loader::discover_models(&models_dir);
    println!("Models directory: {}", models_dir.display());
    println!("Discovered {} {}:", available.len(), if available.len() == 1 { "model" } else { "models" });
    for m in &available {
        println!("  - {} ({})", m.id, m.architecture);
    }

    let manager = Arc::new(tokio::sync::Mutex::new(ModelManager::new(
        models_dir,
        keep_alive,
        memory_budget_bytes,
        cuda_devices,
        max_context_len,
    )));

    let state = Arc::new(AppState {
        manager: Arc::clone(&manager),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/models/running", get(list_running_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        manager::spawn_eviction_task(manager);

        let addr = format!("0.0.0.0:{}", port);
        println!("\nServer listening on http://{}", addr);
        println!(
            "API endpoint:   POST http://localhost:{}/v1/chat/completions",
            port
        );
        println!(
            "Models:         GET  http://localhost:{}/v1/models",
            port
        );
        println!(
            "Running models: GET  http://localhost:{}/v1/models/running",
            port
        );
        println!(
            "Health check:   GET  http://localhost:{}/health",
            port
        );
        println!(
            "\nKeep-alive: {}s (models evicted after idle timeout)",
            keep_alive.as_secs()
        );
        match memory_budget_bytes {
            Some(b) => println!(
                "Memory budget: {:.1} GB (LRU eviction when exceeded)",
                b as f64 / 1_073_741_824.0
            ),
            None => println!("Memory budget: unlimited"),
        }
        println!("Max context length: {} tokens per sequence\n", max_context_len);

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        Ok::<_, anyhow::Error>(())
    })?;

    Ok(())
}
