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

use crate::engine::Engine;
use crate::model_manager::{self, GetResult, ModelManager, SharedModelManager};
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
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
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
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
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
}

pub struct IncomingRequest {
    pub prompt_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub max_tokens: usize,
    pub response_tx: tokio_mpsc::UnboundedSender<EngineEvent>,
}

pub enum EngineEvent {
    Token(String),
    Finish { finish_reason: String },
    StreamEnd,
    Error(String),
}

struct AppState {
    manager: SharedModelManager,
}

pub fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

pub fn engine_loop(
    mut engine: Engine,
    tokenizer: Arc<Tokenizer>,
    mut request_rx: tokio_mpsc::UnboundedReceiver<IncomingRequest>,
) {
    let mut response_channels: HashMap<SequenceId, tokio_mpsc::UnboundedSender<EngineEvent>> =
        HashMap::new();

    loop {
        if engine.has_pending_work() {
            while let Ok(req) = request_rx.try_recv() {
                let seq_id =
                    engine.add_request(req.prompt_tokens, req.sampling_params, req.max_tokens);
                response_channels.insert(seq_id, req.response_tx);
            }
        } else {
            match request_rx.blocking_recv() {
                Some(req) => {
                    let seq_id =
                        engine.add_request(req.prompt_tokens, req.sampling_params, req.max_tokens);
                    response_channels.insert(seq_id, req.response_tx);
                }
                None => break,
            }
        }

        if engine.has_pending_work() {
            match engine.step() {
                Ok(step) => {
                    for tok in &step.new_tokens {
                        if let Some(tx) = response_channels.get(&tok.seq_id) {
                            let text = tokenizer.decode(&[tok.token]).unwrap_or_default();
                            let _ = tx.send(EngineEvent::Token(text));
                        }
                    }
                    for completed in &step.completed {
                        if let Some(tx) = response_channels.remove(&completed.id) {
                            let _ = tx.send(EngineEvent::Finish {
                                finish_reason: "stop".to_string(),
                            });
                            let _ = tx.send(EngineEvent::StreamEnd);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Engine error: {e}");
                    for (_, tx) in response_channels.drain() {
                        let _ = tx.send(EngineEvent::Error(e.to_string()));
                        let _ = tx.send(EngineEvent::StreamEnd);
                    }
                    break;
                }
            }
        }
    }

    for (_, tx) in response_channels.drain() {
        let _ = tx.send(EngineEvent::Error("Model unloaded".to_string()));
        let _ = tx.send(EngineEvent::StreamEnd);
    }
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let models_dir = state.manager.lock().await.models_dir().clone();
    let discovered = crate::model::discover_models(&models_dir);

    let data: Vec<serde_json::Value> = discovered
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "object": "model",
                "architecture": m.architecture,
                "vocab_size": m.vocab_size,
                "num_layers": m.num_layers,
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
    drop(mgr);

    let data: Vec<serde_json::Value> = running
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "object": "model",
                "architecture": m.architecture,
                "vocab_size": m.vocab_size,
                "num_layers": m.num_layers,
                "idle_seconds": m.idle_seconds,
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": data
    }))
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

    let get_result = {
        let mut mgr = state.manager.lock().await;
        mgr.get_or_load(&model_id, Arc::clone(&state.manager))
    };

    let handle = match get_result {
        GetResult::Ready(h) => h,
        GetResult::Wait(rx) => {
            let load_result = rx.await.map_err(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": "Model loader dropped"}})),
                )
            })?;
            load_result.map_err(|e| {
                let status = if e.contains("not found") {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                (status, Json(serde_json::json!({"error": {"message": e}})))
            })?
        }
    };

    let prompt = format_chatml(&body.messages);
    let prompt_tokens = handle.tokenizer.encode(&prompt).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string()}})),
        )
    })?;
    let prompt_len = prompt_tokens.len();

    let sampling_params = SamplingParams {
        temperature: body.temperature.unwrap_or(0.7),
        top_k: body.top_k.unwrap_or(0),
        top_p: body.top_p.unwrap_or(1.0),
        min_p: body.min_p.unwrap_or(0.0),
        repetition_penalty: body.repetition_penalty.unwrap_or(1.0),
    };

    let max_tokens = body
        .max_tokens
        .unwrap_or_else(|| handle.max_seq_len.saturating_sub(prompt_len));

    let (response_tx, response_rx) = tokio_mpsc::unbounded_channel();

    handle
        .request_tx
        .send(IncomingRequest {
            prompt_tokens,
            sampling_params,
            max_tokens,
            response_tx,
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
                            },
                            finish_reason: None,
                        }],
                    };
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                }
                EngineEvent::Finish { finish_reason } => {
                    let chunk = ChatCompletionChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        model: model_id_clone.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: None,
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
        let mut finish_reason = "stop".to_string();
        let mut completion_tokens: usize = 0;

        while let Some(event) = rx.recv().await {
            match event {
                EngineEvent::Token(text) => {
                    content.push_str(&text);
                    completion_tokens += 1;
                }
                EngineEvent::Finish { finish_reason: fr } => {
                    finish_reason = fr;
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

        let response = ChatCompletionResponse {
            id: chat_id,
            object: "chat.completion".to_string(),
            model: model_id,
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens,
                total_tokens: prompt_len + completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

fn make_chat_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("chatcmpl-{:x}{:x}", t.as_secs(), t.subsec_nanos())
}

pub fn start_server(models_dir: PathBuf, port: u16, keep_alive: Duration) -> anyhow::Result<()> {
    if !models_dir.exists() {
        std::fs::create_dir_all(&models_dir)?;
        println!("Created models directory: {}", models_dir.display());
    }
    let available = crate::model::discover_models(&models_dir);
    println!("Models directory: {}", models_dir.display());
    println!("Discovered {} model(s):", available.len());
    for m in &available {
        println!("  - {} ({})", m.id, m.architecture);
    }

    let manager = Arc::new(tokio::sync::Mutex::new(ModelManager::new(
        models_dir,
        keep_alive,
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
        model_manager::spawn_eviction_task(manager);

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
            "\nKeep-alive: {}s (models evicted after idle timeout)\n",
            keep_alive.as_secs()
        );

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        Ok::<_, anyhow::Error>(())
    })?;

    Ok(())
}
