use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokio::sync::mpsc as tokio_mpsc;
use tower::util::ServiceExt;

use super::AppState;
use super::build_router;
use super::types::{EngineEvent, IncomingRequest};
use crate::common::kv_quant::KvQuantMode;
use crate::models::manager::{ModelManager, ModelManagerConfig, ReadyHandle};
use crate::tokenizer::Tokenizer;

#[derive(Clone)]
struct ScriptedReply {
    content: String,
    finish_reason: String,
}

fn create_test_model_dir(root: &Path, model_id: &str) -> anyhow::Result<()> {
    let model_dir = root.join(model_id);
    std::fs::create_dir_all(&model_dir)?;
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string_pretty(&json!({
            "architectures": ["TestForCausalLM"],
            "vocab_size": 32,
            "num_hidden_layers": 1
        }))?,
    )?;

    let model = WordLevel::builder()
        .vocab(
            [
                ("[UNK]".to_string(), 0u32),
                ("System:".to_string(), 1u32),
                ("User:".to_string(), 2u32),
                ("Assistant:".to_string(), 3u32),
                ("Tool".to_string(), 4u32),
                ("result".to_string(), 5u32),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".to_string())
        .build()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer
        .save(model_dir.join("tokenizer.json"), false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    std::fs::write(model_dir.join("tokenizer_config.json"), "{}")?;
    Ok(())
}

fn spawn_scripted_engine(replies: Vec<ScriptedReply>) -> tokio_mpsc::Sender<IncomingRequest> {
    let (request_tx, mut request_rx) = tokio_mpsc::channel::<IncomingRequest>(64);
    let scripted = Arc::new(Mutex::new(VecDeque::from(replies)));
    tokio::spawn(async move {
        while let Some(req) = request_rx.recv().await {
            let reply = scripted
                .lock()
                .expect("scripted replies mutex poisoned")
                .pop_front()
                .expect("missing scripted reply for request");
            let _ = req.response_tx.send(EngineEvent::Token {
                text: reply.content,
                logprob_entries: vec![],
            });
            let _ = req.response_tx.send(EngineEvent::Finish {
                finish_reason: reply.finish_reason,
                completion_tokens: 1,
            });
            let _ = req.response_tx.send(EngineEvent::StreamEnd);
        }
    });
    request_tx
}

// Holds onto each IncomingRequest so response_tx isn't dropped (would close the channel early).
fn spawn_stuck_engine() -> tokio_mpsc::Sender<IncomingRequest> {
    let (request_tx, mut request_rx) = tokio_mpsc::channel::<IncomingRequest>(64);
    tokio::spawn(async move {
        let mut held: Vec<IncomingRequest> = Vec::new();
        while let Some(req) = request_rx.recv().await {
            held.push(req);
        }
    });
    request_tx
}

fn build_test_app(replies: Vec<ScriptedReply>) -> anyhow::Result<(Router, TempDir)> {
    let tmp = tempfile::tempdir()?;
    let model_id = "test-model";
    create_test_model_dir(tmp.path(), model_id)?;

    let tokenizer = Arc::new(Tokenizer::from_dir(
        tmp.path().join(model_id).to_str().expect("utf-8 path"),
    )?);
    let request_tx = spawn_scripted_engine(replies);

    let mut manager = ModelManager::new(ModelManagerConfig {
        models_dir: tmp.path().to_path_buf(),
        keep_alive: Duration::from_secs(60),
        memory_budget_bytes: None,
        cuda_devices: vec![],
        max_context_len: 4096,
        kv_quant: KvQuantMode::Off,
        qjl_quantization: false,
        require_gpu: false,
        max_num_seqs: None,
        max_queued_requests: 200,
        draft_model: None,
        expert_stream_mb: None,
    });
    manager.insert_ready_for_tests(
        model_id,
        ReadyHandle {
            request_tx,
            tokenizer,
            max_seq_len: 4096,
        },
    );

    let state = Arc::new(AppState {
        manager: Arc::new(tokio::sync::Mutex::new(manager)),
        api_key: None,
        request_timeout: None,
    });
    Ok((build_router(state), tmp))
}

fn build_test_app_with_api_key(
    replies: Vec<ScriptedReply>,
    api_key: &str,
) -> anyhow::Result<(Router, TempDir)> {
    let tmp = tempfile::tempdir()?;
    let model_id = "test-model";
    create_test_model_dir(tmp.path(), model_id)?;

    let tokenizer = Arc::new(Tokenizer::from_dir(
        tmp.path().join(model_id).to_str().expect("utf-8 path"),
    )?);
    let request_tx = spawn_scripted_engine(replies);

    let mut manager = ModelManager::new(ModelManagerConfig {
        models_dir: tmp.path().to_path_buf(),
        keep_alive: Duration::from_secs(60),
        memory_budget_bytes: None,
        cuda_devices: vec![],
        max_context_len: 4096,
        kv_quant: KvQuantMode::Off,
        qjl_quantization: false,
        require_gpu: false,
        max_num_seqs: None,
        max_queued_requests: 200,
        draft_model: None,
        expert_stream_mb: None,
    });
    manager.insert_ready_for_tests(
        model_id,
        ReadyHandle {
            request_tx,
            tokenizer,
            max_seq_len: 4096,
        },
    );

    let state = Arc::new(AppState {
        manager: Arc::new(tokio::sync::Mutex::new(manager)),
        api_key: Some(api_key.to_string()),
        request_timeout: None,
    });
    Ok((build_router(state), tmp))
}

fn build_test_app_with_timeout(
    request_tx: tokio_mpsc::Sender<IncomingRequest>,
    timeout: Duration,
) -> anyhow::Result<(Router, TempDir)> {
    let tmp = tempfile::tempdir()?;
    let model_id = "test-model";
    create_test_model_dir(tmp.path(), model_id)?;

    let tokenizer = Arc::new(Tokenizer::from_dir(
        tmp.path().join(model_id).to_str().expect("utf-8 path"),
    )?);

    let mut manager = ModelManager::new(ModelManagerConfig {
        models_dir: tmp.path().to_path_buf(),
        keep_alive: Duration::from_secs(60),
        memory_budget_bytes: None,
        cuda_devices: vec![],
        max_context_len: 4096,
        kv_quant: KvQuantMode::Off,
        qjl_quantization: false,
        require_gpu: false,
        max_num_seqs: None,
        max_queued_requests: 200,
        draft_model: None,
        expert_stream_mb: None,
    });
    manager.insert_ready_for_tests(
        model_id,
        ReadyHandle {
            request_tx,
            tokenizer,
            max_seq_len: 4096,
        },
    );

    let state = Arc::new(AppState {
        manager: Arc::new(tokio::sync::Mutex::new(manager)),
        api_key: None,
        request_timeout: Some(timeout),
    });
    Ok((build_router(state), tmp))
}

fn base_tools() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather lookup",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"],
                    "additionalProperties": false
                },
                "strict": true
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search docs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"],
                    "additionalProperties": false
                },
                "strict": true
            }
        }
    ])
}

async fn post_chat(app: &Router, body: Value) -> axum::response::Response {
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("request failed")
}

async fn post_chat_with_headers(
    app: &Router,
    body: Value,
    headers: &[(&str, &str)],
) -> axum::response::Response {
    let mut req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    app.clone()
        .oneshot(req.body(Body::from(body.to_string())).expect("request"))
        .await
        .expect("request failed")
}

async fn get_request_with_headers(
    app: &Router,
    uri: &str,
    headers: &[(&str, &str)],
) -> axum::response::Response {
    let mut req = Request::builder().method("GET").uri(uri);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    app.clone()
        .oneshot(req.body(Body::empty()).expect("request"))
        .await
        .expect("request failed")
}

async fn get_request(app: &Router, uri: &str) -> axum::response::Response {
    app.clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("request failed")
}

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_text() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    let response = get_request(&app, "/metrics").await;
    assert_eq!(response.status(), StatusCode::OK);

    let headers = response.headers().clone();
    let ct = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("text/plain"),
        "expected text/plain content-type, got: {ct}"
    );

    let body = String::from_utf8(
        to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body")
            .to_vec(),
    )
    .expect("utf-8");

    // Labeled Vecs only appear after at least one observation; check unconditional gauges.
    for metric in &[
        "oxydllm_queue_depth",
        "oxydllm_vram_used_bytes",
        "oxydllm_model_weights_bytes",
        "oxydllm_kv_cache_allocated_bytes",
    ] {
        assert!(
            body.contains(metric),
            "expected metric '{metric}' in /metrics output"
        );
    }
}

#[tokio::test]
async fn metrics_endpoint_shows_test_model_label() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    let response = get_request(&app, "/metrics").await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = String::from_utf8(
        to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body")
            .to_vec(),
    )
    .expect("utf-8");

    assert!(
        body.contains("test-model"),
        "expected test-model label in /metrics output"
    );
}

#[tokio::test]
async fn health_returns_ok() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    let response = get_request(&app, "/health").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body"),
    )
    .expect("json");
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn list_models_returns_list_object() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    let response = get_request(&app, "/v1/models").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body"),
    )
    .expect("json");
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("data array");
    assert!(
        data.iter().any(|m| m["id"] == "test-model"),
        "test-model not in list"
    );
}

#[tokio::test]
async fn get_model_found_returns_model_object() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    let response = get_request(&app, "/v1/models/test-model").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body"),
    )
    .expect("json");
    assert_eq!(body["id"], "test-model");
    assert_eq!(body["object"], "model");
    assert_eq!(body["owned_by"], "local");
}

#[tokio::test]
async fn get_model_not_found_returns_404() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    let response = get_request(&app, "/v1/models/nonexistent").await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body"),
    )
    .expect("json");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("nonexistent")
    );
}

#[tokio::test]
async fn empty_messages_returns_400() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    let response = post_chat(&app, json!({"model": "test-model", "messages": []})).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body"),
    )
    .expect("json");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("messages")
    );
}

// Contract: reasoning_effort accepts only low|medium|high and is rejected
// before any model is resolved or loaded.
#[tokio::test]
async fn invalid_reasoning_effort_returns_400() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    let response = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "maximum"
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body"),
    )
    .expect("json");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("reasoning_effort")
    );
}

#[tokio::test]
async fn missing_model_returns_400() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    let response = post_chat(
        &app,
        json!({"messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body"),
    )
    .expect("json");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("model")
    );
}

#[tokio::test]
async fn n_zero_returns_400() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    let response = post_chat(
        &app,
        json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "n": 0}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body"),
    )
    .expect("json");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("n")
    );
}

#[tokio::test]
async fn invalid_json_body_returns_422() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("not json"))
                .expect("request"),
        )
        .await
        .expect("request failed");
    assert!(
        matches!(
            response.status(),
            StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY
        ),
        "expected 4xx for invalid JSON, got {}",
        response.status()
    );
}

#[tokio::test]
async fn non_streaming_chat_returns_full_response() {
    let (app, _tmp) = build_test_app(vec![ScriptedReply {
        content: "Hello world".to_string(),
        finish_reason: "stop".to_string(),
    }])
    .expect("test app");

    let response = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body"),
    )
    .expect("json");

    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert!(body["choices"][0]["message"]["tool_calls"].is_null());
    assert!(body["usage"].is_object());
}

#[tokio::test]
async fn streaming_chat_emits_content_chunks_and_done() {
    let (app, _tmp) = build_test_app(vec![ScriptedReply {
        content: "Hi there".to_string(),
        finish_reason: "stop".to_string(),
    }])
    .expect("test app");

    let response = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = String::from_utf8(
        to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body")
            .to_vec(),
    )
    .expect("utf-8");

    let mut role_seen = false;
    let mut content = String::new();
    let mut finish_reason = None;
    let mut done_seen = false;

    for line in body.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            done_seen = true;
            continue;
        }
        let chunk: Value = serde_json::from_str(payload).expect("valid SSE chunk");
        assert_eq!(chunk["object"], "chat.completion.chunk");
        let delta = &chunk["choices"][0]["delta"];
        if delta.get("role").and_then(|v| v.as_str()) == Some("assistant") {
            role_seen = true;
        }
        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
            content.push_str(c);
        }
        if let Some(r) = chunk["choices"][0]
            .get("finish_reason")
            .and_then(|v| v.as_str())
        {
            finish_reason = Some(r.to_string());
        }
    }

    assert!(role_seen, "expected assistant role chunk");
    assert_eq!(content, "Hi there");
    assert_eq!(finish_reason.as_deref(), Some("stop"));
    assert!(done_seen, "expected [DONE] marker");
}

#[tokio::test]
async fn forced_function_choice_wraps_direct_json_into_tool_call() {
    let (app, _tmp) = build_test_app(vec![ScriptedReply {
        content: r#"{"location":"Paris"}"#.to_string(),
        finish_reason: "stop".to_string(),
    }])
    .expect("test app");

    let response = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": base_tools(),
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"}
            }
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("json body"),
    )
    .expect("json response");

    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert!(body["choices"][0]["message"]["content"].is_null());
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        r#"{"location":"Paris"}"#
    );
}

#[tokio::test]
async fn allowed_tools_restricts_returned_tool_calls() {
    let (app, _tmp) = build_test_app(vec![ScriptedReply {
        content: r#"{"tool_calls":[{"name":"get_weather","arguments":{"location":"Paris"}},{"name":"search_docs","arguments":{"query":"weather policy"}}]}"#.to_string(),
        finish_reason: "stop".to_string(),
    }])
    .expect("test app");

    let response = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "search"}],
            "tools": base_tools(),
            "tool_choice": {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "required",
                    "tools": [
                        {"type": "function", "function": {"name": "search_docs"}}
                    ]
                }
            }
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("json body"),
    )
    .expect("json response");

    let tool_calls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls array");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0]["function"]["name"], "search_docs");
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn parallel_tool_calls_false_returns_at_most_one_tool_call() {
    let (app, _tmp) = build_test_app(vec![ScriptedReply {
        content: r#"{"tool_calls":[{"name":"get_weather","arguments":{"location":"Paris"}},{"name":"search_docs","arguments":{"query":"forecast"}}]}"#.to_string(),
        finish_reason: "stop".to_string(),
    }])
    .expect("test app");

    let response = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "pick one"}],
            "tools": base_tools(),
            "tool_choice": "auto",
            "parallel_tool_calls": false
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("json body"),
    )
    .expect("json response");

    let tool_calls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls array");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0]["function"]["name"], "get_weather");
}

#[tokio::test]
async fn streaming_tool_calls_emit_incremental_sse_deltas() {
    let (app, _tmp) = build_test_app(vec![ScriptedReply {
        content: r#"{"tool_calls":[{"name":"get_weather","arguments":{"location":"Paris"}}]}"#
            .to_string(),
        finish_reason: "stop".to_string(),
    }])
    .expect("test app");

    let response = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "stream a tool call"}],
            "tools": base_tools(),
            "stream": true
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = String::from_utf8(
        to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("stream body")
            .to_vec(),
    )
    .expect("utf-8 stream body");

    let mut role_seen = false;
    let mut finish_reason = None;
    let mut aggregated_arguments = String::new();
    let mut header_seen = false;
    let mut done_seen = false;

    for line in body.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            done_seen = true;
            continue;
        }
        let chunk: Value = serde_json::from_str(payload).expect("valid SSE chunk JSON");
        let choice = &chunk["choices"][0];
        let delta = &choice["delta"];
        if delta.get("role") == Some(&Value::String("assistant".to_string())) {
            role_seen = true;
        }
        if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tool_call in tool_calls {
                let function = &tool_call["function"];
                if function.get("name") == Some(&Value::String("get_weather".to_string())) {
                    header_seen = true;
                }
                if let Some(args) = function.get("arguments").and_then(|v| v.as_str()) {
                    aggregated_arguments.push_str(args);
                }
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            finish_reason = Some(reason.to_string());
        }
    }

    assert!(role_seen, "expected assistant role chunk");
    assert!(header_seen, "expected a tool call header chunk");
    assert_eq!(aggregated_arguments, r#"{"location":"Paris"}"#);
    assert_eq!(finish_reason.as_deref(), Some("tool_calls"));
    assert!(done_seen, "expected [DONE] marker");
}

#[tokio::test]
async fn auth_disabled_when_no_api_key_configured() {
    let (app, _tmp) = build_test_app(vec![ScriptedReply {
        content: "ok".to_string(),
        finish_reason: "stop".to_string(),
    }])
    .expect("test app");

    let resp = post_chat(
        &app,
        json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_required_request_without_key_returns_401() {
    let (app, _tmp) = build_test_app_with_api_key(vec![], "secret-key").expect("test app");

    let resp = post_chat(
        &app,
        json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.expect("body"))
            .expect("json");
    assert_eq!(body["error"]["type"], "invalid_api_key");
}

#[tokio::test]
async fn auth_required_request_with_wrong_key_returns_401() {
    let (app, _tmp) = build_test_app_with_api_key(vec![], "secret-key").expect("test app");

    let resp = post_chat_with_headers(
        &app,
        json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}),
        &[("authorization", "Bearer wrong-key")],
    )
    .await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_request_with_correct_bearer_succeeds() {
    let (app, _tmp) = build_test_app_with_api_key(
        vec![ScriptedReply {
            content: "ok".to_string(),
            finish_reason: "stop".to_string(),
        }],
        "secret-key",
    )
    .expect("test app");

    let resp = post_chat_with_headers(
        &app,
        json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}),
        &[("authorization", "Bearer secret-key")],
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_request_with_x_api_key_header_succeeds() {
    let (app, _tmp) = build_test_app_with_api_key(
        vec![ScriptedReply {
            content: "ok".to_string(),
            finish_reason: "stop".to_string(),
        }],
        "secret-key",
    )
    .expect("test app");

    let resp = post_chat_with_headers(
        &app,
        json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}),
        &[("x-api-key", "secret-key")],
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_health_endpoint_remains_unauthenticated() {
    let (app, _tmp) = build_test_app_with_api_key(vec![], "secret-key").expect("test app");

    // /health must stay reachable without credentials for liveness probes.
    let resp = get_request(&app, "/health").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_metrics_endpoint_requires_api_key() {
    let (app, _tmp) = build_test_app_with_api_key(vec![], "secret-key").expect("test app");

    let unauth = get_request(&app, "/metrics").await;
    assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

    let ok =
        get_request_with_headers(&app, "/metrics", &[("authorization", "Bearer secret-key")]).await;
    assert_eq!(ok.status(), StatusCode::OK);
}

#[tokio::test]
async fn auth_v1_models_endpoint_requires_api_key() {
    let (app, _tmp) = build_test_app_with_api_key(vec![], "secret-key").expect("test app");

    let unauth = get_request(&app, "/v1/models").await;
    assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);
}

async fn assert_invalid_request(app: &Router, body: Value, field_hint: &str) {
    let resp = post_chat(app, body).await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "expected 400 for invalid {field_hint}"
    );
    let parsed: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.expect("body"))
            .expect("json");
    assert_eq!(
        parsed["error"]["type"], "invalid_request_error",
        "expected invalid_request_error type for {field_hint}; got {:?}",
        parsed["error"]
    );
}

#[tokio::test]
async fn validation_temperature_above_two_rejected() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    assert_invalid_request(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 5.0
        }),
        "temperature",
    )
    .await;
}

#[tokio::test]
async fn validation_temperature_negative_rejected() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    assert_invalid_request(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": -0.5
        }),
        "temperature",
    )
    .await;
}

#[tokio::test]
async fn validation_top_p_above_one_rejected() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    assert_invalid_request(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 2.0
        }),
        "top_p",
    )
    .await;
}

#[tokio::test]
async fn validation_min_p_above_one_rejected() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    assert_invalid_request(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "min_p": 1.5
        }),
        "min_p",
    )
    .await;
}

#[tokio::test]
async fn validation_frequency_penalty_out_of_range_rejected() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    assert_invalid_request(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": 3.0
        }),
        "frequency_penalty",
    )
    .await;
}

#[tokio::test]
async fn validation_presence_penalty_out_of_range_rejected() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    assert_invalid_request(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "presence_penalty": -3.0
        }),
        "presence_penalty",
    )
    .await;
}

#[tokio::test]
async fn validation_top_logprobs_above_twenty_rejected() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    assert_invalid_request(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "top_logprobs": 100,
            "logprobs": true
        }),
        "top_logprobs",
    )
    .await;
}

#[tokio::test]
async fn validation_repetition_penalty_zero_rejected() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    // 0 would produce division-by-zero NaN logits.
    assert_invalid_request(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "repetition_penalty": 0.0
        }),
        "repetition_penalty",
    )
    .await;
}

#[tokio::test]
async fn validation_max_tokens_zero_rejected() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    assert_invalid_request(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0
        }),
        "max_tokens",
    )
    .await;
}

#[tokio::test]
async fn validation_n_above_max_rejected() {
    let (app, _tmp) = build_test_app(vec![]).expect("test app");
    assert_invalid_request(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 1000
        }),
        "n",
    )
    .await;
}

#[tokio::test]
async fn validation_valid_params_still_accepted() {
    // Edge-of-range values must still produce 200 (over-rejection regression).
    let (app, _tmp) = build_test_app(vec![ScriptedReply {
        content: "ok".to_string(),
        finish_reason: "stop".to_string(),
    }])
    .expect("test app");

    let resp = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 2.0,
            "top_p": 1.0,
            "min_p": 0.0,
            "frequency_penalty": -2.0,
            "presence_penalty": 2.0,
            "top_logprobs": 20,
            "logprobs": true,
            "repetition_penalty": 0.0001,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn timeout_non_streaming_returns_408_when_engine_stuck() {
    let request_tx = spawn_stuck_engine();
    let (app, _tmp) =
        build_test_app_with_timeout(request_tx, Duration::from_millis(150)).expect("test app");

    let start = std::time::Instant::now();
    let resp = post_chat(
        &app,
        json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    let elapsed = start.elapsed();

    assert_eq!(resp.status(), StatusCode::REQUEST_TIMEOUT);
    let body: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.expect("body"))
            .expect("json");
    assert_eq!(body["error"]["type"], "request_timeout");
    assert!(
        elapsed < Duration::from_secs(5),
        "handler should return within a few seconds of the deadline; took {elapsed:?}"
    );
}

#[tokio::test]
async fn timeout_streaming_emits_error_chunk_and_done() {
    let request_tx = spawn_stuck_engine();
    let (app, _tmp) =
        build_test_app_with_timeout(request_tx, Duration::from_millis(150)).expect("test app");

    let resp = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }),
    )
    .await;
    // Streaming always 200; error surfaces in body.
    assert_eq!(resp.status(), StatusCode::OK);

    let body = String::from_utf8(
        to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body")
            .to_vec(),
    )
    .expect("utf-8");

    let mut saw_error = false;
    let mut saw_done = false;
    for line in body.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            saw_done = true;
            continue;
        }
        let parsed: Value = serde_json::from_str(payload).expect("each chunk must be valid JSON");
        if parsed.get("error").is_some() {
            assert_eq!(parsed["error"]["type"], "request_timeout");
            saw_error = true;
        }
    }
    assert!(
        saw_error,
        "expected a request_timeout error chunk in stream"
    );
    assert!(saw_done, "expected a [DONE] sentinel after the error chunk");
}

#[tokio::test]
async fn timeout_does_not_fire_when_engine_responds_promptly() {
    let request_tx = spawn_scripted_engine(vec![ScriptedReply {
        content: "fast".to_string(),
        finish_reason: "stop".to_string(),
    }]);
    let (app, _tmp) =
        build_test_app_with_timeout(request_tx, Duration::from_secs(60)).expect("test app");

    let resp = post_chat(
        &app,
        json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.expect("body"))
            .expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "fast");
}

#[tokio::test]
async fn tools_streaming_error_emits_done_sentinel() {
    // Tools error path previously omitted [DONE].
    let (request_tx, mut request_rx) = tokio_mpsc::channel::<IncomingRequest>(64);
    tokio::spawn(async move {
        while let Some(req) = request_rx.recv().await {
            let _ = req
                .response_tx
                .send(EngineEvent::Error("synthetic engine failure".to_string()));
        }
    });

    let tmp = tempfile::tempdir().expect("tmp");
    let model_id = "test-model";
    create_test_model_dir(tmp.path(), model_id).expect("model dir");
    let tokenizer = Arc::new(
        Tokenizer::from_dir(tmp.path().join(model_id).to_str().expect("utf-8 path"))
            .expect("tokenizer"),
    );

    let mut manager = ModelManager::new(ModelManagerConfig {
        models_dir: tmp.path().to_path_buf(),
        keep_alive: Duration::from_secs(60),
        memory_budget_bytes: None,
        cuda_devices: vec![],
        max_context_len: 4096,
        kv_quant: KvQuantMode::Off,
        qjl_quantization: false,
        require_gpu: false,
        max_num_seqs: None,
        max_queued_requests: 200,
        draft_model: None,
        expert_stream_mb: None,
    });
    manager.insert_ready_for_tests(
        model_id,
        ReadyHandle {
            request_tx,
            tokenizer,
            max_seq_len: 4096,
        },
    );
    let state = Arc::new(AppState {
        manager: Arc::new(tokio::sync::Mutex::new(manager)),
        api_key: None,
        request_timeout: None,
    });
    let app = build_router(state);

    let resp = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "tools": base_tools()
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    let body = String::from_utf8(
        to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body")
            .to_vec(),
    )
    .expect("utf-8");

    let mut saw_done = false;
    for line in body.lines() {
        if line.strip_prefix("data: ") == Some("[DONE]") {
            saw_done = true;
        }
    }
    assert!(
        saw_done,
        "tools-streaming error path must end with [DONE]; body was:\n{body}"
    );
}

#[tokio::test]
async fn strict_schema_non_parseable_json_returns_content_filter() {
    // Strict mode: non-JSON output must produce finish_reason="content_filter" + null content.
    let (app, _tmp) = build_test_app(vec![ScriptedReply {
        content: "this is not JSON".to_string(),
        finish_reason: "stop".to_string(),
    }])
    .expect("test app");

    let resp = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": false
                    }
                }
            }
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.expect("body"))
            .expect("json");
    assert_eq!(body["choices"][0]["finish_reason"], "content_filter");
    assert!(
        body["choices"][0]["message"]["content"].is_null(),
        "content must be null under strict schema fail; got {:?}",
        body["choices"][0]["message"]["content"]
    );
}

#[tokio::test]
async fn strict_schema_parseable_but_invalid_returns_content_filter() {
    // Valid JSON missing required "name".
    let (app, _tmp) = build_test_app(vec![ScriptedReply {
        content: r#"{"age": 30}"#.to_string(),
        finish_reason: "stop".to_string(),
    }])
    .expect("test app");

    let resp = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": false
                    }
                }
            }
        }),
    )
    .await;
    let body: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.expect("body"))
            .expect("json");
    assert_eq!(body["choices"][0]["finish_reason"], "content_filter");
}

#[tokio::test]
async fn non_strict_schema_non_parseable_passes_through() {
    // Non-strict mode passes raw model output through even when schema doesn't match.
    let (app, _tmp) = build_test_app(vec![ScriptedReply {
        content: "this is not JSON".to_string(),
        finish_reason: "stop".to_string(),
    }])
    .expect("test app");

    let resp = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "strict": false,
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"]
                    }
                }
            }
        }),
    )
    .await;
    let body: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.expect("body"))
            .expect("json");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        body["choices"][0]["message"]["content"], "this is not JSON",
        "non-strict mode must return raw model output"
    );
}

#[tokio::test]
async fn stream_error_mid_stream_emits_done_after_error_chunk() {
    let (request_tx, mut request_rx) = tokio_mpsc::channel::<IncomingRequest>(64);
    tokio::spawn(async move {
        while let Some(req) = request_rx.recv().await {
            let _ = req.response_tx.send(EngineEvent::Token {
                text: "partial".to_string(),
                logprob_entries: vec![],
            });
            let _ = req.response_tx.send(EngineEvent::Error(
                "synthetic mid-stream failure".to_string(),
            ));
        }
    });

    let tmp = tempfile::tempdir().expect("tmp");
    let model_id = "test-model";
    create_test_model_dir(tmp.path(), model_id).expect("model dir");
    let tokenizer = Arc::new(
        Tokenizer::from_dir(tmp.path().join(model_id).to_str().expect("utf-8 path"))
            .expect("tokenizer"),
    );

    let mut manager = ModelManager::new(ModelManagerConfig {
        models_dir: tmp.path().to_path_buf(),
        keep_alive: Duration::from_secs(60),
        memory_budget_bytes: None,
        cuda_devices: vec![],
        max_context_len: 4096,
        kv_quant: KvQuantMode::Off,
        qjl_quantization: false,
        require_gpu: false,
        max_num_seqs: None,
        max_queued_requests: 200,
        draft_model: None,
        expert_stream_mb: None,
    });
    manager.insert_ready_for_tests(
        model_id,
        ReadyHandle {
            request_tx,
            tokenizer,
            max_seq_len: 4096,
        },
    );
    let state = Arc::new(AppState {
        manager: Arc::new(tokio::sync::Mutex::new(manager)),
        api_key: None,
        request_timeout: None,
    });
    let app = build_router(state);

    let resp = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    let body = String::from_utf8(
        to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body")
            .to_vec(),
    )
    .expect("utf-8");

    let mut saw_content = false;
    let mut saw_error = false;
    let mut saw_done = false;
    for line in body.lines() {
        if let Some(payload) = line.strip_prefix("data: ") {
            if payload == "[DONE]" {
                saw_done = true;
                continue;
            }
            let Ok(parsed) = serde_json::from_str::<Value>(payload) else {
                continue;
            };
            if parsed["choices"][0]["delta"]["content"].as_str() == Some("partial") {
                saw_content = true;
            }
            if parsed.get("error").is_some() {
                saw_error = true;
            }
        }
    }
    assert!(
        saw_content,
        "expected the content chunk emitted before the error, body was:\n{body}"
    );
    assert!(
        saw_error,
        "expected an error chunk after the Token, body was:\n{body}"
    );
    assert!(
        saw_done,
        "non-tools streaming must end with [DONE] even after an error, body was:\n{body}"
    );
}

#[tokio::test]
async fn n_above_one_non_streaming_returns_n_distinct_choices() {
    let (app, _tmp) = build_test_app(vec![
        ScriptedReply {
            content: "alpha".to_string(),
            finish_reason: "stop".to_string(),
        },
        ScriptedReply {
            content: "beta".to_string(),
            finish_reason: "stop".to_string(),
        },
    ])
    .expect("test app");

    let resp = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 2
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.expect("body"))
            .expect("json");

    let choices = body["choices"].as_array().expect("choices array");
    assert_eq!(choices.len(), 2, "expected exactly 2 choices for n=2");

    let mut indices: Vec<i64> = choices
        .iter()
        .map(|c| c["index"].as_i64().expect("index"))
        .collect();
    indices.sort();
    assert_eq!(
        indices,
        vec![0, 1],
        "choices must carry distinct sequential indices"
    );

    let mut contents: Vec<String> = choices
        .iter()
        .map(|c| c["message"]["content"].as_str().unwrap_or("").to_string())
        .collect();
    contents.sort();
    assert_eq!(contents, vec!["alpha".to_string(), "beta".to_string()]);
}

#[tokio::test]
async fn stream_options_include_usage_emits_trailing_usage_chunk() {
    let (app, _tmp) = build_test_app(vec![ScriptedReply {
        content: "hi".to_string(),
        finish_reason: "stop".to_string(),
    }])
    .expect("test app");

    let resp = post_chat(
        &app,
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    let body = String::from_utf8(
        to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body")
            .to_vec(),
    )
    .expect("utf-8");

    let chunks: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .filter_map(|payload| serde_json::from_str::<Value>(payload).ok())
        .collect();

    let usage_chunk = chunks
        .iter()
        .rev()
        .find(|c| c.get("usage").map(|u| !u.is_null()).unwrap_or(false))
        .expect("expected a usage chunk before [DONE]");

    let usage = &usage_chunk["usage"];
    assert!(usage["prompt_tokens"].as_u64().is_some());
    assert!(usage["completion_tokens"].as_u64().is_some());
    assert!(usage["total_tokens"].as_u64().is_some());
    assert!(
        usage_chunk["choices"]
            .as_array()
            .map(|c| c.is_empty())
            .unwrap_or(false),
        "the usage chunk must carry an empty `choices` array per the OpenAI spec, got: {usage_chunk}"
    );
    assert!(
        body.contains("data: [DONE]"),
        "stream must terminate with [DONE], body was:\n{body}"
    );
}
