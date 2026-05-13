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

// ---------------------------------------------------------------------------
// GET endpoint tests
// ---------------------------------------------------------------------------

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

    // Simple Gauges (non-vec) always appear in Prometheus output even with value 0.
    // Labeled Vecs (HistogramVec, CounterVec, GaugeVec) only appear after at least
    // one label combination has been observed or set. We check the gauges that the
    // handler explicitly sets for the test model.
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

    // The handler calls list_running() and sets oxydllm_model_weights_bytes and
    // oxydllm_kv_cache_allocated_bytes for every loaded model. The test model is
    // in SlotState::Ready, so its label must appear in the output.
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

// ---------------------------------------------------------------------------
// Input validation error tests
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Happy-path chat completion tests
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tool call tests
// ---------------------------------------------------------------------------

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
