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

fn spawn_scripted_engine(
    replies: Vec<ScriptedReply>,
) -> tokio_mpsc::UnboundedSender<IncomingRequest> {
    let (request_tx, mut request_rx) = tokio_mpsc::unbounded_channel::<IncomingRequest>();
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
