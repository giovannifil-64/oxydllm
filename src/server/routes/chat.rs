use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use serde::Serialize;
use tokio::sync::mpsc as tokio_mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::AppState;
use super::error_response;
use super::types::{
    ChatCompletionRequest, ChatMessage, EngineEvent, EngineLogprobEntry, IncomingRequest,
    ResponseFormat, StopParam,
};
use crate::chat_template;
use crate::models::manager::GetResult;
use crate::sampling::SamplingParams;
use crate::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Logprob response structs
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone)]
struct TopLogprobItem {
    token: String,
    logprob: f32,
    bytes: Option<Vec<u8>>,
}

#[derive(Serialize, Clone)]
struct TokenLogprob {
    token: String,
    logprob: f32,
    bytes: Option<Vec<u8>>,
    top_logprobs: Vec<TopLogprobItem>,
}

#[derive(Serialize, Clone)]
struct Logprobs {
    content: Vec<TokenLogprob>,
    refusal: Option<String>,
}

// ---------------------------------------------------------------------------
// Completion response structs
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
    system_fingerprint: Option<String>,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
    logprobs: Option<Logprobs>,
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
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
    system_fingerprint: Option<String>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<String>,
    logprobs: Option<Logprobs>,
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn system_fingerprint(model_id: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    model_id.hash(&mut h);
    env!("CARGO_PKG_VERSION").hash(&mut h);
    format!("fp_{:012x}", h.finish() & 0xFFFF_FFFF_FFFF)
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

/// Strip markdown code fences (```json ... ```) that models often wrap JSON in.
fn strip_json_fences(s: &str) -> &str {
    let s = s.trim();
    let inner = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```JSON"))
        .or_else(|| s.strip_prefix("```"))
        .and_then(|t| t.trim_start_matches('\n').strip_suffix("```"))
        .map(|t| t.trim_end_matches('\n').trim());
    inner.unwrap_or(s)
}

/// Build the JSON mode system instruction from a `ResponseFormat`.
fn json_system_instruction(rf: &ResponseFormat) -> String {
    match rf.format_type.as_str() {
        "json_schema" => {
            if let Some(spec) = &rf.json_schema {
                let schema_part = spec
                    .schema
                    .as_ref()
                    .and_then(|s| serde_json::to_string_pretty(s).ok())
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        format!(
                            " that conforms to the following JSON Schema:\n```json\n{}\n```",
                            s
                        )
                    })
                    .unwrap_or_default();
                let name_part = if spec.name.trim().is_empty() {
                    String::new()
                } else {
                    format!(" The schema name is \"{}\".", spec.name.trim())
                };
                let description_part = spec
                    .description
                    .as_ref()
                    .map(|d| d.trim())
                    .filter(|d| !d.is_empty())
                    .map(|d| format!(" Schema description: {}.", d))
                    .unwrap_or_default();
                let strict_part = if spec.strict.unwrap_or(false) {
                    " Follow the schema exactly and do not add unspecified fields.".to_string()
                } else {
                    String::new()
                };
                format!(
                    "You must respond with valid JSON only. Do not include any explanation, markdown, or text outside of the JSON object.{}{}{}{}",
                    schema_part, name_part, description_part, strict_part,
                )
            } else {
                "You must respond with valid JSON only. Do not include any explanation, markdown, or text outside of the JSON object.".to_string()
            }
        }
        _ => {
            // json_object or anything else that signals JSON mode
            "You must respond with valid JSON only. Do not include any explanation, markdown, or text outside of the JSON object.".to_string()
        }
    }
}

pub fn apply_chat_template(
    tokenizer: &Tokenizer,
    messages: &[ChatMessage],
    enable_thinking: bool,
) -> String {
    let Some(template) = tokenizer.chat_template() else {
        if tokenizer.special_token_id("<|turn>").is_some()
            && tokenizer.special_token_id("<turn|>").is_some()
        {
            return chat_template::format_turn_chat(
                messages,
                tokenizer.bos_token(),
                "<|turn>",
                "<turn|>",
                true,
                enable_thinking,
            );
        }

        if tokenizer.special_token_id("<start_of_turn>").is_some()
            && tokenizer.special_token_id("<end_of_turn>").is_some()
        {
            return chat_template::format_turn_chat(
                messages,
                tokenizer.bos_token(),
                "<start_of_turn>",
                "<end_of_turn>",
                true,
                enable_thinking,
            );
        }

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
                    tracing::warn!(
                        "system role not supported by this model template; retrying without system message"
                    );
                    return prompt;
                }
            }

            tracing::warn!(
                error = ?e,
                "chat template rendering failed, falling back to plain text"
            );
            chat_template::format_plain_chat(messages)
        }
    }
}

/// Convert engine logprob entries into the serializable `Logprobs` response struct.
fn build_logprobs_content(entries: &[EngineLogprobEntry], req_top_n: usize) -> Logprobs {
    Logprobs {
        content: entries
            .iter()
            .map(|e| TokenLogprob {
                token: e.token_str.clone(),
                logprob: e.logprob,
                bytes: Some(e.bytes.clone()),
                top_logprobs: e
                    .top_logprobs
                    .iter()
                    .take(req_top_n)
                    .map(|(ts, lp, tb)| TopLogprobItem {
                        token: ts.clone(),
                        logprob: *lp,
                        bytes: Some(tb.clone()),
                    })
                    .collect(),
            })
            .collect(),
        refusal: None,
    }
}

// ---------------------------------------------------------------------------
// chat/completions — helpers
// ---------------------------------------------------------------------------

/// Accumulated result for a single completion (used for n>1).
struct CompletionData {
    content: String,
    reasoning_content: String,
    reasoning_tokens: usize,
    finish_reason: String,
    completion_tokens: usize,
    logprob_entries: Vec<EngineLogprobEntry>,
}

/// Drain one completion channel into a `CompletionData`.
async fn collect_one_completion(
    mut rx: tokio_mpsc::UnboundedReceiver<EngineEvent>,
) -> Result<CompletionData, (StatusCode, Json<serde_json::Value>)> {
    let mut data = CompletionData {
        content: String::new(),
        reasoning_content: String::new(),
        reasoning_tokens: 0,
        finish_reason: "stop".to_string(),
        completion_tokens: 0,
        logprob_entries: Vec::new(),
    };
    while let Some(event) = rx.recv().await {
        match event {
            EngineEvent::Token {
                text,
                logprob_entries,
            } => {
                data.content.push_str(&text);
                data.logprob_entries.extend(logprob_entries);
            }
            EngineEvent::ReasoningToken(text) => {
                data.reasoning_content.push_str(&text);
                data.reasoning_tokens += 1;
            }
            EngineEvent::Finish {
                finish_reason,
                completion_tokens,
            } => {
                data.finish_reason = finish_reason;
                data.completion_tokens = completion_tokens;
            }
            EngineEvent::StreamEnd => break,
            EngineEvent::Error(msg) => {
                return Err(error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    msg,
                    "server_error",
                ));
            }
        }
    }
    Ok(data)
}

// ---------------------------------------------------------------------------
// chat/completions — main handler
// ---------------------------------------------------------------------------

pub(super) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<serde_json::Value>)> {
    if body.messages.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
            "invalid_request_error",
        ));
    }

    let model_id = body.model.as_deref().unwrap_or("").to_string();
    if model_id.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "model field is required",
            "invalid_request_error",
        ));
    }

    let n = body.n.unwrap_or(1);
    if n == 0 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "n must be at least 1",
            "invalid_request_error",
        ));
    }

    let _client_metadata = (&body.user, &body.tools, &body.tool_choice);

    // Validate logit_bias shape early.
    if let Some(ref lb) = body.logit_bias
        && !lb.is_null()
        && !lb.is_object()
    {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "logit_bias must be a JSON object mapping token IDs to biases",
            "invalid_request_error",
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
            tracing::debug!(
                model_id = %model_id,
                manager_ready_ms = t_after_lock.as_secs_f64() * 1000.0,
                "model manager returned ready handle"
            );
            h
        }
        GetResult::Wait(rx) => {
            let load_result = rx.await.map_err(|_| {
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Model loader dropped",
                    "server_error",
                )
            })?;
            let h = load_result.map_err(|e| {
                let status = if e.contains("not found") {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                error_response(status, e, "server_error")
            })?;
            tracing::debug!(
                model_id = %model_id,
                load_completed_ms = t_request.elapsed().as_secs_f64() * 1000.0,
                "model load completed"
            );
            h
        }
    };

    let t_template = std::time::Instant::now();
    let enable_thinking =
        body.enable_thinking.unwrap_or(false) && handle.tokenizer.has_thinking_support();

    // Build the message list, optionally injecting a JSON-mode system instruction.
    let json_mode = body
        .response_format
        .as_ref()
        .map(|rf| rf.format_type == "json_object" || rf.format_type == "json_schema")
        .unwrap_or(false);
    let messages_for_prompt: std::borrow::Cow<[ChatMessage]> =
        if let Some(rf) = &body.response_format {
            if json_mode {
                let instr = json_system_instruction(rf);
                let mut msgs = body.messages.clone();
                if let Some(sys) = msgs.iter_mut().find(|m| m.role == "system") {
                    sys.content = format!("{}\n\n{}", sys.content, instr);
                } else {
                    msgs.insert(
                        0,
                        ChatMessage {
                            role: "system".to_string(),
                            content: instr,
                            reasoning_content: None,
                        },
                    );
                }
                std::borrow::Cow::Owned(msgs)
            } else {
                std::borrow::Cow::Borrowed(&body.messages)
            }
        } else {
            std::borrow::Cow::Borrowed(&body.messages)
        };

    let prompt = apply_chat_template(&handle.tokenizer, &messages_for_prompt, enable_thinking);
    let template_ms = t_template.elapsed().as_secs_f64() * 1000.0;

    let t_encode = std::time::Instant::now();
    let prompt_tokens = handle.tokenizer.encode(&prompt).map_err(|e| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
            "server_error",
        )
    })?;
    let encode_ms = t_encode.elapsed().as_secs_f64() * 1000.0;
    let prompt_len = prompt_tokens.len();

    tracing::debug!(
        model_id = %model_id,
        template_ms,
        encode_ms,
        prompt_tokens = prompt_len,
        pre_engine_total_ms = t_request.elapsed().as_secs_f64() * 1000.0,
        "prompt preparation timings"
    );

    // Parse stop strings into token IDs (single-token stops only).
    let extra_stop_token_ids: Vec<u32> = match &body.stop {
        Some(StopParam::Single(s)) => {
            let ids = handle.tokenizer.encode(s).unwrap_or_default();
            if ids.len() == 1 { ids } else { Vec::new() }
        }
        Some(StopParam::Multiple(strings)) => strings
            .iter()
            .filter_map(|s| {
                let ids = handle.tokenizer.encode(s).unwrap_or_default();
                if ids.len() == 1 { Some(ids[0]) } else { None }
            })
            .collect(),
        None => Vec::new(),
    };

    // Parse logit_bias.
    let logit_bias: Option<Vec<(u32, f32)>> = match &body.logit_bias {
        Some(serde_json::Value::Object(map)) if !map.is_empty() => {
            let pairs: Vec<(u32, f32)> = map
                .iter()
                .filter_map(|(k, v)| {
                    let token_id: u32 = k.parse().ok()?;
                    let bias: f32 = v.as_f64()? as f32;
                    Some((token_id, bias.clamp(-100.0, 100.0)))
                })
                .collect();
            if pairs.is_empty() { None } else { Some(pairs) }
        }
        _ => None,
    };

    // Compute top_logprobs_k for sampling.
    let wants_logprobs = body.logprobs.unwrap_or(false);
    let req_top_n = body.top_logprobs.unwrap_or(0);
    let top_logprobs_k: usize = if wants_logprobs {
        req_top_n.max(1) // at least 1 so we always get the chosen token's logprob
    } else {
        0
    };

    let base_sampling_params = SamplingParams {
        temperature: body.temperature.unwrap_or(0.7),
        top_k: body.top_k.unwrap_or(0),
        top_p: body.top_p.unwrap_or(1.0),
        min_p: body.min_p.unwrap_or(0.0),
        repetition_penalty: body.repetition_penalty.unwrap_or(1.0),
        repetition_window: body.repetition_window.unwrap_or(0),
        frequency_penalty: body.frequency_penalty.unwrap_or(0.0),
        presence_penalty: body.presence_penalty.unwrap_or(0.0),
        seed: body.seed,
        logit_bias,
        top_logprobs_k,
    };

    let remaining = handle.max_seq_len.saturating_sub(prompt_len);
    let max_tokens = body
        .max_completion_tokens
        .or(body.max_tokens)
        .unwrap_or(remaining)
        .min(remaining);

    // Spawn N requests (one per completion).
    let mut completion_rxs: Vec<tokio_mpsc::UnboundedReceiver<EngineEvent>> = Vec::with_capacity(n);

    for i in 0..n {
        let (response_tx, response_rx) = tokio_mpsc::unbounded_channel();
        // Give each completion a distinct seed offset so n>1 produces different outputs.
        let seed = base_sampling_params.seed.map(|s| s.wrapping_add(i as u64));
        let sampling_params = SamplingParams {
            seed,
            ..base_sampling_params.clone()
        };

        handle
            .request_tx
            .send(IncomingRequest {
                prompt_tokens: prompt_tokens.clone(),
                sampling_params,
                max_tokens,
                response_tx,
                model_id: model_id.clone(),
                enqueued_at: std::time::Instant::now(),
                enable_thinking,
                extra_stop_token_ids: extra_stop_token_ids.clone(),
            })
            .map_err(|_| {
                error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Engine unavailable",
                    "server_error",
                )
            })?;
        completion_rxs.push(response_rx);
    }

    let chat_id = make_chat_id();
    let created = unix_timestamp();
    let fp = system_fingerprint(&model_id);
    let stream = body.stream.unwrap_or(false);
    let include_usage = body
        .stream_options
        .as_ref()
        .and_then(|o| o.include_usage)
        .unwrap_or(false);

    if stream {
        // For n=1: use the existing direct-channel approach (avoids merge overhead).
        // For n>1: merge all N receivers into a single (index, event) channel.

        let (sse_tx, sse_rx) =
            tokio_mpsc::unbounded_channel::<Result<Event, std::convert::Infallible>>();

        if n == 1 {
            let mut response_rx = completion_rxs.remove(0);
            let model_id_clone = model_id.clone();

            tokio::spawn(async move {
                // Role chunk.
                let role_chunk = ChatCompletionChunk {
                    id: chat_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_id_clone.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: Some("assistant".to_string()),
                            content: None,
                            reasoning_content: None,
                        },
                        finish_reason: None,
                        logprobs: None,
                    }],
                    usage: None,
                    system_fingerprint: Some(fp.clone()),
                };
                if sse_tx
                    .send(Ok(
                        Event::default().data(serde_json::to_string(&role_chunk).unwrap())
                    ))
                    .is_err()
                {
                    return;
                }

                while let Some(event) = response_rx.recv().await {
                    match event {
                        EngineEvent::Token {
                            text,
                            logprob_entries,
                        } => {
                            if text.is_empty() && logprob_entries.is_empty() {
                                continue;
                            }
                            let chunk_logprobs = if wants_logprobs && !logprob_entries.is_empty() {
                                Some(build_logprobs_content(&logprob_entries, req_top_n))
                            } else if wants_logprobs {
                                Some(Logprobs {
                                    content: vec![],
                                    refusal: None,
                                })
                            } else {
                                None
                            };
                            let chunk = ChatCompletionChunk {
                                id: chat_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_id_clone.clone(),
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: Delta {
                                        role: None,
                                        content: Some(text),
                                        reasoning_content: None,
                                    },
                                    finish_reason: None,
                                    logprobs: chunk_logprobs,
                                }],
                                usage: None,
                                system_fingerprint: Some(fp.clone()),
                            };
                            if sse_tx
                                .send(Ok(
                                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                                ))
                                .is_err()
                            {
                                break;
                            }
                        }
                        EngineEvent::ReasoningToken(text) => {
                            let chunk = ChatCompletionChunk {
                                id: chat_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_id_clone.clone(),
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: Delta {
                                        role: None,
                                        content: None,
                                        reasoning_content: Some(text),
                                    },
                                    finish_reason: None,
                                    logprobs: None,
                                }],
                                usage: None,
                                system_fingerprint: Some(fp.clone()),
                            };
                            if sse_tx
                                .send(Ok(
                                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                                ))
                                .is_err()
                            {
                                break;
                            }
                        }
                        EngineEvent::Finish {
                            finish_reason,
                            completion_tokens,
                        } => {
                            let chunk = ChatCompletionChunk {
                                id: chat_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_id_clone.clone(),
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: Delta {
                                        role: None,
                                        content: None,
                                        reasoning_content: None,
                                    },
                                    finish_reason: Some(finish_reason),
                                    logprobs: None,
                                }],
                                usage: None,
                                system_fingerprint: Some(fp.clone()),
                            };
                            if sse_tx
                                .send(Ok(
                                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                                ))
                                .is_err()
                            {
                                break;
                            }

                            if include_usage {
                                let usage_chunk = ChatCompletionChunk {
                                    id: chat_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_id_clone.clone(),
                                    choices: vec![],
                                    usage: Some(Usage {
                                        prompt_tokens: prompt_len,
                                        completion_tokens,
                                        total_tokens: prompt_len + completion_tokens,
                                        completion_tokens_details: None,
                                    }),
                                    system_fingerprint: Some(fp.clone()),
                                };
                                if sse_tx
                                    .send(Ok(Event::default()
                                        .data(serde_json::to_string(&usage_chunk).unwrap())))
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                        EngineEvent::StreamEnd => {
                            if sse_tx.send(Ok(Event::default().data("[DONE]"))).is_err() {
                                break;
                            }
                            break;
                        }
                        EngineEvent::Error(msg) => {
                            let err = serde_json::json!({
                                "error": { "message": msg, "type": "server_error", "param": null, "code": null }
                            });
                            if sse_tx
                                .send(Ok(
                                    Event::default().data(serde_json::to_string(&err).unwrap())
                                ))
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                }
            });
        } else {
            // n > 1: merge all receivers into a tagged channel.
            let (merged_tx, merged_rx) = tokio_mpsc::unbounded_channel::<(usize, EngineEvent)>();
            for (i, rx) in completion_rxs.into_iter().enumerate() {
                let tx = merged_tx.clone();
                tokio::spawn(async move {
                    let mut rx = rx;
                    while let Some(ev) = rx.recv().await {
                        if tx.send((i, ev)).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(merged_tx);

            let model_id_clone = model_id.clone();

            tokio::spawn(async move {
                // Send role chunk for each completion.
                for i in 0..n {
                    let role_chunk = ChatCompletionChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id_clone.clone(),
                        choices: vec![ChunkChoice {
                            index: i,
                            delta: Delta {
                                role: Some("assistant".to_string()),
                                content: None,
                                reasoning_content: None,
                            },
                            finish_reason: None,
                            logprobs: None,
                        }],
                        usage: None,
                        system_fingerprint: Some(fp.clone()),
                    };
                    if sse_tx
                        .send(Ok(
                            Event::default().data(serde_json::to_string(&role_chunk).unwrap())
                        ))
                        .is_err()
                    {
                        return;
                    }
                }

                let mut rx = merged_rx;
                let mut stream_ends: usize = 0;
                let mut total_completion_tokens: usize = 0;

                while let Some((idx, event)) = rx.recv().await {
                    match event {
                        EngineEvent::Token {
                            text,
                            logprob_entries,
                        } => {
                            if text.is_empty() && logprob_entries.is_empty() {
                                continue;
                            }
                            let chunk_logprobs = if wants_logprobs && !logprob_entries.is_empty() {
                                Some(build_logprobs_content(&logprob_entries, req_top_n))
                            } else if wants_logprobs {
                                Some(Logprobs {
                                    content: vec![],
                                    refusal: None,
                                })
                            } else {
                                None
                            };
                            let chunk = ChatCompletionChunk {
                                id: chat_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_id_clone.clone(),
                                choices: vec![ChunkChoice {
                                    index: idx,
                                    delta: Delta {
                                        role: None,
                                        content: Some(text),
                                        reasoning_content: None,
                                    },
                                    finish_reason: None,
                                    logprobs: chunk_logprobs,
                                }],
                                usage: None,
                                system_fingerprint: Some(fp.clone()),
                            };
                            if sse_tx
                                .send(Ok(
                                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                                ))
                                .is_err()
                            {
                                break;
                            }
                        }
                        EngineEvent::ReasoningToken(text) => {
                            let chunk = ChatCompletionChunk {
                                id: chat_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_id_clone.clone(),
                                choices: vec![ChunkChoice {
                                    index: idx,
                                    delta: Delta {
                                        role: None,
                                        content: None,
                                        reasoning_content: Some(text),
                                    },
                                    finish_reason: None,
                                    logprobs: None,
                                }],
                                usage: None,
                                system_fingerprint: Some(fp.clone()),
                            };
                            if sse_tx
                                .send(Ok(
                                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                                ))
                                .is_err()
                            {
                                break;
                            }
                        }
                        EngineEvent::Finish {
                            finish_reason,
                            completion_tokens,
                        } => {
                            total_completion_tokens += completion_tokens;
                            let chunk = ChatCompletionChunk {
                                id: chat_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_id_clone.clone(),
                                choices: vec![ChunkChoice {
                                    index: idx,
                                    delta: Delta {
                                        role: None,
                                        content: None,
                                        reasoning_content: None,
                                    },
                                    finish_reason: Some(finish_reason),
                                    logprobs: None,
                                }],
                                usage: None,
                                system_fingerprint: Some(fp.clone()),
                            };
                            if sse_tx
                                .send(Ok(
                                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                                ))
                                .is_err()
                            {
                                break;
                            }
                        }
                        EngineEvent::StreamEnd => {
                            stream_ends += 1;
                            if stream_ends == n {
                                if include_usage {
                                    let usage_chunk = ChatCompletionChunk {
                                        id: chat_id.clone(),
                                        object: "chat.completion.chunk".to_string(),
                                        created,
                                        model: model_id_clone.clone(),
                                        choices: vec![],
                                        usage: Some(Usage {
                                            prompt_tokens: prompt_len,
                                            completion_tokens: total_completion_tokens,
                                            total_tokens: prompt_len + total_completion_tokens,
                                            completion_tokens_details: None,
                                        }),
                                        system_fingerprint: Some(fp.clone()),
                                    };
                                    if sse_tx
                                        .send(Ok(Event::default()
                                            .data(serde_json::to_string(&usage_chunk).unwrap())))
                                        .is_err()
                                    {
                                        break;
                                    }
                                }
                                if sse_tx.send(Ok(Event::default().data("[DONE]"))).is_err() {
                                    break;
                                }
                                break;
                            }
                        }
                        EngineEvent::Error(msg) => {
                            let err = serde_json::json!({
                                "error": { "message": msg, "type": "server_error", "param": null, "code": null }
                            });
                            if sse_tx
                                .send(Ok(
                                    Event::default().data(serde_json::to_string(&err).unwrap())
                                ))
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                }
            });
        }

        let sse_stream = UnboundedReceiverStream::new(sse_rx);
        Ok(Sse::new(sse_stream).into_response())
    } else {
        // Non-streaming: collect all N completions concurrently.
        let mut handles = Vec::with_capacity(n);
        for rx in completion_rxs {
            handles.push(tokio::spawn(collect_one_completion(rx)));
        }

        let mut all_choices: Vec<Choice> = Vec::with_capacity(n);
        let mut total_completion_tokens: usize = 0;
        let mut total_reasoning_tokens: usize = 0;

        for (i, handle) in handles.into_iter().enumerate() {
            let data = handle.await.map_err(|_| {
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Task panic",
                    "server_error",
                )
            })??;

            total_completion_tokens += data.completion_tokens;
            total_reasoning_tokens += data.reasoning_tokens;

            let reasoning_opt = if data.reasoning_content.is_empty() {
                None
            } else {
                Some(data.reasoning_content.trim().to_string())
            };

            let logprobs = if wants_logprobs {
                Some(build_logprobs_content(&data.logprob_entries, req_top_n))
            } else {
                None
            };

            // In JSON mode, strip markdown fences and validate syntax.
            let (content, finish_reason) = if json_mode {
                let raw = strip_json_fences(data.content.trim()).to_string();
                let reason = if serde_json::from_str::<serde_json::Value>(&raw).is_ok() {
                    data.finish_reason.clone()
                } else {
                    // Output is not valid JSON — signal this via finish_reason while still
                    // returning the raw text so callers can inspect it.
                    "content_filter".to_string()
                };
                (raw, reason)
            } else {
                (data.content.trim().to_string(), data.finish_reason.clone())
            };

            all_choices.push(Choice {
                index: i,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content,
                    reasoning_content: reasoning_opt,
                },
                finish_reason,
                logprobs,
            });
        }

        let completion_tokens_details = if total_reasoning_tokens > 0 {
            Some(CompletionTokensDetails {
                reasoning_tokens: total_reasoning_tokens,
            })
        } else {
            None
        };

        let response = ChatCompletionResponse {
            id: chat_id,
            object: "chat.completion".to_string(),
            created,
            model: model_id,
            choices: all_choices,
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: total_completion_tokens,
                total_tokens: prompt_len + total_completion_tokens,
                completion_tokens_details,
            },
            system_fingerprint: Some(fp),
        };

        Ok(Json(response).into_response())
    }
}
