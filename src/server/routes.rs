use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc as tokio_mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::chat_template;
use crate::engine::Engine;
use crate::models::manager::{
    self, GetResult, ModelManager, ModelManagerConfig, SharedModelManager,
};
use crate::sampling::SamplingParams;
use crate::scheduler::sequence::SequenceId;
use crate::tokenizer::Tokenizer;

#[derive(Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Deserialize, Clone)]
pub struct JsonSchemaSpec {
    pub name: String,
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: Option<bool>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Deserialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(default)]
    pub json_schema: Option<JsonSchemaSpec>,
}

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
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
    // Extensions (non-OpenAI)
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_window: Option<usize>,
    #[serde(default)]
    pub keep_alive: Option<u64>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum StopParam {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

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
// Engine communication types
// ---------------------------------------------------------------------------

/// Pre-decoded logprob entry for a single generated token.
pub struct EngineLogprobEntry {
    pub token_str: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    /// Top-k alternatives: (token_str, logprob, bytes).
    pub top_logprobs: Vec<(String, f32, Vec<u8>)>,
}

pub struct IncomingRequest {
    pub prompt_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub max_tokens: usize,
    pub response_tx: tokio_mpsc::UnboundedSender<EngineEvent>,
    pub model_id: String,
    pub enqueued_at: std::time::Instant,
    pub enable_thinking: bool,
    pub extra_stop_token_ids: Vec<u32>,
}

pub enum EngineEvent {
    Token {
        text: String,
        logprob_entries: Vec<EngineLogprobEntry>,
    },
    ReasoningToken(String),
    Finish {
        finish_reason: String,
        completion_tokens: usize,
    },
    StreamEnd,
    Error(String),
}

struct AppState {
    manager: SharedModelManager,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn error_response(
    status: StatusCode,
    message: impl Into<String>,
    error_type: &str,
) -> (StatusCode, Json<serde_json::Value>) {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message.into(),
                "type": error_type,
                "param": null,
                "code": null,
            }
        })),
    )
}

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

const TOKEN_DECODE_CACHE_CAP: usize = 8192;

#[derive(Clone)]
struct DecodedToken {
    text: String,
    bytes: Vec<u8>,
}

struct TokenDecodeCache {
    entries: LruCache<u32, DecodedToken>,
}

impl TokenDecodeCache {
    fn new() -> Self {
        Self {
            entries: LruCache::new(NonZeroUsize::new(TOKEN_DECODE_CACHE_CAP).unwrap()),
        }
    }

    fn decode_token(&mut self, tokenizer: &Tokenizer, token_id: u32) -> DecodedToken {
        if let Some(hit) = self.entries.get(&token_id) {
            return hit.clone();
        }
        let text = tokenizer.decode(&[token_id]).unwrap_or_default();
        let decoded = DecodedToken {
            bytes: text.as_bytes().to_vec(),
            text,
        };
        self.entries.push(token_id, decoded.clone());
        decoded
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

/// Build an `EngineLogprobEntry` by decoding token IDs to strings.
fn build_logprob_entry(
    tokenizer: &Tokenizer,
    decode_cache: &mut TokenDecodeCache,
    token_id: u32,
    logprob: f32,
    top_lps: Vec<(u32, f32)>,
) -> EngineLogprobEntry {
    let decoded = decode_cache.decode_token(tokenizer, token_id);
    let token_str = decoded.text;
    let bytes = decoded.bytes;
    let top_logprobs = top_lps
        .into_iter()
        .map(|(tid, lp)| {
            let decoded = decode_cache.decode_token(tokenizer, tid);
            (decoded.text, lp, decoded.bytes)
        })
        .collect();
    EngineLogprobEntry {
        token_str,
        logprob,
        bytes,
        top_logprobs,
    }
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
                    eprintln!(
                        "Warning: system role not supported by this model's template — retrying without system message."
                    );
                    return prompt;
                }
            }

            eprintln!(
                "Warning: chat template rendering failed: {e:#}. Falling back to plain text."
            );
            chat_template::format_plain_chat(messages)
        }
    }
}

// ---------------------------------------------------------------------------
// Engine loop and per-sequence tracking
// ---------------------------------------------------------------------------

type RawTopLogprobs = Vec<(u32, f32)>;
type PendingRawLogprob = (u32, f32, RawTopLogprobs);

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
    /// Raw logprob data buffered for output tokens not yet emitted as text.
    pending_raw_lps: Vec<PendingRawLogprob>,
}

fn abort_sequences(
    engine: &mut Engine,
    trackers: &mut HashMap<SequenceId, SeqTracker>,
    mut seq_ids: Vec<SequenceId>,
    reason: &str,
) {
    seq_ids.sort_unstable();
    seq_ids.dedup();
    for seq_id in seq_ids {
        let model_id = trackers.remove(&seq_id).map(|t| t.model_id);
        let _ = engine.abort_sequence(seq_id);
        match model_id {
            Some(model_id) => eprintln!("[req] {} seq={} aborted ({reason})", model_id, seq_id),
            None => eprintln!("[req] seq={} aborted ({reason})", seq_id),
        }
    }
}

fn enqueue_request(
    req: IncomingRequest,
    engine: &mut Engine,
    trackers: &mut HashMap<SequenceId, SeqTracker>,
) {
    let model_id = req.model_id.clone();
    let enqueued_at = req.enqueued_at;
    let seq_id = engine.add_request_with_stop(
        req.prompt_tokens,
        req.sampling_params,
        req.max_tokens,
        req.extra_stop_token_ids,
    );
    eprintln!("[req] {} seq={} enqueued", model_id, seq_id);
    trackers.insert(
        seq_id,
        SeqTracker {
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
            pending_raw_lps: Vec::new(),
        },
    );
}

fn clamp_to_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn prefix_decode_token(
    tokenizer: &Tokenizer,
    decode_cache: &mut TokenDecodeCache,
    all_ids: &[u32],
    decoded_len: &mut usize,
    token: u32,
) -> Option<String> {
    let single = decode_cache.decode_token(tokenizer, token).text;
    if !single.is_empty() && !single.contains('\u{FFFD}') {
        *decoded_len += single.len();
        return Some(single);
    }
    let full = tokenizer.decode(all_ids).unwrap_or_default();
    let start = clamp_to_char_boundary(&full, *decoded_len);
    let new_text = &full[start..];
    let emit = new_text.trim_end_matches('\u{FFFD}');
    if !emit.is_empty() {
        *decoded_len = start + emit.len();
        Some(emit.to_string())
    } else {
        *decoded_len = start;
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
    let mut consecutive_errors: u32 = 0;
    const MAX_CONSECUTIVE_ERRORS: u32 = 3;

    let think_start_id = tokenizer.special_token_id("<think>");
    let think_end_id = tokenizer.special_token_id("</think>");
    let mut decode_cache = TokenDecodeCache::new();

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

        if !trackers.is_empty() {
            let disconnected_ids: Vec<SequenceId> = trackers
                .iter()
                .filter_map(|(&seq_id, tracker)| tracker.tx.is_closed().then_some(seq_id))
                .collect();
            if !disconnected_ids.is_empty() {
                abort_sequences(
                    &mut engine,
                    &mut trackers,
                    disconnected_ids,
                    "response channel closed",
                );
            }
        }

        if engine.has_pending_work() {
            let step_result = {
                let lock = crate::gpu_lock::gpu_lock();
                let _gpu = lock.acquire();
                engine.step()
            };
            match step_result {
                Ok(step) => {
                    consecutive_errors = 0;
                    let mut disconnected_ids: Vec<SequenceId> = Vec::new();
                    for tok in &step.new_tokens {
                        let mut disconnected = false;
                        if let Some(tracker) = trackers.get_mut(&tok.seq_id) {
                            if tracker.first_token_at.is_none() {
                                let ttft_ms = tracker.enqueued_at.elapsed().as_secs_f64() * 1000.0;
                                eprintln!(
                                    "[timing] {} seq={} TTFT: {:.1}ms",
                                    tracker.model_id, tok.seq_id, ttft_ms
                                );
                                tracker.first_token_at = Some(std::time::Instant::now());
                            }
                            tracker.token_count += 1;

                            // Handle thinking tokens.
                            if tracker.in_thinking {
                                let raw = tokenizer
                                    .decode_with_special(&[tok.token])
                                    .unwrap_or_default();

                                let is_think_start =
                                    think_start_id == Some(tok.token) || raw.contains("<think>");
                                if is_think_start {
                                    continue;
                                }

                                let is_think_end =
                                    think_end_id == Some(tok.token) || raw.contains("</think>");
                                if is_think_end {
                                    tracker.in_thinking = false;
                                    continue;
                                }
                            }

                            if tracker.in_thinking {
                                tracker.thinking_ids.push(tok.token);
                                if let Some(text) = prefix_decode_token(
                                    &tokenizer,
                                    &mut decode_cache,
                                    &tracker.thinking_ids,
                                    &mut tracker.thinking_decoded_len,
                                    tok.token,
                                ) && tracker.tx.send(EngineEvent::ReasoningToken(text)).is_err()
                                {
                                    disconnected = true;
                                }
                            } else {
                                // Buffer logprob raw data (if logprobs were requested).
                                if let Some(logprob) = tok.logprob {
                                    tracker.pending_raw_lps.push((
                                        tok.token,
                                        logprob,
                                        tok.top_logprobs.clone(),
                                    ));
                                }
                                tracker.output_ids.push(tok.token);
                                if let Some(text) = prefix_decode_token(
                                    &tokenizer,
                                    &mut decode_cache,
                                    &tracker.output_ids,
                                    &mut tracker.decoded_len,
                                    tok.token,
                                ) {
                                    let drained = std::mem::take(&mut tracker.pending_raw_lps);
                                    let mut logprob_entries: Vec<EngineLogprobEntry> =
                                        Vec::with_capacity(drained.len());
                                    for (tid, lp, top) in drained {
                                        logprob_entries.push(build_logprob_entry(
                                            &tokenizer,
                                            &mut decode_cache,
                                            tid,
                                            lp,
                                            top,
                                        ));
                                    }
                                    if tracker
                                        .tx
                                        .send(EngineEvent::Token {
                                            text,
                                            logprob_entries,
                                        })
                                        .is_err()
                                    {
                                        disconnected = true;
                                    }
                                }
                            }
                        }
                        if disconnected {
                            disconnected_ids.push(tok.seq_id);
                        }
                    }

                    if !disconnected_ids.is_empty() {
                        abort_sequences(
                            &mut engine,
                            &mut trackers,
                            disconnected_ids,
                            "client disconnected",
                        );
                    }

                    for completed in &step.completed {
                        if let Some(mut tracker) = trackers.remove(&completed.id) {
                            // Flush any remaining buffered text.
                            let remaining_text = if !tracker.output_ids.is_empty() {
                                let full =
                                    tokenizer.decode(&tracker.output_ids).unwrap_or_default();
                                if tracker.decoded_len < full.len() {
                                    let start = clamp_to_char_boundary(&full, tracker.decoded_len);
                                    if start < full.len() {
                                        let rest = full[start..].to_string();
                                        tracker.decoded_len = full.len();
                                        rest
                                    } else {
                                        String::new()
                                    }
                                } else {
                                    String::new()
                                }
                            } else {
                                String::new()
                            };

                            // Drain any buffered logprob entries (tokens whose text was still in the buffer).
                            let drained = std::mem::take(&mut tracker.pending_raw_lps);
                            let mut remaining_lp_entries: Vec<EngineLogprobEntry> =
                                Vec::with_capacity(drained.len());
                            for (tid, lp, top) in drained {
                                remaining_lp_entries.push(build_logprob_entry(
                                    &tokenizer,
                                    &mut decode_cache,
                                    tid,
                                    lp,
                                    top,
                                ));
                            }

                            if !remaining_text.is_empty() || !remaining_lp_entries.is_empty() {
                                let _ = tracker.tx.send(EngineEvent::Token {
                                    text: remaining_text,
                                    logprob_entries: remaining_lp_entries,
                                });
                            }

                            // Flush remaining thinking text.
                            if !tracker.thinking_ids.is_empty() {
                                let full =
                                    tokenizer.decode(&tracker.thinking_ids).unwrap_or_default();
                                if tracker.thinking_decoded_len < full.len() {
                                    let start =
                                        clamp_to_char_boundary(&full, tracker.thinking_decoded_len);
                                    let rest = &full[start..];
                                    if !rest.is_empty() {
                                        let _ = tracker
                                            .tx
                                            .send(EngineEvent::ReasoningToken(rest.to_string()));
                                    }
                                }
                            }

                            let total_ms = tracker.enqueued_at.elapsed().as_secs_f64() * 1000.0;
                            let decode_s = tracker
                                .first_token_at
                                .map(|t| t.elapsed().as_secs_f64())
                                .unwrap_or(0.001);
                            let tps = tracker.token_count as f64 / decode_s.max(0.001);
                            eprintln!(
                                "[timing] {} seq={} done: {} tokens, total={:.1}ms, decode={:.1}ms ({:.1} tok/s)",
                                tracker.model_id,
                                completed.id,
                                tracker.token_count,
                                total_ms,
                                decode_s * 1000.0,
                                tps,
                            );
                            let _ = tracker.tx.send(EngineEvent::Finish {
                                finish_reason: completed
                                    .finish_reason
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
                    consecutive_errors += 1;
                    eprintln!("Engine error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}");

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        eprintln!(
                            "[CRITICAL] {consecutive_errors} consecutive engine errors — aborting all sequences"
                        );
                        let aborted_ids = engine.abort_all();
                        for id in aborted_ids {
                            if let Some(tracker) = trackers.remove(&id) {
                                let _ = tracker.tx.send(EngineEvent::Error(e.to_string()));
                                let _ = tracker.tx.send(EngineEvent::StreamEnd);
                            }
                        }
                    } else {
                        let aborted_ids = engine.abort_running();
                        for id in aborted_ids {
                            if let Some(tracker) = trackers.remove(&id) {
                                let _ = tracker.tx.send(EngineEvent::Error(e.to_string()));
                                let _ = tracker.tx.send(EngineEvent::StreamEnd);
                            }
                        }
                    }
                }
            }
            std::thread::yield_now();
        }
    }

    for (_, tracker) in trackers.drain() {
        let _ = tracker
            .tx
            .send(EngineEvent::Error("Model unloaded".to_string()));
        let _ = tracker.tx.send(EngineEvent::StreamEnd);
    }
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

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
            let size_bytes = registry.get(&m.id).map(|e| e.size_bytes).unwrap_or(0);
            let last_used_secs = registry.get(&m.id).map(|e| e.last_used_secs).unwrap_or(0);
            serde_json::json!({
                "id": m.id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
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

async fn get_model(
    State(state): State<Arc<AppState>>,
    AxumPath(model_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let mgr = state.manager.lock().await;
    let models_dir = mgr.models_dir().clone();
    let registry = mgr.list_registry().clone();
    drop(mgr);

    let discovered = crate::models::loader::discover_models(&models_dir);
    let model = discovered.iter().find(|m| m.id == model_id);

    match model {
        Some(m) => {
            let size_bytes = registry.get(&m.id).map(|e| e.size_bytes).unwrap_or(0);
            Ok(Json(serde_json::json!({
                "id": m.id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "architecture": m.architecture,
                "vocab_size": m.vocab_size,
                "num_layers": m.num_layers,
                "size_bytes": size_bytes,
                "size_gb": (size_bytes as f64 / 1_073_741_824.0 * 100.0).round() / 100.0,
            })))
        }
        None => Err(error_response(
            StatusCode::NOT_FOUND,
            format!("The model '{}' does not exist", model_id),
            "invalid_request_error",
        )),
    }
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
        resp["memory_budget_gb"] =
            ((budget as f64 / 1_073_741_824.0 * 100.0).round() / 100.0).into();
        resp["memory_free_bytes"] = budget.saturating_sub(total_loaded).into();
    }

    Json(resp)
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

async fn chat_completions(
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
            eprintln!(
                "[timing] {} manager lock+ready: {:.1}ms",
                model_id,
                t_after_lock.as_secs_f64() * 1000.0
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
            eprintln!(
                "[timing] {} load completed: {:.1}ms",
                model_id,
                t_request.elapsed().as_secs_f64() * 1000.0
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

    eprintln!(
        "[timing] {} template: {:.1}ms, encode: {:.1}ms ({} tokens), total pre-engine: {:.1}ms",
        model_id,
        template_ms,
        encode_ms,
        prompt_len,
        t_request.elapsed().as_secs_f64() * 1000.0
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

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

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

pub struct StartServerArgs {
    pub models_dir: PathBuf,
    pub port: u16,
    pub keep_alive: Duration,
    pub memory_budget_bytes: Option<usize>,
    pub cuda_devices: Vec<usize>,
    pub max_context_len: usize,
    pub kv_quant: crate::common::kv_quant::KvQuantMode,
    pub qjl_quantization: bool,
    pub require_gpu: bool,
}

pub fn start_server(args: StartServerArgs) -> anyhow::Result<()> {
    let StartServerArgs {
        models_dir,
        port,
        keep_alive,
        memory_budget_bytes,
        cuda_devices,
        max_context_len,
        kv_quant,
        qjl_quantization,
        require_gpu,
    } = args;

    if !models_dir.exists() {
        std::fs::create_dir_all(&models_dir)?;
        println!("Created models directory: {}", models_dir.display());
    }
    let available = crate::models::loader::discover_models(&models_dir);
    println!("Models directory: {}", models_dir.display());
    println!(
        "Discovered {} {}:",
        available.len(),
        if available.len() == 1 {
            "model"
        } else {
            "models"
        }
    );
    for m in &available {
        println!("  - {} ({})", m.id, m.architecture);
    }

    let manager = Arc::new(tokio::sync::Mutex::new(ModelManager::new(
        ModelManagerConfig {
            models_dir,
            keep_alive,
            memory_budget_bytes,
            cuda_devices,
            max_context_len,
            kv_quant,
            qjl_quantization,
            require_gpu,
        },
    )));

    let state = Arc::new(AppState {
        manager: Arc::clone(&manager),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/models/running", get(list_running_models))
        .route("/v1/models/{model_id}", get(get_model))
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
        println!("Models:         GET  http://localhost:{}/v1/models", port);
        println!(
            "Running models: GET  http://localhost:{}/v1/models/running",
            port
        );
        println!("Health check:   GET  http://localhost:{}/health", port);
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
        println!(
            "Max context length: {} tokens per sequence\n",
            max_context_len
        );

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        Ok::<_, anyhow::Error>(())
    })?;

    Ok(())
}
