//! Blocking per-model engine loop: queued requests in, streamed tokens out.
//!
//! Each loaded model runs one [`engine_loop`] on its own OS thread. The loop
//! owns the [`Engine`], pulls [`IncomingRequest`]s from an mpsc channel,
//! steps the engine, and forwards [`EngineEvent`]s to each request's response
//! channel through a per-sequence [`SeqTracker`].
//!
//! Two decoding concerns live here:
//!
//! - Prefix decode: tokenizers with a `Strip` decoder (LLaMA/Phi BPE) lose
//!   inter-word spaces when tokens are decoded one at a time, so an
//!   [`IncrementalDecoder`] per channel re-decodes the accumulated sequence
//!   and emits only the new text suffix.
//! - Channel routing: `<think>` blocks and the GPT-OSS harmony protocol
//!   ([`HarmonyState`]) split tokens between reasoning and content streams.
//!
//! GPU serialization contract: every `engine.step()` runs while holding the
//! per-device lock from [`crate::gpu_lock::gpu_lock_for`], serializing the
//! forward pass against concurrent model loads and other engine loops on the
//! same device; candle's Metal backend has no cross-thread hazard tracking,
//! so data-dependent ops encoded concurrently from two threads can race.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use tokio::sync::mpsc as tokio_mpsc;

use super::metrics;
use super::types::{EngineEvent, EngineLogprobEntry, IncomingRequest};
use crate::common::incremental_decode::{IncrementalDecoder, NeutralPrefix};
use crate::engine::Engine;
use crate::scheduler::sequence::SequenceId;
use crate::tokenizer::Tokenizer;

const TOKEN_DECODE_CACHE_CAP: usize = 8192;

/// Decoded form of a single token id: its text and the raw UTF-8 bytes.
#[derive(Clone)]
struct DecodedToken {
    text: String,
    bytes: Vec<u8>,
}

/// LRU cache of single-token decodes, used when building logprob entries.
///
/// Logprob responses decode the chosen token plus every top-k alternative on
/// each step; caching per-token decodes (up to `TOKEN_DECODE_CACHE_CAP`
/// entries) avoids re-running the tokenizer for the same ids repeatedly.
struct TokenDecodeCache {
    entries: LruCache<u32, DecodedToken>,
}

impl TokenDecodeCache {
    fn new() -> Self {
        Self {
            entries: LruCache::new(NonZeroUsize::new(TOKEN_DECODE_CACHE_CAP).unwrap()),
        }
    }

    /// Returns the cached decode for `token_id`, decoding and caching on miss.
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

/// Builds one [`EngineLogprobEntry`] from a sampled token and its top-k
/// alternatives, decoding token text through `decode_cache`.
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

/// Top-k alternatives for one sampled token, as `(token_id, logprob)` pairs.
type RawTopLogprobs = Vec<(u32, f32)>;

/// A sampled token's `(token_id, logprob, top_k)` held back until its text is
/// emitted, so logprob entries stay aligned with emitted text chunks.
type PendingRawLogprob = (u32, f32, RawTopLogprobs);

/// GPT-OSS harmony channel protocol: the model emits
/// `<|channel|>NAME<|message|>BODY<|end|>` sequences; analysis/commentary
/// bodies are reasoning, the final channel is user-visible content. Activated
/// only when the tokenizer has the harmony marker tokens.
///
/// States: `Markers` sits between messages expecting marker tokens; `Role`
/// (after `<|start|>`) consumes plain role-name tokens as protocol framing
/// until the next marker; `Header` (after `<|channel|>`) collects the channel
/// name until `<|message|>`; `Body` carries whether the current channel is
/// the final (user-visible) one.
#[derive(Clone, Copy, PartialEq)]
enum HarmonyState {
    Markers,
    Role,
    Header,
    Body { final_channel: bool, tool: bool },
}

/// Special-token ids for the harmony protocol markers. `channel` and
/// `message` are required for activation; `end` and `start` are optional in
/// some tokenizer variants.
struct HarmonyIds {
    channel: u32,
    message: u32,
    end: Option<u32>,
    start: Option<u32>,
}

/// What one token means in the harmony protocol.
#[derive(Debug, PartialEq)]
enum HarmonyRoute {
    /// Markers, role names and channel headers: nothing the client sees.
    Framing,
    Reasoning,
    Content,
    /// A commentary header addressed to a function just closed; the body that
    /// follows is the call's arguments and this is the function's name.
    ToolCall(String),
    /// The body of a function call ended.
    ToolCallEnd,
}

/// Unknown structure fails open to content.
fn harmony_route(
    state: &mut HarmonyState,
    header_ids: &mut Vec<u32>,
    ids: &HarmonyIds,
    tokenizer: &Tokenizer,
    tok: u32,
) -> HarmonyRoute {
    let is_end = ids.end == Some(tok) || ids.start == Some(tok);
    match *state {
        HarmonyState::Markers => {
            if tok == ids.channel {
                header_ids.clear();
                *state = HarmonyState::Header;
                return HarmonyRoute::Framing;
            }
            if ids.start == Some(tok) {
                *state = HarmonyState::Role;
                return HarmonyRoute::Framing;
            }
            if ids.end == Some(tok) {
                return HarmonyRoute::Framing;
            }
            *state = HarmonyState::Body {
                final_channel: true,
                tool: false,
            };
            if tok == ids.message {
                HarmonyRoute::Framing
            } else {
                HarmonyRoute::Content
            }
        }
        HarmonyState::Role => {
            if tok == ids.channel {
                header_ids.clear();
                *state = HarmonyState::Header;
            } else if tok == ids.message {
                *state = HarmonyState::Body {
                    final_channel: true,
                    tool: false,
                };
            } else if ids.end == Some(tok) {
                *state = HarmonyState::Markers;
            }
            HarmonyRoute::Framing
        }
        HarmonyState::Header => {
            if tok == ids.message {
                let name = tokenizer.decode(header_ids).unwrap_or_default();
                let recipient = name
                    .split("to=")
                    .nth(1)
                    .and_then(|r| r.split_whitespace().next())
                    .map(|r| r.trim_start_matches("functions.").to_string());
                *state = HarmonyState::Body {
                    final_channel: name.contains("final"),
                    tool: recipient.is_some(),
                };
                return match recipient {
                    Some(function) => HarmonyRoute::ToolCall(function),
                    None => HarmonyRoute::Framing,
                };
            } else if is_end {
                *state = HarmonyState::Markers;
            } else {
                header_ids.push(tok);
            }
            HarmonyRoute::Framing
        }
        HarmonyState::Body {
            final_channel,
            tool,
        } => {
            if is_end || tok == ids.channel {
                if tok == ids.channel {
                    header_ids.clear();
                    *state = HarmonyState::Header;
                } else {
                    *state = HarmonyState::Markers;
                }
                return if tool {
                    HarmonyRoute::ToolCallEnd
                } else {
                    HarmonyRoute::Framing
                };
            }
            if tool || final_channel {
                HarmonyRoute::Content
            } else {
                HarmonyRoute::Reasoning
            }
        }
    }
}

/// Closes the `<tool_call>` block a harmony function call was written into:
/// the body text still held by the decoder, then the brace and tag that
/// complete the JSON the opener began.
fn close_harmony_tool_call(tracker: &mut SeqTracker, tokenizer: &Tokenizer) {
    if !tracker.harmony_tool_open {
        return;
    }
    tracker.harmony_tool_open = false;
    let mut text = tracker
        .output
        .finish(tokenizer, &tracker.output_ids)
        .unwrap_or_default();
    text.push_str("}\n</tool_call>");
    let _ = tracker.tx.send(EngineEvent::Token {
        text,
        logprob_entries: vec![],
    });
}

/// Per-sequence streaming state, mapping engine tokens to one request's
/// response channel.
///
/// Output and reasoning token ids are accumulated (`output_ids`,
/// `thinking_ids`) and re-decoded incrementally through one
/// [`IncrementalDecoder`] per channel (`output`, `thinking`), because
/// decoding tokens one at a time loses inter-word spaces on tokenizers with a
/// `Strip` decoder. Raw logprobs wait in `pending_raw_lps` until text is
/// emitted so entries stay aligned with chunks. `span` is the per-request
/// OpenTelemetry span covering the whole lifecycle and `decode_span` its
/// child covering first token to completion (its start offset from the
/// parent visualizes the TTFT); both close on drop, and `decode_span` is
/// dropped explicitly at completion so it ends before the request span
/// records its outcome fields.
struct SeqTracker {
    tx: tokio_mpsc::UnboundedSender<EngineEvent>,
    request_id: String,
    model_id: String,
    enqueued_at: std::time::Instant,
    first_token_at: Option<std::time::Instant>,
    token_count: usize,
    in_thinking: bool,
    /// True while the reasoning channel's name is still being emitted, for the
    /// models that name it. Gemma 4 opens with `<|channel>thought\n`, and that
    /// name is framing rather than something the model reasoned.
    thinking_header: bool,
    output_ids: Vec<u32>,
    thinking_ids: Vec<u32>,
    output: IncrementalDecoder,
    thinking: IncrementalDecoder,
    pending_raw_lps: Vec<PendingRawLogprob>,
    harmony_state: HarmonyState,
    harmony_header_ids: Vec<u32>,
    /// A harmony function call is being written into the content as a
    /// `<tool_call>` block whose closing is still owed.
    harmony_tool_open: bool,
    /// Whether the request asked for reasoning; a harmony model reasons
    /// regardless, and what it reasoned is dropped when it was not asked for.
    wants_reasoning: bool,
    span: tracing::Span,
    decode_span: Option<tracing::Span>,
}

/// Aborts the given sequences in the engine and drops their trackers,
/// logging each abort with the given reason.
fn abort_sequences(
    engine: &mut Engine,
    trackers: &mut HashMap<SequenceId, SeqTracker>,
    mut seq_ids: Vec<SequenceId>,
    reason: &str,
) {
    seq_ids.sort_unstable();
    seq_ids.dedup();
    for seq_id in seq_ids {
        let info = trackers.remove(&seq_id).map(|t| (t.request_id, t.model_id));
        let _ = engine.abort_sequence(seq_id);
        match info {
            Some((request_id, model_id)) => {
                tracing::debug!(
                    request_id = %request_id,
                    model_id = %model_id,
                    seq_id,
                    reason = %reason,
                    "request aborted"
                )
            }
            None => {
                tracing::debug!(seq_id, reason = %reason, "request aborted")
            }
        }
    }
}

/// Registers `req` with the engine and inserts its [`SeqTracker`].
///
/// A request whose prompt plus completion exceeds what the KV pool holds is
/// answered at once with an error event instead: scheduled, it could never
/// be admitted, and it would block the queue behind it for as long as its
/// timeout allowed.
///
/// The sequence is JSON-constrained when the request asks for it, a token
/// byte table is available, and neither thinking nor harmony is active
/// (those channels would break the byte-level automaton). Also opens the
/// request's lifecycle span, nested under the HTTP-handler span so the trace
/// crosses the HTTP boundary; its empty fields (TTFT, completion tokens,
/// outcome) are recorded as the request progresses.
fn enqueue_request(
    req: IncomingRequest,
    engine: &mut Engine,
    trackers: &mut HashMap<SequenceId, SeqTracker>,
    byte_table: Option<&std::sync::Arc<crate::constrain::TokenByteTable>>,
    harmony: bool,
) {
    let request_id = req.request_id.clone();
    let model_id = req.model_id.clone();
    let enqueued_at = req.enqueued_at;
    let prompt_len = req.prompt_tokens.len();
    let max_tokens = req.max_tokens;
    let capacity = engine.kv_capacity_tokens();
    if prompt_len.saturating_add(max_tokens) > capacity {
        let err = format!(
            "prompt of {prompt_len} tokens plus {max_tokens} completion tokens exceeds the \
             {capacity} tokens the KV cache holds"
        );
        tracing::warn!(
            request_id = %request_id,
            model_id = %model_id,
            error = %err,
            "request refused before scheduling"
        );
        metrics::REQUESTS_TOTAL
            .with_label_values(&[model_id.as_str(), "error"])
            .inc();
        let _ = req.response_tx.send(EngineEvent::Error(err));
        let _ = req.response_tx.send(EngineEvent::StreamEnd);
        return;
    }
    let constraint = match (req.json_mode, byte_table) {
        (Some(spec), Some(table)) if !req.enable_thinking && !req.thinking_open && !harmony => {
            Some(crate::constrain::JsonConstraint::new(
                std::sync::Arc::clone(table),
                spec.mode,
                spec.arena,
            ))
        }
        _ => None,
    };
    let seq_id = match constraint {
        Some(c) => engine.add_request_constrained(
            req.prompt_tokens,
            req.sampling_params,
            req.max_tokens,
            req.extra_stop_token_ids,
            c,
        ),
        None => engine.add_request_with_stop(
            req.prompt_tokens,
            req.sampling_params,
            req.max_tokens,
            req.extra_stop_token_ids,
        ),
    };
    tracing::debug!(
        request_id = %request_id,
        model_id = %model_id,
        seq_id,
        "request enqueued"
    );
    let span = tracing::info_span!(
        parent: &req.parent_span,
        "inference.request",
        request_id = %request_id,
        model_id = %model_id,
        seq_id,
        prompt_tokens = prompt_len,
        max_tokens,
        ttft_ms = tracing::field::Empty,
        completion_tokens = tracing::field::Empty,
        tokens_per_second = tracing::field::Empty,
        finish_reason = tracing::field::Empty,
        outcome = tracing::field::Empty,
    );
    trackers.insert(
        seq_id,
        SeqTracker {
            tx: req.response_tx,
            request_id,
            model_id,
            enqueued_at,
            first_token_at: None,
            token_count: 0,
            in_thinking: req.thinking_open,
            thinking_header: false,
            output_ids: Vec::new(),
            thinking_ids: Vec::new(),
            output: IncrementalDecoder::new(),
            thinking: IncrementalDecoder::new(),
            pending_raw_lps: Vec::new(),
            harmony_state: HarmonyState::Markers,
            harmony_header_ids: Vec::new(),
            harmony_tool_open: false,
            wants_reasoning: req.enable_thinking,
            span,
            decode_span: None,
        },
    );
}

/// Runs a model's blocking request-to-token loop until shutdown, owning the
/// [`Engine`] for the model's whole residency.
///
/// Each iteration drains queued [`IncomingRequest`]s (blocking for the next
/// request when idle), aborts sequences whose response channel closed, then
/// steps the engine and routes every new token through its [`SeqTracker`]:
/// harmony and `<think>` handling split reasoning from content, and prefix
/// decoding turns token ids into text deltas. Completed sequences flush any
/// remaining text and logprobs, record metrics and span fields, and emit
/// `Finish` plus `StreamEnd`.
///
/// `engine.step()` executes under the per-device gpu lock, which serializes
/// the forward pass against concurrent model loads and other engine loops on
/// the same device (candle Metal has no cross-thread hazard tracking for
/// concurrently encoded, data-dependent ops).
///
/// Reaching three consecutive step errors aborts all sequences; fewer abort
/// only the running batch. When `shutdown` is set (model eviction) or the
/// request channel closes, the loop exits and drains remaining trackers with
/// a "Model unloaded" error, dropping the engine and freeing the model.
pub fn engine_loop(
    mut engine: Engine,
    tokenizer: Arc<Tokenizer>,
    mut request_rx: tokio_mpsc::Receiver<IncomingRequest>,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    model_id: String,
) {
    let gpu_lock = crate::gpu_lock::gpu_lock_for(engine.device());
    let mut trackers: HashMap<SequenceId, SeqTracker> = HashMap::new();
    let mut consecutive_errors: u32 = 0;
    const MAX_CONSECUTIVE_ERRORS: u32 = 3;

    let think_start_id = tokenizer
        .special_token_id("<think>")
        .or_else(|| tokenizer.special_token_id("<|channel>"));
    let think_end_id = tokenizer
        .special_token_id("</think>")
        .or_else(|| tokenizer.special_token_id("<channel|>"));
    let harmony_ids = match (
        tokenizer.special_token_id("<|channel|>"),
        tokenizer.special_token_id("<|message|>"),
    ) {
        (Some(channel), Some(message)) => Some(HarmonyIds {
            channel,
            message,
            end: tokenizer.special_token_id("<|end|>"),
            start: tokenizer.special_token_id("<|start|>"),
        }),
        _ => None,
    };
    let mut decode_cache = TokenDecodeCache::new();

    let byte_table = crate::constrain::TokenByteTable::build(&tokenizer);
    if byte_table.is_none() {
        tracing::info!(
            "tokenizer is not byte-level BPE; JSON-constrained decoding unavailable \
             for this model (post-hoc validation still applies)"
        );
    }

    let neutral = NeutralPrefix::new(&tokenizer);

    loop {
        if shutdown.load(std::sync::atomic::Ordering::Acquire) {
            break;
        }

        if engine.has_pending_work() {
            while let Ok(req) = request_rx.try_recv() {
                enqueue_request(
                    req,
                    &mut engine,
                    &mut trackers,
                    byte_table.as_ref(),
                    harmony_ids.is_some(),
                );
            }
        } else {
            match request_rx.blocking_recv() {
                Some(req) => enqueue_request(
                    req,
                    &mut engine,
                    &mut trackers,
                    byte_table.as_ref(),
                    harmony_ids.is_some(),
                ),
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
                let _gpu = gpu_lock.acquire();
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
                                let ttft_ms = (ttft_ms * 10.0).round() / 10.0;
                                tracker.span.record("ttft_ms", ttft_ms);
                                tracing::info!(
                                    parent: &tracker.span,
                                    request_id = %tracker.request_id,
                                    model_id = %tracker.model_id,
                                    seq_id = tok.seq_id,
                                    ttft_ms,
                                    "first token emitted"
                                );
                                metrics::TTFT_HISTOGRAM
                                    .with_label_values(&[&tracker.model_id])
                                    .observe(ttft_ms);
                                tracker.decode_span =
                                    Some(tracing::info_span!(parent: &tracker.span, "decode"));
                                tracker.first_token_at = Some(std::time::Instant::now());
                            }
                            tracker.token_count += 1;

                            let route_reasoning = if let Some(ref hids) = harmony_ids {
                                match harmony_route(
                                    &mut tracker.harmony_state,
                                    &mut tracker.harmony_header_ids,
                                    hids,
                                    &tokenizer,
                                    tok.token,
                                ) {
                                    HarmonyRoute::Framing => continue,
                                    HarmonyRoute::Reasoning if !tracker.wants_reasoning => continue,
                                    HarmonyRoute::Reasoning => true,
                                    HarmonyRoute::Content => false,
                                    HarmonyRoute::ToolCall(function) => {
                                        tracker.harmony_tool_open = true;
                                        let opener = format!(
                                            "<tool_call>\n{{\"name\": \"{function}\", \"arguments\": "
                                        );
                                        if tracker
                                            .tx
                                            .send(EngineEvent::Token {
                                                text: opener,
                                                logprob_entries: vec![],
                                            })
                                            .is_err()
                                        {
                                            disconnected_ids.push(tok.seq_id);
                                        }
                                        continue;
                                    }
                                    HarmonyRoute::ToolCallEnd => {
                                        close_harmony_tool_call(tracker, &tokenizer);
                                        continue;
                                    }
                                }
                            } else {
                                let raw = tokenizer
                                    .decode_with_special(&[tok.token])
                                    .unwrap_or_default();

                                let is_think_start = think_start_id == Some(tok.token)
                                    || raw.contains("<think>")
                                    || raw.contains("<|channel>");
                                if is_think_start {
                                    tracker.thinking_header =
                                        !tracker.in_thinking && raw.contains("<|channel>");
                                    tracker.in_thinking = true;
                                    continue;
                                }
                                if tracker.in_thinking {
                                    let is_think_end = think_end_id == Some(tok.token)
                                        || raw.contains("</think>")
                                        || raw.contains("<channel|>");
                                    if is_think_end {
                                        tracker.in_thinking = false;
                                        tracker.thinking_header = false;
                                        continue;
                                    }
                                    if tracker.thinking_header {
                                        if raw.contains('\n') {
                                            tracker.thinking_header = false;
                                        }
                                        continue;
                                    }
                                }
                                if tracker.in_thinking && !tracker.wants_reasoning {
                                    continue;
                                }
                                tracker.in_thinking
                            };

                            if route_reasoning {
                                tracker.thinking_ids.push(tok.token);
                                if let Some(text) = tracker.thinking.push(
                                    &tokenizer,
                                    &tracker.thinking_ids,
                                    &neutral,
                                ) && tracker.tx.send(EngineEvent::ReasoningToken(text)).is_err()
                                {
                                    disconnected = true;
                                }
                            } else {
                                if let Some(logprob) = tok.logprob {
                                    tracker.pending_raw_lps.push((
                                        tok.token,
                                        logprob,
                                        tok.top_logprobs.clone(),
                                    ));
                                }
                                tracker.output_ids.push(tok.token);
                                if let Some(text) =
                                    tracker
                                        .output
                                        .push(&tokenizer, &tracker.output_ids, &neutral)
                                {
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
                            let remaining_text = tracker
                                .output
                                .finish(&tokenizer, &tracker.output_ids)
                                .unwrap_or_default();

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
                            close_harmony_tool_call(&mut tracker, &tokenizer);

                            if let Some(rest) =
                                tracker.thinking.finish(&tokenizer, &tracker.thinking_ids)
                            {
                                let _ = tracker.tx.send(EngineEvent::ReasoningToken(rest));
                            }

                            let total_ms = tracker.enqueued_at.elapsed().as_secs_f64() * 1000.0;
                            let decode_s = tracker
                                .first_token_at
                                .map(|t| t.elapsed().as_secs_f64())
                                .unwrap_or(0.001);
                            let tps = tracker.token_count as f64 / decode_s.max(0.001);
                            let total_ms = (total_ms * 10.0).round() / 10.0;
                            let decode_ms = ((decode_s * 1000.0) * 10.0).round() / 10.0;
                            let tps = (tps * 100.0).round() / 100.0;
                            let finish_reason =
                                completed.finish_reason.as_deref().unwrap_or("stop");
                            tracker.decode_span = None;
                            tracker
                                .span
                                .record("completion_tokens", tracker.token_count);
                            tracker.span.record("tokens_per_second", tps);
                            tracker.span.record("finish_reason", finish_reason);
                            tracker.span.record("outcome", "ok");
                            tracing::info!(
                                parent: &tracker.span,
                                request_id = %tracker.request_id,
                                model_id = %tracker.model_id,
                                seq_id = completed.id,
                                completion_tokens = tracker.token_count,
                                total_ms,
                                decode_ms,
                                tokens_per_second = tps,
                                "request completed"
                            );
                            metrics::TPS_HISTOGRAM
                                .with_label_values(&[&tracker.model_id])
                                .observe(tps);
                            metrics::REQUESTS_TOTAL
                                .with_label_values(&[tracker.model_id.as_str(), "ok"])
                                .inc();
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

                    if step.prefix_cache_hits + step.prefix_cache_misses > 0 {
                        metrics::PREFIX_CACHE_REQUESTS
                            .with_label_values(&[model_id.as_str(), "hit"])
                            .inc_by(step.prefix_cache_hits as f64);
                        metrics::PREFIX_CACHE_REQUESTS
                            .with_label_values(&[model_id.as_str(), "miss"])
                            .inc_by(step.prefix_cache_misses as f64);
                    }
                }
                Err(e) => {
                    consecutive_errors += 1;
                    tracing::error!(
                        consecutive_errors,
                        max_consecutive_errors = MAX_CONSECUTIVE_ERRORS,
                        error = %e,
                        "engine step failed"
                    );

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        tracing::error!(
                            consecutive_errors,
                            "too many consecutive engine errors, aborting all sequences"
                        );
                        let aborted_ids = engine.abort_all();
                        for id in aborted_ids {
                            if let Some(tracker) = trackers.remove(&id) {
                                metrics::REQUESTS_TOTAL
                                    .with_label_values(&[tracker.model_id.as_str(), "error"])
                                    .inc();
                                tracker.span.record("outcome", "error");
                                tracing::warn!(parent: &tracker.span, error = %e, "request failed");
                                let _ = tracker.tx.send(EngineEvent::Error(e.to_string()));
                                let _ = tracker.tx.send(EngineEvent::StreamEnd);
                            }
                        }
                    } else {
                        let aborted_ids = engine.abort_running();
                        for id in aborted_ids {
                            if let Some(tracker) = trackers.remove(&id) {
                                metrics::REQUESTS_TOTAL
                                    .with_label_values(&[tracker.model_id.as_str(), "error"])
                                    .inc();
                                tracker.span.record("outcome", "error");
                                tracing::warn!(parent: &tracker.span, error = %e, "request failed");
                                let _ = tracker.tx.send(EngineEvent::Error(e.to_string()));
                                let _ = tracker.tx.send(EngineEvent::StreamEnd);
                            }
                        }
                    }
                }
            }
            metrics::QUEUE_DEPTH.set(engine.queue_depth() as f64);
            std::thread::yield_now();
        }
    }

    for (_, tracker) in trackers.drain() {
        metrics::REQUESTS_TOTAL
            .with_label_values(&[tracker.model_id.as_str(), "error"])
            .inc();
        tracker.span.record("outcome", "unloaded");
        let _ = tracker
            .tx
            .send(EngineEvent::Error("Model unloaded".to_string()));
        let _ = tracker.tx.send(EngineEvent::StreamEnd);
    }
}

/// Test doubles shared by the engine-loop and manager tests.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::common::paged::{
        BlockAllocator, DEFAULT_BLOCK_SIZE, PagedKvCache, SharedBlockAllocator,
    };
    use crate::models::traits::BatchModel;
    use candle_core::{DType, Device, Tensor};
    use std::sync::{Arc, Mutex};

    /// Minimal model that drives the real [`engine_loop`](super::engine_loop):
    /// either fails on forward (to exercise the error path) or forces `forced`
    /// as the argmax token so a request completes (to exercise the success and
    /// prefix-cache paths).
    pub(crate) struct StubModel {
        device: Device,
        allocators: Vec<SharedBlockAllocator>,
        /// The argmax token of each forward call in turn; the last one repeats.
        forced: Vec<u32>,
        calls: std::sync::atomic::AtomicUsize,
        fail: bool,
    }

    impl StubModel {
        pub(crate) fn new(forced: u32, fail: bool) -> Self {
            let alloc = BlockAllocator::growing(
                64,
                64,
                DEFAULT_BLOCK_SIZE,
                1,
                8,
                DType::F32,
                &Device::Cpu,
                None,
            )
            .expect("alloc");
            Self {
                device: Device::Cpu,
                allocators: vec![Arc::new(Mutex::new(alloc))],
                forced: vec![forced],
                calls: std::sync::atomic::AtomicUsize::new(0),
                fail,
            }
        }
        pub(crate) fn failing() -> Self {
            Self::new(0, true)
        }
        pub(crate) fn completing(forced: u32) -> Self {
            Self::new(forced, false)
        }
        /// Emits `tokens` one per forward call, the prefill included.
        pub(crate) fn scripted(tokens: Vec<u32>) -> Self {
            let mut stub = Self::new(0, false);
            stub.forced = tokens;
            stub
        }
    }

    impl BatchModel for StubModel {
        fn forward_batch(
            &self,
            token_ids: &Tensor,
            _position_ids: &Tensor,
            _seq_caches: &mut [&mut [PagedKvCache]],
            _token_counts: &[usize],
        ) -> candle_core::Result<Tensor> {
            if self.fail {
                return Err(candle_core::Error::Msg(
                    "stub model forced failure".to_string(),
                ));
            }
            let (_, total_tokens) = token_ids.dims2()?;
            let vocab = self.vocab_size();
            let call = self
                .calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                .min(self.forced.len() - 1);
            let forced = (self.forced[call] as usize).min(vocab - 1);
            let mut logits = vec![0f32; total_tokens * vocab];
            for i in 0..total_tokens {
                logits[i * vocab + forced] = 1.0;
            }
            Tensor::from_vec(logits, (1, total_tokens, vocab), &self.device)
        }
        fn vocab_size(&self) -> usize {
            32
        }
        fn stop_token_ids(&self) -> &[u32] {
            &[]
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn num_layers(&self) -> usize {
            self.allocators.len()
        }
        fn allocators(&self) -> &[SharedBlockAllocator] {
            &self.allocators
        }
    }
}

#[cfg(test)]
mod metrics_loop_tests {
    use super::test_support::StubModel;
    use super::*;
    use crate::common::paged::DEFAULT_BLOCK_SIZE;
    use crate::models::traits::BatchModel;
    use crate::sampling::SamplingParams;
    use crate::scheduler::SchedulerConfig;
    use std::sync::atomic::AtomicBool;
    use tempfile::TempDir;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn make_tokenizer() -> (Arc<Tokenizer>, TempDir) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let model = WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".to_string(), 0u32),
                    ("a".to_string(), 1u32),
                    ("<think>".to_string(), 2u32),
                    ("</think>".to_string(), 3u32),
                    ("b".to_string(), 4u32),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap_or_else(|e| panic!("build wordlevel: {e}"));
        let mut inner = tokenizers::Tokenizer::new(model);
        inner.with_pre_tokenizer(Some(Whitespace {}));
        inner
            .save(tmp.path().join("tokenizer.json"), false)
            .unwrap_or_else(|e| panic!("save tokenizer: {e}"));
        let tok =
            Tokenizer::from_dir(tmp.path().to_str().expect("utf-8 path")).expect("load tokenizer");
        (Arc::new(tok), tmp)
    }

    /// Runs one request through the real `engine_loop` to completion, failure
    /// or refusal, and returns every event the request's channel received;
    /// the loop runs on this thread, so metrics are recorded before the
    /// caller asserts.
    fn drive_one_request(
        stub: StubModel,
        engine_model_id: &str,
        request_model_id: &str,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> Vec<EngineEvent> {
        let (tok, _tmp) = make_tokenizer();
        drive_one_request_with_ids(
            stub,
            tok,
            engine_model_id,
            request_model_id,
            prompt_tokens,
            max_tokens,
            false,
        )
    }

    fn drive_one_request_with(
        stub: StubModel,
        tok: Arc<Tokenizer>,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> Vec<EngineEvent> {
        drive_one_request_with_ids(stub, tok, "m", "m", prompt_tokens, max_tokens, false)
    }

    fn drive_one_request_with_ids(
        stub: StubModel,
        tok: Arc<Tokenizer>,
        engine_model_id: &str,
        request_model_id: &str,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        enable_thinking: bool,
    ) -> Vec<EngineEvent> {
        let engine = Engine::new_with_stop_controls(
            Box::new(stub),
            SchedulerConfig {
                max_num_sequences: 4,
                max_tokens_per_step: 1024,
            },
            &[],
            &[],
        );
        let (req_tx, req_rx) = tokio_mpsc::channel::<IncomingRequest>(8);
        let (resp_tx, mut resp_rx) = tokio_mpsc::unbounded_channel::<EngineEvent>();
        let shutdown = Arc::new(AtomicBool::new(false));

        let req = IncomingRequest {
            request_id: "req-test".to_string(),
            prompt_tokens,
            sampling_params: SamplingParams::default(),
            max_tokens,
            response_tx: resp_tx,
            model_id: request_model_id.to_string(),
            enqueued_at: std::time::Instant::now(),
            enable_thinking,
            thinking_open: false,
            extra_stop_token_ids: vec![],
            json_mode: None,
            parent_span: tracing::Span::none(),
        };
        req_tx.try_send(req).expect("enqueue request");
        // Drop the sender before running the loop: blocking_recv first returns the
        // buffered request, and once all work has drained it returns None so the
        // loop exits. Engine holds a non-Send Box<dyn BatchModel>, so we run the
        // loop on this thread rather than spawning (as production builds it inside
        // its own thread). resp_rx stays alive so the request isn't aborted as a
        // closed channel.
        drop(req_tx);
        engine_loop(engine, tok, req_rx, shutdown, engine_model_id.to_string());
        let mut events = Vec::new();
        while let Ok(event) = resp_rx.try_recv() {
            events.push(event);
        }
        events
    }

    fn harmony_tokenizer() -> (Tokenizer, TempDir) {
        use tokenizers::models::wordlevel::WordLevel;
        let tmp = tempfile::tempdir().expect("tempdir");
        let vocab = [
            "[UNK]",
            "analysis",
            "final",
            "Paris",
            "<|channel|>",
            "<|message|>",
            "<|end|>",
            "<|start|>",
            "assistant",
            "commentary",
            "to=functions.get_weather",
            "{\"city\":\"Rome\"}",
            "<|call|>",
            "<|return|>",
        ];
        let model = WordLevel::builder()
            .vocab(
                vocab
                    .iter()
                    .enumerate()
                    .map(|(i, t)| (t.to_string(), i as u32))
                    .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .expect("wordlevel");
        tokenizers::Tokenizer::new(model)
            .save(tmp.path().join("tokenizer.json"), false)
            .expect("save");
        let specials: serde_json::Map<String, serde_json::Value> = [4, 5, 6, 7, 12, 13]
            .iter()
            .map(|&id| {
                (
                    id.to_string(),
                    serde_json::json!({"content": vocab[id], "special": true}),
                )
            })
            .collect();
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            serde_json::json!({"eos_token": "<|return|>", "added_tokens_decoder": specials})
                .to_string(),
        )
        .expect("write config");
        let tok = Tokenizer::from_dir(tmp.path().to_str().expect("utf-8")).expect("load");
        (tok, tmp)
    }

    /// Contract: in the harmony protocol the analysis body is reasoning, the
    /// final body is content, every marker, role name and channel name is
    /// framing, and a commentary header addressed to a function opens a call
    /// whose body is the arguments, on the token sequences gpt-oss emits.
    #[test]
    fn harmony_routes_analysis_to_reasoning_and_final_to_content() {
        let (tok, _tmp) = harmony_tokenizer();
        let ids = HarmonyIds {
            channel: 4,
            message: 5,
            end: Some(6),
            start: Some(7),
        };
        let mut state = HarmonyState::Markers;
        let mut header = Vec::new();
        let mut route = |t: u32| harmony_route(&mut state, &mut header, &ids, &tok, t);
        let routed: Vec<HarmonyRoute> = [4u32, 1, 5, 3, 6, 7, 8, 4, 2, 5, 3]
            .iter()
            .map(|&t| route(t))
            .collect();
        use HarmonyRoute::*;
        assert_eq!(
            routed,
            vec![
                Framing, Framing, Framing, Reasoning, Framing, Framing, Framing, Framing, Framing,
                Framing, Content
            ]
        );
        let call: Vec<HarmonyRoute> = [4u32, 9, 10, 5, 11, 6].iter().map(|&t| route(t)).collect();
        assert_eq!(
            call,
            vec![
                Framing,
                Framing,
                Framing,
                ToolCall("get_weather".to_string()),
                Content,
                ToolCallEnd
            ]
        );
    }

    /// Contract: a harmony function call reaches the client as the
    /// `<tool_call>` block the route already parses, name and arguments, and
    /// the analysis that preceded it is dropped when reasoning was not asked
    /// for.
    #[test]
    fn a_harmony_function_call_becomes_a_tool_call_block() {
        let (tok, _tmp) = harmony_tokenizer();
        let events = drive_one_request_with(
            StubModel::scripted(vec![4, 1, 5, 3, 6, 4, 9, 10, 5, 11, 6]),
            Arc::new(tok),
            vec![3],
            11,
        );
        let (mut content, mut reasoning) = (String::new(), String::new());
        for event in events {
            match event {
                EngineEvent::Token { text, .. } => content.push_str(&text),
                EngineEvent::ReasoningToken(text) => reasoning.push_str(&text),
                _ => {}
            }
        }
        assert_eq!(
            content,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Rome\"}}\n</tool_call>"
        );
        assert!(reasoning.is_empty(), "{reasoning:?}");
    }

    /// Contract: a thinking block the model opens although thinking is off
    /// never reaches the content, and is dropped from the reply as the
    /// harmony analysis is; asked for, it comes back as reasoning, markers
    /// stripped.
    #[test]
    fn a_thinking_block_is_dropped_or_returned_as_reasoning_as_asked() {
        let (tok, _tmp) = make_tokenizer();
        for (wants, want_reasoning) in [(false, ""), (true, "a")] {
            let events = drive_one_request_with_ids(
                StubModel::scripted(vec![2, 1, 3, 4]),
                Arc::clone(&tok),
                "m",
                "m",
                vec![1],
                4,
                wants,
            );
            let (mut content, mut reasoning) = (String::new(), String::new());
            for event in events {
                match event {
                    EngineEvent::Token { text, .. } => content.push_str(&text),
                    EngineEvent::ReasoningToken(text) => reasoning.push_str(&text),
                    _ => {}
                }
            }
            assert_eq!(
                reasoning.trim(),
                want_reasoning,
                "thinking requested: {wants}"
            );
            assert_eq!(content.trim(), "b", "thinking requested: {wants}");
        }
    }

    fn counter_value(name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        prometheus::gather()
            .into_iter()
            .find(|mf| mf.name() == name)
            .and_then(|mf| {
                mf.metric.iter().find_map(|m| {
                    let matches = labels
                        .iter()
                        .all(|(k, v)| m.label.iter().any(|l| l.name() == *k && l.value() == *v));
                    matches.then(|| m.counter.value())
                })
            })
    }

    // Contract (#2): a request killed by an engine step failure is counted once
    // under status="error". Before the fix only status="ok" was ever recorded.
    #[test]
    fn failed_request_is_counted_as_error_status() {
        let model = "metricstest-error-status-model";
        drive_one_request(StubModel::failing(), model, model, vec![1, 1, 1], 4);
        assert_eq!(
            counter_value(
                "oxydllm_requests_total",
                &[("model", model), ("status", "error")]
            ),
            Some(1.0),
            "an engine step failure must record exactly one error-status request"
        );
    }

    // Contract (#3): prefix-cache counters are labeled with the engine's model id,
    // never an empty string, even when the only in-flight request completes in the
    // same step it prefilled (which drains `trackers` before the cache is recorded).
    #[test]
    fn prefix_cache_uses_engine_model_label_not_empty() {
        let engine_label = "metricstest-engine-label";
        let request_label = "metricstest-request-label";
        drive_one_request(
            StubModel::completing(5),
            engine_label,
            request_label,
            vec![1, 1, 1],
            1,
        );

        let misses = counter_value(
            "oxydllm_prefix_cache_requests_total",
            &[("model", engine_label), ("result", "miss")],
        );
        assert!(
            misses.map(|v| v >= 1.0).unwrap_or(false),
            "cold prefill misses must be labeled with the engine model id, got {misses:?}"
        );
        assert_eq!(
            counter_value(
                "oxydllm_prefix_cache_requests_total",
                &[("model", ""), ("result", "miss")]
            ),
            None,
            "prefix-cache must never record under an empty model label"
        );
        assert_eq!(
            counter_value(
                "oxydllm_requests_total",
                &[("model", request_label), ("status", "ok")]
            ),
            Some(1.0),
            "the successful completion is still counted under the request's model id"
        );
    }

    /// Contract: a request whose prompt and completion cannot fit the KV
    /// pool is refused on arrival with an error event, never scheduled.
    ///
    /// Enqueued, such a prompt is never admitted (or outgrows the pool after
    /// a preemption and is never re-admitted), so it sat at the head of the
    /// queue blocking every later request while the loop spun until the
    /// request timeout.
    #[test]
    fn a_request_the_kv_pool_cannot_hold_is_refused_at_enqueue() {
        let model = "metricstest-refused-model";
        let stub = StubModel::completing(5);
        let capacity = stub.allocators()[0].lock().unwrap().num_total() * DEFAULT_BLOCK_SIZE;
        let events = drive_one_request(stub, model, model, vec![1; capacity - 2], 4);
        assert_eq!(events.len(), 2, "exactly an error and the end of stream");
        assert!(
            matches!(&events[0], EngineEvent::Error(msg) if msg.contains("KV cache")),
            "the refusal names the KV cache"
        );
        assert!(matches!(events[1], EngineEvent::StreamEnd));
        assert_eq!(
            counter_value(
                "oxydllm_requests_total",
                &[("model", model), ("status", "error")]
            ),
            Some(1.0)
        );

        let fits = drive_one_request(
            StubModel::completing(5),
            model,
            model,
            vec![1; capacity - 4],
            4,
        );
        assert!(
            fits.iter().any(|e| matches!(e, EngineEvent::Finish { .. })),
            "a request that exactly fits is served"
        );
    }
}
