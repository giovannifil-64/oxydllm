use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use tokio::sync::mpsc as tokio_mpsc;

use super::metrics;
use super::types::{EngineEvent, EngineLogprobEntry, IncomingRequest};
use crate::engine::Engine;
use crate::scheduler::sequence::SequenceId;
use crate::tokenizer::Tokenizer;

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

type RawTopLogprobs = Vec<(u32, f32)>;
type PendingRawLogprob = (u32, f32, RawTopLogprobs);

/// GPT-OSS harmony channel protocol: the model emits
/// `<|channel|>NAME<|message|>BODY<|end|>` sequences; analysis/commentary
/// bodies are reasoning, the final channel is user-visible content. Activated
/// only when the tokenizer has the harmony marker tokens.
#[derive(Clone, Copy, PartialEq)]
enum HarmonyState {
    /// Between messages — expecting marker tokens.
    Markers,
    /// After `<|start|>` — plain role-name tokens until the next marker.
    Role,
    /// After `<|channel|>` — collecting the channel name until `<|message|>`.
    Header,
    Body {
        final_channel: bool,
    },
}

struct HarmonyIds {
    channel: u32,
    message: u32,
    end: Option<u32>,
    start: Option<u32>,
}

/// Returns `None` when the token is protocol framing (skip it), otherwise
/// whether the token belongs to reasoning (`true`) or content (`false`).
/// Unknown structure fails open to content.
fn harmony_route(
    state: &mut HarmonyState,
    header_ids: &mut Vec<u32>,
    ids: &HarmonyIds,
    tokenizer: &Tokenizer,
    tok: u32,
) -> Option<bool> {
    let is_end = ids.end == Some(tok) || ids.start == Some(tok);
    match *state {
        HarmonyState::Markers => {
            if tok == ids.channel {
                header_ids.clear();
                *state = HarmonyState::Header;
                return None;
            }
            if ids.start == Some(tok) {
                *state = HarmonyState::Role;
                return None;
            }
            if ids.end == Some(tok) {
                return None;
            }
            if tok == ids.message {
                *state = HarmonyState::Body {
                    final_channel: true,
                };
                return None;
            }
            // Plain token where a marker was expected: content.
            *state = HarmonyState::Body {
                final_channel: true,
            };
            Some(false)
        }
        HarmonyState::Role => {
            if tok == ids.channel {
                header_ids.clear();
                *state = HarmonyState::Header;
            } else if tok == ids.message {
                *state = HarmonyState::Body {
                    final_channel: true,
                };
            } else if ids.end == Some(tok) {
                *state = HarmonyState::Markers;
            }
            // Plain tokens here are the role name — protocol framing, skip.
            None
        }
        HarmonyState::Header => {
            if tok == ids.message {
                let name = tokenizer.decode(header_ids).unwrap_or_default();
                *state = HarmonyState::Body {
                    final_channel: name.contains("final"),
                };
            } else if is_end {
                *state = HarmonyState::Markers;
            } else {
                header_ids.push(tok);
            }
            None
        }
        HarmonyState::Body { final_channel } => {
            if is_end {
                *state = HarmonyState::Markers;
                return None;
            }
            if tok == ids.channel {
                header_ids.clear();
                *state = HarmonyState::Header;
                return None;
            }
            Some(!final_channel)
        }
    }
}

struct SeqTracker {
    tx: tokio_mpsc::UnboundedSender<EngineEvent>,
    request_id: String,
    model_id: String,
    enqueued_at: std::time::Instant,
    first_token_at: Option<std::time::Instant>,
    token_count: usize,
    in_thinking: bool,
    output_ids: Vec<u32>,
    thinking_ids: Vec<u32>,
    decoded_len: usize,
    thinking_decoded_len: usize,
    pending_raw_lps: Vec<PendingRawLogprob>,
    out_stable_end: usize,
    out_stable_text: String,
    think_stable_end: usize,
    think_stable_text: String,
    harmony_state: HarmonyState,
    harmony_header_ids: Vec<u32>,
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

fn enqueue_request(
    req: IncomingRequest,
    engine: &mut Engine,
    trackers: &mut HashMap<SequenceId, SeqTracker>,
) {
    let request_id = req.request_id.clone();
    let model_id = req.model_id.clone();
    let enqueued_at = req.enqueued_at;
    let seq_id = engine.add_request_with_stop(
        req.prompt_tokens,
        req.sampling_params,
        req.max_tokens,
        req.extra_stop_token_ids,
    );
    tracing::debug!(
        request_id = %request_id,
        model_id = %model_id,
        seq_id,
        "request enqueued"
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
            in_thinking: req.enable_thinking,
            output_ids: Vec::new(),
            thinking_ids: Vec::new(),
            decoded_len: 0,
            thinking_decoded_len: 0,
            pending_raw_lps: Vec::new(),
            out_stable_end: 0,
            out_stable_text: String::new(),
            think_stable_end: 0,
            think_stable_text: String::new(),
            harmony_state: HarmonyState::Markers,
            harmony_header_ids: Vec::new(),
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

// Trailing `U+FFFD` is held back so multi-byte UTF-8 split across tokens is
// buffered until the continuation token arrives.
fn emit_suffix(full: &str, decoded_len: &mut usize) -> Option<String> {
    let start = clamp_to_char_boundary(full, *decoded_len);
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

const DECODE_STABLE_LOOKBACK: usize = 4;

fn decode_with_neutral_prefix(
    tokenizer: &Tokenizer,
    ids: &[u32],
    neutral_id: u32,
    neutral_text: &str,
) -> String {
    if ids.is_empty() {
        return String::new();
    }
    let mut prefixed = Vec::with_capacity(ids.len() + 1);
    prefixed.push(neutral_id);
    prefixed.extend_from_slice(ids);
    let with_prefix = tokenizer.decode(&prefixed).unwrap_or_default();
    with_prefix
        .strip_prefix(neutral_text)
        .map(|s| s.to_string())
        .unwrap_or_else(|| tokenizer.decode(ids).unwrap_or_default())
}

fn advance_stable_window(
    tokenizer: &Tokenizer,
    all_ids: &[u32],
    stable_end: &mut usize,
    stable_text: &mut String,
    neutral_id: u32,
    neutral_text: &str,
) {
    let new_stable_end = all_ids.len().saturating_sub(DECODE_STABLE_LOOKBACK);
    if new_stable_end <= *stable_end {
        return;
    }
    if *stable_end == 0 {
        *stable_text = tokenizer
            .decode(&all_ids[..new_stable_end])
            .unwrap_or_default();
    } else {
        let delta_ids = &all_ids[*stable_end..new_stable_end];
        let delta = decode_with_neutral_prefix(tokenizer, delta_ids, neutral_id, neutral_text);
        stable_text.push_str(&delta);
    }
    *stable_end = new_stable_end;
}

fn prefix_decode_incremental(
    tokenizer: &Tokenizer,
    all_ids: &[u32],
    decoded_len: &mut usize,
    stable_end: &mut usize,
    stable_text: &mut String,
    neutral_id: u32,
    neutral_text: &str,
) -> Option<String> {
    advance_stable_window(
        tokenizer,
        all_ids,
        stable_end,
        stable_text,
        neutral_id,
        neutral_text,
    );
    let window_ids = &all_ids[*stable_end..];
    let window_text = if *stable_end == 0 {
        tokenizer.decode(window_ids).unwrap_or_default()
    } else {
        decode_with_neutral_prefix(tokenizer, window_ids, neutral_id, neutral_text)
    };
    let full = format!("{}{}", stable_text, window_text);
    emit_suffix(&full, decoded_len)
}

#[cfg(test)]
mod streaming_decode_tests {
    use super::*;

    fn run_stream(steps: &[&str]) -> String {
        let mut decoded_len = 0;
        let mut out = String::new();
        for full in steps {
            if let Some(s) = emit_suffix(full, &mut decoded_len) {
                out.push_str(&s);
            }
        }
        out
    }

    #[test]
    fn cumulative_emission_matches_final_canonical_text() {
        let steps = [
            "Sono",
            "Sono un",
            "Sono un assistente",
            "Sono un assistente virtuale",
        ];
        assert_eq!(run_stream(&steps), *steps.last().unwrap());
    }

    #[test]
    fn space_only_token_followed_by_g_prefixed_token_does_not_double_space() {
        // Regression: Ġ-only space token followed by a Ġ-prefixed token must
        // not emit a double space.
        let steps = [
            "Sono un assistente",
            "Sono un assistente ",
            "Sono un assistente virtuale",
        ];
        assert_eq!(run_stream(&steps), "Sono un assistente virtuale");
    }

    #[test]
    fn no_character_loss_across_partial_utf8_token_boundary() {
        let steps = ["caf\u{FFFD}", "café", "café 🎉"];
        assert_eq!(run_stream(&steps), "café 🎉");
    }

    #[test]
    fn empty_step_emits_nothing() {
        let steps = ["hello", "hello", "hello world"];
        assert_eq!(run_stream(&steps), "hello world");
    }

    #[test]
    fn idempotent_when_decoded_len_matches_full_len() {
        let mut decoded_len = 5;
        let out = emit_suffix("hello", &mut decoded_len);
        assert!(out.is_none());
        assert_eq!(decoded_len, 5);
    }

    #[test]
    fn clamps_decoded_len_past_end_of_full() {
        let mut decoded_len = 100;
        let out = emit_suffix("short", &mut decoded_len);
        assert!(out.is_none());
        assert_eq!(decoded_len, 5);
    }

    #[test]
    fn clamps_decoded_len_to_char_boundary_inside_multibyte() {
        // decoded_len starts inside é (2-byte UTF-8); must back off to boundary.
        let mut decoded_len = 4;
        let out = emit_suffix("café!", &mut decoded_len);
        assert_eq!(out.as_deref(), Some("é!"));
        assert_eq!(decoded_len, 6);
    }
}

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

    let think_start_id = tokenizer.special_token_id("<think>");
    let think_end_id = tokenizer.special_token_id("</think>");
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

    let neutral_id = tokenizer
        .encode("a")
        .ok()
        .and_then(|ids| ids.into_iter().next())
        .unwrap_or(1);
    let neutral_text = tokenizer.decode(&[neutral_id]).unwrap_or_default();

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
                                tracing::info!(
                                    request_id = %tracker.request_id,
                                    model_id = %tracker.model_id,
                                    seq_id = tok.seq_id,
                                    ttft_ms,
                                    "first token emitted"
                                );
                                metrics::TTFT_HISTOGRAM
                                    .with_label_values(&[&tracker.model_id])
                                    .observe(ttft_ms);
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
                                    None => continue,
                                    Some(r) => r,
                                }
                            } else {
                                if tracker.in_thinking {
                                    let raw = tokenizer
                                        .decode_with_special(&[tok.token])
                                        .unwrap_or_default();

                                    let is_think_start = think_start_id == Some(tok.token)
                                        || raw.contains("<think>");
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
                                tracker.in_thinking
                            };

                            if route_reasoning {
                                tracker.thinking_ids.push(tok.token);
                                if let Some(text) = prefix_decode_incremental(
                                    &tokenizer,
                                    &tracker.thinking_ids,
                                    &mut tracker.thinking_decoded_len,
                                    &mut tracker.think_stable_end,
                                    &mut tracker.think_stable_text,
                                    neutral_id,
                                    &neutral_text,
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
                                if let Some(text) = prefix_decode_incremental(
                                    &tokenizer,
                                    &tracker.output_ids,
                                    &mut tracker.decoded_len,
                                    &mut tracker.out_stable_end,
                                    &mut tracker.out_stable_text,
                                    neutral_id,
                                    &neutral_text,
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
                            let total_ms = (total_ms * 10.0).round() / 10.0;
                            let decode_ms = ((decode_s * 1000.0) * 10.0).round() / 10.0;
                            let tps = (tps * 100.0).round() / 100.0;
                            tracing::info!(
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
        let _ = tracker
            .tx
            .send(EngineEvent::Error("Model unloaded".to_string()));
        let _ = tracker.tx.send(EngineEvent::StreamEnd);
    }
}

#[cfg(test)]
mod metrics_loop_tests {
    use super::*;
    use crate::common::paged::{
        BlockAllocator, DEFAULT_BLOCK_SIZE, PagedKvCache, SharedBlockAllocator,
    };
    use crate::models::traits::BatchModel;
    use crate::sampling::SamplingParams;
    use crate::scheduler::SchedulerConfig;
    use candle_core::{DType, Device, Tensor};
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;
    use tempfile::TempDir;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    // Minimal model that drives the real `engine_loop`: either fails on forward
    // (to exercise the error path) or forces `forced` as the argmax token so a
    // request completes (to exercise the success + prefix-cache path).
    struct StubModel {
        device: Device,
        allocators: Vec<SharedBlockAllocator>,
        forced: u32,
        fail: bool,
    }

    impl StubModel {
        fn new(forced: u32, fail: bool) -> Self {
            let alloc =
                BlockAllocator::new(64, DEFAULT_BLOCK_SIZE, 1, 8, DType::F32, &Device::Cpu, None)
                    .expect("alloc");
            Self {
                device: Device::Cpu,
                allocators: vec![Arc::new(Mutex::new(alloc))],
                forced,
                fail,
            }
        }
        fn failing() -> Self {
            Self::new(0, true)
        }
        fn completing(forced: u32) -> Self {
            Self::new(forced, false)
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
            let forced = (self.forced as usize).min(vocab - 1);
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
        fn max_seq_len(&self) -> usize {
            1024
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

    fn make_tokenizer() -> (Arc<Tokenizer>, TempDir) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let model = WordLevel::builder()
            .vocab(
                [("[UNK]".to_string(), 0u32), ("a".to_string(), 1u32)]
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

    // Run a single request through the real engine_loop to completion (or failure)
    // and join the loop thread, so metrics are fully recorded before assertions.
    fn drive_one_request(
        stub: StubModel,
        engine_model_id: &str,
        request_model_id: &str,
        max_tokens: usize,
    ) {
        let engine = Engine::new_with_stop_controls(
            Box::new(stub),
            SchedulerConfig {
                max_num_sequences: 4,
                max_tokens_per_step: 1024,
            },
            &[],
            &[],
        );
        let (tok, _tmp) = make_tokenizer();
        let (req_tx, req_rx) = tokio_mpsc::channel::<IncomingRequest>(8);
        let (resp_tx, resp_rx) = tokio_mpsc::unbounded_channel::<EngineEvent>();
        let shutdown = Arc::new(AtomicBool::new(false));

        let req = IncomingRequest {
            request_id: "req-test".to_string(),
            prompt_tokens: vec![1, 1, 1],
            sampling_params: SamplingParams::default(),
            max_tokens,
            response_tx: resp_tx,
            model_id: request_model_id.to_string(),
            enqueued_at: std::time::Instant::now(),
            enable_thinking: false,
            extra_stop_token_ids: vec![],
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
        drop(resp_rx);
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
        drive_one_request(StubModel::failing(), model, model, 4);
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
        drive_one_request(StubModel::completing(5), engine_label, request_label, 1);

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
}
