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

type RawTopLogprobs = Vec<(u32, f32)>;
type PendingRawLogprob = (u32, f32, RawTopLogprobs);

struct SeqTracker {
    tx: tokio_mpsc::UnboundedSender<EngineEvent>,
    /// UUID from the originating HTTP request. Shared across all N completions
    /// of the same API call. Use this to correlate log lines end-to-end.
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

/// Pure suffix-emission logic given the canonical full decode and the bytes
/// already emitted.  Trailing `U+FFFD` is held back so multi-byte UTF-8
/// split across tokens is buffered until the continuation token arrives.
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

    /// Drive `emit_suffix` with the canonical decode at each step (`steps[i]`
    /// = what `tokenizer.decode(all_ids[..=i])` would return) and return the
    /// concatenation of every emitted chunk.
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
        // Regression: the old fast-path heuristic ("piece starts with Ġ ⇒
        // prepend space") emitted an extra space when a Ġ-only space token
        // was followed by a Ġ-prefixed token, producing "...assistente
        // virtuale" → "...assistente  virtuale" with drift that could later
        // swallow a character (the original "  irtuale" report).  Canonical
        // suffix-emission must reproduce the canonical text exactly.
        let steps = [
            "Sono un assistente",
            "Sono un assistente ",
            "Sono un assistente virtuale",
        ];
        assert_eq!(run_stream(&steps), "Sono un assistente virtuale");
    }

    #[test]
    fn no_character_loss_across_partial_utf8_token_boundary() {
        // Token N decodes to a partial multi-byte UTF-8 (FFFD trailing); the
        // next token completes the codepoint.  The emitter must hold the
        // partial char back and emit the complete sequence on the next step.
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
        // Defensive: if some upstream bug bumps decoded_len past full.len(),
        // the clamp should bound it and emit nothing rather than panic.
        let mut decoded_len = 100;
        let out = emit_suffix("short", &mut decoded_len);
        assert!(out.is_none());
        assert_eq!(decoded_len, 5);
    }

    #[test]
    fn clamps_decoded_len_to_char_boundary_inside_multibyte() {
        // decoded_len starts in the middle of a 2-byte UTF-8 char (é = 2 bytes).
        // Clamp must back off to the previous char boundary.
        let mut decoded_len = 4; // "café" = 5 bytes; offset 4 is inside é
        let out = emit_suffix("café!", &mut decoded_len);
        // Should clamp to 3 (start of é) and emit "é!"
        assert_eq!(out.as_deref(), Some("é!"));
        assert_eq!(decoded_len, 6);
    }
}

pub fn engine_loop(
    mut engine: Engine,
    tokenizer: Arc<Tokenizer>,
    mut request_rx: tokio_mpsc::Receiver<IncomingRequest>,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let gpu_lock = crate::gpu_lock::gpu_lock_for(engine.device());
    let mut trackers: HashMap<SequenceId, SeqTracker> = HashMap::new();
    let mut consecutive_errors: u32 = 0;
    const MAX_CONSECUTIVE_ERRORS: u32 = 3;

    let think_start_id = tokenizer.special_token_id("<think>");
    let think_end_id = tokenizer.special_token_id("</think>");
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
                                // Buffer logprob raw data (if logprobs were requested).
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

                    // Update prefix cache counters from this step's prefill results.
                    if step.prefix_cache_hits + step.prefix_cache_misses > 0 {
                        let model_label = trackers
                            .values()
                            .next()
                            .map(|t| t.model_id.clone())
                            .unwrap_or_default();
                        metrics::PREFIX_CACHE_REQUESTS
                            .with_label_values(&[model_label.as_str(), "hit"])
                            .inc_by(step.prefix_cache_hits as f64);
                        metrics::PREFIX_CACHE_REQUESTS
                            .with_label_values(&[model_label.as_str(), "miss"])
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
            // Update queue depth after each step (reflects post-step state).
            metrics::QUEUE_DEPTH.set(engine.queue_depth() as f64);
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
