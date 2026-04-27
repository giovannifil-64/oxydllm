use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use tokio::sync::mpsc as tokio_mpsc;

use super::types::{EngineEvent, EngineLogprobEntry, IncomingRequest};
use crate::engine::Engine;
use crate::scheduler::sequence::SequenceId;
use crate::tokenizer::Tokenizer;

const TOKEN_DECODE_CACHE_CAP: usize = 8192;

#[derive(Clone)]
struct DecodedToken {
    text: String,
    bytes: Vec<u8>,
    piece: Option<String>,
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
        let piece = tokenizer.id_to_token(token_id);
        let decoded = DecodedToken {
            bytes: text.as_bytes().to_vec(),
            text,
            piece,
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
            Some(model_id) => {
                tracing::debug!(
                    model_id = %model_id,
                    request_id = seq_id,
                    seq_id,
                    reason = %reason,
                    "request aborted"
                )
            }
            None => {
                tracing::debug!(request_id = seq_id, seq_id, reason = %reason, "request aborted")
            }
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
    tracing::debug!(model_id = %model_id, request_id = seq_id, seq_id, "request enqueued");
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
    let decoded = decode_cache.decode_token(tokenizer, token);
    let mut single = decoded.text;
    if !single.is_empty() && !single.contains('\u{FFFD}') {
        let has_leading_ws = single
            .chars()
            .next()
            .map(char::is_whitespace)
            .unwrap_or(false);

        // Some tokenizers (e.g. SentencePiece) keep word-boundary markers in
        // vocab entries and drop them when decoding a single token. Reinsert
        // a leading space for non-initial tokens to keep incremental decoding
        // consistent with full-sequence decoding.
        if *decoded_len > 0
            && !has_leading_ws
            && decoded
                .piece
                .as_deref()
                .map(|p| p.starts_with('▁') || p.starts_with('Ġ'))
                .unwrap_or(false)
        {
            single.insert(0, ' ');
        }

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
    let gpu_lock = crate::gpu_lock::gpu_lock_for(engine.device());
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
                                    model_id = %tracker.model_id,
                                    request_id = tok.seq_id,
                                    seq_id = tok.seq_id,
                                    ttft_ms,
                                    "first token emitted"
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
                            let total_ms = (total_ms * 10.0).round() / 10.0;
                            let decode_ms = ((decode_s * 1000.0) * 10.0).round() / 10.0;
                            let tps = (tps * 100.0).round() / 100.0;
                            tracing::info!(
                                model_id = %tracker.model_id,
                                request_id = completed.id,
                                seq_id = completed.id,
                                completion_tokens = tracker.token_count,
                                total_ms,
                                decode_ms,
                                tokens_per_second = tps,
                                "request completed"
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
