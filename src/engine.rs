//! Batched inference engine: one [`step`](Engine::step) advances every
//! scheduled sequence by one token (or several under speculative decoding).
//!
//! [`Engine`] owns the target [`BatchModel`], an optional draft model, the
//! [`Scheduler`], and the block-level [`PrefixCache`]. Each step:
//! 1. asks the scheduler which sequences to run and splits them into prefill
//!    and decode sets,
//! 2. plans prefill inputs against the prefix cache so cached leading blocks
//!    are skipped,
//! 3. runs one batched forward over all prefill tokens plus one token per
//!    normal-decode sequence (large prefill batches run as two chunks),
//! 4. samples the next token per sequence, registers freshly filled prompt
//!    blocks in the prefix cache, and applies stop rules,
//! 5. runs the greedy speculative cycle for eligible decode sequences, then
//!    retires finished ones.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{Device, Result, Tensor};

use crate::common::block::flush_caches;
use crate::common::paged::{DEFAULT_BLOCK_SIZE, PagedKvCache, SharedBlockAllocator};
use crate::common::prefix_cache::PrefixCache;
use crate::models::traits::BatchModel;
use crate::sampling::{self, SamplingParams};
use crate::scheduler::sequence::*;
use crate::scheduler::*;

/// A token emitted for a sequence during [`Engine::step`], with logprob data
/// when the request asked for it.
pub struct NewToken {
    pub seq_id: SequenceId,
    pub token: u32,
    pub logprob: Option<f32>,
    pub top_logprobs: Vec<(u32, f32)>,
}

/// Result of one [`Engine::step`]: emitted tokens, sequences that finished
/// this step, and prefix-cache hit/miss counters for this step's prefills.
pub struct StepOutput {
    pub new_tokens: Vec<NewToken>,
    pub completed: Vec<CompletedSequence>,
    pub prefix_cache_hits: usize,
    pub prefix_cache_misses: usize,
}

/// Moves every scheduled sequence's KV caches out of the [`Scheduler`] for
/// the duration of a forward pass and guarantees they are moved back.
///
/// [`BatchModel::forward_batch`] needs simultaneous `&mut` access to the cache
/// vectors of many sequences, which the scheduler's storage cannot hand out
/// directly. The caches are taken with `mem::take` on construction and
/// restored either explicitly via [`restore`](Self::restore) or on drop, so a
/// forward error cannot leave sequences with empty caches (leaking their KV
/// blocks).
struct CacheRestoreGuard<'a> {
    scheduler: &'a mut Scheduler,
    prefill_ids: Vec<SequenceId>,
    decode_ids: Vec<SequenceId>,
    cache_vecs: Vec<Vec<PagedKvCache>>,
    restored: bool,
}

impl<'a> CacheRestoreGuard<'a> {
    fn new(
        scheduler: &'a mut Scheduler,
        prefill_ids: &[SequenceId],
        decode_ids: &[SequenceId],
    ) -> Self {
        let mut cache_vecs: Vec<Vec<PagedKvCache>> =
            Vec::with_capacity(prefill_ids.len() + decode_ids.len());

        for &seq_id in prefill_ids {
            let seq = scheduler.get_running_mut(seq_id).unwrap();
            cache_vecs.push(std::mem::take(&mut seq.caches));
        }
        for &seq_id in decode_ids {
            let seq = scheduler.get_running_mut(seq_id).unwrap();
            cache_vecs.push(std::mem::take(&mut seq.caches));
        }

        Self {
            scheduler,
            prefill_ids: prefill_ids.to_vec(),
            decode_ids: decode_ids.to_vec(),
            cache_vecs,
            restored: false,
        }
    }

    /// Mutable slice view over the taken cache vectors, prefills first then
    /// decodes, in scheduling order.
    fn cache_slices(&mut self) -> Vec<&mut [PagedKvCache]> {
        self.cache_vecs
            .iter_mut()
            .map(|v| v.as_mut_slice())
            .collect()
    }

    fn restore_inner(&mut self) {
        if self.restored {
            return;
        }

        for (i, &seq_id) in self.prefill_ids.iter().enumerate() {
            if let Some(seq) = self.scheduler.get_running_mut(seq_id) {
                seq.caches = std::mem::take(&mut self.cache_vecs[i]);
            } else {
                debug_assert!(
                    false,
                    "missing prefill sequence {seq_id} while restoring caches"
                );
            }
        }
        let offset = self.prefill_ids.len();
        for (i, &seq_id) in self.decode_ids.iter().enumerate() {
            if let Some(seq) = self.scheduler.get_running_mut(seq_id) {
                seq.caches = std::mem::take(&mut self.cache_vecs[offset + i]);
            } else {
                debug_assert!(
                    false,
                    "missing decode sequence {seq_id} while restoring caches"
                );
            }
        }

        self.restored = true;
    }

    /// Consumes the guard, restoring the caches immediately instead of at end
    /// of scope.
    fn restore(mut self) {
        self.restore_inner();
    }
}

impl Drop for CacheRestoreGuard<'_> {
    fn drop(&mut self) {
        self.restore_inner();
    }
}

/// Number of tokens the draft model proposes per speculative step.
const SPEC_DRAFT_K: usize = 4;

/// Synchronous inference core: owns the target model, the [`Scheduler`], and
/// the [`PrefixCache`], and advances all admitted sequences one
/// [`step`](Self::step) at a time.
///
/// `draft_model` is the speculative-decoding proposer (`None` disables
/// speculation). `stop_token_ids` and `stop_token_sequences` merge the
/// model's stop tokens with engine-level extras; `allocators` and
/// `block_size` mirror the model's paged-KV layout for prefix-cache
/// registration.
pub struct Engine {
    model: Box<dyn BatchModel>,
    draft_model: Option<Box<dyn BatchModel>>,
    scheduler: Scheduler,
    device: Device,
    stop_token_ids: HashSet<u32>,
    stop_token_sequences: Vec<Vec<u32>>,
    allocators: Vec<SharedBlockAllocator>,
    prefix_cache: PrefixCache,
    block_size: usize,
    pacer: PrefillPacer,
}

/// True when `tokens` ends with any of the configured multi-token stop
/// sequences.
fn has_matching_stop_sequence(tokens: &[u32], stop_sequences: &[Vec<u32>]) -> bool {
    stop_sequences.iter().any(|seq| {
        let n = seq.len();
        n > 0 && tokens.len() >= n && tokens[tokens.len() - n..] == seq[..]
    })
}

/// How long one forward should hold the GPU.
///
/// A forward is submitted as one uninterrupted stretch of GPU work, and while
/// it runs the system compositor cannot draw. macOS kills its window server
/// after forty seconds of that, which a single multi-thousand-token prefill
/// reaches on a laptop GPU. A quarter of a second keeps the desktop responsive
/// with room to spare while still being long enough to amortize a dispatch.
/// A chunk this slow is submitting too much work at once whatever it buys in
/// throughput: the cap stops growing there.
const CHUNK_CEILING: Duration = Duration::from_secs(3);

/// Tokens a chunk will never exceed, so one forward's activations stay bounded
/// however fast the machine is.
///
/// Measured on a 12B checkpoint at 1658 tokens of prompt: 512 tokens a chunk
/// gives 15.5 s to first token and 1024 gives 15.1, so the climb is over well
/// before here, and the layers that materialise their attention scores make a
/// larger chunk a memory risk rather than a gain.
const CHUNK_MAX: usize = 1024;

/// Tokens per forward, climbed toward the throughput the machine actually
/// delivers.
///
/// Splitting a prompt into chunks does not let anything else run: the whole
/// batch belongs to one step either way. What the split buys is bounded
/// activations and a bounded GPU submission, and what it costs is throughput,
/// since a small chunk pays the same per-layer overhead for less work. Chasing a
/// fixed per-chunk latency therefore optimises nothing a caller can see: on a
/// 12B checkpoint that target walked the cap down to forty tokens and spent
/// forty percent of the time to first token on the walk.
///
/// So the cap climbs while the measured tokens per second improve, and holds at
/// the best size it has seen once doubling stops paying. It also stops growing
/// at [`CHUNK_CEILING`] and never passes [`CHUNK_MAX`], and it halves whenever a
/// chunk fails outright, which is what a chunk too large for the device to back
/// looks like.
pub struct PrefillPacer {
    cap: usize,
    best: Option<(usize, f64)>,
    settled: bool,
}

impl Default for PrefillPacer {
    /// Starts at 512 tokens and climbs: a couple of doublings reach the size
    /// this machine likes, and starting there rather than at one token means
    /// even a short prompt runs in one or two chunks.
    fn default() -> Self {
        Self {
            cap: 512,
            best: None,
            settled: false,
        }
    }
}

impl PrefillPacer {
    fn cap(&self) -> usize {
        self.cap
    }

    /// Folds one chunk's measured throughput into the cap.
    ///
    /// Only full chunks are evidence: a short tail says nothing about what the
    /// cap should be.
    fn observe(&mut self, tokens: usize, elapsed: Duration) {
        if tokens < self.cap || self.settled {
            return;
        }
        let rate = tokens as f64 / elapsed.as_secs_f64().max(1e-9);

        match self.best {
            Some((best_tokens, best_rate)) if rate <= best_rate * 1.05 => {
                self.cap = best_tokens;
                self.settled = true;
            }
            _ => {
                self.best = Some((tokens, rate));
                if elapsed >= CHUNK_CEILING || self.cap >= CHUNK_MAX {
                    self.settled = true;
                } else {
                    self.cap = (self.cap * 2).min(CHUNK_MAX);
                }
            }
        }
    }

    /// Halves the cap after a chunk the device could not back, and stops the
    /// climb: whatever the throughput would have been, this size does not run.
    fn shrink(&mut self) -> bool {
        if self.cap <= 32 {
            return false;
        }
        self.cap /= 2;
        self.best = None;
        self.settled = true;
        true
    }
}

/// One forward's worth of a batch: which sequences it carries, how many tokens
/// of each, and which of them end here and so contribute a logits row.
#[derive(Debug, PartialEq, Eq)]
struct ForwardChunk {
    start: usize,
    counts: Vec<usize>,
    seq_base: usize,
    finished: Vec<usize>,
}

impl ForwardChunk {
    fn len(&self) -> usize {
        self.counts.iter().sum()
    }
}

/// Walks a batch, handing out one forward at a time.
///
/// The cap is asked for per chunk rather than fixed up front, so a prompt that
/// turns out slower than expected shrinks its own remaining chunks instead of
/// only the next request's. Sequences keep their order and may span chunks: a
/// prompt longer than the cap is carried across several forwards, which is
/// what bounds the work in any one of them, and a sequence contributes its
/// logits row in the chunk where its last token lands.
struct ChunkPlanner<'a> {
    counts: &'a [usize],
    seq: usize,
    done_in_seq: usize,
    start: usize,
}

impl<'a> ChunkPlanner<'a> {
    fn new(counts: &'a [usize]) -> Self {
        Self {
            counts,
            seq: 0,
            done_in_seq: 0,
            start: 0,
        }
    }

    fn next(&mut self, cap: usize) -> Option<ForwardChunk> {
        if self.seq >= self.counts.len() {
            return None;
        }
        let seq_base = self.seq;
        let start = self.start;
        let mut counts: Vec<usize> = Vec::new();
        let mut finished: Vec<usize> = Vec::new();
        let mut room = cap.max(1);

        while self.seq < self.counts.len() && room > 0 {
            let take = (self.counts[self.seq] - self.done_in_seq).min(room);
            counts.push(take);
            room -= take;
            self.done_in_seq += take;
            if self.done_in_seq == self.counts[self.seq] {
                finished.push(counts.len() - 1);
                self.seq += 1;
                self.done_in_seq = 0;
            }
        }

        self.start += counts.iter().sum::<usize>();
        Some(ForwardChunk {
            start,
            counts,
            seq_base,
            finished,
        })
    }
}

/// Every chunk a batch is split into at a fixed cap. The engine drives the
/// planner directly, asking for a fresh cap per chunk; this is how the tests
/// look at a whole plan at once.
#[cfg(test)]
fn plan_forward_chunks(token_counts: &[usize], cap: usize) -> Vec<ForwardChunk> {
    let mut planner = ChunkPlanner::new(token_counts);
    std::iter::from_fn(|| planner.next(cap)).collect()
}

/// Per-sequence prefill plan: how many leading tokens/blocks the prefix cache
/// already covers, the uncached suffix that must run through the model
/// (`input_tokens`, `uncached_len`), and the total count of full blocks the
/// finished prompt occupies.
struct PrefillInfo {
    seq_id: SequenceId,
    num_cached_tokens: usize,
    num_cached_blocks: usize,
    num_full_blocks_total: usize,
    uncached_len: usize,
    input_tokens: Vec<u32>,
}

/// Flattened model input for one step: token ids and positions for all
/// prefill suffixes followed by one token per decode sequence, with
/// `token_counts` giving each sequence's share.
struct BatchInput {
    all_token_ids: Vec<u32>,
    all_positions: Vec<u32>,
    token_counts: Vec<usize>,
}

/// Engine-level stop criteria: merged stop token ids plus multi-token stop
/// sequences.
struct StopRules<'a> {
    token_ids: &'a HashSet<u32>,
    sequences: &'a [Vec<u32>],
}

impl StopRules<'_> {
    /// True when `token` is a stop token (engine-wide or in the sequence's
    /// extras) or the tail of `all_tokens` matches a stop sequence.
    fn matches(&self, token: u32, seq_extra_ids: &[u32], all_tokens: &[u32]) -> bool {
        self.token_ids.contains(&token)
            || seq_extra_ids.contains(&token)
            || has_matching_stop_sequence(all_tokens, self.sequences)
    }
}

/// Borrow bundle for registering freshly filled prompt blocks in the
/// [`PrefixCache`].
struct PrefixRegistry<'a> {
    cache: &'a mut PrefixCache,
    allocators: &'a [SharedBlockAllocator],
    block_size: usize,
}

impl PrefixRegistry<'_> {
    /// Registers the blocks a prefill filled beyond its cached prefix.
    fn register(
        &mut self,
        all_tokens: &[u32],
        num_cached_blocks: usize,
        new_block_ids: &[Vec<usize>],
    ) {
        self.cache.register(
            all_tokens,
            num_cached_blocks,
            new_block_ids,
            self.allocators,
            self.block_size,
        );
    }
}

/// Builds a [`PrefillInfo`] per prefill sequence, consulting the prefix
/// cache.
///
/// Matched leading blocks are prepopulated into the sequence's caches and
/// their tokens skipped; at least one token is always left uncached so the
/// forward still produces logits to sample from.
fn plan_prefill_inputs(
    scheduler: &mut Scheduler,
    prefix_cache: &mut PrefixCache,
    prefill_ids: &[SequenceId],
    block_size: usize,
) -> Vec<PrefillInfo> {
    let mut infos = Vec::with_capacity(prefill_ids.len());
    for &seq_id in prefill_ids {
        let seq = scheduler.get_running_mut(seq_id).unwrap();
        let (all_tokens, _, caches) = seq.tokens_and_caches();
        let seq_len = all_tokens.len();

        let (mut num_cached_blocks, matched_block_ids) =
            prefix_cache.lookup(all_tokens, block_size);
        let max_cacheable = (seq_len.saturating_sub(1)) / block_size;
        if num_cached_blocks > max_cacheable {
            num_cached_blocks = max_cacheable;
        }
        let num_cached_tokens = num_cached_blocks * block_size;

        if num_cached_blocks > 0 {
            for (layer_idx, cache) in caches.iter_mut().enumerate() {
                for layer_block_ids in matched_block_ids[..num_cached_blocks].iter() {
                    if let Some(&bid) = layer_block_ids.get(layer_idx) {
                        cache.prepopulate_block(bid);
                    }
                }
            }
            for cache in caches.iter_mut() {
                cache.set_num_tokens(num_cached_tokens);
            }
            tracing::debug!(
                seq_id,
                cached_blocks = num_cached_blocks,
                total_blocks = seq_len / block_size,
                skipped_tokens = num_cached_tokens,
                "prefix cache hit"
            );
        }

        let uncached_len = seq_len - num_cached_tokens;
        let input_tokens = all_tokens[num_cached_tokens..].to_vec();
        let num_full_blocks_total = seq_len / block_size;

        infos.push(PrefillInfo {
            seq_id,
            num_cached_tokens,
            num_cached_blocks,
            num_full_blocks_total,
            uncached_len,
            input_tokens,
        });
    }
    infos
}

/// Flattens the planned prefill suffixes and the decode sequences' last
/// tokens into a single [`BatchInput`], prefills first.
fn build_batch_input(
    scheduler: &mut Scheduler,
    prefill_infos: &[PrefillInfo],
    decode_ids: &[SequenceId],
) -> BatchInput {
    let mut all_token_ids: Vec<u32> = Vec::new();
    let mut all_positions: Vec<u32> = Vec::new();
    let mut token_counts: Vec<usize> = Vec::new();

    for info in prefill_infos {
        all_token_ids.extend_from_slice(&info.input_tokens);
        for local_idx in 0..info.uncached_len {
            all_positions.push((info.num_cached_tokens + local_idx) as u32);
        }
        token_counts.push(info.uncached_len);
    }

    for &seq_id in decode_ids {
        let seq = scheduler.get_running_mut(seq_id).unwrap();
        all_token_ids.push(*seq.all_tokens.last().unwrap());
        all_positions.push(seq.num_processed_tokens as u32);
        token_counts.push(1);
    }

    BatchInput {
        all_token_ids,
        all_positions,
        token_counts,
    }
}

/// Runs the batch as one or more forwards and returns one logits row per
/// sequence, in batch order.
///
/// The batch is split by [`ChunkPlanner`] into forwards the GPU can finish
/// promptly, so a long prompt no longer submits one uninterrupted stretch of
/// work; `pacer` learns the size from what the machine measures.
/// Caches are moved out through a [`CacheRestoreGuard`] so they are restored
/// even when a forward fails, and each chunk's staged pool writes are flushed
/// before the next one starts.
///
/// ## Errors
///
/// Propagates tensor-construction and model forward errors.
fn run_forward_pass(
    model: &dyn BatchModel,
    device: &Device,
    scheduler: &mut Scheduler,
    prefill_ids: &[SequenceId],
    decode_ids: &[SequenceId],
    pacer: &mut PrefillPacer,
    batch: BatchInput,
) -> Result<Tensor> {
    let BatchInput {
        all_token_ids,
        all_positions,
        token_counts,
    } = batch;

    let mut cache_guard = CacheRestoreGuard::new(scheduler, prefill_ids, decode_ids);
    let mut cache_slices = cache_guard.cache_slices();

    let combined_result: Result<Tensor> = (|| {
        let mut rows: Vec<Tensor> = Vec::with_capacity(token_counts.len());

        let mut planner = ChunkPlanner::new(&token_counts);
        while let Some(chunk) = planner.next(pacer.cap()) {
            let len = chunk.len();
            let input = Tensor::from_vec(
                all_token_ids[chunk.start..chunk.start + len].to_vec(),
                (1, len),
                device,
            )?;
            let positions = Tensor::from_vec(
                all_positions[chunk.start..chunk.start + len].to_vec(),
                (len,),
                device,
            )?;
            let caches = &mut cache_slices[chunk.seq_base..chunk.seq_base + chunk.counts.len()];

            let started = Instant::now();
            let out = match model.forward_batch_last(&input, &positions, caches, &chunk.counts) {
                Ok(out) => out,
                Err(e) => {
                    if pacer.shrink() {
                        tracing::warn!(
                            tokens = len,
                            new_cap = pacer.cap(),
                            error = %e,
                            "prefill chunk failed; the cap comes down for the next request"
                        );
                    }
                    return Err(e);
                }
            };
            let out = out.squeeze(0)?;
            for &row in &chunk.finished {
                rows.push(out.narrow(0, row, 1)?);
            }
            flush_caches(caches)?;
            // A forward returns once the work is queued, not once it has run,
            // so the chunk has to be drained before it can be timed. Draining
            // is also what keeps the submission bounded: without it the GPU
            // would accumulate every chunk of the prompt as one stretch of
            // work, which is the state this split exists to avoid.
            // A forward returns once the work is queued, not once it has run,
            // so a chunk has to be drained before it can be timed, and draining
            // is also what keeps the submission bounded. Decode steps are
            // exempt: one token per sequence is already short, and the sampler
            // reads the logits back immediately, which drains them anyway.
            if len > chunk.counts.len() {
                device.synchronize()?;
                pacer.observe(len, started.elapsed());
            }
        }

        Tensor::cat(&rows, 0)
    })();

    drop(cache_slices);
    cache_guard.restore();
    combined_result
}

/// Samples the first generated token for each prefill sequence from its last
/// logit row, advances constraint and stop state, and registers the prompt's
/// newly filled blocks in the prefix cache.
///
/// ## Errors
///
/// Propagates logit indexing and sampling failures.
fn sample_prefill_outputs(
    scheduler: &mut Scheduler,
    prefix: &mut PrefixRegistry,
    prefill_infos: &[PrefillInfo],
    batch_logits: &Tensor,
    stop: &StopRules,
    new_tokens: &mut Vec<NewToken>,
) -> Result<()> {
    for (row, info) in prefill_infos.iter().enumerate() {
        let seq_logits = batch_logits.get(row)?;

        let sample_out = {
            let seq = scheduler.get_running(info.seq_id).unwrap();
            let vocab = seq_logits.dims1()?;
            let allowed = seq.constraint.as_ref().map(|c| c.allowed(vocab));
            sampling::sample(
                &seq_logits,
                &seq.sampling_params,
                &seq.all_tokens,
                Some(&seq.token_counts),
                allowed.as_deref(),
            )?
        };
        let next_token = sample_out.token;
        let emit = {
            let seq = scheduler.get_running_mut(info.seq_id).unwrap();
            if let Some(c) = seq.constraint.as_mut() {
                c.advance(next_token);
            }
            seq.append_token(next_token);
            seq.num_processed_tokens = seq.all_tokens.len() - 1;
            seq.phase = SequencePhase::Decode;
            let is_stop = stop.matches(next_token, &seq.extra_stop_token_ids, &seq.all_tokens);
            seq.apply_token(next_token, is_stop)
        };

        if emit {
            new_tokens.push(NewToken {
                seq_id: info.seq_id,
                token: next_token,
                logprob: sample_out.logprob,
                top_logprobs: sample_out.top_logprobs,
            });
        }

        if info.num_full_blocks_total > info.num_cached_blocks {
            let new_block_ids: Vec<Vec<usize>> = {
                let seq = scheduler.get_running_mut(info.seq_id).unwrap();
                (info.num_cached_blocks..info.num_full_blocks_total)
                    .map(|block_idx| {
                        seq.caches
                            .iter()
                            .filter_map(|c| c.block_id_at(block_idx))
                            .collect::<Vec<usize>>()
                    })
                    .collect()
            };
            let seq = scheduler.get_running(info.seq_id).unwrap();
            prefix.register(&seq.all_tokens, info.num_cached_blocks, &new_block_ids);
        }
    }
    Ok(())
}

/// Samples one token per normal-decode sequence from the logit rows after the
/// prefill section, advancing constraint and stop state.
///
/// ## Errors
///
/// Propagates logit indexing and sampling failures.
fn sample_decode_outputs(
    scheduler: &mut Scheduler,
    decode_ids: &[SequenceId],
    batch_logits: &Tensor,
    num_prefills: usize,
    stop: &StopRules,
    new_tokens: &mut Vec<NewToken>,
) -> Result<()> {
    for (i, &seq_id) in decode_ids.iter().enumerate() {
        let seq_logits = batch_logits.get(num_prefills + i)?;
        let sample_out = {
            let seq = scheduler.get_running(seq_id).unwrap();
            let vocab = seq_logits.dims1()?;
            let allowed = seq.constraint.as_ref().map(|c| c.allowed(vocab));
            sampling::sample(
                &seq_logits,
                &seq.sampling_params,
                &seq.all_tokens,
                Some(&seq.token_counts),
                allowed.as_deref(),
            )?
        };
        let next_token = sample_out.token;
        let seq = scheduler.get_running_mut(seq_id).unwrap();
        if let Some(c) = seq.constraint.as_mut() {
            c.advance(next_token);
        }
        seq.append_token(next_token);
        seq.num_processed_tokens = seq.all_tokens.len() - 1;
        let is_stop = stop.matches(next_token, &seq.extra_stop_token_ids, &seq.all_tokens);

        if seq.apply_token(next_token, is_stop) {
            new_tokens.push(NewToken {
                seq_id: seq.id,
                token: next_token,
                logprob: sample_out.logprob,
                top_logprobs: sample_out.top_logprobs,
            });
        }
    }
    Ok(())
}

/// Runs one forward on a single sequence's caches and returns the greedy
/// argmax token id at each input position.
///
/// Greedy speculative decoding only needs argmaxes, so no full logits leave
/// the GPU.
///
/// ## Errors
///
/// Propagates tensor-construction and model forward errors.
fn greedy_forward(
    model: &dyn BatchModel,
    device: &Device,
    caches: &mut [PagedKvCache],
    tokens: &[u32],
    positions: &[u32],
) -> Result<Vec<u32>> {
    let t = tokens.len();
    let input = Tensor::from_vec(tokens.to_vec(), (1, t), device)?;
    let pos = Tensor::from_vec(positions.to_vec(), (t,), device)?;
    let mut slices: Vec<&mut [PagedKvCache]> = vec![caches];
    let logits = model.forward_batch(&input, &pos, &mut slices, &[t])?;
    let argmax = logits.squeeze(0)?.argmax(candle_core::D::Minus1)?;
    argmax.to_vec1::<u32>()
}

/// Greedy speculative decode for the given decode sequences, one at a time.
///
/// Cycle per sequence: the draft proposes [`SPEC_DRAFT_K`] tokens
/// autoregressively; the target verifies `[last, d_1..d_K]` in one forward,
/// yielding K+1 argmaxes where `target_am[i]` is the target's token after
/// `verify[i]` (its prediction of `verify[i+1]`). The longest draft prefix
/// matching the target's argmax is accepted, plus the target's own token at
/// the first divergence (correction) or after the last accepted token
/// (bonus); a stop or length cap drops the remaining accepted tokens. Output
/// is identical to plain greedy decoding.
///
/// Cache invariants: on entry the target cache holds `l - 1` confirmed
/// tokens (all but the last, which is fed this step); the draft cache is
/// lazily brought up to the same confirmed length, covering the first step's
/// prompt and the gap left by a fully accepted draft (`m == K`). After
/// acceptance both caches are rolled back to the new confirmed length minus
/// the last token, and pending speculative verify writes are discarded so
/// decode tokens never enter the prefix-cache block pool.
///
/// ## Errors
///
/// Propagates draft/target forward and cache truncation errors.
fn run_speculative_decode(
    model: &dyn BatchModel,
    draft: &dyn BatchModel,
    device: &Device,
    scheduler: &mut Scheduler,
    decode_ids: &[SequenceId],
    stop: &StopRules,
    new_tokens: &mut Vec<NewToken>,
) -> Result<()> {
    for &seq_id in decode_ids {
        let seq = scheduler.get_running_mut(seq_id).unwrap();
        let l = seq.all_tokens.len();
        let target_have = l - 1;
        let draft_have = seq.draft_caches.first().map_or(0, |c| c.num_tokens());
        if draft_have < target_have {
            let gap: Vec<u32> = seq.all_tokens[draft_have..target_have].to_vec();
            let pos: Vec<u32> = (draft_have as u32..target_have as u32).collect();
            greedy_forward(draft, device, &mut seq.draft_caches, &gap, &pos)?;
        }

        let last = seq.all_tokens[l - 1];
        let mut drafts: Vec<u32> = Vec::with_capacity(SPEC_DRAFT_K);
        let mut cur = last;
        for pos in ((l - 1) as u32..).take(SPEC_DRAFT_K) {
            let out = greedy_forward(draft, device, &mut seq.draft_caches, &[cur], &[pos])?;
            cur = out[0];
            drafts.push(cur);
        }

        let mut verify: Vec<u32> = Vec::with_capacity(SPEC_DRAFT_K + 1);
        verify.push(last);
        verify.extend_from_slice(&drafts);
        let vpos: Vec<u32> = ((l - 1) as u32..(l + SPEC_DRAFT_K) as u32).collect();
        let target_am = greedy_forward(model, device, &mut seq.caches, &verify, &vpos)?;

        let mut m = 0usize;
        while m < SPEC_DRAFT_K && target_am[m] == drafts[m] {
            m += 1;
        }
        let mut accepted: Vec<u32> = drafts[..m].to_vec();
        accepted.push(target_am[m]);

        for tok in accepted {
            seq.append_token(tok);
            seq.num_processed_tokens = seq.all_tokens.len() - 1;
            let is_stop = stop.matches(tok, &seq.extra_stop_token_ids, &seq.all_tokens);
            if seq.apply_token(tok, is_stop) {
                new_tokens.push(NewToken {
                    seq_id: seq.id,
                    token: tok,
                    logprob: None,
                    top_logprobs: Vec::new(),
                });
            } else {
                break;
            }
        }

        let keep = seq.all_tokens.len() - 1;
        for c in &mut seq.caches {
            c.discard_pending();
            c.truncate_to(keep)?;
        }
        for c in &mut seq.draft_caches {
            c.discard_pending();
            c.truncate_to(keep)?;
        }
    }
    Ok(())
}

impl Engine {
    /// Creates an engine over `model`, merging `extra_stop_ids` and
    /// `extra_stop_sequences` (deduplicated) with the model's own stop
    /// tokens.
    ///
    /// The prefix cache is disabled for recurrent (hybrid linear-attention)
    /// models, whose per-sequence state cannot be shared block-wise.
    pub fn new_with_stop_controls(
        model: Box<dyn BatchModel>,
        config: SchedulerConfig,
        extra_stop_ids: &[u32],
        extra_stop_sequences: &[Vec<u32>],
    ) -> Self {
        let allocators: Vec<SharedBlockAllocator> =
            model.allocators().iter().map(Arc::clone).collect();
        let num_layers = model.num_layers();
        let device = model.device().clone();
        let block_size = if allocators.is_empty() {
            DEFAULT_BLOCK_SIZE
        } else {
            allocators[0].lock().unwrap().block_size()
        };
        let mut stop_token_ids: HashSet<u32> = model.stop_token_ids().iter().copied().collect();
        for &id in extra_stop_ids {
            stop_token_ids.insert(id);
        }
        let mut stop_token_sequences: Vec<Vec<u32>> = Vec::new();
        for seq in extra_stop_sequences {
            if !seq.is_empty() && !stop_token_sequences.contains(seq) {
                stop_token_sequences.push(seq.clone());
            }
        }
        let scheduler_allocators: Vec<SharedBlockAllocator> =
            allocators.iter().map(Arc::clone).collect();
        let scheduler = Scheduler::new(config, scheduler_allocators, num_layers);
        let prefix_cache = if model.has_recurrent_state() {
            tracing::info!("recurrent (hybrid linear-attention) model: prefix cache disabled");
            PrefixCache::disabled()
        } else {
            PrefixCache::new(512)
        };
        Self {
            model,
            draft_model: None,
            scheduler,
            device,
            stop_token_ids,
            stop_token_sequences,
            allocators,
            prefix_cache,
            block_size,
            pacer: PrefillPacer::default(),
        }
    }

    /// Enables greedy speculative decoding with `draft` as the proposer.
    ///
    /// The draft must share the target's tokenizer/vocab and run on the same
    /// device. Recurrent (hybrid linear-attention) targets are refused and
    /// the draft ignored: rejected speculative tokens roll the KV cache back
    /// via truncation, which a recurrent state cannot do, silently
    /// corrupting generation.
    pub fn with_draft_model(mut self, draft: Box<dyn BatchModel>) -> Self {
        if self.model.has_recurrent_state() {
            tracing::error!(
                "speculative decoding is not supported for recurrent (hybrid \
                 linear-attention) models; draft model ignored"
            );
            return self;
        }
        let draft_allocators: Vec<SharedBlockAllocator> =
            draft.allocators().iter().map(Arc::clone).collect();
        self.scheduler.set_draft_allocators(draft_allocators);
        self.draft_model = Some(draft);
        self
    }

    /// The device the target model runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Queues a prompt for generation and returns its sequence id.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
    ) -> SequenceId {
        self.scheduler
            .add_request(prompt_tokens, sampling_params, max_tokens)
    }

    /// [`add_request`](Self::add_request) with request-specific extra stop
    /// token ids.
    pub fn add_request_with_stop(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
        extra_stop_token_ids: Vec<u32>,
    ) -> SequenceId {
        self.scheduler.add_request_with_stop(
            prompt_tokens,
            sampling_params,
            max_tokens,
            extra_stop_token_ids,
        )
    }

    /// [`add_request_with_stop`](Self::add_request_with_stop) with a grammar
    /// constraint attached; constrained sequences never join the speculative
    /// cycle (the mask changes the sampled distribution).
    pub fn add_request_constrained(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
        extra_stop_token_ids: Vec<u32>,
        constraint: crate::constrain::JsonConstraint,
    ) -> SequenceId {
        self.scheduler.add_request_full(
            prompt_tokens,
            sampling_params,
            max_tokens,
            extra_stop_token_ids,
            Some(constraint),
        )
    }

    /// Runs one engine iteration and returns the tokens and completions it
    /// produced.
    ///
    /// Scheduled sequences split three ways: prefills and normal decodes
    /// share one batched forward (prefills consult the prefix cache first),
    /// while plain-greedy, unconstrained decode sequences take the
    /// speculative cycle when a draft model is configured. Only plain greedy
    /// can use the (exact) greedy spec cycle; anything with temperature,
    /// penalties, logprobs, bias, or a grammar constraint decodes normally
    /// so its distribution is honored. Without a draft, everything decodes
    /// in the batch.
    ///
    /// ## Errors
    ///
    /// Propagates model forward, sampling, and cache errors.
    pub fn step(&mut self) -> Result<StepOutput> {
        let Engine {
            model,
            draft_model,
            scheduler,
            device,
            stop_token_ids,
            stop_token_sequences,
            allocators,
            prefix_cache,
            block_size,
            pacer,
        } = self;
        let block_size = *block_size;
        let draft = draft_model.as_deref();

        let output = scheduler.schedule(Some(prefix_cache));

        let mut prefill_ids: Vec<SequenceId> = Vec::new();
        let mut decode_ids: Vec<SequenceId> = Vec::new();
        for sched_seq in &output.scheduled {
            match sched_seq.phase {
                SequencePhase::Prefill => prefill_ids.push(sched_seq.id),
                SequencePhase::Decode => decode_ids.push(sched_seq.id),
            }
        }

        let (spec_ids, normal_decode_ids): (Vec<SequenceId>, Vec<SequenceId>) = if draft.is_some() {
            decode_ids.iter().copied().partition(|&id| {
                scheduler
                    .get_running(id)
                    .is_some_and(|s| s.sampling_params.is_plain_greedy() && s.constraint.is_none())
            })
        } else {
            (Vec::new(), decode_ids.clone())
        };

        let decode_for_batch: &[SequenceId] = &normal_decode_ids;
        let has_spec_work = !spec_ids.is_empty();

        let prefill_infos = plan_prefill_inputs(scheduler, prefix_cache, &prefill_ids, block_size);
        let batch = build_batch_input(scheduler, &prefill_infos, decode_for_batch);

        if batch.all_token_ids.is_empty() && !has_spec_work {
            let completed = scheduler.retire_finished();
            return Ok(StepOutput {
                new_tokens: Vec::new(),
                completed,
                prefix_cache_hits: 0,
                prefix_cache_misses: 0,
            });
        }

        let stop = StopRules {
            token_ids: stop_token_ids,
            sequences: stop_token_sequences,
        };
        let mut new_tokens = Vec::new();

        if !batch.all_token_ids.is_empty() {
            let batch_logits = run_forward_pass(
                model.as_ref(),
                device,
                scheduler,
                &prefill_ids,
                decode_for_batch,
                pacer,
                batch,
            )?;
            let mut prefix = PrefixRegistry {
                cache: prefix_cache,
                allocators,
                block_size,
            };
            sample_prefill_outputs(
                scheduler,
                &mut prefix,
                &prefill_infos,
                &batch_logits,
                &stop,
                &mut new_tokens,
            )?;
            sample_decode_outputs(
                scheduler,
                &normal_decode_ids,
                &batch_logits,
                prefill_infos.len(),
                &stop,
                &mut new_tokens,
            )?;
        }

        if let Some(draft) = draft
            && !spec_ids.is_empty()
        {
            run_speculative_decode(
                model.as_ref(),
                draft,
                device,
                scheduler,
                &spec_ids,
                &stop,
                &mut new_tokens,
            )?;
        }

        let completed = scheduler.retire_finished();
        let prefix_cache_hits = prefill_infos
            .iter()
            .filter(|i| i.num_cached_blocks > 0)
            .count();
        let prefix_cache_misses = prefill_infos
            .iter()
            .filter(|i| i.num_cached_blocks == 0)
            .count();
        Ok(StepOutput {
            new_tokens,
            completed,
            prefix_cache_hits,
            prefix_cache_misses,
        })
    }

    /// Aborts every running sequence, freeing its KV blocks; returns their
    /// ids.
    pub fn abort_running(&mut self) -> Vec<SequenceId> {
        self.scheduler.abort_all_running()
    }

    /// Aborts one sequence (running or waiting); returns whether it existed.
    pub fn abort_sequence(&mut self, seq_id: SequenceId) -> bool {
        self.scheduler.abort_sequence(seq_id)
    }

    /// Aborts all running and waiting sequences; returns their ids.
    pub fn abort_all(&mut self) -> Vec<SequenceId> {
        self.scheduler.abort_all()
    }

    /// True while any sequence is waiting or running.
    pub fn has_pending_work(&self) -> bool {
        self.scheduler.has_pending_work()
    }

    /// Number of sequences currently waiting or running.
    pub fn queue_depth(&self) -> usize {
        self.scheduler.queue_depth()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::paged::BlockAllocator;
    use candle_core::DType;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Contract: a prompt longer than one forward's cap is carried across
    /// consecutive forwards instead of being submitted whole, and it yields its
    /// logits row in the chunk where its last token lands.
    ///
    /// Submitting it whole is what let a multi-thousand-token prefill hold the
    /// GPU for tens of seconds, long enough for macOS to kill its window
    /// server.
    #[test]
    fn a_long_prompt_is_carried_across_forwards() {
        let chunks = plan_forward_chunks(&[1000], 256);
        assert_eq!(chunks.len(), 4);
        assert!(
            chunks.iter().all(|c| c.len() <= 256),
            "no forward may exceed the cap"
        );
        assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), 1000);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[3].start, 768);
        assert!(
            chunks[..3].iter().all(|c| c.finished.is_empty()),
            "the sequence ends only once"
        );
        assert_eq!(chunks[3].finished, vec![0]);
    }

    /// Contract: sequences that fit share a forward, so batching survives the
    /// cap, and each one is placed against the right cache.
    #[test]
    fn sequences_that_fit_share_one_forward() {
        let chunks = plan_forward_chunks(&[100, 60, 40], 256);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].counts, vec![100, 60, 40]);
        assert_eq!(chunks[0].seq_base, 0);
        assert_eq!(chunks[0].finished, vec![0, 1, 2]);
    }

    /// Contract: a chunk boundary inside one sequence does not disturb the
    /// sequences after it; the next forward resumes at the right cache and the
    /// token offsets stay contiguous.
    #[test]
    fn a_split_sequence_keeps_the_batch_aligned() {
        let chunks = plan_forward_chunks(&[300, 50], 256);
        assert_eq!(chunks[0].counts, vec![256]);
        assert_eq!(chunks[0].seq_base, 0);
        assert!(chunks[0].finished.is_empty());
        assert_eq!(chunks[1].counts, vec![44, 50]);
        assert_eq!(chunks[1].seq_base, 0);
        assert_eq!(chunks[1].finished, vec![0, 1]);
        assert_eq!(chunks[1].start, 256);
        assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), 350);
    }

    /// Contract: decode steps, one token per sequence, still go out as a single
    /// forward.
    #[test]
    fn a_decode_step_stays_one_forward() {
        let chunks = plan_forward_chunks(&[1; 8], 512);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].finished.len(), 8);
    }

    /// A pseudo-random source with the seed in hand, so a failure reproduces by
    /// rerunning the same seed.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
        }

        fn between(&mut self, lo: usize, hi: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + ((self.0 >> 33) as usize) % (hi - lo + 1)
        }
    }

    /// Contract: splitting a batch into forwards neither loses a token nor
    /// computes one twice, keeps every sequence's tokens in order, and gives
    /// each sequence exactly one logits row.
    ///
    /// The planner is where a long prompt gets cut apart, and a cut in the wrong
    /// place does not crash: it feeds the model a prompt with a hole in it, or
    /// samples from the wrong row, and the answer is merely wrong. Chunk
    /// boundaries land inside sequences by design, so the arithmetic deserves
    /// more than the handful of examples a person thinks to write down.
    #[test]
    fn fuzz_chunking_preserves_the_batch() {
        for seed in 0u64..256 {
            let mut rng = Rng::new(seed);
            let cap = rng.between(1, 64);
            let counts: Vec<usize> = (0..rng.between(1, 8))
                .map(|_| rng.between(1, 200))
                .collect();

            let chunks = plan_forward_chunks(&counts, cap);

            let mut offset = 0usize;
            let mut per_sequence = vec![0usize; counts.len()];
            let mut finished = vec![0usize; counts.len()];

            for chunk in &chunks {
                assert_eq!(
                    chunk.start, offset,
                    "seed {seed}: chunk starts at {} but {offset} tokens came before it",
                    chunk.start
                );
                assert!(
                    chunk.len() <= cap,
                    "seed {seed}: chunk of {} tokens over a cap of {cap}",
                    chunk.len()
                );
                assert!(!chunk.counts.is_empty(), "seed {seed}: empty chunk");
                offset += chunk.len();

                for (i, &n) in chunk.counts.iter().enumerate() {
                    let seq = chunk.seq_base + i;
                    assert!(
                        seq < counts.len(),
                        "seed {seed}: chunk names sequence {seq}"
                    );
                    per_sequence[seq] += n;
                }
                for &row in &chunk.finished {
                    finished[chunk.seq_base + row] += 1;
                }
            }

            assert_eq!(
                offset,
                counts.iter().sum::<usize>(),
                "seed {seed}: chunks cover {offset} of {} tokens",
                counts.iter().sum::<usize>()
            );
            assert_eq!(
                per_sequence, counts,
                "seed {seed}: a sequence was fed the wrong number of tokens"
            );
            assert!(
                finished.iter().all(|&f| f == 1),
                "seed {seed}: sequences ended {finished:?} times, expected once each"
            );
        }
    }

    /// Contract: the cap follows the machine. A chunk that overran the target
    /// shrinks it, a full chunk well inside the target grows it, and a chunk
    /// Contract: the cap climbs while a bigger chunk delivers more tokens per
    /// second, and holds at the size that delivered the most.
    ///
    /// This replaces a cap that chased a fixed per-chunk latency. Splitting a
    /// prompt does not let anything else run, so that target optimised nothing a
    /// caller can see: on a 12B checkpoint it walked the cap down to forty
    /// tokens and spent forty percent of the time to first token doing it.
    #[test]
    fn the_cap_climbs_while_a_bigger_chunk_pays() {
        let mut p = PrefillPacer {
            cap: 64,
            best: None,
            settled: false,
        };
        let start = p.cap();

        p.observe(start, Duration::from_millis(100));
        assert_eq!(p.cap(), start * 2, "a chunk that paid doubles the cap");

        p.observe(start * 2, Duration::from_millis(100));
        assert_eq!(p.cap(), start * 4, "and again while it keeps paying");

        p.observe(start * 4, Duration::from_millis(400));
        assert_eq!(
            p.cap(),
            start * 2,
            "the cap returns to the size that delivered the most"
        );

        for _ in 0..8 {
            p.observe(p.cap(), Duration::from_millis(1));
        }
        assert_eq!(p.cap(), start * 2, "and stays there");
    }

    /// Contract: a chunk that is not full says nothing about the cap.
    #[test]
    fn a_short_tail_does_not_move_the_cap() {
        let mut p = PrefillPacer::default();
        let start = p.cap();
        p.observe(start / 4, Duration::from_millis(1));
        assert_eq!(p.cap(), start);
    }

    /// Contract: the climb stops at a chunk slow enough to be one submission too
    /// many, however good its throughput, and never passes the hard ceiling.
    #[test]
    fn the_climb_stops_at_the_ceiling() {
        let mut p = PrefillPacer::default();
        p.observe(p.cap(), CHUNK_CEILING + Duration::from_millis(1));
        let settled = p.cap();
        p.observe(settled, Duration::from_millis(1));
        assert_eq!(p.cap(), settled, "a chunk that slow ends the climb");

        let mut q = PrefillPacer::default();
        for _ in 0..12 {
            q.observe(q.cap(), Duration::from_millis(1));
        }
        assert!(q.cap() <= CHUNK_MAX, "the cap never passes {CHUNK_MAX}");
    }

    /// Contract: a chunk the device could not back brings the cap down and ends
    /// the climb, so the next request does not repeat it.
    #[test]
    fn a_failed_chunk_brings_the_cap_down() {
        let mut p = PrefillPacer::default();
        let start = p.cap();
        assert!(p.shrink());
        assert_eq!(p.cap(), start / 2);

        p.observe(p.cap(), Duration::from_millis(1));
        assert_eq!(p.cap(), start / 2);

        let mut q = PrefillPacer::default();
        while q.shrink() {}
        assert!(q.cap() <= 32, "shrinking stops rather than reaching zero");
    }

    struct FakeModel {
        device: Device,
        vocab_size: usize,
        stop_tokens: Vec<u32>,
        allocators: Vec<SharedBlockAllocator>,
        num_layers: usize,
        forced_token: u32,
        forward_calls: Arc<AtomicUsize>,
    }

    impl FakeModel {
        fn new(
            stop_tokens: Vec<u32>,
            forced_token: u32,
            allocators: Vec<SharedBlockAllocator>,
        ) -> (Self, Arc<AtomicUsize>) {
            let forward_calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    device: Device::Cpu,
                    vocab_size: 32,
                    stop_tokens,
                    num_layers: allocators.len(),
                    allocators,
                    forced_token,
                    forward_calls: Arc::clone(&forward_calls),
                },
                forward_calls,
            )
        }
    }

    impl BatchModel for FakeModel {
        fn forward_batch(
            &self,
            token_ids: &Tensor,
            _position_ids: &Tensor,
            _seq_caches: &mut [&mut [PagedKvCache]],
            _token_counts: &[usize],
        ) -> Result<Tensor> {
            self.forward_calls.fetch_add(1, Ordering::Relaxed);
            let (_, total_tokens) = token_ids.dims2()?;
            let forced = (self.forced_token as usize).min(self.vocab_size.saturating_sub(1));
            let mut logits = vec![0f32; total_tokens * self.vocab_size];
            for i in 0..total_tokens {
                logits[i * self.vocab_size + forced] = 1.0;
            }
            Tensor::from_vec(logits, (1, total_tokens, self.vocab_size), &self.device)
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn stop_token_ids(&self) -> &[u32] {
            &self.stop_tokens
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn num_layers(&self) -> usize {
            self.num_layers
        }

        fn allocators(&self) -> &[SharedBlockAllocator] {
            &self.allocators
        }
    }

    struct KvFakeModel {
        device: Device,
        vocab_size: usize,
        stop_tokens: Vec<u32>,
        allocators: Vec<SharedBlockAllocator>,
        num_layers: usize,
        forced_token: u32,
    }

    impl KvFakeModel {
        fn new(
            stop_tokens: Vec<u32>,
            forced_token: u32,
            allocators: Vec<SharedBlockAllocator>,
        ) -> Self {
            let num_layers = allocators.len();
            Self {
                device: Device::Cpu,
                vocab_size: 32,
                stop_tokens,
                num_layers,
                allocators,
                forced_token,
            }
        }
    }

    impl BatchModel for KvFakeModel {
        fn forward_batch(
            &self,
            token_ids: &Tensor,
            _position_ids: &Tensor,
            seq_caches: &mut [&mut [PagedKvCache]],
            token_counts: &[usize],
        ) -> Result<Tensor> {
            let (_, total_tokens) = token_ids.dims2()?;
            // append expects shape (batch=1, n_kv_heads=1, seq, head_dim=8)
            for (seq_idx, seq_cache) in seq_caches.iter_mut().enumerate() {
                let n = token_counts[seq_idx];
                let k = Tensor::zeros((1usize, 1usize, n, 8usize), DType::F32, &self.device)?;
                let v = Tensor::zeros((1usize, 1usize, n, 8usize), DType::F32, &self.device)?;
                for layer_cache in seq_cache.iter_mut() {
                    layer_cache.append(&k, &v)?;
                }
            }
            let forced = (self.forced_token as usize).min(self.vocab_size.saturating_sub(1));
            let mut logits = vec![0f32; total_tokens * self.vocab_size];
            for i in 0..total_tokens {
                logits[i * self.vocab_size + forced] = 1.0;
            }
            Tensor::from_vec(logits, (1, total_tokens, self.vocab_size), &self.device)
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn stop_token_ids(&self) -> &[u32] {
            &self.stop_tokens
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn num_layers(&self) -> usize {
            self.num_layers
        }

        fn allocators(&self) -> &[SharedBlockAllocator] {
            &self.allocators
        }
    }

    struct ErrorModel {
        device: Device,
        allocators: Vec<SharedBlockAllocator>,
    }

    impl ErrorModel {
        fn new(allocators: Vec<SharedBlockAllocator>) -> Self {
            Self {
                device: Device::Cpu,
                allocators,
            }
        }
    }

    impl BatchModel for ErrorModel {
        fn forward_batch(
            &self,
            _token_ids: &Tensor,
            _position_ids: &Tensor,
            _seq_caches: &mut [&mut [PagedKvCache]],
            _token_counts: &[usize],
        ) -> Result<Tensor> {
            candle_core::bail!("simulated forward failure")
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

    fn make_allocators(num_layers: usize, num_blocks: usize) -> Vec<SharedBlockAllocator> {
        (0..num_layers)
            .map(|_| {
                Arc::new(Mutex::new(
                    BlockAllocator::new(
                        num_blocks,
                        DEFAULT_BLOCK_SIZE,
                        1,
                        8,
                        DType::F32,
                        &Device::Cpu,
                        None,
                    )
                    .unwrap(),
                ))
            })
            .collect()
    }

    fn test_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 1024,
        }
    }

    #[test]
    fn new_with_stop_tokens_merges_and_deduplicates() {
        let allocators = make_allocators(1, 8);
        let (model, _) = FakeModel::new(vec![2, 4], 0, allocators);
        let engine =
            Engine::new_with_stop_controls(Box::new(model), test_scheduler_config(), &[4, 9], &[]);

        assert!(engine.stop_token_ids.contains(&2));
        assert!(engine.stop_token_ids.contains(&4));
        assert!(engine.stop_token_ids.contains(&9));
        assert_eq!(engine.stop_token_ids.len(), 3);
    }

    #[test]
    fn add_request_and_abort_all_clears_pending_work() {
        let allocators = make_allocators(1, 8);
        let (model, _) = FakeModel::new(vec![2], 3, allocators);
        let mut engine =
            Engine::new_with_stop_controls(Box::new(model), test_scheduler_config(), &[], &[]);

        let id0 = engine.add_request(vec![10, 11], SamplingParams::default(), 4);
        let id1 =
            engine.add_request_with_stop(vec![12, 13], SamplingParams::default(), 4, vec![99]);

        assert!(engine.has_pending_work());
        let mut ids = engine.abort_all();
        ids.sort_unstable();
        assert_eq!(ids, vec![id0, id1]);
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn abort_sequence_clears_specific_request() {
        let allocators = make_allocators(1, 8);
        let (model, _) = FakeModel::new(vec![2], 3, allocators);
        let mut engine =
            Engine::new_with_stop_controls(Box::new(model), test_scheduler_config(), &[], &[]);

        let id0 = engine.add_request(vec![10, 11], SamplingParams::default(), 4);
        let id1 = engine.add_request(vec![20, 21], SamplingParams::default(), 4);

        assert!(engine.abort_sequence(id0));
        assert!(engine.has_pending_work());
        assert!(engine.abort_sequence(id1));
        assert!(!engine.has_pending_work());
        assert!(!engine.abort_sequence(999_999));
    }

    #[test]
    fn step_without_work_returns_empty_and_skips_forward() {
        let allocators = make_allocators(1, 8);
        let (model, forward_calls) = FakeModel::new(vec![2], 3, allocators);
        let mut engine =
            Engine::new_with_stop_controls(Box::new(model), test_scheduler_config(), &[], &[]);

        let out = engine.step().unwrap();
        assert!(out.new_tokens.is_empty());
        assert!(out.completed.is_empty());
        assert_eq!(forward_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn step_completes_sequence_when_model_stop_token_is_sampled() {
        let allocators = make_allocators(1, 8);
        let forced_stop = 7;
        let (model, forward_calls) = FakeModel::new(vec![forced_stop], forced_stop, allocators);
        let mut engine =
            Engine::new_with_stop_controls(Box::new(model), test_scheduler_config(), &[], &[]);

        let seq_id = engine.add_request(vec![1, 2, 3], SamplingParams::default(), 4);
        let out = engine.step().unwrap();

        assert_eq!(forward_calls.load(Ordering::Relaxed), 1);
        assert!(out.new_tokens.is_empty());
        assert_eq!(out.completed.len(), 1);
        assert_eq!(out.completed[0].id, seq_id);
        assert_eq!(out.completed[0].finish_reason.as_deref(), Some("stop"));
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn step_respects_request_specific_stop_tokens() {
        let allocators = make_allocators(1, 8);
        let forced_token = 11;
        let (model, _) = FakeModel::new(Vec::new(), forced_token, allocators);
        let mut engine =
            Engine::new_with_stop_controls(Box::new(model), test_scheduler_config(), &[], &[]);

        let seq_id = engine.add_request_with_stop(
            vec![21, 22],
            SamplingParams::default(),
            4,
            vec![forced_token],
        );
        let out = engine.step().unwrap();

        assert!(out.new_tokens.is_empty());
        assert_eq!(out.completed.len(), 1);
        assert_eq!(out.completed[0].id, seq_id);
        assert_eq!(out.completed[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn stop_sequence_matches_suffix_only() {
        assert!(has_matching_stop_sequence(&[1, 2, 3, 4], &[vec![3, 4]]));
        assert!(!has_matching_stop_sequence(&[1, 2, 3, 4], &[vec![2, 3]]));
        assert!(!has_matching_stop_sequence(&[1, 2], &[vec![1, 2, 3]]));
    }

    #[test]
    fn step_forward_error_propagates() {
        let allocators = make_allocators(1, 8);
        let mut engine = Engine::new_with_stop_controls(
            Box::new(ErrorModel::new(allocators)),
            test_scheduler_config(),
            &[],
            &[],
        );
        engine.add_request(vec![1, 2, 3], SamplingParams::default(), 4);
        assert!(
            engine.step().is_err(),
            "forward error must propagate out of step()"
        );
    }

    #[test]
    fn step_mixes_prefill_and_decode_in_same_step() {
        let allocators = make_allocators(1, 8);
        let forced_token = 5;
        let (model, forward_calls) = FakeModel::new(vec![1], forced_token, allocators);
        let mut engine =
            Engine::new_with_stop_controls(Box::new(model), test_scheduler_config(), &[], &[]);

        let id_a = engine.add_request(vec![10, 11], SamplingParams::default(), 8);
        let out = engine.step().unwrap();
        assert_eq!(forward_calls.load(Ordering::Relaxed), 1);
        assert_eq!(out.new_tokens.len(), 1);
        assert_eq!(out.new_tokens[0].seq_id, id_a);

        let id_b = engine.add_request(vec![20, 21], SamplingParams::default(), 8);
        let out = engine.step().unwrap();
        assert_eq!(
            forward_calls.load(Ordering::Relaxed),
            2,
            "mixed prefill+decode must be batched into one forward call"
        );
        assert_eq!(out.new_tokens.len(), 2);
        let ids: Vec<u64> = out.new_tokens.iter().map(|t| t.seq_id).collect();
        assert!(ids.contains(&id_a), "decode seq A must emit a token");
        assert!(ids.contains(&id_b), "prefill seq B must emit a token");
    }

    #[test]
    fn step_finishes_with_length_when_max_tokens_reached() {
        let allocators = make_allocators(1, 8);
        let forced_token = 5;
        let (model, _) = FakeModel::new(vec![], forced_token, allocators);
        let mut engine =
            Engine::new_with_stop_controls(Box::new(model), test_scheduler_config(), &[], &[]);

        let seq_id = engine.add_request(vec![1, 2, 3], SamplingParams::default(), 1);
        let out = engine.step().unwrap();

        assert_eq!(out.new_tokens.len(), 1);
        assert_eq!(out.new_tokens[0].seq_id, seq_id);
        assert_eq!(out.completed.len(), 1);
        assert_eq!(out.completed[0].id, seq_id);
        assert_eq!(out.completed[0].finish_reason.as_deref(), Some("length"));
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn step_respects_engine_stop_sequences() {
        let allocators = make_allocators(1, 8);
        let forced_token = 11;
        let (model, _) = FakeModel::new(Vec::new(), forced_token, allocators);
        let mut engine = Engine::new_with_stop_controls(
            Box::new(model),
            test_scheduler_config(),
            &[],
            &[vec![forced_token]],
        );

        let seq_id = engine.add_request(vec![21, 22], SamplingParams::default(), 4);
        let out = engine.step().unwrap();

        assert!(out.new_tokens.is_empty());
        assert_eq!(out.completed.len(), 1);
        assert_eq!(out.completed[0].id, seq_id);
        assert_eq!(out.completed[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn engine_prefix_cache_hits_on_repeated_prompt() {
        // 64-token shared prefix = exactly 4 full blocks (DEFAULT_BLOCK_SIZE=16).
        let allocators = make_allocators(1, 256);
        let model = KvFakeModel::new(vec![31], 1, allocators);
        let mut engine = Engine::new_with_stop_controls(
            Box::new(model),
            SchedulerConfig {
                max_num_sequences: 8,
                max_tokens_per_step: 4096,
            },
            &[],
            &[],
        );

        let shared_prefix: Vec<u32> = vec![42u32; 64];

        let first_prompt: Vec<u32> = shared_prefix.iter().copied().chain([0u32]).collect();
        engine.add_request(first_prompt, SamplingParams::default(), 5);
        while engine.has_pending_work() {
            engine.step().unwrap();
        }

        for i in 1u32..8 {
            let prompt: Vec<u32> = shared_prefix.iter().copied().chain([i]).collect();
            engine.add_request(prompt, SamplingParams::default(), 5);
        }

        let mut total_hits = 0usize;
        let mut total_misses = 0usize;
        while engine.has_pending_work() {
            let out = engine.step().unwrap();
            total_hits += out.prefix_cache_hits;
            total_misses += out.prefix_cache_misses;
        }

        assert!(
            total_hits >= 7,
            "expected >=7 prefix cache hits (7 warm requests), got {total_hits}"
        );
        assert_eq!(
            total_misses, 0,
            "expected 0 misses for warm requests, got {total_misses}"
        );
    }
}
