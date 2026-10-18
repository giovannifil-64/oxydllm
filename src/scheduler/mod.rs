//! FCFS continuous-batching scheduler over paged KV memory.
//!
//! [`Scheduler`] owns every [`SequenceState`] across its lifecycle (waiting,
//! running, finished) and decides which sequences advance each engine step.
//! [`schedule`](Scheduler::schedule) keeps running decodes going, preempts
//! them back to the waiting queue when no KV block is free, and admits
//! waiting prompts first-come-first-served within the per-step token budget
//! and the free-block supply.

pub mod sequence;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::common::paged::{DEFAULT_BLOCK_SIZE, PagedKvCache, SharedBlockAllocator};
use crate::common::prefix_cache::PrefixCache;
use crate::sampling::SamplingParams;
use sequence::*;

/// Admission limits: at most `max_num_sequences` running concurrently and at
/// most `max_tokens_per_step` fed to the model per step.
pub struct SchedulerConfig {
    pub max_num_sequences: usize,
    pub max_tokens_per_step: usize,
}

/// A sequence selected for the current step, tagged with the phase it runs
/// in.
pub struct ScheduledSequence {
    pub id: SequenceId,
    pub phase: SequencePhase,
}

/// The set of sequences [`Scheduler::schedule`] selected for one engine step.
pub struct SchedulerOutput {
    pub scheduled: Vec<ScheduledSequence>,
}

/// A retired sequence and its finish reason (`"stop"` or `"length"`).
pub struct CompletedSequence {
    pub id: SequenceId,
    pub finish_reason: Option<String>,
}

/// Sequence lifecycle owner and per-step admission controller.
///
/// `waiting` is the FCFS queue and `running` the active set, with
/// `running_index` mapping ids to slots. `allocators` (and
/// `draft_allocators` when speculative decoding is enabled; empty otherwise)
/// are the per-layer KV block allocators new sequences draw their caches
/// from; `block_size` mirrors the allocators'.
pub struct Scheduler {
    config: SchedulerConfig,
    waiting: VecDeque<SequenceState>,
    running: Vec<SequenceState>,
    running_index: HashMap<SequenceId, usize>,
    allocators: Vec<SharedBlockAllocator>,
    draft_allocators: Vec<SharedBlockAllocator>,
    draft_num_layers: usize,
    next_id: SequenceId,
    num_layers: usize,
    block_size: usize,
}

impl Scheduler {
    /// Creates an empty scheduler drawing KV blocks from `allocators` (one
    /// per model layer).
    pub fn new(
        config: SchedulerConfig,
        allocators: Vec<SharedBlockAllocator>,
        num_layers: usize,
    ) -> Self {
        let block_size = if allocators.is_empty() {
            DEFAULT_BLOCK_SIZE
        } else {
            allocators[0].lock().unwrap().block_size()
        };
        Self {
            config,
            waiting: VecDeque::new(),
            running: Vec::new(),
            running_index: HashMap::new(),
            allocators,
            draft_allocators: Vec::new(),
            draft_num_layers: 0,
            next_id: 0,
            num_layers,
            block_size,
        }
    }

    /// Enable speculative decoding by registering the draft model's per-layer
    /// allocators; subsequently-added sequences get matching `draft_caches`.
    pub fn set_draft_allocators(&mut self, allocators: Vec<SharedBlockAllocator>) {
        self.draft_num_layers = allocators.len();
        self.draft_allocators = allocators;
    }

    fn push_to_running(&mut self, seq: SequenceState) {
        let idx = self.running.len();
        self.running_index.insert(seq.id, idx);
        self.running.push(seq);
    }

    /// Removes the sequence at `vec_idx` via `swap_remove`, fixing the moved
    /// sequence's index entry.
    fn remove_from_running(&mut self, vec_idx: usize) -> SequenceState {
        let seq_id = self.running[vec_idx].id;
        self.running_index.remove(&seq_id);
        let seq = self.running.swap_remove(vec_idx);
        if vec_idx < self.running.len() {
            self.running_index.insert(self.running[vec_idx].id, vec_idx);
        }
        seq
    }

    /// Queues a prompt; returns the new sequence id.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
    ) -> SequenceId {
        self.add_request_full(prompt_tokens, sampling_params, max_tokens, None)
    }

    /// Queues a prompt with every per-request control: sampling parameters,
    /// generation cap, and an optional grammar constraint.
    ///
    /// The sequence starts `Waiting` in `Prefill` phase with fresh per-layer
    /// caches (target, plus draft when speculative decoding is enabled) and
    /// prompt token counts prebuilt for the penalty samplers.
    pub fn add_request_full(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
        constraint: Option<crate::constrain::JsonConstraint>,
    ) -> SequenceId {
        let id = self.next_id;
        self.next_id += 1;

        let mut token_counts: HashMap<u32, u32> = HashMap::new();
        for &tok in &prompt_tokens {
            *token_counts.entry(tok).or_insert(0) += 1;
        }

        let caches = (0..self.num_layers)
            .map(|i| PagedKvCache::new(Arc::clone(&self.allocators[i])))
            .collect();

        let draft_caches = (0..self.draft_num_layers)
            .map(|i| PagedKvCache::new(Arc::clone(&self.draft_allocators[i])))
            .collect();

        let seq = SequenceState {
            id,
            num_generated: 0,
            all_tokens: prompt_tokens,
            token_counts,
            sampling_params,
            status: SequenceStatus::Waiting,
            phase: SequencePhase::Prefill,
            caches,
            draft_caches,
            num_processed_tokens: 0,
            max_tokens,
            finish_reason: None,
            constraint,
        };

        self.waiting.push_back(seq);
        id
    }

    /// Free blocks in the layer-0 allocator (all layers allocate in
    /// lockstep).
    fn num_free_blocks(&self) -> usize {
        if self.allocators.is_empty() {
            return 0;
        }
        self.allocators[0].lock().unwrap().num_free()
    }

    /// KV blocks a waiting prompt still needs, discounting blocks the prefix
    /// cache would supply (capped so at least one token always runs).
    fn blocks_needed_for_prefill(
        &self,
        seq: &SequenceState,
        prefix_cache: Option<&PrefixCache>,
    ) -> usize {
        let total_tokens = seq.all_tokens.len();
        let total_blocks = total_tokens.div_ceil(self.block_size);
        let cached = prefix_cache.map_or(0, |pc| {
            let max_cacheable = total_tokens.saturating_sub(1) / self.block_size;
            pc.count_cached_blocks(&seq.all_tokens, self.block_size)
                .min(max_cacheable)
        });
        total_blocks.saturating_sub(cached)
    }

    /// True when the sequence's next decode token lands on a fresh block
    /// boundary.
    fn decode_needs_new_block(&self, seq: &SequenceState) -> bool {
        seq.num_processed_tokens.is_multiple_of(self.block_size) && seq.num_processed_tokens > 0
    }

    /// Selects the sequences to run this step.
    ///
    /// Running decodes are scheduled first (one token each). A decode
    /// sitting on a block boundary needs a free block; when none is
    /// available the sequence is preempted: its caches are cleared, its
    /// progress reset, and it is pushed back to the front of the waiting
    /// queue to re-prefill later. Waiting prompts are then admitted FCFS
    /// while the running-set, token-budget, and free-block limits allow;
    /// admission stops at the first prompt that does not fit.
    ///
    /// Every block a scheduled decode will take is reserved as it is
    /// scheduled, and admission counts only what is left: the forward takes
    /// the blocks the scheduler promised, so a step never promises one block
    /// twice. Before a sequence is preempted or a prompt turned away for want
    /// of blocks, the prefix cache gives up its least recently used entries,
    /// whose blocks it alone holds; the entries the prompt itself reuses are
    /// refreshed first so they go last.
    ///
    /// The token budget bundles work into a step, it does not cap what a
    /// sequence may be: a prompt longer than a whole step's budget is admitted
    /// as the only prefill of its step rather than deferred, because admission
    /// is first-come-first-served and deferring it would starve both it and
    /// every request queued behind it. Running decodes still ride along, one
    /// token each. Callers therefore see steps that exceed
    /// [`SchedulerConfig::max_tokens_per_step`] when a single prompt does.
    pub fn schedule(&mut self, mut prefix_cache: Option<&mut PrefixCache>) -> SchedulerOutput {
        let mut scheduled = Vec::new();
        let mut budget = self.config.max_tokens_per_step;

        let mut seqs_needing_block: Vec<SequenceId> = Vec::new();
        for seq in self.running.iter() {
            if self.decode_needs_new_block(seq) {
                seqs_needing_block.push(seq.id);
            } else if budget > 0 {
                scheduled.push(ScheduledSequence {
                    id: seq.id,
                    phase: SequencePhase::Decode,
                });
                budget -= 1;
            }
        }

        let mut reserved = 0usize;
        for &seq_id in &seqs_needing_block {
            if let Some(pc) = prefix_cache.as_deref_mut() {
                while self.num_free_blocks() <= reserved && pc.evict_lru(&self.allocators) {}
            }
            if self.num_free_blocks() > reserved {
                if budget > 0 {
                    scheduled.push(ScheduledSequence {
                        id: seq_id,
                        phase: SequencePhase::Decode,
                    });
                    budget -= 1;
                    reserved += 1;
                }
            } else {
                let idx = *self.running_index.get(&seq_id).unwrap();
                let mut seq = self.remove_from_running(idx);
                tracing::warn!(
                    seq_id = seq.id,
                    "sequence preempted due to memory pressure; KV cache cleared and re-queued"
                );
                seq.clear_caches();
                seq.num_processed_tokens = 0;
                seq.phase = SequencePhase::Prefill;
                seq.status = SequenceStatus::Waiting;
                self.waiting.push_front(seq);
            }
        }

        let mut free_blocks = self.num_free_blocks().saturating_sub(reserved);
        let mut prefill_scheduled = false;

        while self.running.len() < self.config.max_num_sequences && budget > 0 {
            let seq = match self.waiting.front() {
                Some(s) => s,
                None => break,
            };

            let mut blocks_needed = self.blocks_needed_for_prefill(seq, prefix_cache.as_deref());
            if blocks_needed > free_blocks
                && let Some(pc) = prefix_cache.as_deref_mut()
            {
                pc.touch(&seq.all_tokens, self.block_size);
                while blocks_needed > free_blocks && pc.evict_lru(&self.allocators) {
                    free_blocks = self.num_free_blocks().saturating_sub(reserved);
                    blocks_needed = self.blocks_needed_for_prefill(seq, Some(pc));
                }
            }
            if blocks_needed > free_blocks {
                break;
            }

            let tokens_needed = seq.all_tokens.len();
            if tokens_needed > budget && prefill_scheduled {
                break;
            }

            let mut seq = self.waiting.pop_front().unwrap();
            seq.status = SequenceStatus::Running;
            seq.phase = SequencePhase::Prefill;

            scheduled.push(ScheduledSequence {
                id: seq.id,
                phase: SequencePhase::Prefill,
            });
            budget = budget.saturating_sub(tokens_needed);
            prefill_scheduled = true;
            free_blocks = free_blocks.saturating_sub(blocks_needed);

            self.push_to_running(seq);
        }

        SchedulerOutput { scheduled }
    }

    /// The running sequence with this id, if any.
    pub fn get_running(&self, seq_id: SequenceId) -> Option<&SequenceState> {
        let idx = *self.running_index.get(&seq_id)?;
        self.running.get(idx)
    }

    /// Mutable access to the running sequence with this id, if any.
    pub fn get_running_mut(&mut self, seq_id: SequenceId) -> Option<&mut SequenceState> {
        let idx = *self.running_index.get(&seq_id)?;
        self.running.get_mut(idx)
    }

    /// Removes every `Finished` running sequence, frees its KV blocks, and
    /// reports the completions.
    pub fn retire_finished(&mut self) -> Vec<CompletedSequence> {
        let finished: Vec<SequenceId> = self
            .running
            .iter()
            .filter(|s| s.status == SequenceStatus::Finished)
            .map(|s| s.id)
            .collect();

        let mut completed = Vec::with_capacity(finished.len());
        for seq_id in finished {
            let idx = *self.running_index.get(&seq_id).unwrap();
            let mut seq = self.remove_from_running(idx);
            seq.clear_caches();
            completed.push(CompletedSequence {
                id: seq.id,
                finish_reason: seq.finish_reason.clone(),
            });
        }
        completed
    }

    /// Removes all running sequences and frees their blocks; returns their
    /// ids.
    pub fn abort_all_running(&mut self) -> Vec<SequenceId> {
        let ids: Vec<SequenceId> = self.running.iter().map(|s| s.id).collect();
        for mut seq in self.running.drain(..) {
            seq.clear_caches();
        }
        self.running_index.clear();
        ids
    }

    /// Removes one sequence from running or waiting, freeing its blocks;
    /// returns whether it was found.
    pub fn abort_sequence(&mut self, seq_id: SequenceId) -> bool {
        if let Some(&idx) = self.running_index.get(&seq_id) {
            let mut seq = self.remove_from_running(idx);
            seq.clear_caches();
            return true;
        }

        if let Some(wait_idx) = self.waiting.iter().position(|s| s.id == seq_id)
            && let Some(mut seq) = self.waiting.remove(wait_idx)
        {
            seq.clear_caches();
            return true;
        }

        false
    }

    /// Aborts everything, running and waiting; returns all ids.
    pub fn abort_all(&mut self) -> Vec<SequenceId> {
        let mut ids = self.abort_all_running();
        for mut seq in self.waiting.drain(..) {
            ids.push(seq.id);
            seq.clear_caches();
        }
        ids
    }

    /// True while any sequence is waiting or running.
    pub fn has_pending_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    /// Number of sequences currently waiting or running.
    pub fn queue_depth(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    #[cfg(test)]
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    #[cfg(test)]
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::paged::BlockAllocator;
    use candle_core::{DType, Device};
    use std::sync::Mutex;

    fn make_allocators(num_layers: usize, num_blocks: usize) -> Vec<SharedBlockAllocator> {
        (0..num_layers)
            .map(|_| {
                Arc::new(Mutex::new(
                    BlockAllocator::growing(
                        num_blocks,
                        num_blocks,
                        DEFAULT_BLOCK_SIZE,
                        2,
                        4,
                        DType::F32,
                        &Device::Cpu,
                        None,
                    )
                    .unwrap(),
                ))
            })
            .collect()
    }

    #[test]
    fn admission_and_retirement() {
        let allocators = make_allocators(2, 64);
        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 1024,
        };
        let mut sched = Scheduler::new(config, allocators, 2);

        let id0 = sched.add_request(vec![1, 2, 3], SamplingParams::default(), 10);
        let id1 = sched.add_request(vec![4, 5], SamplingParams::default(), 10);
        assert_eq!(sched.num_waiting(), 2);

        let output = sched.schedule(None);
        assert_eq!(output.scheduled.len(), 2);
        assert_eq!(output.scheduled[0].id, id0);
        assert_eq!(output.scheduled[0].phase, SequencePhase::Prefill);
        assert_eq!(output.scheduled[1].id, id1);
        assert_eq!(sched.num_running(), 2);
        assert_eq!(sched.num_waiting(), 0);

        sched.get_running_mut(id0).unwrap().status = SequenceStatus::Finished;
        let completed = sched.retire_finished();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].id, id0);
        assert_eq!(sched.num_running(), 1);
    }

    #[test]
    fn max_sequences_limit() {
        let allocators = make_allocators(2, 64);
        let config = SchedulerConfig {
            max_num_sequences: 2,
            max_tokens_per_step: 1024,
        };
        let mut sched = Scheduler::new(config, allocators, 2);

        sched.add_request(vec![1, 2], SamplingParams::default(), 10);
        sched.add_request(vec![3, 4], SamplingParams::default(), 10);
        sched.add_request(vec![5, 6], SamplingParams::default(), 10);

        let output = sched.schedule(None);
        assert_eq!(output.scheduled.len(), 2);
        assert_eq!(sched.num_running(), 2);
        assert_eq!(sched.num_waiting(), 1);
    }

    #[test]
    fn preemption_under_memory_pressure() {
        // 2 blocks per layer: very tight memory
        let allocators = make_allocators(2, 2);
        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 1024,
        };
        let mut sched = Scheduler::new(config, allocators.clone(), 2);

        // Request with 16 tokens (fills 1 block exactly)
        let tokens: Vec<u32> = (0..DEFAULT_BLOCK_SIZE as u32).collect();
        let id0 = sched.add_request(tokens.clone(), SamplingParams::default(), 100);

        let output = sched.schedule(None);
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(output.scheduled[0].id, id0);

        // Simulate that prefill processed all tokens, allocating 1 block
        let seq = sched.get_running_mut(id0).unwrap();
        seq.num_processed_tokens = DEFAULT_BLOCK_SIZE;
        seq.phase = SequencePhase::Decode;

        // Add a second request that needs 1 block
        let id1 = sched.add_request(tokens, SamplingParams::default(), 100);
        let output = sched.schedule(None);

        assert!(sched.has_pending_work());
        let _ = (id1, output);
    }

    #[test]
    fn block_conservation() {
        let allocators = make_allocators(2, 16);
        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 1024,
        };
        let mut sched = Scheduler::new(config, allocators.clone(), 2);

        let initial_free = allocators[0].lock().unwrap().num_free();

        let id0 = sched.add_request(vec![1, 2, 3], SamplingParams::default(), 10);
        sched.schedule(None);

        sched.get_running_mut(id0).unwrap().status = SequenceStatus::Finished;
        sched.retire_finished();

        let final_free = allocators[0].lock().unwrap().num_free();
        assert_eq!(final_free, initial_free);
    }

    /// Contract: a prompt too long for one step's token budget still runs.
    ///
    /// This replaces the earlier rule, that such a prompt was simply not
    /// admitted. Because admission is first-come-first-served, that rule left
    /// the prompt at the head of the queue for every subsequent step, so the
    /// request hung with no error and blocked everything queued behind it.
    /// Reachable in a supported configuration: `--max-context-len 16384`
    /// accepts prompts far above the 4096-token step budget.
    #[test]
    fn oversized_prompt_runs_alone_rather_than_starving() {
        let allocators = make_allocators(1, 16);
        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 2,
        };
        let mut sched = Scheduler::new(config, allocators, 1);

        sched.add_request(vec![1, 2, 3, 4, 5], SamplingParams::default(), 10);

        let out = sched.schedule(None);
        assert_eq!(out.scheduled.len(), 1, "5-token prompt, budget of 2");
        assert_eq!(out.scheduled[0].phase, SequencePhase::Prefill);
        assert_eq!(sched.num_running(), 1);
        assert_eq!(sched.num_waiting(), 0);
    }

    /// Contract: the budget still bundles a step, so an oversized prompt does
    /// not join a step that already carries another prefill. It is deferred by
    /// one step, not starved: the next call admits it, alongside the decode of
    /// the sequence ahead of it. Deferring it until the running set drained
    /// instead would starve it under continuous traffic.
    #[test]
    fn oversized_prompt_waits_only_for_the_next_step() {
        let allocators = make_allocators(1, 16);
        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 4,
        };
        let mut sched = Scheduler::new(config, allocators, 1);

        sched.add_request(vec![1, 2], SamplingParams::default(), 10);
        sched.add_request(vec![1, 2, 3, 4, 5, 6], SamplingParams::default(), 10);

        let first = sched.schedule(None);
        assert_eq!(first.scheduled.len(), 1, "only the prompt that fits");
        assert_eq!(sched.num_waiting(), 1);

        let second = sched.schedule(None);
        assert!(
            second
                .scheduled
                .iter()
                .any(|s| s.phase == SequencePhase::Prefill),
            "the oversized prompt is admitted on a later step"
        );
        assert_eq!(sched.num_waiting(), 0);
    }

    /// A pseudo-random source with the seed in hand, so a failure reproduces by
    /// rerunning the same seed rather than by luck.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
        }

        /// Uniform in `[lo, hi]`.
        fn between(&mut self, lo: usize, hi: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + ((self.0 >> 33) as usize) % (hi - lo + 1)
        }
    }

    /// Contract (liveness): every prompt admitted to the queue eventually runs,
    /// whatever its length relative to a step's token budget.
    ///
    /// This is the property the scheduler broke by refusing to admit a prompt
    /// longer than one step: nothing crashed, the queue simply never advanced,
    /// and the request hung with no error while blocking everything behind it.
    /// A liveness property finds that in seconds; no example-based test did,
    /// because each one asserted about a batch it had chosen to be schedulable.
    #[test]
    fn fuzz_every_admitted_prompt_eventually_runs() {
        for seed in 0u64..64 {
            let mut rng = Rng::new(seed);
            let budget = [4usize, 16, 64, 256][rng.between(0, 3)];
            let config = SchedulerConfig {
                max_num_sequences: rng.between(1, 4),
                max_tokens_per_step: budget,
            };
            // Blocks are deliberately plentiful: this property is about the
            // token budget, and starving the pool would make preemption, not
            // admission, the reason a queue stalls.
            let allocators = make_allocators(1, 4096);
            let mut sched = Scheduler::new(config, allocators, 1);

            let n = rng.between(1, 6);
            let mut ids = Vec::new();
            for _ in 0..n {
                // Lengths straddle the budget on purpose, since that is exactly
                // where admission used to give up.
                let prompt_len = rng.between(1, budget * 3);
                let max_tokens = rng.between(1, 4);
                ids.push(sched.add_request(
                    vec![7u32; prompt_len],
                    SamplingParams::default(),
                    max_tokens,
                ));
            }

            let mut ran: std::collections::HashSet<SequenceId> = std::collections::HashSet::new();
            let mut steps = 0usize;
            let limit = 4096;
            while sched.has_pending_work() && steps < limit {
                steps += 1;
                let out = sched.schedule(None);
                for s in &out.scheduled {
                    ran.insert(s.id);
                    let seq = sched.get_running_mut(s.id).unwrap();
                    match s.phase {
                        SequencePhase::Prefill => {
                            seq.num_processed_tokens = seq.all_tokens.len();
                            seq.phase = SequencePhase::Decode;
                        }
                        SequencePhase::Decode => seq.num_processed_tokens += 1,
                    }
                    seq.append_token(7);
                    seq.apply_token(7, false);
                }
                sched.retire_finished();
            }

            assert_eq!(
                ran.len(),
                ids.len(),
                "seed {seed}: {} of {} prompts never ran (budget {budget})",
                ran.len(),
                ids.len()
            );
            assert!(
                !sched.has_pending_work(),
                "seed {seed}: queue still holds work after {steps} steps (budget {budget})"
            );
        }
    }

    /// Contract: a step never feeds the model more than one prompt's worth over
    /// the budget, so the exemption that keeps an oversized prompt moving does
    /// not turn into an unbounded batch.
    #[test]
    fn fuzz_a_step_never_exceeds_the_budget_by_more_than_one_prompt() {
        for seed in 0u64..64 {
            let mut rng = Rng::new(seed ^ 0x5eed);
            let budget = [4usize, 16, 64][rng.between(0, 2)];
            let config = SchedulerConfig {
                max_num_sequences: rng.between(1, 4),
                max_tokens_per_step: budget,
            };
            let allocators = make_allocators(1, 4096);
            let mut sched = Scheduler::new(config, allocators, 1);

            let mut lengths = std::collections::HashMap::new();
            for _ in 0..rng.between(1, 6) {
                let len = rng.between(1, budget * 3);
                let id = sched.add_request(vec![7u32; len], SamplingParams::default(), 2);
                lengths.insert(id, len);
            }

            let mut steps = 0usize;
            while sched.has_pending_work() && steps < 4096 {
                steps += 1;
                let out = sched.schedule(None);
                let mut prefill_tokens = 0usize;
                let mut oversized = 0usize;
                for s in &out.scheduled {
                    if s.phase == SequencePhase::Prefill {
                        let len = lengths[&s.id];
                        prefill_tokens += len;
                        if len > budget {
                            oversized = oversized.max(len);
                        }
                    }
                }
                assert!(
                    prefill_tokens <= budget.max(oversized),
                    "seed {seed}: step carried {prefill_tokens} prefill tokens on a budget of \
                     {budget} with the largest oversized prompt at {oversized}"
                );
                for s in &out.scheduled {
                    let seq = sched.get_running_mut(s.id).unwrap();
                    match s.phase {
                        SequencePhase::Prefill => {
                            seq.num_processed_tokens = seq.all_tokens.len();
                            seq.phase = SequencePhase::Decode;
                        }
                        SequencePhase::Decode => seq.num_processed_tokens += 1,
                    }
                    seq.append_token(7);
                    seq.apply_token(7, false);
                }
                sched.retire_finished();
            }
        }
    }

    /// Contract: no sequence lifecycle leaks blocks.
    ///
    /// With the pool deliberately too small, sequences get preempted, requeued
    /// and re-prefilled, and every one of those paths has to give its blocks
    /// back. A leak here is invisible until the pool runs dry and the server
    /// quietly stops admitting anything, which is a support case rather than a
    /// crash.
    #[test]
    fn fuzz_preemption_and_retirement_give_every_block_back() {
        for seed in 0u64..64 {
            let mut rng = Rng::new(seed ^ 0xb10c);
            // Small enough that preemption is the normal case, not the corner.
            let capacity = rng.between(2, 8);
            let allocators = make_allocators(1, capacity);
            let config = SchedulerConfig {
                max_num_sequences: rng.between(1, 3),
                max_tokens_per_step: [8usize, 32, 128][rng.between(0, 2)],
            };
            let mut sched = Scheduler::new(config, allocators.clone(), 1);

            for _ in 0..rng.between(1, 5) {
                let len = rng.between(1, 4 * DEFAULT_BLOCK_SIZE);
                sched.add_request(
                    vec![7u32; len],
                    SamplingParams::default(),
                    rng.between(1, 3),
                );
            }

            for _ in 0..256 {
                let out = sched.schedule(None);
                for s in &out.scheduled {
                    // Stand in for the engine, which fills the cache and lets it
                    // take the blocks it needs. Going through the cache is the
                    // point: it is the thing that must give them back.
                    let (id, phase) = (s.id, s.phase);
                    let seq = sched.get_running_mut(id).unwrap();
                    let tokens = match phase {
                        SequencePhase::Prefill => seq.all_tokens.len(),
                        SequencePhase::Decode => 1,
                    };
                    let kv = candle_core::Tensor::zeros(
                        (1, 1, tokens, 4),
                        candle_core::DType::F32,
                        &candle_core::Device::Cpu,
                    )
                    .unwrap();
                    // An exhausted pool fails the append, exactly as it would in
                    // the engine; the scheduler preempts on the next step.
                    let _ = seq.caches[0].append(&kv, &kv);
                    if phase == SequencePhase::Prefill {
                        seq.num_processed_tokens = seq.all_tokens.len();
                        seq.phase = SequencePhase::Decode;
                    } else {
                        seq.num_processed_tokens += 1;
                    }
                    seq.append_token(7);
                    seq.apply_token(7, false);
                }
                sched.retire_finished();
                if !sched.has_pending_work() {
                    break;
                }
            }

            sched.abort_all();
            assert_eq!(
                allocators[0].lock().unwrap().num_free(),
                capacity,
                "seed {seed}: pool came back short after every sequence was gone"
            );
        }
    }

    #[test]
    fn preempted_sequence_is_requeued_as_prefill() {
        let allocators = make_allocators(1, 2);
        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 1024,
        };
        let mut sched = Scheduler::new(config, allocators.clone(), 1);

        let id = sched.add_request(vec![42], SamplingParams::default(), 100);

        // Admit the sequence (1 block needed, 2 available locally)
        let out = sched.schedule(None);
        assert_eq!(out.scheduled[0].phase, SequencePhase::Prefill);
        assert_eq!(sched.num_running(), 1);

        // Exhaust the allocator to simulate engine block usage
        let b0 = allocators[0].lock().unwrap().allocate().unwrap();
        let b1 = allocators[0].lock().unwrap().allocate().unwrap();
        assert_eq!(allocators[0].lock().unwrap().num_free(), 0);

        // Simulate prefill completed: seq now decodes at a block boundary
        {
            let seq = sched.get_running_mut(id).unwrap();
            seq.num_processed_tokens = DEFAULT_BLOCK_SIZE;
            seq.phase = SequencePhase::Decode;
        }

        // schedule(): decode needs a new block, none free, so preemption
        sched.schedule(None);
        assert_eq!(sched.num_running(), 0);
        assert_eq!(
            sched.num_waiting(),
            1,
            "preempted seq should be back in waiting"
        );

        // Release blocks and re-schedule: seq re-admitted as Prefill with reset state
        allocators[0].lock().unwrap().free(b0);
        allocators[0].lock().unwrap().free(b1);

        let out = sched.schedule(None);
        assert_eq!(out.scheduled.len(), 1);
        assert_eq!(out.scheduled[0].id, id);
        assert_eq!(out.scheduled[0].phase, SequencePhase::Prefill);
        assert_eq!(
            sched.get_running(id).unwrap().num_processed_tokens,
            0,
            "preempted seq must have num_processed_tokens reset to 0"
        );
    }

    #[test]
    fn abort_sequence_removes_from_running_and_waiting() {
        let allocators = make_allocators(2, 32);
        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 1024,
        };
        let mut sched = Scheduler::new(config, allocators, 2);

        let running_id = sched.add_request(vec![1, 2, 3], SamplingParams::default(), 8);
        let waiting_id = sched.add_request(vec![4, 5, 6], SamplingParams::default(), 8);
        let _ = sched.schedule(None);

        assert!(sched.get_running(running_id).is_some());
        assert!(sched.abort_sequence(running_id));
        assert!(sched.get_running(running_id).is_none());

        assert!(sched.abort_sequence(waiting_id));
        assert!(!sched.abort_sequence(999_999));
        assert!(!sched.has_pending_work());
    }

    /// Stands in for the engine after a prefill: the cache takes the blocks
    /// the prompt fills and the sequence moves on to decoding.
    fn simulate_prefill(sched: &mut Scheduler, id: SequenceId) {
        let seq = sched.get_running_mut(id).unwrap();
        let tokens = seq.all_tokens.len();
        let kv = candle_core::Tensor::zeros(
            (1, 2, tokens, 4),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        seq.caches[0].append(&kv, &kv).unwrap();
        seq.num_processed_tokens = tokens;
        seq.phase = SequencePhase::Decode;
        seq.append_token(7);
    }

    /// Contract: a free block is promised to at most one sequence per step.
    ///
    /// Scheduling re-read the free count for every decode at a block boundary
    /// without reserving anything, so two sequences crossing a boundary in the
    /// same step were both scheduled against one free block. The second
    /// allocation then failed inside the forward, and the engine aborted every
    /// running request instead of preempting one sequence.
    #[test]
    fn the_last_free_block_is_promised_to_one_decode_only() {
        let allocators = make_allocators(1, 3);
        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 1024,
        };
        let mut sched = Scheduler::new(config, allocators.clone(), 1);
        let a = sched.add_request(vec![1; DEFAULT_BLOCK_SIZE], SamplingParams::default(), 10);
        let b = sched.add_request(vec![2; DEFAULT_BLOCK_SIZE], SamplingParams::default(), 10);
        assert_eq!(sched.schedule(None).scheduled.len(), 2);
        simulate_prefill(&mut sched, a);
        simulate_prefill(&mut sched, b);
        assert_eq!(allocators[0].lock().unwrap().num_free(), 1);

        let out = sched.schedule(None);
        assert_eq!(
            out.scheduled.len(),
            1,
            "one free block lets exactly one decode cross its boundary"
        );
        assert_eq!(out.scheduled[0].phase, SequencePhase::Decode);
        assert_eq!(
            sched.num_waiting(),
            1,
            "the other sequence is preempted instead of being promised the same block"
        );
    }

    /// Contract: a block promised to a decode this step is not counted as
    /// free when the step admits prompts.
    #[test]
    fn a_block_promised_to_a_decode_is_not_handed_to_a_prefill() {
        let allocators = make_allocators(1, 2);
        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 1024,
        };
        let mut sched = Scheduler::new(config, allocators.clone(), 1);
        let a = sched.add_request(vec![1; DEFAULT_BLOCK_SIZE], SamplingParams::default(), 10);
        assert_eq!(sched.schedule(None).scheduled.len(), 1);
        simulate_prefill(&mut sched, a);
        assert_eq!(allocators[0].lock().unwrap().num_free(), 1);
        let b = sched.add_request(vec![2; DEFAULT_BLOCK_SIZE], SamplingParams::default(), 10);

        let out = sched.schedule(None);
        assert!(
            out.scheduled
                .iter()
                .any(|s| s.id == a && s.phase == SequencePhase::Decode),
            "the running decode keeps its block"
        );
        assert!(
            !out.scheduled.iter().any(|s| s.id == b),
            "the prompt must wait: the only free block is already promised"
        );
        assert_eq!(sched.num_waiting(), 1);
    }

    /// Pins `blocks` blocks of a finished prompt in the prefix cache, the way
    /// a completed request leaves them: held by the cache alone.
    fn pin_in_prefix_cache(
        cache: &mut PrefixCache,
        allocators: &[SharedBlockAllocator],
        tokens: &[u32],
    ) {
        let blocks = tokens.len() / DEFAULT_BLOCK_SIZE;
        let ids: Vec<Vec<usize>> = (0..blocks)
            .map(|_| vec![allocators[0].lock().unwrap().allocate().unwrap()])
            .collect();
        cache.register(tokens, 0, &ids, allocators, DEFAULT_BLOCK_SIZE);
        for id in &ids {
            allocators[0].lock().unwrap().free(id[0]);
        }
    }

    /// Contract: blocks held only by the prefix cache never keep a prompt
    /// out; under pressure the cache yields them, oldest first.
    ///
    /// The cache had no eviction other than overflowing its own capacity, so
    /// on a pool it could cover it pinned every block a finished prompt had
    /// filled, the next prompt with a different prefix was never admitted,
    /// and the engine loop spun on a queue it could not advance.
    #[test]
    fn a_prompt_kept_out_only_by_the_prefix_cache_is_admitted() {
        let allocators = make_allocators(1, 4);
        let mut cache = PrefixCache::new(512);
        pin_in_prefix_cache(&mut cache, &allocators, &[1u32; 3 * DEFAULT_BLOCK_SIZE]);
        assert_eq!(allocators[0].lock().unwrap().num_free(), 1);

        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 1024,
        };
        let mut sched = Scheduler::new(config, allocators.clone(), 1);
        let id = sched.add_request(
            vec![2; 2 * DEFAULT_BLOCK_SIZE],
            SamplingParams::default(),
            10,
        );

        let out = sched.schedule(Some(&mut cache));
        assert_eq!(
            out.scheduled.len(),
            1,
            "the cache yields what the prompt needs"
        );
        assert_eq!(out.scheduled[0].id, id);
        assert_eq!(out.scheduled[0].phase, SequencePhase::Prefill);
        assert!(
            allocators[0].lock().unwrap().num_free() >= 2,
            "the yielded blocks are free for the prefill to take"
        );
    }

    /// Contract: the cache yields a block to a decode at a block boundary
    /// before the scheduler preempts that sequence.
    #[test]
    fn a_decode_takes_a_block_from_the_prefix_cache_before_being_preempted() {
        let allocators = make_allocators(1, 2);
        let mut cache = PrefixCache::new(512);
        pin_in_prefix_cache(&mut cache, &allocators, &[1u32; DEFAULT_BLOCK_SIZE]);

        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 1024,
        };
        let mut sched = Scheduler::new(config, allocators.clone(), 1);
        let a = sched.add_request(vec![2; DEFAULT_BLOCK_SIZE], SamplingParams::default(), 10);
        assert_eq!(sched.schedule(Some(&mut cache)).scheduled.len(), 1);
        simulate_prefill(&mut sched, a);
        assert_eq!(allocators[0].lock().unwrap().num_free(), 0);

        let out = sched.schedule(Some(&mut cache));
        assert_eq!(
            out.scheduled.len(),
            1,
            "the decode continues on the yielded block"
        );
        assert_eq!(out.scheduled[0].phase, SequencePhase::Decode);
        assert_eq!(sched.num_waiting(), 0, "nothing was preempted");
    }

    /// Contract: under pressure the cache keeps the blocks the waiting prompt
    /// itself reuses and yields the others, so admission does not cost the
    /// prompt its own prefix hit.
    #[test]
    fn pressure_eviction_spares_the_prompt_s_own_cached_prefix() {
        let allocators = make_allocators(1, 4);
        let mut cache = PrefixCache::new(512);
        let shared = [1u32; 2 * DEFAULT_BLOCK_SIZE];
        pin_in_prefix_cache(&mut cache, &allocators, &shared);
        pin_in_prefix_cache(&mut cache, &allocators, &[2u32; 2 * DEFAULT_BLOCK_SIZE]);
        assert_eq!(allocators[0].lock().unwrap().num_free(), 0);

        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 1024,
        };
        let mut sched = Scheduler::new(config, allocators.clone(), 1);
        let prompt: Vec<u32> = shared
            .iter()
            .copied()
            .chain([3u32; DEFAULT_BLOCK_SIZE])
            .collect();
        sched.add_request(prompt.clone(), SamplingParams::default(), 10);

        let out = sched.schedule(Some(&mut cache));
        assert_eq!(out.scheduled.len(), 1);
        assert_eq!(
            cache.count_cached_blocks(&prompt, DEFAULT_BLOCK_SIZE),
            2,
            "the prompt's own prefix stays cached; the other prompt's blocks were yielded"
        );
    }
}
