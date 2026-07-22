pub mod sequence;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::common::paged::{DEFAULT_BLOCK_SIZE, PagedKvCache, SharedBlockAllocator};
use crate::common::prefix_cache::PrefixCache;
use crate::sampling::SamplingParams;
use sequence::*;

pub struct SchedulerConfig {
    pub max_num_sequences: usize,
    pub max_tokens_per_step: usize,
}

pub struct ScheduledSequence {
    pub id: SequenceId,
    pub phase: SequencePhase,
}

pub struct SchedulerOutput {
    pub scheduled: Vec<ScheduledSequence>,
}

pub struct CompletedSequence {
    pub id: SequenceId,
    pub finish_reason: Option<String>,
}

pub struct Scheduler {
    config: SchedulerConfig,
    waiting: VecDeque<SequenceState>,
    running: Vec<SequenceState>,
    running_index: HashMap<SequenceId, usize>,
    allocators: Vec<SharedBlockAllocator>,
    /// Draft-model allocators for speculative decoding; empty when no draft.
    draft_allocators: Vec<SharedBlockAllocator>,
    draft_num_layers: usize,
    next_id: SequenceId,
    num_layers: usize,
    block_size: usize,
}

impl Scheduler {
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

    fn remove_from_running(&mut self, vec_idx: usize) -> SequenceState {
        let seq_id = self.running[vec_idx].id;
        self.running_index.remove(&seq_id);
        let seq = self.running.swap_remove(vec_idx);
        if vec_idx < self.running.len() {
            self.running_index.insert(self.running[vec_idx].id, vec_idx);
        }
        seq
    }

    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
    ) -> SequenceId {
        self.add_request_with_stop(prompt_tokens, sampling_params, max_tokens, Vec::new())
    }

    pub fn add_request_with_stop(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
        extra_stop_token_ids: Vec<u32>,
    ) -> SequenceId {
        self.add_request_full(
            prompt_tokens,
            sampling_params,
            max_tokens,
            extra_stop_token_ids,
            None,
        )
    }

    pub fn add_request_full(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
        extra_stop_token_ids: Vec<u32>,
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
            extra_stop_token_ids,
            constraint,
        };

        self.waiting.push_back(seq);
        id
    }

    fn num_free_blocks(&self) -> usize {
        if self.allocators.is_empty() {
            return 0;
        }
        self.allocators[0].lock().unwrap().num_free()
    }

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

    fn decode_needs_new_block(&self, seq: &SequenceState) -> bool {
        seq.num_processed_tokens.is_multiple_of(self.block_size) && seq.num_processed_tokens > 0
    }

    pub fn schedule(&mut self, prefix_cache: Option<&PrefixCache>) -> SchedulerOutput {
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

        for &seq_id in &seqs_needing_block {
            if self.num_free_blocks() > 0 {
                if budget > 0 {
                    scheduled.push(ScheduledSequence {
                        id: seq_id,
                        phase: SequencePhase::Decode,
                    });
                    budget -= 1;
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

        let mut free_blocks = self.num_free_blocks();

        while self.running.len() < self.config.max_num_sequences && budget > 0 {
            let seq = match self.waiting.front() {
                Some(s) => s,
                None => break,
            };

            let blocks_needed = self.blocks_needed_for_prefill(seq, prefix_cache);
            if blocks_needed > free_blocks {
                break;
            }

            let tokens_needed = seq.all_tokens.len();
            if tokens_needed > budget {
                break;
            }

            let mut seq = self.waiting.pop_front().unwrap();
            seq.status = SequenceStatus::Running;
            seq.phase = SequencePhase::Prefill;

            scheduled.push(ScheduledSequence {
                id: seq.id,
                phase: SequencePhase::Prefill,
            });
            budget -= tokens_needed;
            free_blocks = free_blocks.saturating_sub(blocks_needed);

            self.push_to_running(seq);
        }

        SchedulerOutput { scheduled }
    }

    pub fn get_running(&self, seq_id: SequenceId) -> Option<&SequenceState> {
        let idx = *self.running_index.get(&seq_id)?;
        self.running.get(idx)
    }

    pub fn get_running_mut(&mut self, seq_id: SequenceId) -> Option<&mut SequenceState> {
        let idx = *self.running_index.get(&seq_id)?;
        self.running.get_mut(idx)
    }

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

    pub fn abort_all_running(&mut self) -> Vec<SequenceId> {
        let ids: Vec<SequenceId> = self.running.iter().map(|s| s.id).collect();
        for mut seq in self.running.drain(..) {
            seq.clear_caches();
        }
        self.running_index.clear();
        ids
    }

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

    pub fn abort_all(&mut self) -> Vec<SequenceId> {
        let mut ids = self.abort_all_running();
        for mut seq in self.waiting.drain(..) {
            ids.push(seq.id);
            seq.clear_caches();
        }
        ids
    }

    pub fn has_pending_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

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
                    BlockAllocator::new(
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

        // Mark one as finished
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
        // Only 2 should be admitted
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

        // Admit and "prefill" it
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

    #[test]
    fn schedule_does_not_admit_prompt_exceeding_token_budget() {
        let allocators = make_allocators(1, 16);
        let config = SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 2, // tight budget
        };
        let mut sched = Scheduler::new(config, allocators, 1);

        sched.add_request(vec![1, 2, 3, 4, 5], SamplingParams::default(), 10);

        let out = sched.schedule(None);
        assert_eq!(out.scheduled.len(), 0, "5-token prompt exceeds budget of 2");
        assert_eq!(sched.num_running(), 0);
        assert_eq!(sched.num_waiting(), 1);
    }

    #[test]
    fn preempted_sequence_is_requeued_as_prefill() {
        let allocators = make_allocators(1, 2); // 1 layer, 2 blocks
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
}
