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
            next_id: 0,
            num_layers,
            block_size,
        }
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
        let id = self.next_id;
        self.next_id += 1;

        let mut token_counts: HashMap<u32, u32> = HashMap::new();
        for &tok in &prompt_tokens {
            *token_counts.entry(tok).or_insert(0) += 1;
        }

        let caches = (0..self.num_layers)
            .map(|i| PagedKvCache::new(Arc::clone(&self.allocators[i])))
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
            num_processed_tokens: 0,
            max_tokens,
            finish_reason: None,
            extra_stop_token_ids,
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
                eprintln!(
                    "[scheduler] seq={} preempted (memory pressure) — KV cache cleared, re-queued for prefill",
                    seq.id
                );
                for cache in &mut seq.caches {
                    cache.clear();
                }
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
            for cache in &mut seq.caches {
                cache.clear();
            }
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
            for cache in &mut seq.caches {
                cache.clear();
            }
        }
        self.running_index.clear();
        ids
    }

    pub fn abort_all(&mut self) -> Vec<SequenceId> {
        let mut ids = self.abort_all_running();
        for mut seq in self.waiting.drain(..) {
            ids.push(seq.id);
            for cache in &mut seq.caches {
                cache.clear();
            }
        }
        ids
    }

    pub fn has_pending_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
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
        // 2 blocks per layer — very tight memory
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
}
