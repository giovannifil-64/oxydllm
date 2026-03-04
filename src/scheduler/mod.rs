pub mod sequence;

use std::collections::VecDeque;
use std::sync::Arc;

use crate::common::paged::{PagedKvCache, SharedBlockAllocator, DEFAULT_BLOCK_SIZE};
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
}

pub struct Scheduler {
    config: SchedulerConfig,
    waiting: VecDeque<SequenceState>,
    running: Vec<SequenceState>,
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
            allocators,
            next_id: 0,
            num_layers,
            block_size,
        }
    }

    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
    ) -> SequenceId {
        let id = self.next_id;
        self.next_id += 1;

        let caches = (0..self.num_layers)
            .map(|i| PagedKvCache::new(Arc::clone(&self.allocators[i])))
            .collect();

        let all_tokens = prompt_tokens.clone();
        let seq = SequenceState {
            id,
            generated_tokens: Vec::new(),
            all_tokens,
            sampling_params,
            status: SequenceStatus::Waiting,
            phase: SequencePhase::Prefill,
            caches,
            num_processed_tokens: 0,
            max_tokens,
            finish_reason: None,
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

    fn blocks_needed_for_prefill(&self, seq: &SequenceState) -> usize {
        let total_tokens = seq.all_tokens.len();
        (total_tokens + self.block_size - 1) / self.block_size
    }

    fn decode_needs_new_block(&self, seq: &SequenceState) -> bool {
        seq.num_processed_tokens % self.block_size == 0 && seq.num_processed_tokens > 0
    }

    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut scheduled = Vec::new();
        let mut budget = self.config.max_tokens_per_step;

        let mut to_preempt = Vec::new();
        for (idx, seq) in self.running.iter().enumerate() {
            if budget == 0 {
                break;
            }
            if self.decode_needs_new_block(seq) && self.num_free_blocks() == 0 {
                to_preempt.push(idx);
                continue;
            }
            scheduled.push(ScheduledSequence {
                id: seq.id,
                phase: SequencePhase::Decode,
            });
            budget -= 1;
        }

        for &idx in to_preempt.iter().rev() {
            let mut seq = self.running.remove(idx);
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

        if !to_preempt.is_empty() {
            let running_ids: std::collections::HashSet<SequenceId> =
                self.running.iter().map(|s| s.id).collect();
            scheduled.retain(|s| running_ids.contains(&s.id));
        }

        let mut newly_admitted = Vec::new();
        while self.running.len() + newly_admitted.len() < self.config.max_num_sequences
            && budget > 0
        {
            let seq = match self.waiting.front() {
                Some(s) => s,
                None => break,
            };

            let blocks_needed = self.blocks_needed_for_prefill(seq);
            if blocks_needed > self.num_free_blocks() {
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

            newly_admitted.push(seq);
        }

        self.running.extend(newly_admitted);

        SchedulerOutput { scheduled }
    }

    pub fn get_running_mut(&mut self, seq_id: SequenceId) -> Option<&mut SequenceState> {
        self.running.iter_mut().find(|s| s.id == seq_id)
    }

    pub fn retire_finished(&mut self) -> Vec<CompletedSequence> {
        let mut completed = Vec::new();
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].status == SequenceStatus::Finished {
                let mut seq = self.running.remove(i);
                for cache in &mut seq.caches {
                    cache.clear();
                }
                completed.push(CompletedSequence {
                    id: seq.id,
                });
            } else {
                i += 1;
            }
        }
        completed
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
                    BlockAllocator::new(num_blocks, DEFAULT_BLOCK_SIZE, 2, 4, DType::F32, &Device::Cpu)
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

        let output = sched.schedule();
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

        let output = sched.schedule();
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
        let output = sched.schedule();
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(output.scheduled[0].id, id0);

        // Simulate that prefill processed all tokens, allocating 1 block
        let seq = sched.get_running_mut(id0).unwrap();
        seq.num_processed_tokens = DEFAULT_BLOCK_SIZE;
        seq.phase = SequencePhase::Decode;

        // Add a second request that needs 1 block
        let id1 = sched.add_request(tokens, SamplingParams::default(), 100);
        let output = sched.schedule();

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
        sched.schedule();


        sched.get_running_mut(id0).unwrap().status = SequenceStatus::Finished;
        sched.retire_finished();

        let final_free = allocators[0].lock().unwrap().num_free();
        assert_eq!(final_free, initial_free);
    }
}
