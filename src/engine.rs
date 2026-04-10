use std::collections::HashSet;
use std::sync::Arc;

use candle_core::{Device, Result, Tensor};

use crate::common::paged::{DEFAULT_BLOCK_SIZE, PagedKvCache, SharedBlockAllocator};
use crate::common::prefix_cache::PrefixCache;
use crate::models::traits::BatchModel;
use crate::sampling::{self, SamplingParams};
use crate::scheduler::sequence::*;
use crate::scheduler::*;

pub struct NewToken {
    pub seq_id: SequenceId,
    pub token: u32,
    pub logprob: Option<f32>,
    pub top_logprobs: Vec<(u32, f32)>,
}

pub struct StepOutput {
    pub new_tokens: Vec<NewToken>,
    pub completed: Vec<CompletedSequence>,
}

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

    fn restore(mut self) {
        self.restore_inner();
    }
}

impl Drop for CacheRestoreGuard<'_> {
    fn drop(&mut self) {
        self.restore_inner();
    }
}

pub struct Engine {
    model: Box<dyn BatchModel>,
    scheduler: Scheduler,
    device: Device,
    stop_token_ids: HashSet<u32>,
    allocators: Vec<SharedBlockAllocator>,
    prefix_cache: PrefixCache,
    block_size: usize,
}

impl Engine {
    pub fn new_with_stop_tokens(
        model: Box<dyn BatchModel>,
        config: SchedulerConfig,
        extra_stop_ids: &[u32],
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
        let scheduler_allocators: Vec<SharedBlockAllocator> =
            allocators.iter().map(Arc::clone).collect();
        let scheduler = Scheduler::new(config, scheduler_allocators, num_layers);
        let prefix_cache = PrefixCache::new(512);
        Self {
            model,
            scheduler,
            device,
            stop_token_ids,
            allocators,
            prefix_cache,
            block_size,
        }
    }

    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
    ) -> SequenceId {
        self.scheduler
            .add_request(prompt_tokens, sampling_params, max_tokens)
    }

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

    pub fn step(&mut self) -> Result<StepOutput> {
        let Engine {
            model,
            scheduler,
            device,
            stop_token_ids,
            allocators,
            prefix_cache,
            block_size,
            ..
        } = self;
        let block_size = *block_size;

        let output = scheduler.schedule(Some(prefix_cache));
        let mut new_tokens = Vec::new();

        let mut prefill_ids: Vec<SequenceId> = Vec::new();
        let mut decode_ids: Vec<SequenceId> = Vec::new();

        for sched_seq in &output.scheduled {
            match sched_seq.phase {
                SequencePhase::Prefill => prefill_ids.push(sched_seq.id),
                SequencePhase::Decode => decode_ids.push(sched_seq.id),
            }
        }

        struct PrefillInfo {
            seq_id: SequenceId,
            num_cached_tokens: usize,
            num_cached_blocks: usize,
            num_full_blocks_total: usize,
            uncached_len: usize,
            input_tokens: Vec<u32>,
        }

        let mut prefill_infos: Vec<PrefillInfo> = Vec::with_capacity(prefill_ids.len());

        for &seq_id in &prefill_ids {
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
                eprintln!(
                    "[prefix_cache] seq={} hit {}/{} blocks ({} tokens skipped)",
                    seq_id,
                    num_cached_blocks,
                    seq_len / block_size,
                    num_cached_tokens
                );
            }

            let uncached_len = seq_len - num_cached_tokens;
            let input_tokens = all_tokens[num_cached_tokens..].to_vec();
            let num_full_blocks_total = seq_len / block_size;

            prefill_infos.push(PrefillInfo {
                seq_id,
                num_cached_tokens,
                num_cached_blocks,
                num_full_blocks_total,
                uncached_len,
                input_tokens,
            });
        }

        let mut all_token_ids: Vec<u32> = Vec::new();
        let mut all_positions: Vec<u32> = Vec::new();
        let mut token_counts: Vec<usize> = Vec::new();

        for info in &prefill_infos {
            all_token_ids.extend_from_slice(&info.input_tokens);
            for local_idx in 0..info.uncached_len {
                all_positions.push((info.num_cached_tokens + local_idx) as u32);
            }
            token_counts.push(info.uncached_len);
        }

        let total_prefill_tokens = all_token_ids.len();

        for &seq_id in &decode_ids {
            let seq = scheduler.get_running_mut(seq_id).unwrap();
            all_token_ids.push(*seq.all_tokens.last().unwrap());
            all_positions.push(seq.num_processed_tokens as u32);
            token_counts.push(1);
        }

        let total_tokens = all_token_ids.len();

        if total_tokens == 0 {
            let completed = scheduler.retire_finished();
            return Ok(StepOutput {
                new_tokens,
                completed,
            });
        }

        let input = Tensor::from_vec(all_token_ids, (1, total_tokens), device)?;
        let position_ids = Tensor::from_vec(all_positions, (total_tokens,), device)?;

        // Ensure sequence caches are restored even if forward_batch panics.
        let logits = {
            let mut cache_guard = CacheRestoreGuard::new(scheduler, &prefill_ids, &decode_ids);
            let mut cache_slices = cache_guard.cache_slices();
            let logits_result =
                model.forward_batch(&input, &position_ids, &mut cache_slices, &token_counts);
            drop(cache_slices);
            cache_guard.restore();
            logits_result?
        };

        let batch_logits = logits.squeeze(0)?;

        let mut logit_offset = 0usize;
        for info in &prefill_infos {
            let last_idx = logit_offset + info.uncached_len - 1;
            let seq_logits = batch_logits.get(last_idx)?;
            logit_offset += info.uncached_len;

            let sample_out = {
                let seq = scheduler.get_running(info.seq_id).unwrap();
                sampling::sample(
                    &seq_logits,
                    &seq.sampling_params,
                    &seq.all_tokens,
                    Some(&seq.token_counts),
                )?
            };
            let next_token = sample_out.token;
            let is_stop = stop_token_ids.contains(&next_token) || {
                let seq = scheduler.get_running(info.seq_id).unwrap();
                seq.extra_stop_token_ids.contains(&next_token)
            };

            let emit = {
                let seq = scheduler.get_running_mut(info.seq_id).unwrap();
                seq.append_token(next_token);
                seq.num_processed_tokens = seq.all_tokens.len() - 1;
                seq.phase = SequencePhase::Decode;
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
                prefix_cache.register(
                    &seq.all_tokens,
                    info.num_cached_blocks,
                    &new_block_ids,
                    allocators,
                    block_size,
                );
            }
        }

        for (i, &seq_id) in decode_ids.iter().enumerate() {
            let seq_logits = batch_logits.get(total_prefill_tokens + i)?;
            let sample_out = {
                let seq = scheduler.get_running(seq_id).unwrap();
                sampling::sample(
                    &seq_logits,
                    &seq.sampling_params,
                    &seq.all_tokens,
                    Some(&seq.token_counts),
                )?
            };
            let next_token = sample_out.token;
            let seq = scheduler.get_running_mut(seq_id).unwrap();
            let is_stop = stop_token_ids.contains(&next_token)
                || seq.extra_stop_token_ids.contains(&next_token);

            seq.append_token(next_token);
            seq.num_processed_tokens = seq.all_tokens.len() - 1;

            if seq.apply_token(next_token, is_stop) {
                new_tokens.push(NewToken {
                    seq_id: seq.id,
                    token: next_token,
                    logprob: sample_out.logprob,
                    top_logprobs: sample_out.top_logprobs,
                });
            }
        }

        let completed = scheduler.retire_finished();
        Ok(StepOutput {
            new_tokens,
            completed,
        })
    }

    pub fn abort_running(&mut self) -> Vec<SequenceId> {
        self.scheduler.abort_all_running()
    }

    pub fn abort_all(&mut self) -> Vec<SequenceId> {
        self.scheduler.abort_all()
    }

    pub fn has_pending_work(&self) -> bool {
        self.scheduler.has_pending_work()
    }
}
