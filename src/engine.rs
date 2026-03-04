use std::sync::Arc;

use candle_core::{Device, Result, Tensor};

use crate::common::paged::{SharedBlockAllocator, DEFAULT_BLOCK_SIZE};
use crate::common::prefix_cache::PrefixCache;
use crate::models::traits::BatchModel;
use crate::sampling::{self, SamplingParams};
use crate::scheduler::sequence::*;
use crate::scheduler::*;

pub struct NewToken {
    pub seq_id: SequenceId,
    pub token: u32,
}

pub struct StepOutput {
    pub new_tokens: Vec<NewToken>,
    pub completed: Vec<CompletedSequence>,
}

pub struct Engine {
    model: Box<dyn BatchModel>,
    scheduler: Scheduler,
    device: Device,
    stop_token_ids: Vec<u32>,
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
            model.allocators().iter().map(|a| Arc::clone(a)).collect();
        let num_layers = model.num_layers();
        let device = model.device().clone();
        let block_size = if allocators.is_empty() {
            DEFAULT_BLOCK_SIZE
        } else {
            allocators[0].lock().unwrap().block_size()
        };
        let mut stop_token_ids = model.stop_token_ids().to_vec();
        for &id in extra_stop_ids {
            if !stop_token_ids.contains(&id) {
                stop_token_ids.push(id);
            }
        }
        let scheduler_allocators: Vec<SharedBlockAllocator> =
            allocators.iter().map(Arc::clone).collect();
        let scheduler = Scheduler::new(config, scheduler_allocators, num_layers);
        let prefix_cache = PrefixCache::new(512);
        Self { model, scheduler, device, stop_token_ids, allocators, prefix_cache, block_size }
    }

    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
    ) -> SequenceId {
        self.scheduler.add_request(prompt_tokens, sampling_params, max_tokens)
    }

    pub fn step(&mut self) -> Result<StepOutput> {
        let output = self.scheduler.schedule();
        let mut new_tokens = Vec::new();

        let mut prefill_ids: Vec<SequenceId> = Vec::new();
        let mut decode_ids: Vec<SequenceId> = Vec::new();

        for sched_seq in &output.scheduled {
            match sched_seq.phase {
                SequencePhase::Prefill => prefill_ids.push(sched_seq.id),
                SequencePhase::Decode => decode_ids.push(sched_seq.id),
            }
        }

        for &seq_id in &prefill_ids {
            let (all_tokens, sampling_params) = {
                let seq = self.scheduler.get_running_mut(seq_id).unwrap();
                (seq.all_tokens.clone(), seq.sampling_params.clone())
            };
            let seq_len = all_tokens.len();
            let block_size = self.block_size;

            let (mut num_cached_blocks, matched_block_ids) =
                self.prefix_cache.lookup(&all_tokens, block_size);

            let max_cacheable = (seq_len.saturating_sub(1)) / block_size;
            if num_cached_blocks > max_cacheable {
                num_cached_blocks = max_cacheable;
            }

            let num_cached_tokens = num_cached_blocks * block_size;

            if num_cached_blocks > 0 {
                let seq = self.scheduler.get_running_mut(seq_id).unwrap();
                for (layer_idx, cache) in seq.caches.iter_mut().enumerate() {
                    for layer_block_ids in matched_block_ids[..num_cached_blocks].iter() {
                        if let Some(&bid) = layer_block_ids.get(layer_idx) {
                            cache.prepopulate_block(bid);
                        }
                    }
                }
                for cache in seq.caches.iter_mut() {
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

            let t_prefill = std::time::Instant::now();
            let input = Tensor::from_vec(input_tokens, (1, uncached_len), &self.device)?;
            let logits = {
                let seq = self.scheduler.get_running_mut(seq_id).unwrap();
                self.model.forward_with_cache(&input, num_cached_tokens, &mut seq.caches)?
            };
            eprintln!(
                "[timing] prefill forward: {:.1}ms ({}/{} tokens, {} cached)",
                t_prefill.elapsed().as_secs_f64() * 1000.0,
                uncached_len,
                seq_len,
                num_cached_tokens
            );

            let last_logits = logits.squeeze(0)?.get(uncached_len - 1)?;
            let next_token =
                sampling::sample(&last_logits, &sampling_params, &all_tokens)?;
            let is_stop = self.stop_token_ids.contains(&next_token);

            {
                let seq = self.scheduler.get_running_mut(seq_id).unwrap();
                seq.all_tokens.push(next_token);
                seq.num_processed_tokens = seq.all_tokens.len() - 1;
                seq.phase = SequencePhase::Decode;

                if is_stop {
                    seq.status = SequenceStatus::Finished;
                    seq.finish_reason = Some("stop".to_string());
                } else {
                    seq.generated_tokens.push(next_token);
                    if seq.generated_tokens.len() >= seq.max_tokens {
                        seq.status = SequenceStatus::Finished;
                        seq.finish_reason = Some("length".to_string());
                    }
                }
            }

            if !is_stop {
                new_tokens.push(NewToken { seq_id, token: next_token });
            }

            let num_full_blocks_total = seq_len / block_size;
            if num_full_blocks_total > num_cached_blocks {
                let new_block_ids: Vec<Vec<usize>> = {
                    let seq = self.scheduler.get_running_mut(seq_id).unwrap();
                    (num_cached_blocks..num_full_blocks_total)
                        .map(|block_idx| {
                            seq.caches
                                .iter()
                                .filter_map(|c| c.block_id_at(block_idx))
                                .collect::<Vec<usize>>()
                        })
                        .collect()
                };
                self.prefix_cache.register(
                    &all_tokens,
                    num_cached_blocks,
                    &new_block_ids,
                    &self.allocators,
                    block_size,
                );
            }
        }

        if decode_ids.len() == 1 {
            let seq_id = decode_ids[0];
            let seq = self.scheduler.get_running_mut(seq_id).unwrap();
            let last_token = *seq.all_tokens.last().unwrap();
            let start_pos = seq.num_processed_tokens;
            let is_first_decode = seq.generated_tokens.len() == 1;

            let t_decode = std::time::Instant::now();
            let input = Tensor::from_vec(vec![last_token], (1, 1), &self.device)?;
            let logits = self.model.forward_with_cache(&input, start_pos, &mut seq.caches)?;
            if is_first_decode {
                eprintln!(
                    "[timing] decode step 1: {:.1}ms",
                    t_decode.elapsed().as_secs_f64() * 1000.0
                );
            }
            let last_logits = logits.squeeze(0)?.get(0)?;

            let next_token =
                sampling::sample(&last_logits, &seq.sampling_params, &seq.all_tokens)?;
            let is_stop = self.stop_token_ids.contains(&next_token);

            seq.all_tokens.push(next_token);
            seq.num_processed_tokens = seq.all_tokens.len() - 1;

            if is_stop {
                seq.status = SequenceStatus::Finished;
                seq.finish_reason = Some("stop".to_string());
            } else {
                seq.generated_tokens.push(next_token);
                new_tokens.push(NewToken { seq_id: seq.id, token: next_token });
                if seq.generated_tokens.len() >= seq.max_tokens {
                    seq.status = SequenceStatus::Finished;
                    seq.finish_reason = Some("length".to_string());
                }
            }
        } else if decode_ids.len() > 1 {
            let mut all_token_ids: Vec<u32> = Vec::with_capacity(decode_ids.len());
            let mut all_positions: Vec<u32> = Vec::with_capacity(decode_ids.len());
            let mut token_counts: Vec<usize> = Vec::with_capacity(decode_ids.len());

            for &seq_id in &decode_ids {
                let seq = self.scheduler.get_running_mut(seq_id).unwrap();
                let last_token = *seq.all_tokens.last().unwrap();
                let position = seq.num_processed_tokens as u32;
                all_token_ids.push(last_token);
                all_positions.push(position);
                token_counts.push(1);
            }

            let total_tokens = all_token_ids.len();
            let input = Tensor::from_vec(all_token_ids, (1, total_tokens), &self.device)?;
            let position_ids =
                Tensor::from_vec(all_positions, (total_tokens,), &self.device)?;

            let mut cache_vecs: Vec<Vec<_>> = Vec::with_capacity(decode_ids.len());
            for &seq_id in &decode_ids {
                let seq = self.scheduler.get_running_mut(seq_id).unwrap();
                cache_vecs.push(std::mem::take(&mut seq.caches));
            }

            let mut cache_slices: Vec<&mut [_]> =
                cache_vecs.iter_mut().map(|v| v.as_mut_slice()).collect();

            let logits = self.model.forward_batch(
                &input,
                &position_ids,
                &mut cache_slices,
                &token_counts,
            )?;

            for (i, &seq_id) in decode_ids.iter().enumerate() {
                let seq = self.scheduler.get_running_mut(seq_id).unwrap();
                seq.caches = std::mem::take(&mut cache_vecs[i]);
            }

            let batch_logits = logits.squeeze(0)?;

            for (i, &seq_id) in decode_ids.iter().enumerate() {
                let seq_logits = batch_logits.get(i)?;
                let seq = self.scheduler.get_running_mut(seq_id).unwrap();
                let next_token =
                    sampling::sample(&seq_logits, &seq.sampling_params, &seq.all_tokens)?;
                let is_stop = self.stop_token_ids.contains(&next_token);

                seq.all_tokens.push(next_token);
                seq.num_processed_tokens = seq.all_tokens.len() - 1;

                if is_stop {
                    seq.status = SequenceStatus::Finished;
                    seq.finish_reason = Some("stop".to_string());
                } else {
                    seq.generated_tokens.push(next_token);
                    new_tokens.push(NewToken { seq_id: seq.id, token: next_token });
                    if seq.generated_tokens.len() >= seq.max_tokens {
                        seq.status = SequenceStatus::Finished;
                        seq.finish_reason = Some("length".to_string());
                    }
                }
            }
        }

        let completed = self.scheduler.retire_finished();
        Ok(StepOutput { new_tokens, completed })
    }

    pub fn has_pending_work(&self) -> bool {
        self.scheduler.has_pending_work()
    }
}
