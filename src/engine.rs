use std::collections::HashSet;
use std::sync::Arc;

use candle_core::{Device, Result, Tensor};

use crate::common::block::flush_caches;
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
    pub prefix_cache_hits: usize,
    pub prefix_cache_misses: usize,
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
    stop_token_sequences: Vec<Vec<u32>>,
    allocators: Vec<SharedBlockAllocator>,
    prefix_cache: PrefixCache,
    block_size: usize,
}

fn has_matching_stop_sequence(tokens: &[u32], stop_sequences: &[Vec<u32>]) -> bool {
    stop_sequences.iter().any(|seq| {
        let n = seq.len();
        n > 0 && tokens.len() >= n && tokens[tokens.len() - n..] == seq[..]
    })
}

const PREFILL_CHUNK_THRESHOLD: usize = 1024;

fn pick_prefill_chunk_split(uncached_lens: &[usize], total_prefill_tokens: usize) -> Option<usize> {
    if uncached_lens.len() < 2 || total_prefill_tokens < PREFILL_CHUNK_THRESHOLD {
        return None;
    }
    let target = total_prefill_tokens / 2;
    let mut acc = 0usize;
    for (i, &n) in uncached_lens.iter().enumerate() {
        acc += n;
        if acc >= target {
            // Need at least one prefill in each half for a meaningful split.
            return if i + 1 < uncached_lens.len() {
                Some(i + 1)
            } else {
                None
            };
        }
    }
    None
}

struct PrefillInfo {
    seq_id: SequenceId,
    num_cached_tokens: usize,
    num_cached_blocks: usize,
    num_full_blocks_total: usize,
    uncached_len: usize,
    input_tokens: Vec<u32>,
}

struct BatchInput {
    all_token_ids: Vec<u32>,
    all_positions: Vec<u32>,
    token_counts: Vec<usize>,
    total_prefill_tokens: usize,
}

struct StopRules<'a> {
    token_ids: &'a HashSet<u32>,
    sequences: &'a [Vec<u32>],
}

impl StopRules<'_> {
    fn matches(&self, token: u32, seq_extra_ids: &[u32], all_tokens: &[u32]) -> bool {
        self.token_ids.contains(&token)
            || seq_extra_ids.contains(&token)
            || has_matching_stop_sequence(all_tokens, self.sequences)
    }
}

struct PrefixRegistry<'a> {
    cache: &'a mut PrefixCache,
    allocators: &'a [SharedBlockAllocator],
    block_size: usize,
}

impl PrefixRegistry<'_> {
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

    let total_prefill_tokens = all_token_ids.len();

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
        total_prefill_tokens,
    }
}

fn run_forward_pass(
    model: &dyn BatchModel,
    device: &Device,
    scheduler: &mut Scheduler,
    prefill_ids: &[SequenceId],
    decode_ids: &[SequenceId],
    prefill_uncached_lens: &[usize],
    batch: BatchInput,
) -> Result<Tensor> {
    let BatchInput {
        all_token_ids,
        all_positions,
        token_counts,
        total_prefill_tokens,
    } = batch;
    let total_tokens = all_token_ids.len();
    let split_seq_idx = pick_prefill_chunk_split(prefill_uncached_lens, total_prefill_tokens);

    let mut cache_guard = CacheRestoreGuard::new(scheduler, prefill_ids, decode_ids);
    let mut cache_slices = cache_guard.cache_slices();

    let combined_result: Result<Tensor> = (|| {
        let logits = if let Some(split) = split_seq_idx {
            let prefill_a_tokens: usize = prefill_uncached_lens[..split].iter().sum();
            let chunk_b_tokens = total_tokens - prefill_a_tokens;

            let input_a = Tensor::from_vec(
                all_token_ids[..prefill_a_tokens].to_vec(),
                (1, prefill_a_tokens),
                device,
            )?;
            let positions_a = Tensor::from_vec(
                all_positions[..prefill_a_tokens].to_vec(),
                (prefill_a_tokens,),
                device,
            )?;
            let input_b = Tensor::from_vec(
                all_token_ids[prefill_a_tokens..].to_vec(),
                (1, chunk_b_tokens),
                device,
            )?;
            let positions_b = Tensor::from_vec(
                all_positions[prefill_a_tokens..].to_vec(),
                (chunk_b_tokens,),
                device,
            )?;

            let counts_a: Vec<usize> = token_counts[..split].to_vec();
            let counts_b: Vec<usize> = token_counts[split..].to_vec();

            let (caches_a, caches_b) = cache_slices.split_at_mut(split);

            let logits_a = model.forward_batch(&input_a, &positions_a, caches_a, &counts_a)?;
            let logits_b = model.forward_batch(&input_b, &positions_b, caches_b, &counts_b)?;

            Tensor::cat(&[&logits_a, &logits_b], 1)?
        } else {
            let input = Tensor::from_vec(all_token_ids, (1, total_tokens), device)?;
            let position_ids = Tensor::from_vec(all_positions, (total_tokens,), device)?;
            model.forward_batch(&input, &position_ids, &mut cache_slices, &token_counts)?
        };
        flush_caches(&mut cache_slices)?;
        Ok(logits)
    })();

    drop(cache_slices);
    cache_guard.restore();
    combined_result?.squeeze(0)
}

fn sample_prefill_outputs(
    scheduler: &mut Scheduler,
    prefix: &mut PrefixRegistry,
    prefill_infos: &[PrefillInfo],
    batch_logits: &Tensor,
    stop: &StopRules,
    new_tokens: &mut Vec<NewToken>,
) -> Result<()> {
    let mut logit_offset = 0usize;
    for info in prefill_infos {
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
        let emit = {
            let seq = scheduler.get_running_mut(info.seq_id).unwrap();
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

fn sample_decode_outputs(
    scheduler: &mut Scheduler,
    decode_ids: &[SequenceId],
    batch_logits: &Tensor,
    total_prefill_tokens: usize,
    stop: &StopRules,
    new_tokens: &mut Vec<NewToken>,
) -> Result<()> {
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

impl Engine {
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
        let prefix_cache = PrefixCache::new(512);
        Self {
            model,
            scheduler,
            device,
            stop_token_ids,
            stop_token_sequences,
            allocators,
            prefix_cache,
            block_size,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
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
            stop_token_sequences,
            allocators,
            prefix_cache,
            block_size,
            ..
        } = self;
        let block_size = *block_size;

        let output = scheduler.schedule(Some(prefix_cache));

        let mut prefill_ids: Vec<SequenceId> = Vec::new();
        let mut decode_ids: Vec<SequenceId> = Vec::new();
        for sched_seq in &output.scheduled {
            match sched_seq.phase {
                SequencePhase::Prefill => prefill_ids.push(sched_seq.id),
                SequencePhase::Decode => decode_ids.push(sched_seq.id),
            }
        }

        let prefill_infos = plan_prefill_inputs(scheduler, prefix_cache, &prefill_ids, block_size);
        let batch = build_batch_input(scheduler, &prefill_infos, &decode_ids);

        if batch.all_token_ids.is_empty() {
            let completed = scheduler.retire_finished();
            return Ok(StepOutput {
                new_tokens: Vec::new(),
                completed,
                prefix_cache_hits: 0,
                prefix_cache_misses: 0,
            });
        }

        let prefill_uncached_lens: Vec<usize> =
            prefill_infos.iter().map(|i| i.uncached_len).collect();
        let total_prefill_tokens = batch.total_prefill_tokens;

        let batch_logits = run_forward_pass(
            model.as_ref(),
            device,
            scheduler,
            &prefill_ids,
            &decode_ids,
            &prefill_uncached_lens,
            batch,
        )?;

        let stop = StopRules {
            token_ids: stop_token_ids,
            sequences: stop_token_sequences,
        };
        let mut prefix = PrefixRegistry {
            cache: prefix_cache,
            allocators,
            block_size,
        };
        let mut new_tokens = Vec::new();
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
            &decode_ids,
            &batch_logits,
            total_prefill_tokens,
            &stop,
            &mut new_tokens,
        )?;

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

    pub fn abort_running(&mut self) -> Vec<SequenceId> {
        self.scheduler.abort_all_running()
    }

    pub fn abort_sequence(&mut self, seq_id: SequenceId) -> bool {
        self.scheduler.abort_sequence(seq_id)
    }

    pub fn abort_all(&mut self) -> Vec<SequenceId> {
        self.scheduler.abort_all()
    }

    pub fn has_pending_work(&self) -> bool {
        self.scheduler.has_pending_work()
    }

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

    #[test]
    fn pick_chunk_split_skips_small_batches() {
        assert_eq!(pick_prefill_chunk_split(&[256, 256], 512), None);
        assert_eq!(pick_prefill_chunk_split(&[4096], 4096), None);
        assert_eq!(pick_prefill_chunk_split(&[], 0), None);
    }

    #[test]
    fn pick_chunk_split_balances_two_equal_seqs() {
        assert_eq!(pick_prefill_chunk_split(&[1024, 1024], 2048), Some(1));
    }

    #[test]
    fn pick_chunk_split_handles_uneven_seqs() {
        assert_eq!(pick_prefill_chunk_split(&[200, 1500], 1700), None);
        assert_eq!(pick_prefill_chunk_split(&[800, 800, 800], 2400), Some(2));
    }

    #[test]
    fn pick_chunk_split_returns_none_when_first_seq_dominates() {
        assert_eq!(pick_prefill_chunk_split(&[3000, 100], 3100), Some(1));
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

        fn max_seq_len(&self) -> usize {
            1024
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

        fn max_seq_len(&self) -> usize {
            2048
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
