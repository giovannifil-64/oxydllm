use candle_core::{DType, Device, Result, Tensor};
use crate::sampling::{self, SamplingParams};
use super::common::paged::{PagedKvCache, SharedBlockAllocator};

pub trait Model {
    fn forward(&mut self, tokens: &Tensor, start_pos: usize) -> Result<Tensor>;
    fn clear_cache(&mut self);
    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> u32;
    fn max_seq_len(&self) -> usize;
    fn device(&self) -> &Device;
}

pub trait BatchModel {
    fn forward_with_cache(
        &self,
        tokens: &Tensor,
        start_pos: usize,
        caches: &mut [PagedKvCache],
    ) -> Result<Tensor>;

    fn forward_batch(
        &self,
        token_ids: &Tensor,
        position_ids: &Tensor,
        seq_caches: &mut [&mut [PagedKvCache]],
        token_counts: &[usize],
    ) -> Result<Tensor>;

    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> u32;
    fn max_seq_len(&self) -> usize;
    fn device(&self) -> &Device;
    fn num_layers(&self) -> usize;
    fn n_kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn dtype(&self) -> DType;

    fn allocators(&self) -> &[SharedBlockAllocator];
}

pub fn generate(
    model: &mut dyn Model,
    prompt_tokens: Vec<u32>,
    params: &SamplingParams,
) -> Result<Vec<u32>> {
    let device = model.device().clone();
    let max_tokens = model.max_seq_len().saturating_sub(prompt_tokens.len());
    let prompt_len = prompt_tokens.len();

    model.clear_cache();

    let input = Tensor::from_vec(prompt_tokens.clone(), (1, prompt_len), &device)?;
    let logits = model.forward(&input, 0)?;
    let last_logits = logits.squeeze(0)?.get(prompt_len - 1)?;

    let mut all_tokens = prompt_tokens;
    let mut next = sampling::sample(&last_logits, params, &all_tokens)?;
    all_tokens.push(next);
    let mut generated = vec![next];

    if next == model.eos_token_id() {
        return Ok(generated);
    }

    for _ in 1..max_tokens {
        let start_pos = all_tokens.len() - 1;
        let input = Tensor::from_vec(vec![next], (1, 1), &device)?;
        let logits = model.forward(&input, start_pos)?;
        let last_logits = logits.squeeze(0)?.get(0)?;
        next = sampling::sample(&last_logits, params, &all_tokens)?;

        all_tokens.push(next);
        generated.push(next);

        if next == model.eos_token_id() {
            break;
        }
    }

    Ok(generated)
}
