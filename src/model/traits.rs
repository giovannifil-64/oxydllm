use candle_core::{Device, Result, Tensor};

pub trait Model {
    fn forward(&self, tokens: &Tensor, start_pos: usize) -> Result<Tensor>;
    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> u32;
    fn max_seq_len(&self) -> usize;
    fn device(&self) -> &Device;
}

fn greedy_sample(logits: &Tensor) -> Result<u32> {
    Ok(logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?)
}

pub fn generate(
    model: &dyn Model,
    prompt_tokens: Vec<u32>,
) -> Result<Vec<u32>> {
    let device = model.device();
    let max_tokens = model.max_seq_len().saturating_sub(prompt_tokens.len());
    let mut tokens = prompt_tokens;
    let mut generated = Vec::new();

    for _ in 0..max_tokens {
        let input = Tensor::from_vec(tokens.clone(), (1, tokens.len()), device)?;
        let logits = model.forward(&input, 0)?;
        let last_logits = logits.squeeze(0)?.get(tokens.len() - 1)?;
        let next = greedy_sample(&last_logits)?;

        tokens.push(next);
        generated.push(next);

        if next == model.eos_token_id() {
            break;
        }
    }

    Ok(generated)
}
