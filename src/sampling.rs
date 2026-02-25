use candle_core::{Result, Tensor, D};

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
        }
    }
}

pub fn sample(logits: &Tensor, params: &SamplingParams, prev_tokens: &[u32]) -> Result<u32> {
    if params.temperature == 0.0 {
        return Ok(logits.argmax(D::Minus1)?.to_scalar::<u32>()?);
    }

    let mut logits = logits.to_dtype(candle_core::DType::F32)?;

    if params.repetition_penalty != 1.0 {
        logits = apply_repetition_penalty(&logits, prev_tokens, params.repetition_penalty)?;
    }

    logits = (&logits / params.temperature as f64)?;

    let max_logit = logits.max(D::Minus1)?.to_scalar::<f32>()?;
    let logits = (&logits - max_logit as f64)?;
    let exp = logits.exp()?;
    let sum_exp = exp.sum(D::Minus1)?.to_scalar::<f32>()?;
    let probs = (&exp / sum_exp as f64)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    let probs_vec = if params.min_p > 0.0 {
        apply_min_p(&probs_vec, params.min_p)
    } else {
        probs_vec
    };

    let probs_vec = if params.top_k > 0 {
        apply_top_k(&probs_vec, params.top_k)
    } else {
        probs_vec
    };

    let probs_vec = if params.top_p < 1.0 {
        apply_top_p(&probs_vec, params.top_p)
    } else {
        probs_vec
    };

    categorical_sample(&probs_vec)
}

fn apply_repetition_penalty(
    logits: &Tensor,
    prev_tokens: &[u32],
    penalty: f32,
) -> Result<Tensor> {
    let mut logits_vec: Vec<f32> = logits.to_vec1()?;
    for &tok in prev_tokens {
        let idx = tok as usize;
        if idx < logits_vec.len() {
            let l = logits_vec[idx];
            logits_vec[idx] = if l > 0.0 { l / penalty } else { l * penalty };
        }
    }
    Tensor::from_vec(logits_vec, logits.shape(), logits.device())
}

fn apply_min_p(probs: &[f32], min_p: f32) -> Vec<f32> {
    let max_prob = probs.iter().cloned().fold(0.0_f32, f32::max);
    let threshold = min_p * max_prob;
    let mut filtered: Vec<f32> = probs.iter().map(|&p| if p >= threshold { p } else { 0.0 }).collect();
    renormalize(&mut filtered);
    filtered
}

fn apply_top_k(probs: &[f32], k: usize) -> Vec<f32> {
    if k >= probs.len() {
        return probs.to_vec();
    }
    let mut sorted: Vec<f32> = probs.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold = sorted[k];

    let mut filtered: Vec<f32> = probs.iter().map(|&p| if p > threshold { p } else { 0.0 }).collect();

    let mut count = filtered.iter().filter(|&&p| p > 0.0).count();
    if count < k {
        for i in 0..filtered.len() {
            if count >= k {
                break;
            }
            if filtered[i] == 0.0 && probs[i] == threshold {
                filtered[i] = probs[i];
                count += 1;
            }
        }
    }

    renormalize(&mut filtered);
    filtered
}

fn apply_top_p(probs: &[f32], top_p: f32) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumulative = 0.0;
    let mut filtered = vec![0.0_f32; probs.len()];

    for (idx, prob) in indexed {
        if cumulative >= top_p {
            break;
        }
        filtered[idx] = prob;
        cumulative += prob;
    }

    renormalize(&mut filtered);
    filtered
}

fn renormalize(probs: &mut [f32]) {
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

fn categorical_sample(probs: &[f32]) -> Result<u32> {
    let r: f32 = fastrand_f32();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return Ok(i as u32);
        }
    }
    for (i, &p) in probs.iter().enumerate().rev() {
        if p > 0.0 {
            return Ok(i as u32);
        }
    }
    Ok(0)
}

fn fastrand_f32() -> f32 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    use std::time::SystemTime;

    thread_local! {
        static STATE: std::cell::Cell<u32> = std::cell::Cell::new({
            let mut hasher = DefaultHasher::new();
            SystemTime::now().hash(&mut hasher);
            std::thread::current().id().hash(&mut hasher);
            let h = hasher.finish();
            let s = (h ^ (h >> 32)) as u32;
            if s == 0 { 1 } else { s }
        });
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        s.set(x);
        (x as f32) / (u32::MAX as f32)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_returns_argmax() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0_f32, 5.0, 3.0, 2.0], (4,), &device).unwrap();
        let params = SamplingParams::default();
        let token = sample(&logits, &params, &[]).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn temperature_sampling_returns_valid_token() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0_f32, 5.0, 3.0, 2.0], (4,), &device).unwrap();
        let params = SamplingParams {
            temperature: 0.8,
            ..Default::default()
        };
        let token = sample(&logits, &params, &[]).unwrap();
        assert!(token < 4);
    }

    #[test]
    fn top_k_filters_correctly() {
        let probs = vec![0.1, 0.5, 0.3, 0.1];
        let filtered = apply_top_k(&probs, 2);
        let nonzero: Vec<_> = filtered.iter().filter(|&&p| p > 0.0).collect();
        assert_eq!(nonzero.len(), 2);
    }

    #[test]
    fn repetition_penalty_reduces_repeated_logits() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![2.0_f32, 3.0, 1.0], (3,), &device).unwrap();
        let penalized = apply_repetition_penalty(&logits, &[1], 2.0).unwrap();
        let vals: Vec<f32> = penalized.to_vec1().unwrap();
        assert!(vals[1] < 3.0);
        assert_eq!(vals[0], 2.0);
    }

    #[test]
    fn min_p_filters_low_prob() {
        let probs = vec![0.5, 0.3, 0.1, 0.05, 0.05];
        let filtered = apply_min_p(&probs, 0.2);
        assert_eq!(filtered[3], 0.0);
        assert_eq!(filtered[4], 0.0);
        assert!(filtered[0] > 0.0);
    }
}
