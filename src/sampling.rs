use candle_core::{Result, Tensor, D};

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: None,
        }
    }
}

pub fn sample(logits: &Tensor, params: &SamplingParams, prev_tokens: &[u32]) -> Result<u32> {
    if params.temperature == 0.0 {
        return logits.argmax(D::Minus1)?.to_scalar::<u32>();
    }

    let mut logits_vec: Vec<f32> = logits.to_dtype(candle_core::DType::F32)?.to_vec1()?;

    if params.repetition_penalty != 1.0 {
        apply_repetition_penalty_cpu(&mut logits_vec, prev_tokens, params.repetition_penalty);
    }

    let inv_temp = 1.0 / params.temperature;
    for l in logits_vec.iter_mut() {
        *l *= inv_temp;
    }

    let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for l in logits_vec.iter_mut() {
        *l = (*l - max_logit).exp();
        sum += *l;
    }
    let inv_sum = 1.0 / sum;
    for p in logits_vec.iter_mut() {
        *p *= inv_sum;
    }

    let probs_vec = logits_vec;

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

    categorical_sample(&probs_vec, params.seed)
}

fn apply_repetition_penalty_cpu(logits: &mut [f32], prev_tokens: &[u32], penalty: f32) {
    for &tok in prev_tokens {
        let idx = tok as usize;
        if idx < logits.len() {
            let l = logits[idx];
            logits[idx] = if l > 0.0 { l / penalty } else { l * penalty };
        }
    }
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
    let mut temp: Vec<f32> = probs.to_vec();
    temp.select_nth_unstable_by(k, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = temp[k];

    let mut filtered: Vec<f32> = probs.iter().map(|&p| if p > threshold { p } else { 0.0 }).collect();
    let mut count = filtered.iter().filter(|&&p| p > 0.0).count();

    if count < k {
        for (i, &p) in probs.iter().enumerate() {
            if count >= k {
                break;
            }
            if filtered[i] == 0.0 && p == threshold && p > 0.0 {
                filtered[i] = p;
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

fn categorical_sample(probs: &[f32], seed: Option<u64>) -> Result<u32> {
    let r: f32 = match seed {
        Some(s) => seeded_rand_f32(s),
        None => thread_rand_f32(),
    };
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

fn seeded_rand_f32(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z = z ^ (z >> 31);
    (z >> 40) as f32 / ((1u64 << 24) as f32)
}

fn thread_rand_f32() -> f32 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    use std::time::SystemTime;

    thread_local! {
        static STATE: std::cell::Cell<u64> = std::cell::Cell::new({
            let mut hasher = DefaultHasher::new();
            SystemTime::now().hash(&mut hasher);
            std::thread::current().id().hash(&mut hasher);
            let s = hasher.finish();
            if s == 0 { 1 } else { s }
        });
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 40) as f32 / ((1u64 << 24) as f32)
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
        let mut logits = vec![2.0_f32, 3.0, 1.0];
        apply_repetition_penalty_cpu(&mut logits, &[1], 2.0);
        assert!(logits[1] < 3.0);
        assert_eq!(logits[0], 2.0);
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
