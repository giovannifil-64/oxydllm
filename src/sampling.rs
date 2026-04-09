use candle_core::{D, Result, Tensor};

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    /// Number of trailing tokens considered for repetition penalty.
    /// 0 means full history (current default behavior).
    pub repetition_window: usize,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: Option<u64>,
    pub logit_bias: Option<Vec<(u32, f32)>>,
    pub top_logprobs_k: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            repetition_window: 0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            logit_bias: None,
            top_logprobs_k: 0,
        }
    }
}

pub struct SampleOutput {
    pub token: u32,
    pub logprob: Option<f32>,
    pub top_logprobs: Vec<(u32, f32)>,
}

pub fn sample(
    logits: &Tensor,
    params: &SamplingParams,
    prev_tokens: &[u32],
    token_counts: Option<&std::collections::HashMap<u32, u32>>,
) -> Result<SampleOutput> {
    let no_mods = params.repetition_penalty == 1.0
        && params.frequency_penalty == 0.0
        && params.presence_penalty == 0.0
        && params.logit_bias.is_none();

    if params.temperature == 0.0 && params.top_logprobs_k == 0 && no_mods {
        let token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        return Ok(SampleOutput {
            token,
            logprob: None,
            top_logprobs: Vec::new(),
        });
    }

    let mut logits_vec: Vec<f32> = logits.to_dtype(candle_core::DType::F32)?.to_vec1()?;

    if params.repetition_penalty != 1.0 {
        let repetition_tokens = repetition_window_slice(prev_tokens, params.repetition_window);
        let can_use_full_counts = repetition_tokens.len() == prev_tokens.len();
        if can_use_full_counts {
            if let Some(counts) = token_counts {
                apply_repetition_penalty_with_counts(
                    &mut logits_vec,
                    counts,
                    params.repetition_penalty,
                );
            } else {
                apply_repetition_penalty_cpu(
                    &mut logits_vec,
                    repetition_tokens,
                    params.repetition_penalty,
                );
            }
        } else {
            apply_repetition_penalty_cpu(
                &mut logits_vec,
                repetition_tokens,
                params.repetition_penalty,
            );
        }
    }
    if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
        if let Some(counts) = token_counts {
            apply_frequency_presence_penalty_with_counts(
                &mut logits_vec,
                counts,
                params.frequency_penalty,
                params.presence_penalty,
            );
        } else {
            apply_frequency_presence_penalty(
                &mut logits_vec,
                prev_tokens,
                params.frequency_penalty,
                params.presence_penalty,
            );
        }
    }
    if let Some(ref bias) = params.logit_bias {
        for &(tok, b) in bias {
            let idx = tok as usize;
            if idx < logits_vec.len() {
                logits_vec[idx] += b;
            }
        }
    }

    if params.temperature == 0.0 {
        let token = logits_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        if params.top_logprobs_k > 0 {
            let (log_probs, _) = compute_log_probs(&logits_vec, 1.0);
            let lp = log_probs
                .get(token as usize)
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            let top = top_k_by_logprob(&log_probs, params.top_logprobs_k);
            Ok(SampleOutput {
                token,
                logprob: Some(lp),
                top_logprobs: top,
            })
        } else {
            Ok(SampleOutput {
                token,
                logprob: None,
                top_logprobs: Vec::new(),
            })
        }
    } else {
        let (log_probs, probs_vec) = compute_log_probs(&logits_vec, params.temperature);

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

        let token = categorical_sample(&probs_vec, params.seed, prev_tokens.len() as u64)?;

        if params.top_logprobs_k > 0 {
            let lp = log_probs
                .get(token as usize)
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            let top = top_k_by_logprob(&log_probs, params.top_logprobs_k);
            Ok(SampleOutput {
                token,
                logprob: Some(lp),
                top_logprobs: top,
            })
        } else {
            Ok(SampleOutput {
                token,
                logprob: None,
                top_logprobs: Vec::new(),
            })
        }
    }
}

fn repetition_window_slice(prev_tokens: &[u32], repetition_window: usize) -> &[u32] {
    if repetition_window == 0 || repetition_window >= prev_tokens.len() {
        prev_tokens
    } else {
        &prev_tokens[prev_tokens.len() - repetition_window..]
    }
}

fn compute_log_probs(logits: &[f32], temperature: f32) -> (Vec<f32>, Vec<f32>) {
    let temp = temperature.max(1e-8_f32);
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max_logit) / temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    let inv_sum = 1.0 / sum.max(1e-10_f32);
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }
    let log_probs: Vec<f32> = probs.iter().map(|&p| p.ln().max(-100.0_f32)).collect();
    (log_probs, probs)
}

fn top_k_by_logprob(log_probs: &[f32], k: usize) -> Vec<(u32, f32)> {
    if k == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(u32, f32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &lp)| (i as u32, lp))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
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

fn apply_repetition_penalty_with_counts(
    logits: &mut [f32],
    token_counts: &std::collections::HashMap<u32, u32>,
    penalty: f32,
) {
    for (&tok, &count) in token_counts {
        let idx = tok as usize;
        if idx < logits.len() {
            let l = logits[idx];
            let factor = penalty.powf(count as f32);
            logits[idx] = if l > 0.0 { l / factor } else { l * factor };
        }
    }
}

fn apply_frequency_presence_penalty(
    logits: &mut [f32],
    prev_tokens: &[u32],
    frequency_penalty: f32,
    presence_penalty: f32,
) {
    let mut counts = std::collections::HashMap::<u32, u32>::new();
    for &tok in prev_tokens {
        *counts.entry(tok).or_insert(0) += 1;
    }
    apply_frequency_presence_penalty_with_counts(
        logits,
        &counts,
        frequency_penalty,
        presence_penalty,
    );
}

fn apply_frequency_presence_penalty_with_counts(
    logits: &mut [f32],
    counts: &std::collections::HashMap<u32, u32>,
    frequency_penalty: f32,
    presence_penalty: f32,
) {
    for (&tok, &count) in counts {
        let idx = tok as usize;
        if idx < logits.len() {
            logits[idx] -= frequency_penalty * count as f32 + presence_penalty;
        }
    }
}

fn apply_min_p(probs: &[f32], min_p: f32) -> Vec<f32> {
    let max_prob = probs.iter().cloned().fold(0.0_f32, f32::max);
    let threshold = min_p * max_prob;
    let mut filtered: Vec<f32> = probs
        .iter()
        .map(|&p| if p >= threshold { p } else { 0.0 })
        .collect();
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

    let mut filtered: Vec<f32> = probs
        .iter()
        .map(|&p| if p > threshold { p } else { 0.0 })
        .collect();
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
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0;
    let mut filtered = vec![0.0_f32; probs.len()];

    for (idx, prob) in indexed {
        if cumulative >= top_p && cumulative > 0.0 {
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

fn categorical_sample(probs: &[f32], seed: Option<u64>, step: u64) -> Result<u32> {
    let r: f32 = match seed {
        Some(s) => splitmix64_f32(s.wrapping_add(step)),
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

fn splitmix64_f32(x: u64) -> f32 {
    let mut z = x.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z = z ^ (z >> 31);
    (z >> 40) as f32 / ((1u64 << 24) as f32)
}

fn thread_rand_f32() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
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
        let token = sample(&logits, &params, &[], None).unwrap().token;
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
        let token = sample(&logits, &params, &[], None).unwrap().token;
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
    fn repetition_window_limits_penalty_scope() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![10.0_f32, 9.0, 8.0], (3,), &device).unwrap();
        let prev_tokens = vec![0_u32, 0, 0, 1];
        let mut counts = std::collections::HashMap::new();
        counts.insert(0_u32, 3_u32);
        counts.insert(1_u32, 1_u32);

        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            repetition_window: 1,
            ..Default::default()
        };

        let out = sample(&logits, &params, &prev_tokens, Some(&counts)).unwrap();
        assert_eq!(
            out.token, 0,
            "only the last token should be repetition-penalized"
        );
    }

    #[test]
    fn repetition_window_zero_uses_full_history() {
        let prev_tokens = vec![1_u32, 2, 3, 4];
        let full = repetition_window_slice(&prev_tokens, 0);
        assert_eq!(full, prev_tokens.as_slice());
    }

    #[test]
    fn seeded_sampling_varies_across_steps() {
        let probs = vec![0.25_f32, 0.25, 0.25, 0.25];
        let seed = Some(42u64);
        let r0 = categorical_sample(&probs, seed, 0).unwrap();
        let r1 = categorical_sample(&probs, seed, 1).unwrap();
        let r2 = categorical_sample(&probs, seed, 2).unwrap();

        assert!(
            !(r0 == r1 && r1 == r2),
            "seeded sampling returned same token for 3 consecutive steps"
        );
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let probs = vec![0.1_f32, 0.5, 0.3, 0.1];
        let seed = Some(123u64);
        let a = categorical_sample(&probs, seed, 7).unwrap();
        let b = categorical_sample(&probs, seed, 7).unwrap();
        assert_eq!(a, b, "same seed+step must produce same result");
    }

    #[test]
    fn min_p_filters_low_prob() {
        let probs = vec![0.5, 0.3, 0.1, 0.05, 0.05];
        let filtered = apply_min_p(&probs, 0.2);
        assert_eq!(filtered[3], 0.0);
        assert_eq!(filtered[4], 0.0);
        assert!(filtered[0] > 0.0);
    }

    #[test]
    fn logprobs_returned_when_requested() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0_f32, 5.0, 3.0, 2.0], (4,), &device).unwrap();
        let params = SamplingParams {
            temperature: 0.8,
            top_logprobs_k: 3,
            ..Default::default()
        };
        let out = sample(&logits, &params, &[], None).unwrap();
        assert!(out.logprob.is_some(), "logprob should be set");
        assert_eq!(out.top_logprobs.len(), 3, "should have 3 top logprobs");
        let lps: Vec<f32> = out.top_logprobs.iter().map(|&(_, lp)| lp).collect();
        for i in 1..lps.len() {
            assert!(
                lps[i - 1] >= lps[i],
                "top_logprobs should be sorted descending"
            );
        }
    }

    #[test]
    fn logit_bias_shifts_logits() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![10.0_f32, 5.0, 3.0, 2.0], (4,), &device).unwrap();
        let params = SamplingParams {
            temperature: 0.0,
            logit_bias: Some(vec![(0, -100.0)]),
            ..Default::default()
        };
        let out = sample(&logits, &params, &[], None).unwrap();
        assert_ne!(out.token, 0, "token 0 should be suppressed by logit_bias");
    }

    #[test]
    fn greedy_logprobs_sum_to_one() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0_f32, 5.0, 3.0, 2.0], (4,), &device).unwrap();
        let params = SamplingParams {
            temperature: 0.0,
            top_logprobs_k: 4,
            ..Default::default()
        };
        let out = sample(&logits, &params, &[], None).unwrap();
        let sum: f32 = out.top_logprobs.iter().map(|&(_, lp)| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-4, "exp(logprobs) should sum to ~1");
    }
}
