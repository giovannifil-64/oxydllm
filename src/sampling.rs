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

impl SamplingParams {
    /// True when sampling reduces to a plain argmax — no temperature, logprobs,
    /// penalties, or logit bias. This is the only case greedy speculative
    /// decoding can serve exactly; everything else must use the normal sampler.
    pub fn is_plain_greedy(&self) -> bool {
        self.temperature == 0.0
            && self.top_logprobs_k == 0
            && self.repetition_penalty == 1.0
            && self.frequency_penalty == 0.0
            && self.presence_penalty == 0.0
            && self.logit_bias.is_none()
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
    if params.is_plain_greedy() {
        let token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        // NaN guard reduced in native dtype (no full-vocab F32 cast per token).
        let sum: f32 = logits
            .sum_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_scalar()?;
        if sum.is_nan() {
            candle_core::bail!("model returned NaN logits — numerical instability detected");
        }
        return Ok(SampleOutput {
            token,
            logprob: None,
            top_logprobs: Vec::new(),
        });
    }

    let mut logits_vec: Vec<f32> = logits.to_dtype(candle_core::DType::F32)?.to_vec1()?;

    if logits_vec.iter().any(|l| l.is_nan()) {
        candle_core::bail!("model returned NaN logits — numerical instability detected");
    }

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
        let mut probs_vec = softmax_probs(&logits_vec, params.temperature);
        // Log-probs must reflect the PRE-filter distribution; computed lazily —
        // skipping the full-vocab ln pass when logprobs aren't requested.
        let log_probs: Option<Vec<f32>> =
            (params.top_logprobs_k > 0).then(|| log_probs_from(&probs_vec));

        if params.min_p > 0.0 {
            apply_min_p(&mut probs_vec, params.min_p);
        }
        if params.top_k > 0 {
            apply_top_k(
                &mut probs_vec,
                params.top_k,
                params.seed,
                prev_tokens.len() as u64,
            );
        }
        if params.top_p < 1.0 {
            apply_top_p(&mut probs_vec, params.top_p);
        }

        let token = categorical_sample(&probs_vec, params.seed, prev_tokens.len() as u64)?;

        if let Some(log_probs) = log_probs {
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

fn softmax_probs(logits: &[f32], temperature: f32) -> Vec<f32> {
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
    probs
}

fn log_probs_from(probs: &[f32]) -> Vec<f32> {
    probs.iter().map(|&p| p.ln().max(-100.0_f32)).collect()
}

fn compute_log_probs(logits: &[f32], temperature: f32) -> (Vec<f32>, Vec<f32>) {
    let probs = softmax_probs(logits, temperature);
    (log_probs_from(&probs), probs)
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

thread_local! {
    static F32_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static IDX_SCRATCH: std::cell::RefCell<Vec<(usize, f32)>> = const { std::cell::RefCell::new(Vec::new()) };
    static TIE_IDX_SCRATCH: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

const TIE_BREAK_NAMESPACE: u64 = 0xa55a_a55a_1234_5678;

fn renormalize_in_place(probs: &mut [f32], sum: f32) {
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

fn apply_min_p(probs: &mut [f32], min_p: f32) {
    let max_prob = probs.iter().cloned().fold(0.0_f32, f32::max);
    let threshold = min_p * max_prob;
    let mut sum = 0.0;
    for p in probs.iter_mut() {
        if *p >= threshold {
            sum += *p;
        } else {
            *p = 0.0;
        }
    }
    renormalize_in_place(probs, sum);
}

fn apply_top_k(probs: &mut [f32], k: usize, seed: Option<u64>, step: u64) {
    if k >= probs.len() {
        return;
    }
    F32_SCRATCH.with(|s| {
        let mut scratch = s.borrow_mut();
        scratch.clear();
        scratch.extend_from_slice(probs);
        scratch.select_nth_unstable_by(k, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = scratch[k];

        let count_strict = probs.iter().filter(|&&p| p > threshold).count();
        let need_ties = k.saturating_sub(count_strict);

        if need_ties == 0 || threshold == 0.0 {
            let mut sum = 0.0;
            for p in probs.iter_mut() {
                if *p > threshold {
                    sum += *p;
                } else {
                    *p = 0.0;
                }
            }
            renormalize_in_place(probs, sum);
            return;
        }

        TIE_IDX_SCRATCH.with(|t| {
            let mut tied = t.borrow_mut();
            tied.clear();
            for (i, &p) in probs.iter().enumerate() {
                if p == threshold && p > 0.0 {
                    tied.push(i);
                }
            }

            if tied.len() <= need_ties {
                // Trivial case: every tied position is kept anyway.
                let mut sum = 0.0;
                for p in probs.iter_mut() {
                    if *p >= threshold && *p > 0.0 {
                        sum += *p;
                    } else {
                        *p = 0.0;
                    }
                }
                renormalize_in_place(probs, sum);
                return;
            }

            let base = tie_break_seed(seed, step);
            tied.sort_unstable_by_key(|&i| {
                std::cmp::Reverse(splitmix64_u64(base.wrapping_add(i as u64)))
            });
            tied.truncate(need_ties);
            tied.sort_unstable();

            let mut sum = 0.0;
            let mut tie_cursor = 0usize;
            for (i, p) in probs.iter_mut().enumerate() {
                if *p > threshold {
                    sum += *p;
                } else if tie_cursor < tied.len() && tied[tie_cursor] == i {
                    tie_cursor += 1;
                    sum += *p;
                } else {
                    *p = 0.0;
                }
            }
            renormalize_in_place(probs, sum);
        });
    });
}

fn tie_break_seed(seed: Option<u64>, step: u64) -> u64 {
    let base = seed.unwrap_or_else(thread_rand_u64);
    base.wrapping_add(step) ^ TIE_BREAK_NAMESPACE
}

// Given the candidate list sorted descending by prob, keep the smallest prefix
// whose cumulative mass reaches top_p; zero everything else and renormalize.
fn keep_top_p_prefix(probs: &mut [f32], sorted: &[(usize, f32)], top_p: f32) {
    let mut cumulative = 0.0;
    let mut keep_count = 0;
    for &(_, prob) in sorted.iter() {
        if cumulative >= top_p && cumulative > 0.0 {
            break;
        }
        cumulative += prob;
        keep_count += 1;
    }

    for p in probs.iter_mut() {
        *p = 0.0;
    }
    let mut sum = 0.0;
    for &(idx, prob) in sorted.iter().take(keep_count) {
        probs[idx] = prob;
        sum += prob;
    }
    renormalize_in_place(probs, sum);
}

fn apply_top_p(probs: &mut [f32], top_p: f32) {
    IDX_SCRATCH.with(|s| {
        let mut indexed = s.borrow_mut();
        indexed.clear();

        // Fast path: LLM distributions are concentrated, so the tokens above a
        // max-relative threshold (typically hundreds, not the full vocab) almost
        // always carry >= top_p mass. Every candidate prob >= threshold > every
        // non-candidate prob, so when the candidate mass covers top_p the global
        // descending-order prefix is provably contained in the candidates — the
        // full-vocab sort reduces to sorting just the candidates.
        let max_prob = probs.iter().cloned().fold(0.0_f32, f32::max);
        let threshold = max_prob * 1e-4;
        let mut candidate_mass = 0.0_f32;
        for (i, &p) in probs.iter().enumerate() {
            if p >= threshold {
                indexed.push((i, p));
                candidate_mass += p;
            }
        }
        let fast_hit = candidate_mass >= top_p;
        if !fast_hit {
            // Flat distribution: candidates don't cover top_p — full sort.
            indexed.clear();
            indexed.extend(probs.iter().enumerate().map(|(i, &p)| (i, p)));
        }
        top_p_probe(fast_hit, indexed.len());
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        keep_top_p_prefix(probs, &indexed, top_p);
    });
}

// Gated diagnostics (OXYDLLM_PROFILE_DECODE=1): fast-path hit rate + candidate
// count for apply_top_p on real model logits, reported every 256 calls.
fn top_p_probe(fast_hit: bool, candidates: usize) {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    static EN: OnceLock<bool> = OnceLock::new();
    if !*EN.get_or_init(|| std::env::var("OXYDLLM_PROFILE_DECODE").as_deref() == Ok("1")) {
        return;
    }
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static HITS: AtomicU64 = AtomicU64::new(0);
    static CAND: AtomicU64 = AtomicU64::new(0);
    let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if fast_hit {
        HITS.fetch_add(1, Ordering::Relaxed);
    }
    CAND.fetch_add(candidates as u64, Ordering::Relaxed);
    if n.is_multiple_of(256) {
        eprintln!(
            "top_p probe: calls={n} fast_hit={:.1}% avg_candidates={}",
            HITS.load(Ordering::Relaxed) as f64 / n as f64 * 100.0,
            CAND.load(Ordering::Relaxed) / n
        );
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
    candle_core::bail!("sampling distribution is empty after filtering (all probs zero)")
}

fn splitmix64_u64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn splitmix64_f32(x: u64) -> f32 {
    (splitmix64_u64(x) >> 40) as f32 / ((1u64 << 24) as f32)
}

thread_local! {
    static THREAD_RNG_STATE: std::cell::Cell<u64> = std::cell::Cell::new({
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::SystemTime;
        let mut hasher = DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        std::thread::current().id().hash(&mut hasher);
        let s = hasher.finish();
        if s == 0 { 1 } else { s }
    });
}

fn thread_rand_u64() -> u64 {
    THREAD_RNG_STATE.with(|s| {
        let mut x = s.get();
        // xorshift64 — fine here; we don't need cryptographic quality.
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        x
    })
}

fn thread_rand_f32() -> f32 {
    (thread_rand_u64() >> 40) as f32 / ((1u64 << 24) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Decomposition bench for the non-greedy per-token sampling cost (E2E measured
    // ~2.2 ms/token vs greedy). Times each stage on a realistic peaked 151936-vocab
    // distribution so the dominant cost is identified by measurement, not guessed.
    #[cfg(feature = "metal")]
    #[test]
    fn sampling_cost_decomposition() {
        use std::time::Instant;
        let Ok(dev) = candle_core::Device::new_metal(0) else {
            return;
        };
        let vocab = 151_936usize;
        // Peaked, LLM-like spectrum: a strong head + noise tail.
        let logits_f32: Vec<f32> = (0..vocab)
            .map(|i| {
                let noise = ((i * 2654435761) % 1000) as f32 / 1000.0;
                if i < 512 {
                    8.0 - (i as f32) * 0.02 + noise
                } else {
                    noise
                }
            })
            .collect();
        let logits_gpu = Tensor::from_vec(logits_f32.clone(), (vocab,), &dev)
            .unwrap()
            .to_dtype(candle_core::DType::BF16)
            .unwrap();

        let iters = 100;
        let timeit = |f: &mut dyn FnMut()| -> f64 {
            for _ in 0..5 {
                f();
            }
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.elapsed().as_secs_f64() * 1e6 / iters as f64
        };

        let cast_copy = timeit(&mut || {
            let v: Vec<f32> = logits_gpu
                .to_dtype(candle_core::DType::F32)
                .unwrap()
                .to_vec1()
                .unwrap();
            std::hint::black_box(v);
        });
        let v = logits_f32.clone();
        let nan_scan = timeit(&mut || {
            std::hint::black_box(v.iter().any(|l| l.is_nan()));
        });
        let softmax_full = timeit(&mut || {
            std::hint::black_box(compute_log_probs(&v, 0.8));
        });
        // The ln pass alone (the part wasted when top_logprobs_k == 0).
        let probs_only: Vec<f32> = {
            let (_, p) = compute_log_probs(&v, 0.8);
            p
        };
        let ln_pass = timeit(&mut || {
            let lp: Vec<f32> = probs_only.iter().map(|&p| p.ln().max(-100.0)).collect();
            std::hint::black_box(lp);
        });
        let top_p = timeit(&mut || {
            let mut p = probs_only.clone();
            apply_top_p(&mut p, 0.95);
            std::hint::black_box(p);
        });
        let clone_cost = timeit(&mut || {
            std::hint::black_box(probs_only.clone());
        });
        let categorical = timeit(&mut || {
            let _ = categorical_sample(&probs_only, Some(7), 0);
        });

        println!("sampling decomposition (vocab={vocab}, us/token):");
        println!("  gpu cast+copy : {cast_copy:8.1}");
        println!("  nan scan      : {nan_scan:8.1}");
        println!("  softmax+ln    : {softmax_full:8.1}  (ln pass alone: {ln_pass:.1})");
        println!(
            "  top_p (sort)  : {:8.1}  (minus clone {clone_cost:.1})",
            top_p - clone_cost
        );
        println!("  categorical   : {categorical:8.1}");
    }

    // Contract: the threshold fast path in apply_top_p produces EXACTLY the same
    // filtered+renormalized distribution as the reference full-vocab sort, across
    // peaked (fast path) and near-flat (fallback path) continuous distributions.
    #[test]
    fn top_p_fast_path_matches_full_sort_reference() {
        fn reference_top_p(probs: &mut [f32], top_p: f32) {
            let mut indexed: Vec<(usize, f32)> =
                probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            keep_top_p_prefix(probs, &indexed, top_p);
        }

        let mut rng: u64 = 0x1234_5678;
        let mut next_f32 = move || {
            rng = splitmix64_u64(rng);
            (rng >> 40) as f32 / (1u64 << 24) as f32
        };

        for (case, vocab, peak) in [
            ("peaked-small", 512usize, 12.0f32),
            ("peaked-large", 50_000, 10.0),
            ("mild", 50_000, 3.0),
            ("near-flat (fallback)", 50_000, 0.01),
        ] {
            // Continuous random logits (ties have measure zero) + a decaying head.
            let logits: Vec<f32> = (0..vocab)
                .map(|i| {
                    let head = if i < 64 { peak - i as f32 * 0.1 } else { 0.0 };
                    head + next_f32()
                })
                .collect();
            for top_p in [0.5f32, 0.9, 0.95, 0.99] {
                let base = softmax_probs(&logits, 0.8);
                let mut fast = base.clone();
                let mut reference = base;
                apply_top_p(&mut fast, top_p);
                reference_top_p(&mut reference, top_p);
                assert_eq!(
                    fast, reference,
                    "{case} top_p={top_p}: fast path diverged from full-sort reference"
                );
            }
        }
    }

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
        let mut probs = vec![0.1, 0.5, 0.3, 0.1];
        apply_top_k(&mut probs, 2, Some(7), 0);
        let nonzero: Vec<_> = probs.iter().filter(|&&p| p > 0.0).collect();
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
        let mut probs = vec![0.5, 0.3, 0.1, 0.05, 0.05];
        apply_min_p(&mut probs, 0.2);
        assert_eq!(probs[3], 0.0);
        assert_eq!(probs[4], 0.0);
        assert!(probs[0] > 0.0);
    }

    #[test]
    fn top_p_keeps_smallest_prefix_above_cutoff() {
        let mut probs = vec![0.4_f32, 0.3, 0.2, 0.1];
        apply_top_p(&mut probs, 0.6);
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
        assert_eq!(probs[2], 0.0);
        assert_eq!(probs[3], 0.0);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "top_p must renormalize: sum={sum}"
        );
    }

    #[test]
    fn top_k_promotes_ties_to_reach_k() {
        let mut probs = vec![0.1_f32, 0.1, 0.1, 0.7];
        apply_top_k(&mut probs, 3, Some(123), 0);
        let nonzero = probs.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(nonzero, 3, "top-k must include ties up to k: {probs:?}");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn top_k_tie_break_is_unbiased_over_many_seeds() {
        let mut counts = [0usize; 4];
        let trials = 4_000;
        for step in 0..trials as u64 {
            let mut probs = vec![0.25_f32, 0.25, 0.25, 0.25];
            apply_top_k(&mut probs, 2, Some(42), step);
            for (i, &p) in probs.iter().enumerate() {
                if p > 0.0 {
                    counts[i] += 1;
                }
            }
        }
        let expected = trials / 2;
        let band = expected / 5;
        for (i, &c) in counts.iter().enumerate() {
            assert!(
                c.abs_diff(expected) < band,
                "index {i} biased: {c} vs expected ~{expected} (counts={counts:?})"
            );
        }
    }

    #[test]
    fn top_k_tie_break_is_reproducible_with_seed() {
        let mut a = vec![0.1_f32, 0.1, 0.1, 0.1, 0.6];
        let mut b = a.clone();
        apply_top_k(&mut a, 3, Some(99), 5);
        apply_top_k(&mut b, 3, Some(99), 5);
        assert_eq!(a, b, "same seed+step must produce identical tie selection");
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
