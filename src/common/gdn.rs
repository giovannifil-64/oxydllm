//! Gated DeltaNet linear attention (Qwen3.5 / Qwen3-Next family).
//!
//! Math follows `transformers/models/qwen3_5/modeling_qwen3_5.py`:
//!
//! 1. Causal depthwise conv1d (no bias) + SiLU over the packed q|k|v stream.
//! 2. q, k L2-normalized (eps on the sum of squares), q scaled by `dk^-0.5`.
//! 3. `β = σ(b·x)`; `g = -exp(A_log)·softplus(a·x + dt_bias)`, all F32.
//! 4. Recurrence `S_t = S_{t-1}·exp(g_t) + k_t ⊗ ((v_t - S_{t-1}ᵀk_t)·β_t)`
//!    with output `o_t = S_tᵀq_t`.
//! 5. Gated RMSNorm (norm before gate, plain weight): `rms(o)·w·silu(z)`.
//!
//! Prefill uses the chunked parallel scan ([`chunk_gated_delta_rule`], chunk
//! size 64). The reference inverts the per-chunk unit-lower-triangular system
//! with a sequential row loop; that is O(C) kernel launches per chunk and
//! unusable on Metal, so [`invert_unit_lower`] uses a blocked inversion:
//! doubling product on 16×16 diagonal blocks plus pairwise block combination,
//! O(log C) batched matmuls with bounded intermediates. Decode uses the O(1)
//! recurrent step ([`recurrent_delta_step`]). Per-sequence state (conv window
//! plus S) lives in [`super::paged::PagedKvCache::recurrent_mut`].

use super::config::{BlockConfig, LinearAttnConfig};
use super::gguf_weights::GgufWeights;
use super::linear::{AnyLinear, QLinear, sigmoid, silu};
use super::paged::RecurrentState;
use super::weights::ModelWeights;
use candle_core::{D, DType, Result, Tensor};

/// Prefill chunk size (matches the transformers reference). Override with
/// `OXYDLLM_GDN_CHUNK` for experiments.
const CHUNK_SIZE: usize = 64;
/// Base size for the blocked triangular inversion, see `invert_unit_lower`.
const INVERT_BLOCK: usize = 16;
const L2_NORM_EPS: f64 = 1e-6;

/// Debug probe (env `OXYDLLM_GDN_DEBUG=1`): logs NaN/Inf count and max |x|
/// per stage. Syncs the device, debugging only.
fn debug_probe(label: &str, t: &Tensor) {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("OXYDLLM_GDN_DEBUG").is_ok_and(|v| v == "1")) {
        return;
    }
    let stats = (|| -> Result<(usize, usize, f32)> {
        let v: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let nan = v.iter().filter(|x| x.is_nan()).count();
        let inf = v.iter().filter(|x| x.is_infinite()).count();
        let max = v
            .iter()
            .filter(|x| x.is_finite())
            .fold(0f32, |a, &b| a.max(b.abs()));
        Ok((nan, inf, max))
    })();
    match stats {
        Ok((nan, inf, max)) => {
            if nan > 0 || inf > 0 {
                tracing::warn!("GDN probe {label}: NaN={nan} Inf={inf} max|x|={max:.3e}");
            } else {
                tracing::info!("GDN probe {label}: max|x|={max:.3e}");
            }
        }
        Err(e) => tracing::warn!("GDN probe {label}: stats failed: {e}"),
    }
}

fn chunk_size() -> usize {
    use std::sync::OnceLock;
    static SIZE: OnceLock<usize> = OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("OXYDLLM_GDN_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(CHUNK_SIZE)
    })
}

enum BaProjection {
    Fused(AnyLinear),
    Separate { b: AnyLinear, a: AnyLinear },
}

/// Gated DeltaNet: the linear-attention token mixer of hybrid models (Qwen3.5).
///
/// Replaces softmax attention on the linear layers with a gated delta-rule
/// recurrence: a short causal depthwise convolution over the q/k/v projections,
/// then a per-head state update gated by a learned decay. The state is a
/// fixed-size matrix carried across decode steps (in the paged cache) instead of
/// a growing KV cache. The head geometry comes from
/// [`super::config::LinearAttnConfig`].
///
/// Besides the projections (`in_proj_*`, `out_proj`), the parameters are:
/// `conv_w`, the depthwise conv weight (`[kernel, conv_dim]`); `neg_exp_a_log`
/// and `dt_bias`, the decay `-exp(A_log)` and timestep bias of the recurrence
/// (both F32 `[num_v_heads]`, kept in F32 because the recurrence is
/// precision-sensitive); and `norm_w`, the gated RMSNorm weight (F32
/// `[head_v_dim]`).
///
/// `v_heads_tiled` records the checkpoint's V-head ordering. HuggingFace groups
/// V heads by K head (the k-head of v-head `h` is `h / r`, so q/k expand by
/// `repeat_interleave`), while the llama.cpp GGUF converter reorders them to
/// tiled order (k-head `h % nk`, so q/k expand by whole-block tiling). Both are
/// consistent permutations of the same computation.
pub struct GatedDeltaNet {
    in_proj_qkv: AnyLinear,
    in_proj_z: AnyLinear,
    in_proj_ba: BaProjection,
    out_proj: AnyLinear,
    conv_w: Tensor,
    neg_exp_a_log: Tensor,
    dt_bias: Tensor,
    norm_w: Tensor,
    cfg: LinearAttnConfig,
    rms_norm_eps: f64,
    v_heads_tiled: bool,
}

/// Numerically stable ln(1+eˣ) = relu(x) + ln(1 + e^(-|x|)).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let ln1p = ((x.abs()?.neg()?.exp()? + 1.0)?).log()?;
    x.relu()? + ln1p
}

/// x · rsqrt(Σx² + eps) over the last dim, FLA's l2norm (eps inside the sum).
fn l2norm(x: &Tensor) -> Result<Tensor> {
    let sq = x.sqr()?.sum_keepdim(D::Minus1)?;
    x.broadcast_div(&(sq + L2_NORM_EPS)?.sqrt()?)
}

fn eye(n: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut v = vec![0f32; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    Tensor::from_vec(v, (1, n, n), device)
}

/// Lower-triangular ones mask [1, n, n]; `strict` excludes the diagonal.
fn tril_mask(n: usize, strict: bool, device: &candle_core::Device) -> Result<Tensor> {
    let mut v = vec![0f32; n * n];
    for i in 0..n {
        let bound = if strict { i } else { i + 1 };
        for j in 0..bound {
            v[i * n + j] = 1.0;
        }
    }
    Tensor::from_vec(v, (1, n, n), device)
}

/// (I - A)⁻¹ for strictly-lower-triangular A [b, n, n] via the doubling
/// product ∏ (I + A^(2^j)), exact because A is nilpotent (Aⁿ = 0).
///
/// Only safe for small n: the explicit powers A^(2^j) grow combinatorially
/// (path counts) even when the true inverse is small, and the result relies
/// on cancellation across factors. At n=16 the growth (≤ C(15,7)·ρ⁸ ≈ 4e3)
/// is harmless in F32; at n=64 it overflowed on real prompts with highly
/// correlated keys (repeated tokens) and produced NaN logits.
fn invert_unit_lower_base(a: &Tensor, n: usize) -> Result<Tensor> {
    let id = eye(n, a.device())?;
    let mut inv = a.broadcast_add(&id)?;
    let mut p = a.clone();
    let mut covered = 2usize;
    while covered < n {
        p = p.matmul(&p)?;
        inv = (&inv + p.matmul(&inv)?)?;
        covered *= 2;
    }
    Ok(inv)
}

/// Stable (I - A)⁻¹ for strictly-lower-triangular A [b, n, n]: invert the
/// 16×16 diagonal blocks with the doubling product, then combine pairs
/// level by level with the block identity
///   [[N₁₁, 0], [N₂₁, N₂₂]]⁻¹ = [[M₁₁, 0], [M₂₂·A₂₁·M₁₁, M₂₂]]
/// (N = I - A, M = N⁻¹). Every intermediate is a product of true-inverse
/// blocks and A sub-blocks, all bounded, unlike the explicit large powers
/// of the plain doubling form. This mirrors the delta-rule's own stability
/// (β < 1 keeps N well conditioned), matching the torch reference's
/// sequential forward substitution at ~log₂(n) batched-matmul cost.
fn invert_unit_lower(a: &Tensor, n: usize) -> Result<Tensor> {
    let b = a.dim(0)?;
    if n <= INVERT_BLOCK || !n.is_multiple_of(INVERT_BLOCK) || !n.is_power_of_two() {
        return invert_unit_lower_base(a, n);
    }

    // Diagonal blocks, shaped (b·nb, B, B), inverted batched.
    let nb = n / INVERT_BLOCK;
    let mut diag = Vec::with_capacity(nb);
    for i in 0..nb {
        diag.push(
            a.narrow(1, i * INVERT_BLOCK, INVERT_BLOCK)?
                .narrow(2, i * INVERT_BLOCK, INVERT_BLOCK)?
                .contiguous()?,
        );
    }
    let diag = Tensor::stack(&diag, 1)?.reshape((b * nb, INVERT_BLOCK, INVERT_BLOCK))?;
    let inv = invert_unit_lower_base(&diag, INVERT_BLOCK)?;
    // (b, nblocks, s, s) per level.
    let mut m = inv.reshape((b, nb, INVERT_BLOCK, INVERT_BLOCK))?;

    let mut s = INVERT_BLOCK;
    while s < n {
        let pairs = n / (2 * s);
        // A₂₁ sub-blocks of each pair: rows (2p+1)s.., cols 2p·s.. of A.
        let mut a21 = Vec::with_capacity(pairs);
        for p in 0..pairs {
            a21.push(
                a.narrow(1, (2 * p + 1) * s, s)?
                    .narrow(2, 2 * p * s, s)?
                    .contiguous()?,
            );
        }
        let a21 = Tensor::stack(&a21, 1)?.reshape((b * pairs, s, s))?;

        let m_flat = m.reshape((b, pairs, 2, s, s))?;
        let m11 = m_flat
            .narrow(2, 0, 1)?
            .reshape((b * pairs, s, s))?
            .contiguous()?;
        let m22 = m_flat
            .narrow(2, 1, 1)?
            .reshape((b * pairs, s, s))?
            .contiguous()?;
        let x = m22.matmul(&a21.matmul(&m11)?)?; // M₂₂·A₂₁·M₁₁

        let zero = Tensor::zeros((b * pairs, s, s), DType::F32, a.device())?;
        let top = Tensor::cat(&[&m11, &zero], 2)?;
        let bottom = Tensor::cat(&[&x, &m22], 2)?;
        let merged = Tensor::cat(&[&top, &bottom], 1)?; // (b·pairs, 2s, 2s)
        m = merged.reshape((b, pairs, 2 * s, 2 * s))?;
        s *= 2;
    }
    m.reshape((b, n, n))
}

/// Inclusive prefix sum over the last dim of `[b, n]` via mask matmul
/// (`out[i]` = Σ_{j≤i} `x[j]`).
fn cumsum_last(x: &Tensor, n: usize) -> Result<Tensor> {
    let mut v = vec![0f32; n * n];
    for j in 0..n {
        for i in j..n {
            v[j * n + i] = 1.0;
        }
    }
    let u = Tensor::from_vec(v, (n, n), x.device())?;
    x.matmul(&u)
}

/// One recurrent DeltaNet step. q,k: [h, 1, dk]; v: [h, 1, dv]; g,beta: [h, 1];
/// state: [h, dk, dv]. All F32. Returns ([h, 1, dv], new state).
fn recurrent_delta_step(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dk = q.dim(D::Minus1)?;
    let q = (l2norm(q)? * (1.0 / (dk as f64).sqrt()))?;
    let k = l2norm(k)?;

    let g_exp = g.exp()?.unsqueeze(D::Minus1)?; // [h,1,1]
    let s = state.broadcast_mul(&g_exp)?;
    let kv_mem = k.matmul(&s)?; // [h,1,dv]
    let delta = (v - kv_mem)?.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
    let s = (s + k.transpose(D::Minus2, D::Minus1)?.matmul(&delta)?)?;
    let out = q.matmul(&s)?;
    Ok((out, s))
}

/// Chunked parallel scan. q,k: [h, t, dk]; v: [h, t, dv]; g,beta: [h, t];
/// all F32, raw (pre-l2norm). Returns ([h, t, dv], final state [h, dk, dv]).
fn chunk_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: Option<&Tensor>,
    chunk_size: usize,
) -> Result<(Tensor, Tensor)> {
    let device = q.device().clone();
    let (h, t, dk) = q.dims3()?;
    let dv = v.dim(D::Minus1)?;
    let c = chunk_size;
    let n = t.div_ceil(c);
    let pad = n * c - t;

    let q = (l2norm(q)? * (1.0 / (dk as f64).sqrt()))?;
    let k = l2norm(k)?;

    let pad3 = |x: &Tensor, d: usize| -> Result<Tensor> {
        if pad == 0 {
            return x.contiguous();
        }
        let z = Tensor::zeros((h, pad, d), DType::F32, &device)?;
        Tensor::cat(&[x, &z], 1)
    };
    let pad2 = |x: &Tensor| -> Result<Tensor> {
        if pad == 0 {
            return x.contiguous();
        }
        let z = Tensor::zeros((h, pad), DType::F32, &device)?;
        Tensor::cat(&[x, &z], 1)
    };

    // Flatten (head, chunk) into one batch dim for the intra-chunk math.
    let hn = h * n;
    let q_c = pad3(&q, dk)?.reshape((hn, c, dk))?;
    let k_c = pad3(&k, dk)?.reshape((hn, c, dk))?;
    let v_c = pad3(v, dv)?.reshape((hn, c, dv))?;
    let g_c = cumsum_last(&pad2(g)?.reshape((hn, c))?, c)?;
    let beta_c = pad2(beta)?.reshape((hn, c))?;

    let tril = tril_mask(c, false, &device)?;
    let strict = tril_mask(c, true, &device)?;

    // decay[i][j] = exp(g_i - g_j) on the lower triangle (incl. diagonal).
    let decay = g_c
        .unsqueeze(D::Minus1)?
        .broadcast_sub(&g_c.unsqueeze(D::Minus2)?)?
        .broadcast_mul(&tril)?
        .exp()?
        .broadcast_mul(&tril)?;

    let k_beta = k_c.broadcast_mul(&beta_c.unsqueeze(D::Minus1)?)?;
    let v_beta = v_c.broadcast_mul(&beta_c.unsqueeze(D::Minus1)?)?;

    let a = (k_beta.matmul(&k_c.transpose(D::Minus2, D::Minus1)?)? * decay.clone())?
        .broadcast_mul(&strict)?
        .neg()?;
    debug_probe("chunk.a", &a);
    let inv = invert_unit_lower(&a, c)?;
    debug_probe("chunk.inv", &inv);

    let v_t = inv.matmul(&v_beta)?;
    let k_cumdecay = inv.matmul(&k_beta.broadcast_mul(&g_c.exp()?.unsqueeze(D::Minus1)?)?)?;
    debug_probe("chunk.v_t", &v_t);
    debug_probe("chunk.k_cumdecay", &k_cumdecay);

    let g_exp = g_c.exp()?.unsqueeze(D::Minus1)?; // (hn, c, 1)
    let attn_all = (q_c.matmul(&k_c.transpose(D::Minus2, D::Minus1)?)? * decay)?; // (hn, c, c)
    let qg_all = q_c.broadcast_mul(&g_exp)?; // (hn, c, dk)
    let g_last = g_c.narrow(1, c - 1, 1)?; // (hn, 1)
    let eg_last = g_last.exp()?.reshape((h, n))?; // (h, n)
    let carry_t = k_c
        .broadcast_mul(&g_last.broadcast_sub(&g_c)?.exp()?.unsqueeze(D::Minus1)?)?
        .transpose(D::Minus2, D::Minus1)?
        .contiguous()?;

    let chunked = |x: &Tensor, a: usize, b: usize| x.reshape((h, n, a, b));
    let v_n = chunked(&v_t, c, dv)?;
    let kcd_n = chunked(&k_cumdecay, c, dk)?;
    let qg_n = chunked(&qg_all, c, dk)?;
    let attn_n = chunked(&attn_all, c, c)?;
    let carry_n = chunked(&carry_t, dk, c)?;

    let mut s = match initial_state {
        Some(st) => st.clone(),
        None => Tensor::zeros((h, dk, dv), DType::F32, &device)?,
    };
    let mut outs: Vec<Tensor> = Vec::with_capacity(n);
    for i in 0..n {
        let take = |x: &Tensor| -> Result<Tensor> { x.narrow(1, i, 1)?.squeeze(1)?.contiguous() };
        let v_new = (take(&v_n)? - take(&kcd_n)?.matmul(&s)?)?;
        outs.push((take(&qg_n)?.matmul(&s)? + take(&attn_n)?.matmul(&v_new)?)?);
        let eg_i = eg_last.narrow(1, i, 1)?.unsqueeze(D::Minus1)?; // (h, 1, 1)
        s = (s.broadcast_mul(&eg_i)? + take(&carry_n)?.matmul(&v_new)?)?;
    }

    let out = Tensor::cat(&outs, 1)?.narrow(1, 0, t)?;
    debug_probe("chunk.out", &out);
    debug_probe("chunk.final_state", &s);
    Ok((out, s))
}

impl GatedDeltaNet {
    pub fn load(cfg: &BlockConfig, layer_idx: usize, weights: &ModelWeights) -> Result<Self> {
        let la = cfg.linear_attn.ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "GatedDeltaNet::load on layer {layer_idx} without linear_attn config"
            ))
        })?;
        let p = format!("model.layers.{layer_idx}.linear_attn");
        let key_dim = la.num_k_heads * la.head_k_dim;
        let value_dim = la.num_v_heads * la.head_v_dim;
        let conv_dim = 2 * key_dim + value_dim;

        let proj = |name: &str, expect_out: Option<usize>| -> Result<AnyLinear> {
            let prefix = format!("{p}.{name}");
            // Quantized checkpoints (compressed-tensors / AWQ) pack the large
            // DeltaNet projections; small ones (in_proj_a/b) stay dense.
            if let Some(raw) = weights.try_get_quant(&prefix) {
                let got = raw
                    .out_features()
                    .map_err(|e| candle_core::Error::Msg(format!("{prefix}: {e:#}")))?;
                if let Some(expect) = expect_out
                    && got != expect
                {
                    candle_core::bail!("{prefix}: expected out_features {expect}, got {got}");
                }
                let device = raw.scales.device().clone();
                let dtype = raw.scales.dtype();
                return AnyLinear::from_quant(&raw, None, &device, dtype);
            }
            let weight_name = format!("{prefix}.weight");
            let w = weights.get(&weight_name)?.clone();
            if let Some(expect) = expect_out
                && w.dim(0)? != expect
            {
                candle_core::bail!(
                    "{weight_name}: expected out_features {expect}, got {}",
                    w.dim(0)?
                );
            }
            let scale_inv = weights.try_get_scale_inv(&weight_name).cloned();
            if w.dtype() == DType::F8E4M3 && scale_inv.is_none() {
                candle_core::bail!("missing '{weight_name}_scale_inv' required by FP8 tensor");
            }
            AnyLinear::from_weight_with_scale_inv(w, scale_inv, None)
        };

        let in_proj_qkv = proj("in_proj_qkv", Some(conv_dim))?;
        let in_proj_z = proj("in_proj_z", Some(value_dim))?;
        let out_proj = proj("out_proj", None)?;

        let b_dense = weights.try_get_quant(&format!("{p}.in_proj_b")).is_none()
            && weights
                .try_get(&format!("{p}.in_proj_b.weight"))
                .is_some_and(|w| w.dtype() != DType::F8E4M3);
        let a_dense = weights.try_get_quant(&format!("{p}.in_proj_a")).is_none()
            && weights
                .try_get(&format!("{p}.in_proj_a.weight"))
                .is_some_and(|w| w.dtype() != DType::F8E4M3);
        let in_proj_ba = if b_dense && a_dense {
            let b_w = weights.get(&format!("{p}.in_proj_b.weight"))?;
            let a_w = weights.get(&format!("{p}.in_proj_a.weight"))?;
            if b_w.dim(0)? != la.num_v_heads || a_w.dim(0)? != la.num_v_heads {
                candle_core::bail!(
                    "{p}.in_proj_b/a: expected out_features {}, got {}/{}",
                    la.num_v_heads,
                    b_w.dim(0)?,
                    a_w.dim(0)?
                );
            }
            let ba_w = Tensor::cat(&[b_w, a_w], 0)?;
            BaProjection::Fused(AnyLinear::from_weight_with_scale_inv(ba_w, None, None)?)
        } else {
            BaProjection::Separate {
                b: proj("in_proj_b", Some(la.num_v_heads))?,
                a: proj("in_proj_a", Some(la.num_v_heads))?,
            }
        };

        let conv_raw = weights.get(&format!("{p}.conv1d.weight"))?; // [conv_dim, 1, k]
        if conv_raw.dims() != [conv_dim, 1, la.conv_kernel] {
            candle_core::bail!(
                "{p}.conv1d.weight: expected [{conv_dim}, 1, {}], got {:?}",
                la.conv_kernel,
                conv_raw.dims()
            );
        }
        let conv_w = conv_raw.squeeze(1)?.transpose(0, 1)?.contiguous()?; // [k, conv_dim]

        let f32_vec = |name: &str, expect: usize| -> Result<Tensor> {
            let t = weights.get(&format!("{p}.{name}"))?;
            if t.elem_count() != expect {
                candle_core::bail!(
                    "{p}.{name}: expected {expect} elements, got {}",
                    t.elem_count()
                );
            }
            t.to_dtype(DType::F32)
        };
        let neg_exp_a_log = f32_vec("A_log", la.num_v_heads)?.exp()?.neg()?;
        let dt_bias = f32_vec("dt_bias", la.num_v_heads)?;
        let norm_w = f32_vec("norm.weight", la.head_v_dim)?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_ba,
            out_proj,
            conv_w,
            neg_exp_a_log,
            dt_bias,
            norm_w,
            cfg: la,
            rms_norm_eps: cfg.rms_norm_eps,
            v_heads_tiled: false,
        })
    }

    pub fn load_gguf(
        cfg: &BlockConfig,
        layer_idx: usize,
        gguf: &GgufWeights,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        let la = cfg.linear_attn.ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "GatedDeltaNet::load_gguf on layer {layer_idx} without linear_attn config"
            ))
        })?;
        let prefix = format!("blk.{layer_idx}");
        let key_dim = la.num_k_heads * la.head_k_dim;
        let value_dim = la.num_v_heads * la.head_v_dim;
        let conv_dim = 2 * key_dim + value_dim;

        let qproj = |name: &str, expect_out: usize| -> Result<AnyLinear> {
            let qt = gguf.get(&format!("{prefix}.{name}.weight"))?;
            let got = qt.shape().dims()[0];
            if got != expect_out {
                candle_core::bail!(
                    "{prefix}.{name}.weight: expected out_features {expect_out}, got {got}"
                );
            }
            Ok(AnyLinear::Quantized(QLinear::from_arc(qt, dtype)?))
        };
        let in_proj_qkv = qproj("attn_qkv", conv_dim)?;
        let in_proj_z = qproj("attn_gate", value_dim)?;
        let in_proj_ba = BaProjection::Separate {
            b: qproj("ssm_beta", la.num_v_heads)?,
            a: qproj("ssm_alpha", la.num_v_heads)?,
        };
        let out_proj = {
            let qt = gguf.get(&format!("{prefix}.ssm_out.weight"))?;
            AnyLinear::Quantized(QLinear::from_arc(qt, dtype)?)
        };

        let conv_raw = gguf
            .get(&format!("{prefix}.ssm_conv1d.weight"))?
            .dequantize(device)?;
        let conv_w = match *conv_raw.dims() {
            [c, k] if c == conv_dim && k == la.conv_kernel => {
                conv_raw.transpose(0, 1)?.contiguous()?
            }
            [k, c] if c == conv_dim && k == la.conv_kernel => conv_raw.contiguous()?,
            _ => candle_core::bail!(
                "{prefix}.ssm_conv1d.weight: expected [{conv_dim}, {}] or transposed, got {:?}",
                la.conv_kernel,
                conv_raw.dims()
            ),
        }
        .to_dtype(dtype)?;

        let f32_vec = |name: &str, expect: usize| -> Result<Tensor> {
            let t = gguf.get(&format!("{prefix}.{name}"))?.dequantize(device)?;
            if t.elem_count() != expect {
                candle_core::bail!(
                    "{prefix}.{name}: expected {expect} elements, got {}",
                    t.elem_count()
                );
            }
            t.to_dtype(DType::F32)
        };
        // llama.cpp's converter bakes the transform: GGUF `ssm_a` already
        // holds -exp(A_log), use it verbatim.
        let neg_exp_a_log = f32_vec("ssm_a", la.num_v_heads)?;
        let dt_bias = f32_vec("ssm_dt.bias", la.num_v_heads)?;
        let norm_w = f32_vec("ssm_norm.weight", la.head_v_dim)?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_ba,
            out_proj,
            conv_w,
            neg_exp_a_log,
            dt_bias,
            norm_w,
            cfg: la,
            rms_norm_eps: cfg.rms_norm_eps,
            v_heads_tiled: true,
        })
    }

    /// Causal depthwise conv + SiLU on the packed stream [1, t, conv_dim].
    /// Returns (activated stream, new conv window [1, k-1, conv_dim]).
    fn conv_forward(&self, qkv: &Tensor, prev_window: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (_, t, conv_dim) = qkv.dims3()?;
        let k = self.cfg.conv_kernel;

        if let Some(window) = prev_window {
            // Single-token decode: slide the window, one dot per channel.
            debug_assert_eq!(t, 1, "conv window path requires t == 1");
            let window = Tensor::cat(&[window, qkv], 1)?; // [1, k, c]
            let out = window
                .broadcast_mul(&self.conv_w.unsqueeze(0)?)?
                .sum_keepdim(1)?; // [1, 1, c]
            let new_window = window.narrow(1, 1, k - 1)?.contiguous()?;
            return Ok((silu(&out)?, new_window));
        }

        // Fresh prefill: left-pad k-1 zeros, sum k shifted broadcast products.
        let zeros = Tensor::zeros((1, k - 1, conv_dim), qkv.dtype(), qkv.device())?;
        let padded = Tensor::cat(&[&zeros, qkv], 1)?; // [1, t+k-1, c]
        let mut acc: Option<Tensor> = None;
        for j in 0..k {
            let term = padded
                .narrow(1, j, t)?
                .broadcast_mul(&self.conv_w.narrow(0, j, 1)?)?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        let new_window = padded.narrow(1, t, k - 1)?.contiguous()?;
        Ok((silu(&acc.expect("conv_kernel >= 1"))?, new_window))
    }

    fn project(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let nv = self.cfg.num_v_heads;
        let qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?;

        let to_h_t = |x: &Tensor| -> Result<Tensor> {
            x.squeeze(0)?
                .transpose(0, 1)?
                .to_dtype(DType::F32)?
                .contiguous()
        };
        let (b_t, a_t) = match &self.in_proj_ba {
            BaProjection::Fused(p) => {
                let ba = to_h_t(&p.forward(x)?)?; // [2·nv, T]
                (ba.narrow(0, 0, nv)?, ba.narrow(0, nv, nv)?)
            }
            BaProjection::Separate { b, a } => (to_h_t(&b.forward(x)?)?, to_h_t(&a.forward(x)?)?),
        };
        let beta = sigmoid(&b_t)?;
        let g = softplus(&a_t.broadcast_add(&self.dt_bias.unsqueeze(1)?)?)?
            .broadcast_mul(&self.neg_exp_a_log.unsqueeze(1)?)?;
        Ok((qkv, z, beta, g))
    }

    fn mix_segment(
        &self,
        qkv: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        state: &mut Option<RecurrentState>,
    ) -> Result<Tensor> {
        let la = &self.cfg;
        let t = qkv.dim(1)?;
        let key_dim = la.num_k_heads * la.head_k_dim;
        let value_dim = la.num_v_heads * la.head_v_dim;
        let rep = la.num_v_heads / la.num_k_heads;

        if state.is_some() && t > 1 {
            // The engine never splits a prompt across forwards (prefill is
            // whole-prompt, decode is single-token), so this cannot happen.
            candle_core::bail!("GatedDeltaNet: multi-token continuation is not supported");
        }

        let prev = state.as_ref().map(|s| &s.conv);
        let (qkv_act, new_conv) = self.conv_forward(qkv, prev)?;
        let qkv_f32 = qkv_act.to_dtype(DType::F32)?;

        let to_heads = |x: &Tensor, n_heads: usize, hd: usize| -> Result<Tensor> {
            x.reshape((1, t, n_heads, hd))?
                .squeeze(0)?
                .transpose(0, 1)?
                .contiguous()
        };
        let q = to_heads(
            &qkv_f32.narrow(2, 0, key_dim)?,
            la.num_k_heads,
            la.head_k_dim,
        )?;
        let k = to_heads(
            &qkv_f32.narrow(2, key_dim, key_dim)?,
            la.num_k_heads,
            la.head_k_dim,
        )?;
        let v = to_heads(
            &qkv_f32.narrow(2, 2 * key_dim, value_dim)?,
            la.num_v_heads,
            la.head_v_dim,
        )?;

        let expand_heads = |x: &Tensor| -> Result<Tensor> {
            if rep == 1 {
                return Ok(x.clone());
            }
            let (hk, tt, d) = x.dims3()?;
            if self.v_heads_tiled {
                let copies: Vec<&Tensor> = std::iter::repeat_n(x, rep).collect();
                Tensor::cat(&copies, 0)
            } else {
                x.unsqueeze(1)?
                    .expand((hk, rep, tt, d))?
                    .contiguous()?
                    .reshape((hk * rep, tt, d))
            }
        };
        let q = expand_heads(&q)?;
        let k = expand_heads(&k)?;

        debug_probe("fwd.qkv_act", &qkv_act);
        debug_probe("fwd.g", g);
        debug_probe("fwd.v", &v);
        let (core, new_s) = match state.as_ref() {
            Some(st) if t == 1 => recurrent_delta_step(&q, &k, &v, g, beta, &st.s)?,
            _ => chunk_gated_delta_rule(&q, &k, &v, g, beta, None, chunk_size())?,
        };
        debug_probe("fwd.core", &core);
        *state = Some(RecurrentState {
            conv: new_conv,
            s: new_s,
        });

        core.transpose(0, 1)?
            .reshape((1, t, la.num_v_heads, la.head_v_dim))
    }

    fn finish(&self, core: &Tensor, z: &Tensor, in_dtype: DType) -> Result<Tensor> {
        let la = &self.cfg;
        let t = core.dim(1)?;
        let value_dim = la.num_v_heads * la.head_v_dim;
        let z_f32 = z
            .reshape((1, t, la.num_v_heads, la.head_v_dim))?
            .to_dtype(DType::F32)?;

        #[cfg(feature = "metal")]
        let gated = if core.device().is_metal() {
            let normed = super::metal_ops::rms_norm_fused(
                &core.contiguous()?,
                &self.norm_w,
                self.rms_norm_eps as f32,
            )?;
            super::metal_ops::silu_mul_fused(&z_f32.contiguous()?, &normed)?
        } else {
            Self::gated_norm_fallback(core, &z_f32, &self.norm_w, self.rms_norm_eps)?
        };
        #[cfg(not(feature = "metal"))]
        let gated = Self::gated_norm_fallback(core, &z_f32, &self.norm_w, self.rms_norm_eps)?;

        let out = gated.reshape((1, t, value_dim))?.to_dtype(in_dtype)?;
        self.out_proj.forward(&out)
    }

    fn gated_norm_fallback(
        core: &Tensor,
        z_f32: &Tensor,
        norm_w: &Tensor,
        eps: f64,
    ) -> Result<Tensor> {
        let var = core.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = core
            .broadcast_div(&(var + eps)?.sqrt()?)?
            .broadcast_mul(norm_w)?;
        normed * silu(z_f32)?
    }

    pub fn forward_segment(
        &self,
        x: &Tensor,
        state: &mut Option<RecurrentState>,
    ) -> Result<Tensor> {
        let (qkv, z, beta, g) = self.project(x)?;
        let core = self.mix_segment(&qkv, &beta, &g, state)?;
        self.finish(&core, &z, x.dtype())
    }

    pub fn forward_batch(
        &self,
        x: &Tensor,
        token_counts: &[usize],
        states: &mut [&mut Option<RecurrentState>],
    ) -> Result<Tensor> {
        debug_assert_eq!(token_counts.len(), states.len());
        if states.len() == 1 {
            return self.forward_segment(x, states[0]);
        }

        let (qkv, z, beta, g) = self.project(x)?;
        let mut cores = Vec::with_capacity(states.len());
        let mut offset = 0usize;
        for (state, &t) in states.iter_mut().zip(token_counts) {
            let qkv_seg = qkv.narrow(1, offset, t)?.contiguous()?;
            let beta_seg = beta.narrow(1, offset, t)?.contiguous()?;
            let g_seg = g.narrow(1, offset, t)?.contiguous()?;
            cores.push(self.mix_segment(&qkv_seg, &beta_seg, &g_seg, state)?);
            offset += t;
        }
        let core = Tensor::cat(&cores, 1)?;
        self.finish(&core, &z, x.dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use serde_json::Value;

    /// Golden reference values computed with the VERBATIM torch fallback
    /// functions of transformers `modeling_qwen3_5.py` (main @ 2026-06-11):
    /// seeded random delta-rule inputs with their expected outputs/states,
    /// plus a full small-dims layer (prefill + cached decode step). Embedding
    /// them pins our scan/recurrence against the official math without
    /// requiring PyTorch at test time. The generator (a ~250-line script that
    /// copies `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule`
    /// verbatim and dumps inputs+outputs as JSON) lives in the untracked
    /// `scripts/` folder as local dev tooling.
    const FIXTURES: &str = include_str!("gdn_fixtures.json");

    fn vecf(v: &Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    /// [B=1, T, H, D] (torch layout) to [H, T, D].
    fn to_htd(data: Vec<f32>, t: usize, h: usize, d: usize) -> Tensor {
        Tensor::from_vec(data, (1, t, h, d), &Device::Cpu)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .transpose(0, 1)
            .unwrap()
            .contiguous()
            .unwrap()
    }

    /// [B=1, T, H] to [H, T].
    fn to_ht(data: Vec<f32>, t: usize, h: usize) -> Tensor {
        Tensor::from_vec(data, (1, t, h), &Device::Cpu)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .transpose(0, 1)
            .unwrap()
            .contiguous()
            .unwrap()
    }

    fn assert_close(got: &Tensor, want: &[f32], tol: f32, what: &str) {
        let got: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).abs() < tol,
                "{what}[{i}]: got {a}, want {b} (tol {tol})"
            );
        }
    }

    /// Contract: the chunked scan reproduces the transformers reference
    /// (torch_chunk_gated_delta_rule) bit-for-bit up to F32 tolerance,
    /// including the final recurrent state used to seed decode.
    #[test]
    fn chunked_scan_matches_transformers_reference() {
        let fx: Value = serde_json::from_str(FIXTURES).unwrap();
        let f = &fx["delta_rule"];
        let (t, h, dk, dv) = (
            f["dims"]["t"].as_u64().unwrap() as usize,
            f["dims"]["h"].as_u64().unwrap() as usize,
            f["dims"]["dk"].as_u64().unwrap() as usize,
            f["dims"]["dv"].as_u64().unwrap() as usize,
        );
        let q = to_htd(vecf(&f["q"]), t, h, dk);
        let k = to_htd(vecf(&f["k"]), t, h, dk);
        let v = to_htd(vecf(&f["v"]), t, h, dv);
        let g = to_ht(vecf(&f["g"]), t, h);
        let beta = to_ht(vecf(&f["beta"]), t, h);

        // chunk_size 4 exercises multiple chunks + padding (t=10 to 3 chunks).
        let (out, state) = chunk_gated_delta_rule(&q, &k, &v, &g, &beta, None, 4).unwrap();
        let out_bthd = out.transpose(0, 1).unwrap().contiguous().unwrap(); // [T,H,DV]
        assert_close(&out_bthd, &vecf(&f["out"]), 1e-4, "chunk out");
        assert_close(&state, &vecf(&f["final_state"]), 1e-4, "chunk final state");

        // Chunk size must not change the result (exact algebraic identity).
        let (out64, state64) = chunk_gated_delta_rule(&q, &k, &v, &g, &beta, None, 64).unwrap();
        let out64 = out64.transpose(0, 1).unwrap().contiguous().unwrap();
        assert_close(&out64, &vecf(&f["out"]), 1e-4, "chunk64 out");
        assert_close(&state64, &vecf(&f["final_state"]), 1e-4, "chunk64 state");
    }

    /// Contract: the recurrent decode step matches the reference both
    /// token-by-token over a fresh sequence and as a continuation from a
    /// chunked-prefill state (the prefill-to-decode handoff).
    #[test]
    fn recurrent_step_matches_reference_and_chunked_prefill() {
        let fx: Value = serde_json::from_str(FIXTURES).unwrap();
        let f = &fx["delta_rule"];
        let (t, h, dk, dv) = (
            f["dims"]["t"].as_u64().unwrap() as usize,
            f["dims"]["h"].as_u64().unwrap() as usize,
            f["dims"]["dk"].as_u64().unwrap() as usize,
            f["dims"]["dv"].as_u64().unwrap() as usize,
        );
        let q = to_htd(vecf(&f["q"]), t, h, dk);
        let k = to_htd(vecf(&f["k"]), t, h, dk);
        let v = to_htd(vecf(&f["v"]), t, h, dv);
        let g = to_ht(vecf(&f["g"]), t, h);
        let beta = to_ht(vecf(&f["beta"]), t, h);

        // Token-by-token recurrence over the whole sequence.
        let mut s = Tensor::zeros((h, dk, dv), DType::F32, &Device::Cpu).unwrap();
        let mut outs = Vec::new();
        for i in 0..t {
            let sl3 = |x: &Tensor| x.narrow(1, i, 1).unwrap().contiguous().unwrap();
            let (o, ns) =
                recurrent_delta_step(&sl3(&q), &sl3(&k), &sl3(&v), &sl3(&g), &sl3(&beta), &s)
                    .unwrap();
            s = ns;
            outs.push(o);
        }
        let out = Tensor::cat(&outs, 1).unwrap();
        let out_bthd = out.transpose(0, 1).unwrap().contiguous().unwrap();
        assert_close(&out_bthd, &vecf(&f["out"]), 1e-4, "recurrent out");
        assert_close(&s, &vecf(&f["final_state"]), 1e-4, "recurrent state");

        // Continuation step from the reference final state.
        let st = &f["step"];
        let q1 = to_htd(vecf(&st["q"]), 1, h, dk);
        let k1 = to_htd(vecf(&st["k"]), 1, h, dk);
        let v1 = to_htd(vecf(&st["v"]), 1, h, dv);
        let g1 = to_ht(vecf(&st["g"]), 1, h);
        let b1 = to_ht(vecf(&st["beta"]), 1, h);
        let (o1, s1) = recurrent_delta_step(&q1, &k1, &v1, &g1, &b1, &s).unwrap();
        let o1 = o1.transpose(0, 1).unwrap().contiguous().unwrap();
        assert_close(&o1, &vecf(&st["out"]), 1e-4, "step out");
        assert_close(&s1, &vecf(&st["final_state"]), 1e-4, "step state");
    }

    fn fixture_layer(f: &Value) -> (GatedDeltaNet, usize, usize) {
        use rustc_hash::FxHashMap;
        let d = &f["dims"];
        let (hidden, nk, nv, dk, dv, conv_k, t) = (
            d["hidden"].as_u64().unwrap() as usize,
            d["nk"].as_u64().unwrap() as usize,
            d["nv"].as_u64().unwrap() as usize,
            d["dk"].as_u64().unwrap() as usize,
            d["dv"].as_u64().unwrap() as usize,
            d["conv_k"].as_u64().unwrap() as usize,
            d["t"].as_u64().unwrap() as usize,
        );
        let dev = Device::Cpu;
        let key_dim = nk * dk;
        let value_dim = nv * dv;
        let conv_dim = 2 * key_dim + value_dim;

        let mut tensors = FxHashMap::default();
        let mut ins = |name: &str, data: Vec<f32>, shape: Vec<usize>| {
            tensors.insert(
                format!("model.layers.0.linear_attn.{name}"),
                Tensor::from_vec(data, shape, &dev).unwrap(),
            );
        };
        ins(
            "in_proj_qkv.weight",
            vecf(&f["w_qkv"]),
            vec![conv_dim, hidden],
        );
        ins("in_proj_z.weight", vecf(&f["w_z"]), vec![value_dim, hidden]);
        ins("in_proj_b.weight", vecf(&f["w_b"]), vec![nv, hidden]);
        ins("in_proj_a.weight", vecf(&f["w_a"]), vec![nv, hidden]);
        ins(
            "out_proj.weight",
            vecf(&f["w_out"]),
            vec![hidden, value_dim],
        );
        ins(
            "conv1d.weight",
            vecf(&f["conv_w"]),
            vec![conv_dim, 1, conv_k],
        );
        ins("A_log", vecf(&f["a_log"]), vec![nv]);
        ins("dt_bias", vecf(&f["dt_bias"]), vec![nv]);
        ins("norm.weight", vecf(&f["norm_w"]), vec![dv]);
        let weights = ModelWeights::from_tensors(tensors);

        let cfg = BlockConfig {
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 1,
            rms_norm_eps: 1e-6,
            qk_norm: false,
            attention_scale: None,
            activation: super::super::config::Activation::SiLU,
            norm_type: super::super::config::NormType::Standard,
            attn_softcap: None,
            residual_multiplier: None,
            v_norm: false,
            has_ffn_norms: false,
            sliding_window: None,
            moe: None,
            linear_attn: Some(LinearAttnConfig {
                num_k_heads: nk,
                num_v_heads: nv,
                head_k_dim: dk,
                head_v_dim: dv,
                conv_kernel: conv_k,
            }),
            attn_output_gate: false,
            rotary_dim: None,
            gguf_qk_permuted: false,
        };
        (GatedDeltaNet::load(&cfg, 0, &weights).unwrap(), hidden, t)
    }

    /// Contract: the full layer (projections, causal conv, delta rule, gated
    /// norm, out_proj) matches the transformers reference on a fresh
    /// prefill AND on the subsequent cached decode step, including the conv
    /// window handoff between the two.
    #[test]
    fn full_layer_matches_reference_prefill_then_decode() {
        let fx: Value = serde_json::from_str(FIXTURES).unwrap();
        let f = &fx["gdn_layer"];
        let (layer, hidden, t) = fixture_layer(f);
        let dev = Device::Cpu;

        let x = Tensor::from_vec(vecf(&f["x_prefill"]), (1, t, hidden), &dev).unwrap();
        let mut state: Option<RecurrentState> = None;
        let out = layer.forward_segment(&x, &mut state).unwrap();
        assert_close(&out, &vecf(&f["out_prefill"]), 2e-4, "layer prefill out");
        assert!(state.is_some(), "prefill must seed the recurrent state");

        let x1 = Tensor::from_vec(vecf(&f["x_step"]), (1, 1, hidden), &dev).unwrap();
        let out1 = layer.forward_segment(&x1, &mut state).unwrap();
        assert_close(&out1, &vecf(&f["out_step"]), 2e-4, "layer decode out");
        assert_close(
            &state.as_ref().unwrap().s,
            &vecf(&f["final_state"]),
            2e-4,
            "layer final state",
        );
    }

    /// Contract: the chunked scan stays numerically stable on adversarial
    /// inputs. Long runs of identical tokens make every in-chunk key pair
    /// fully correlated; the plain doubling inversion computed explicit
    /// matrix powers that overflowed F32 here (NaN logits on real prompts)
    /// even though the true inverse is well conditioned. With the blocked
    /// inversion the scan at the shipped chunk size must agree with the
    /// unconditionally stable recurrent path.
    #[test]
    fn chunked_scan_stable_on_fully_correlated_keys() {
        let dev = Device::Cpu;
        let (h, t, dk, dv) = (2usize, 128usize, 16usize, 16usize);

        // Identical key vector at every position (the worst case), strong β,
        // weak decay, mirrors a long run of one repeated token.
        let k_one: Vec<f32> = (0..dk).map(|i| (i as f32 * 0.37).sin() + 0.1).collect();
        let k_data: Vec<f32> = (0..h * t).flat_map(|_| k_one.clone()).collect();
        let k = Tensor::from_vec(k_data, (h, t, dk), &dev).unwrap();
        let mk = |seed: f32| {
            let d: Vec<f32> = (0..h * t * dk)
                .map(|i| ((i as f32 * seed).sin()) * 0.8)
                .collect();
            Tensor::from_vec(d, (h, t, dk), &dev).unwrap()
        };
        let q = mk(0.13);
        let v = mk(0.29);
        let beta = Tensor::full(0.95f32, (h, t), &dev).unwrap();
        let g = Tensor::full(-0.02f32, (h, t), &dev).unwrap();

        let (out, state) = chunk_gated_delta_rule(&q, &k, &v, &g, &beta, None, CHUNK_SIZE).unwrap();
        let out_v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            out_v.iter().all(|x| x.is_finite()),
            "chunked scan must stay finite on fully correlated keys"
        );

        // Reference: token-by-token recurrence (no large intermediates).
        let mut s = Tensor::zeros((h, dk, dv), DType::F32, &dev).unwrap();
        let mut outs = Vec::new();
        for i in 0..t {
            let sl = |x: &Tensor| x.narrow(1, i, 1).unwrap().contiguous().unwrap();
            let (o, ns) =
                recurrent_delta_step(&sl(&q), &sl(&k), &sl(&v), &sl(&g), &sl(&beta), &s).unwrap();
            s = ns;
            outs.push(o);
        }
        let ref_out = Tensor::cat(&outs, 1).unwrap();
        let ref_v: Vec<f32> = ref_out.flatten_all().unwrap().to_vec1().unwrap();
        let max_diff = out_v
            .iter()
            .zip(ref_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "chunked vs recurrent diverged on correlated keys: max diff {max_diff}"
        );
        let state_v: Vec<f32> = state.flatten_all().unwrap().to_vec1().unwrap();
        assert!(state_v.iter().all(|x| x.is_finite()));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_gated_norm_matches_fallback_on_metal() {
        let Ok(dev) = Device::new_metal(0) else {
            return; // no Metal device available (CI)
        };
        let (t, nv, dv) = (5usize, 4usize, 8usize);
        let core = Tensor::randn(0f32, 2f32, (1, t, nv, dv), &dev).unwrap();
        let z = Tensor::randn(0f32, 2f32, (1, t, nv, dv), &dev).unwrap();
        let w = Tensor::randn(0f32, 1f32, (dv,), &dev).unwrap();
        let eps = 1e-6f64;

        let fallback = GatedDeltaNet::gated_norm_fallback(&core, &z, &w, eps).unwrap();
        let fused = {
            let normed = crate::common::metal_ops::rms_norm_fused(
                &core.contiguous().unwrap(),
                &w,
                eps as f32,
            )
            .unwrap();
            crate::common::metal_ops::silu_mul_fused(&z.contiguous().unwrap(), &normed).unwrap()
        };
        let a: Vec<f32> = fallback.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-5, "fused vs fallback max diff {max_diff}");
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore]
    fn gdn_decode_step_cost_decomposition() {
        use rustc_hash::FxHashMap;
        use std::time::Instant;

        let dev = Device::new_metal(0).unwrap();
        let dtype = DType::BF16;
        let (hidden, nk, nv, dk, dv, conv_k) =
            (4096usize, 16usize, 32usize, 128usize, 128usize, 4usize);
        let key_dim = nk * dk;
        let value_dim = nv * dv;
        let conv_dim = 2 * key_dim + value_dim;

        let mut tensors = FxHashMap::default();
        let mut ins = |name: &str, shape: Vec<usize>, f32_dtype: bool| {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) * 1e-3).collect();
            let t = Tensor::from_vec(data, shape, &dev).unwrap();
            let t = if f32_dtype {
                t
            } else {
                t.to_dtype(dtype).unwrap()
            };
            tensors.insert(format!("model.layers.0.linear_attn.{name}"), t);
        };
        ins("in_proj_qkv.weight", vec![conv_dim, hidden], false);
        ins("in_proj_z.weight", vec![value_dim, hidden], false);
        ins("in_proj_b.weight", vec![nv, hidden], false);
        ins("in_proj_a.weight", vec![nv, hidden], false);
        ins("out_proj.weight", vec![hidden, value_dim], false);
        ins("conv1d.weight", vec![conv_dim, 1, conv_k], false);
        ins("A_log", vec![nv], true);
        ins("dt_bias", vec![nv], true);
        ins("norm.weight", vec![dv], true);
        let weights = ModelWeights::from_tensors(tensors);

        let cfg = BlockConfig {
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 1,
            rms_norm_eps: 1e-6,
            qk_norm: false,
            attention_scale: None,
            activation: super::super::config::Activation::SiLU,
            norm_type: super::super::config::NormType::Standard,
            attn_softcap: None,
            residual_multiplier: None,
            v_norm: false,
            has_ffn_norms: false,
            sliding_window: None,
            moe: None,
            linear_attn: Some(LinearAttnConfig {
                num_k_heads: nk,
                num_v_heads: nv,
                head_k_dim: dk,
                head_v_dim: dv,
                conv_kernel: conv_k,
            }),
            attn_output_gate: false,
            rotary_dim: None,
            gguf_qk_permuted: false,
        };
        let layer = GatedDeltaNet::load(&cfg, 0, &weights).unwrap();

        let x1 = Tensor::randn(0f32, 1f32, (1, 1, hidden), &dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let sync = |label: &str, start: Instant, iters: u32| {
            dev.synchronize().unwrap();
            println!(
                "{label}: {:.2} ms/iter",
                start.elapsed().as_secs_f64() * 1e3 / iters as f64
            );
        };

        // Prefill to seed the state, then warm decode steps.
        let mut state: Option<RecurrentState> = None;
        let x64 = Tensor::randn(0f32, 1f32, (1, 64, hidden), &dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let t0 = Instant::now();
        layer.forward_segment(&x64, &mut state).unwrap();
        sync("prefill T=64 (cold)", t0, 1);
        let t0 = Instant::now();
        let mut state2: Option<RecurrentState> = None;
        layer.forward_segment(&x64, &mut state2).unwrap();
        sync("prefill T=64 (warm)", t0, 1);

        for _ in 0..3 {
            layer.forward_segment(&x1, &mut state).unwrap();
        }
        dev.synchronize().unwrap();

        const N: u32 = 10;
        let t0 = Instant::now();
        for _ in 0..N {
            layer.forward_segment(&x1, &mut state).unwrap();
        }
        sync("decode step (full)", t0, N);

        let t0 = Instant::now();
        for _ in 0..N {
            let _ = layer.project(&x1).unwrap();
        }
        sync("projections+βg only", t0, N);

        // Sub-part: recurrent delta step on F32 tensors.
        let q = Tensor::randn(0f32, 1f32, (nv, 1, dk), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (nv, 1, dk), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (nv, 1, dv), &dev).unwrap();
        let g = Tensor::randn(0f32, 1f32, (nv, 1), &dev).unwrap();
        let beta = Tensor::randn(0f32, 1f32, (nv, 1), &dev).unwrap();
        let s0 = Tensor::zeros((nv, dk, dv), DType::F32, &dev).unwrap();
        let t0 = Instant::now();
        for _ in 0..N {
            let _ = recurrent_delta_step(&q, &k, &v, &g, &beta, &s0).unwrap();
        }
        sync("delta step only", t0, N);

        // Sub-part: conv window update.
        let window = Tensor::randn(0f32, 1f32, (1, conv_k - 1, conv_dim), &dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let qkv1 = Tensor::randn(0f32, 1f32, (1, 1, conv_dim), &dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let t0 = Instant::now();
        for _ in 0..N {
            let _ = layer.conv_forward(&qkv1, Some(&window)).unwrap();
        }
        sync("conv update only", t0, N);

        // Sub-part: sigmoid/softplus scalar chains.
        let small = Tensor::randn(0f32, 1f32, (nv, 1), &dev).unwrap();
        let t0 = Instant::now();
        for _ in 0..N {
            let _ = sigmoid(&small).unwrap();
            let _ = softplus(&small).unwrap();
        }
        sync("sigmoid+softplus only", t0, N);
    }

    /// Contract: a continuation with t > 1 (which the engine never produces)
    /// fails loudly instead of silently corrupting the recurrence.
    #[test]
    fn multi_token_continuation_is_rejected() {
        let fx: Value = serde_json::from_str(FIXTURES).unwrap();
        let f = &fx["gdn_layer"];
        let (layer, hidden, t) = fixture_layer(f);
        let dev = Device::Cpu;

        let x = Tensor::from_vec(vecf(&f["x_prefill"]), (1, t, hidden), &dev).unwrap();
        let mut state: Option<RecurrentState> = None;
        layer.forward_segment(&x, &mut state).unwrap();

        let x2 = Tensor::zeros((1, 2, hidden), DType::F32, &dev).unwrap();
        assert!(layer.forward_segment(&x2, &mut state).is_err());
    }
}
