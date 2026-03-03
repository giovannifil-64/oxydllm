// ─────────────────────────────────────────────────────────────────────────────
// metal_ops.rs  — Metal-accelerated SDPA for rLLM
// ─────────────────────────────────────────────────────────────────────────────
//
// Wraps candle-metal-kernels' built-in Scaled Dot Product Attention (SDPA)
// shaders — derived from MLX — via candle's `CustomOp3` trait so rLLM can
// call fused Flash-Attention-like kernels on Apple Silicon without custom
// .metal files.
//
// Approach inspired by Crane (https://github.com/lucasjinreal/Crane), which
// also leverages candle's built-in Metal SDPA kernels for fast inference on
// Apple Silicon.
//
// The SDPA kernel has two paths:
//   • **vector** (seq_q ≤ 8, used in decode):
//       – `call_sdpa_vector` for short KV, or
//       – `call_sdpa_vector_2pass` when KV length ≥ 1024
//   • **full** (seq_q > 8, used in prefill):
//       – `call_sdpa_full` with optional mask and causal flag
//
// Both support GQA natively (n_heads can be a multiple of n_kv_heads),
// so `repeat_kv` is NOT needed when using SDPA.
// ─────────────────────────────────────────────────────────────────────────────

use candle_core::{
    backend::BackendStorage, CpuStorage, CustomOp3, DType, Layout,
    MetalStorage, Result, Shape, Tensor, D,
};
use candle_metal_kernels::SdpaDType;

// ─── Sdpa CustomOp3 ──────────────────────────────────────────────────────────

struct Sdpa {
    scale: f32,
    softcapping: f32,
    mask: Option<Tensor>,
    do_causal: bool,
}

impl CustomOp3 for Sdpa {
    fn name(&self) -> &'static str {
        "metal-sdpa"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("SDPA: Metal-only — use standard attention path on CPU")
    }

    fn metal_fwd(
        &self,
        q: &MetalStorage,
        q_l: &Layout,
        k: &MetalStorage,
        k_l: &Layout,
        v: &MetalStorage,
        v_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = q.device();

        let out_dims = vec![q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, v_l.dim(3)?];
        let elem_count: usize = out_dims.iter().product();
        let out_shape = Shape::from_dims(&out_dims);
        let out_layout = Layout::contiguous(out_shape.clone());

        let output = device.new_buffer(elem_count, q.dtype(), "sdpa_o")?;

        // ── Validate shapes ──────────────────────────────────────────
        if q_l.dim(D::Minus1)? != k_l.dim(D::Minus1)? {
            candle_core::bail!("`q` and `k` last dims must match");
        }
        if v_l.dim(D::Minus(3))? != k_l.dim(D::Minus(3))? {
            candle_core::bail!("`k` and `v` head dims must match");
        }
        if q_l.dim(D::Minus(3))? % k_l.dim(D::Minus(3))? != 0 {
            candle_core::bail!("query `n_heads` must be a multiple of `n_kv_heads`");
        }

        let q_head = q_l.dim(D::Minus1)?;
        let q_seq = q_l.dim(2)?;
        let k_seq = k_l.dim(2)?;

        let supported_head_dim = matches!(q_head, 32 | 64 | 72 | 80 | 96 | 128 | 256);

        let supports_sdpa_full_mask = self.mask.is_none() || q_seq <= k_seq;
        let supports_sdpa_full =
            q_seq > 8 && supported_head_dim && supports_sdpa_full_mask;
        let supports_sdpa_vector =
            q_seq <= 8 && supported_head_dim && q_seq <= k_seq;

        if !supported_head_dim || !(supports_sdpa_full || supports_sdpa_vector) {
            candle_core::bail!(
                "Metal SDPA does not support q dims {:?}, k dims {:?}, v dims {:?}",
                q_l.dims(), k_l.dims(), v_l.dims()
            );
        }

        for t in [k.dtype(), v.dtype()] {
            if q.dtype() != t {
                candle_core::bail!("all q, k, v dtypes must match for SDPA");
            }
        }

        let itype = match q.dtype() {
            DType::BF16 => SdpaDType::BF16,
            DType::F16 => SdpaDType::F16,
            DType::F32 => SdpaDType::F32,
            other => candle_core::bail!("unsupported SDPA dtype {other:?}"),
        };

        // ── Dispatch ─────────────────────────────────────────────────
        let encoder = device.command_encoder()?;

        if supports_sdpa_vector {
            const TWO_PASS_K_THRESHOLD: usize = 1024;

            if k_seq >= TWO_PASS_K_THRESHOLD {
                let mut intermediate_shape = [
                    &out_dims[0..out_dims.len() - 2],
                    &[candle_metal_kernels::SDPA_2PASS_BLOCKS],
                    &[out_dims[out_dims.len() - 1]],
                ]
                .concat();
                let intermediate = device.new_buffer(
                    intermediate_shape.iter().product::<usize>(),
                    DType::F32,
                    "sdpa_2pass_intermediate",
                )?;
                let _ = intermediate_shape.pop().unwrap();
                let sums = device.new_buffer(
                    intermediate_shape.iter().product::<usize>(),
                    DType::F32,
                    "sdpa_2pass_sums",
                )?;
                let maxs = device.new_buffer(
                    intermediate_shape.iter().product::<usize>(),
                    DType::F32,
                    "sdpa_2pass_maxs",
                )?;

                candle_metal_kernels::call_sdpa_vector_2pass(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    q_l.start_offset(),
                    q_l.dims(),
                    q.buffer(),
                    k_l.start_offset(),
                    k_l.dims(),
                    k_l.stride(),
                    k.buffer(),
                    v_l.start_offset(),
                    v_l.stride(),
                    v.buffer(),
                    &output,
                    &intermediate,
                    &sums,
                    &maxs,
                    self.scale,
                    self.softcapping,
                    itype,
                )
                .map_err(candle_core::Error::wrap)?;
            } else {
                candle_metal_kernels::call_sdpa_vector(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    q_l.start_offset(),
                    q_l.dims(),
                    q.buffer(),
                    k_l.start_offset(),
                    k_l.dims(),
                    k_l.stride(),
                    k.buffer(),
                    v_l.start_offset(),
                    v_l.stride(),
                    v.buffer(),
                    &output,
                    self.scale,
                    self.softcapping,
                    itype,
                )
                .map_err(candle_core::Error::wrap)?;
            }
        } else if supports_sdpa_full {
            if self.softcapping != 1.0 {
                candle_core::bail!("SDPA full requires softcapping to be 1.0");
            }

            let mask_s_l = self.mask.as_ref().map(|m| m.storage_and_layout());

            let (mask_type, mask_buffer, mask_strides) = if let Some(mask) = &self.mask {
                let (mask_s, mask_l) = mask_s_l.as_ref().unwrap();

                let mask_buffer = match &**mask_s {
                    candle_core::Storage::Metal(m) => m.buffer(),
                    _ => candle_core::bail!("Expected Metal device for mask"),
                };

                let mask_type = match mask.dtype() {
                    DType::BF16 => SdpaDType::BF16,
                    DType::F16 => SdpaDType::F16,
                    DType::F32 => SdpaDType::F32,
                    other => candle_core::bail!("unsupported mask dtype {other:?}"),
                };
                if mask_type != itype {
                    candle_core::bail!("Mask dtype {mask_type:?} must match q dtype {itype:?}");
                }

                if mask_l.dims() != [q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, k_seq] {
                    candle_core::bail!(
                        "Mask shape must be {:?}, got {:?}",
                        [q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, k_seq],
                        mask_l.dims()
                    );
                }

                (
                    Some(mask_type),
                    Some(mask_buffer),
                    Some(mask_l.stride().to_vec()),
                )
            } else {
                (None, None, None)
            };

            candle_metal_kernels::call_sdpa_full(
                device.device(),
                &encoder,
                device.kernels(),
                q_l.start_offset(),
                q_l.dims(),
                q_l.stride(),
                q.buffer(),
                k_l.start_offset(),
                k_l.dims(),
                k_l.stride(),
                k.buffer(),
                v_l.start_offset(),
                v.buffer(),
                v_l.stride(),
                mask_type,
                mask_buffer,
                mask_strides.as_deref(),
                &output,
                out_layout.stride(),
                self.scale,
                self.do_causal,
                itype,
            )
            .map_err(candle_core::Error::wrap)?;
        } else {
            candle_core::bail!("must be vector or full SDPA path");
        }

        Ok((
            MetalStorage::new(output, device.clone(), elem_count, q.dtype()),
            out_shape,
        ))
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Scaled Dot Product Attention via Metal fused kernels.
///
/// Computes `softmax(Q·Kᵀ × scale) · V` on-GPU without materialising the
/// full (seq_q × seq_kv) attention matrix.
///
/// **Input shapes** (all `[B, H, seq, head_dim]`):
/// - `q`: `(bs, n_heads, seq_q, head_dim)`
/// - `k`: `(bs, n_kv_heads, seq_kv, head_dim)`
/// - `v`: `(bs, n_kv_heads, seq_kv, head_dim)`
///
/// GQA is supported natively: `n_heads` must be a multiple of `n_kv_heads`.
/// `repeat_kv` is NOT needed — the kernel handles head expansion internally.
///
/// Returns `(bs, n_heads, seq_q, head_dim)`.
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    do_causal: bool,
    scale: f32,
) -> Result<Tensor> {
    q.apply_op3_no_bwd(
        k,
        v,
        &Sdpa {
            scale,
            softcapping: 1.0,
            mask: mask.cloned(),
            do_causal,
        },
    )
}

/// Check whether SDPA is usable for the given configuration.
///
/// Returns `true` when:
/// - tensor is on a Metal device
/// - dtype is F16, BF16, or F32
/// - head_dim is one of 32, 64, 72, 80, 96, 128, 256
pub fn sdpa_available(tensor: &Tensor, head_dim: usize) -> bool {
    tensor.device().is_metal()
        && matches!(
            tensor.dtype(),
            DType::F16 | DType::BF16 | DType::F32
        )
        && matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256)
}
