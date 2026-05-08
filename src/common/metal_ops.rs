// ─────────────────────────────────────────────────────────────────────────────
// metal_ops.rs  — Metal-accelerated kernels for oxydLLM
// ─────────────────────────────────────────────────────────────────────────────
//
// All ops wrap candle-metal-kernels (already a project dependency) via
// candle's `CustomOp` traits, matching the same pattern used for SDPA.
//
// Kernels provided:
//   • SDPA        — fused Flash-Attention (vector + full paths, GQA-native)
//   • RMSNorm     — single-pass fused normalisation + scale
//   • Softmax     — fused softmax over last dimension
//   • RoPE        — fused rotary embedding (standard non-interleaved layout)
//   • GatedSiLU   — fused silu(gate)*up from a single interleaved [*, 2N] tensor
//   • SiLU-Mul    — fused silu(gate)*up from two separate contiguous tensors
// ─────────────────────────────────────────────────────────────────────────────

use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, D, DType, Layout, MetalStorage, Result, Shape,
    Tensor, backend::BackendStorage,
};
use candle_metal_kernels::{
    SdpaDType,
    metal::{ComputeCommandEncoder, ComputePipeline},
};
use objc2_metal::{MTLResourceUsage, MTLSize};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// ─── SDPA ────────────────────────────────────────────────────────────────────

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
            q_seq > 8 && supported_head_dim && supports_sdpa_full_mask && self.softcapping == 1.0;
        let supports_sdpa_vector = q_seq <= 8 && supported_head_dim && q_seq <= k_seq;

        if !supported_head_dim || !(supports_sdpa_full || supports_sdpa_vector) {
            candle_core::bail!(
                "Metal SDPA does not support q dims {:?}, k dims {:?}, v dims {:?}",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
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
        } else {
            // supports_sdpa_full (softcapping already checked == 1.0 above)
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
        }

        Ok((
            MetalStorage::new(output, device.clone(), elem_count, q.dtype()),
            out_shape,
        ))
    }
}

// ─── RMSNorm ─────────────────────────────────────────────────────────────────

struct RmsNormOp {
    eps: f32,
}

impl CustomOp2 for RmsNormOp {
    fn name(&self) -> &'static str {
        "metal-rms-norm"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("RmsNormOp: Metal-only")
    }

    fn metal_fwd(
        &self,
        x: &MetalStorage,
        l_x: &Layout,
        w: &MetalStorage,
        l_w: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = x.device();

        let name = match (x.dtype(), w.dtype()) {
            (DType::F32, DType::F32) => "rmsnorm_f32",
            (DType::F16, DType::F16) => "rmsnorm_f16",
            (DType::BF16, DType::BF16) => "rmsnorm_bf16",
            (dt1, dt2) => candle_core::bail!("rms_norm dtype mismatch: x={dt1:?} w={dt2:?}"),
        };

        if !(l_x.is_contiguous() && l_w.is_contiguous()) {
            candle_core::bail!("RmsNormOp: both input and weight must be contiguous");
        }

        let last_dim = l_x.dims()[l_x.shape().rank() - 1];
        let elem_count = l_x.shape().elem_count();
        let output = device.new_buffer(elem_count, x.dtype(), "rms_norm_out")?;
        let encoder = device.command_encoder()?;

        candle_metal_kernels::call_rms_norm(
            device.device(),
            &encoder,
            device.kernels(),
            name,
            elem_count,
            last_dim,
            self.eps,
            x.buffer(),
            l_x.start_offset() * x.dtype().size_in_bytes(),
            w.buffer(),
            l_w.start_offset() * w.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok((
            MetalStorage::new(output, device.clone(), elem_count, x.dtype()),
            l_x.shape().clone(),
        ))
    }
}

// ─── Softmax ─────────────────────────────────────────────────────────────────

struct SoftmaxOp;

impl CustomOp1 for SoftmaxOp {
    fn name(&self) -> &'static str {
        "metal-softmax"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("SoftmaxOp: Metal-only")
    }

    fn metal_fwd(&self, x: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = x.device();

        let name = match x.dtype() {
            DType::F32 => "softmax_f32",
            DType::F16 => "softmax_f16",
            DType::BF16 => "softmax_bf16",
            other => candle_core::bail!("softmax not implemented for {other:?}"),
        };

        if !l.is_contiguous() {
            candle_core::bail!("SoftmaxOp: input must be contiguous");
        }

        let last_dim = l.dims()[l.shape().rank() - 1];
        let elem_count = l.shape().elem_count();
        let output = device.new_buffer(elem_count, x.dtype(), "softmax_out")?;
        let encoder = device.command_encoder()?;

        candle_metal_kernels::call_last_softmax(
            device.device(),
            &encoder,
            device.kernels(),
            name,
            elem_count,
            last_dim,
            x.buffer(),
            l.start_offset() * x.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok((
            MetalStorage::new(output, device.clone(), elem_count, x.dtype()),
            l.shape().clone(),
        ))
    }
}

// ─── RoPE ────────────────────────────────────────────────────────────────────
//
// Uses the standard (non-interleaved) layout: [x_first_half | x_second_half].
// Input  x:   [b, h, seq, d]          (contiguous)
// Input  cos: [seq, d/2]              (contiguous, pre-gathered for positions)
// Input  sin: [seq, d/2]              (contiguous, pre-gathered for positions)
// Output:     [b, h, seq, d]

struct RopeOp;

impl CustomOp3 for RopeOp {
    fn name(&self) -> &'static str {
        "metal-rope"
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
        candle_core::bail!("RopeOp: Metal-only")
    }

    fn metal_fwd(
        &self,
        src: &MetalStorage,
        l_src: &Layout,
        cos: &MetalStorage,
        l_cos: &Layout,
        sin: &MetalStorage,
        l_sin: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = src.device();

        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle_core::bail!(
                "RopeOp dtype mismatch: src={:?} cos={:?} sin={:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            );
        }

        let name = match src.dtype() {
            DType::F32 => "rope_f32",
            DType::F16 => "rope_f16",
            DType::BF16 => "rope_bf16",
            other => candle_core::bail!("RopeOp not implemented for {other:?}"),
        };

        if !(l_src.is_contiguous() && l_cos.is_contiguous() && l_sin.is_contiguous()) {
            candle_core::bail!("RopeOp: all inputs must be contiguous");
        }

        let (b, h, t, d) = l_src.shape().dims4()?;
        let el = b * h * t * d;
        let output = device.new_buffer(el, src.dtype(), "rope_out")?;
        let encoder = device.command_encoder()?;

        // stride_b = 0: cos/sin are [seq, d/2] shared across all batch/head dims
        candle_metal_kernels::call_rope(
            device.device(),
            &encoder,
            device.kernels(),
            name,
            b * h, // bh
            t * d, // td
            d,     // full head_dim
            0,     // stride_b = 0 (2D cos/sin, no per-batch offset)
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok((
            MetalStorage::new(output, device.clone(), el, src.dtype()),
            l_src.shape().clone(),
        ))
    }
}

// ─── FFN Fused Kernels ────────────────────────────────────────────────────────
//
// Two complementary kernels cover every FFN variant:
//
//   GatedSiluOp  — takes the combined matmul output [*, 2*N] produced by
//                  Fused/Packed gate_up projections and computes
//                  silu(gate)*up in a single pass, avoiding two extra
//                  encoder creations and an intermediate buffer.
//
//   SiluMulOp    — takes two separate contiguous tensors [*, N] (from the
//                  GGUF Separate path) and computes silu(gate)*up in-place.
//
// Both kernels promote F16/BF16 arithmetic to F32 for the SiLU computation
// and cast back, matching the precision of the scalar fallback path.

const FFN_METAL_SOURCE: &str = include_str!("ffn_kernels.metal");

// Pipeline cache keyed by (device registry ID, kernel function name).
// Compilation (first call per device+kernel) takes ~50-100 ms; subsequent
// calls just clone the cached Arc-wrapped pipeline state.
static FFN_PIPELINES: OnceLock<Mutex<HashMap<(u64, &'static str), ComputePipeline>>> =
    OnceLock::new();

fn get_or_compile_ffn_pipeline(
    device: &candle_metal_kernels::metal::Device,
    kernel_name: &'static str,
) -> Result<ComputePipeline> {
    let cache = FFN_PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.registry_id(), kernel_name);

    let mut guard = cache
        .lock()
        .map_err(|e| candle_core::Error::Msg(format!("FFN pipeline cache poisoned: {e}")))?;

    if let Some(p) = guard.get(&key) {
        return Ok(p.clone());
    }

    let lib = device
        .new_library_with_source(FFN_METAL_SOURCE, None)
        .map_err(|e| candle_core::Error::Msg(format!("FFN Metal compile: {e}")))?;

    let func = lib
        .get_function(kernel_name, None)
        .map_err(|e| candle_core::Error::Msg(format!("FFN kernel '{kernel_name}': {e}")))?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| candle_core::Error::Msg(format!("FFN pipeline: {e}")))?;

    guard.insert(key, pipeline.clone());
    Ok(pipeline)
}

fn ffn_dispatch(pipeline: &ComputePipeline, encoder: &ComputeCommandEncoder, elem_count: usize) {
    let tg_size = pipeline.max_total_threads_per_threadgroup().min(1024);
    let tg_count = elem_count.div_ceil(tg_size);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: tg_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        },
    );
}

// ─── GatedSiluOp ─────────────────────────────────────────────────────────────
//
// Reads a single contiguous [*, 2*N] tensor where the first N columns are the
// gate projection output and the second N are the up projection output.
// Computes silu(gate) * up and writes [*, N].

struct GatedSiluOp {
    intermediate_size: usize,
}

impl CustomOp1 for GatedSiluOp {
    fn name(&self) -> &'static str {
        "metal-gated-silu"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("GatedSiluOp: Metal-only")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout) -> Result<(MetalStorage, Shape)> {
        if !lx.is_contiguous() {
            candle_core::bail!("GatedSiluOp: input must be contiguous");
        }
        let rank = lx.shape().rank();
        let last_dim = lx.dims()[rank - 1];
        if last_dim != 2 * self.intermediate_size {
            candle_core::bail!(
                "GatedSiluOp: last dim {last_dim} != 2*intermediate_size={}",
                2 * self.intermediate_size
            );
        }

        let out_elems = lx.shape().elem_count() / 2;
        let mut out_dims = lx.dims().to_vec();
        *out_dims.last_mut().unwrap() = self.intermediate_size;
        let out_shape = Shape::from_dims(&out_dims);

        let kernel_name: &'static str = match x.dtype() {
            DType::F32 => "gated_silu_f32",
            DType::F16 => "gated_silu_f16",
            DType::BF16 => "gated_silu_bf16",
            other => candle_core::bail!("GatedSiluOp: unsupported dtype {other:?}"),
        };

        let device = x.device();
        let output = device.new_buffer(out_elems, x.dtype(), "gated_silu_out")?;
        let pipeline = get_or_compile_ffn_pipeline(device.device(), kernel_name)?;
        let encoder = device.command_encoder()?;

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            Some(x.buffer()),
            lx.start_offset() * x.dtype().size_in_bytes(),
        );
        encoder.set_buffer(1, Some(&*output), 0);
        encoder.set_bytes(2, &(out_elems as u32));
        encoder.set_bytes(3, &(self.intermediate_size as u32));
        encoder.use_resource(x.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(&*output, MTLResourceUsage::Write);
        ffn_dispatch(&pipeline, &encoder, out_elems);

        Ok((
            MetalStorage::new(output, device.clone(), out_elems, x.dtype()),
            out_shape,
        ))
    }
}

// ─── SiluMulOp ───────────────────────────────────────────────────────────────
//
// Takes two separate contiguous tensors `gate` and `up` of identical shape and
// computes silu(gate[i]) * up[i] element-wise.

struct SiluMulOp;

impl CustomOp2 for SiluMulOp {
    fn name(&self) -> &'static str {
        "metal-silu-mul"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("SiluMulOp: Metal-only")
    }

    fn metal_fwd(
        &self,
        gate: &MetalStorage,
        lg: &Layout,
        up: &MetalStorage,
        lu: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if !lg.is_contiguous() || !lu.is_contiguous() {
            candle_core::bail!("SiluMulOp: both inputs must be contiguous");
        }
        if gate.dtype() != up.dtype() {
            candle_core::bail!(
                "SiluMulOp: dtype mismatch {:?} vs {:?}",
                gate.dtype(),
                up.dtype()
            );
        }
        let elem_count = lg.shape().elem_count();
        if elem_count != lu.shape().elem_count() {
            candle_core::bail!("SiluMulOp: shape mismatch");
        }

        let kernel_name: &'static str = match gate.dtype() {
            DType::F32 => "silu_mul_f32",
            DType::F16 => "silu_mul_f16",
            DType::BF16 => "silu_mul_bf16",
            other => candle_core::bail!("SiluMulOp: unsupported dtype {other:?}"),
        };

        let device = gate.device();
        let output = device.new_buffer(elem_count, gate.dtype(), "silu_mul_out")?;
        let pipeline = get_or_compile_ffn_pipeline(device.device(), kernel_name)?;
        let encoder = device.command_encoder()?;

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            Some(gate.buffer()),
            lg.start_offset() * gate.dtype().size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(up.buffer()),
            lu.start_offset() * up.dtype().size_in_bytes(),
        );
        encoder.set_buffer(2, Some(&*output), 0);
        encoder.set_bytes(3, &(elem_count as u32));
        encoder.use_resource(gate.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(up.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(&*output, MTLResourceUsage::Write);
        ffn_dispatch(&pipeline, &encoder, elem_count);

        Ok((
            MetalStorage::new(output, device.clone(), elem_count, gate.dtype()),
            lg.shape().clone(),
        ))
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Scaled Dot Product Attention via Metal fused kernels.
///
/// - Vector path (seq_q ≤ 8): supports `softcapping`
/// - Full path   (seq_q > 8): requires `softcapping == 1.0`
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    do_causal: bool,
    scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    q.apply_op3_no_bwd(
        k,
        v,
        &Sdpa {
            scale,
            softcapping,
            mask: mask.cloned(),
            do_causal,
        },
    )
}

/// Fused RMSNorm via Metal kernel.
///
/// Equivalent to `x / rms(x) * weight` in a single GPU pass.
/// Both `x` and `weight` must be contiguous and have the same dtype.
pub fn rms_norm_fused(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    x.apply_op2_no_bwd(weight, &RmsNormOp { eps })
}

/// Fused softmax over the last dimension via Metal kernel.
///
/// Input must be contiguous.
pub fn softmax_fused(x: &Tensor) -> Result<Tensor> {
    x.apply_op1_no_bwd(&SoftmaxOp)
}

/// Fused RoPE (standard non-interleaved layout) via Metal kernel.
///
/// - `x`:   `[b, h, seq, d]`      — contiguous
/// - `cos`: `[seq, d/2]`          — pre-gathered for the active positions
/// - `sin`: `[seq, d/2]`          — pre-gathered for the active positions
pub fn rope_fused(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    x.apply_op3_no_bwd(cos, sin, &RopeOp)
}

/// Check whether SDPA is usable for the given configuration.
///
/// Returns `true` when:
/// - tensor is on a Metal device
/// - dtype is F16, BF16, or F32
/// - head_dim is one of 32, 64, 72, 80, 96, 128, 256
pub fn sdpa_available(tensor: &Tensor, head_dim: usize) -> bool {
    tensor.device().is_metal()
        && matches!(tensor.dtype(), DType::F16 | DType::BF16 | DType::F32)
        && matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256)
}

/// Fused gated-SiLU via Metal kernel.
///
/// `x` must be a contiguous tensor of shape `[*, 2*intermediate_size]` where
/// the first `intermediate_size` columns are the gate projection and the second
/// are the up projection.  Returns `[*, intermediate_size]` = silu(gate) * up.
pub fn gated_silu_fused(x: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    x.apply_op1_no_bwd(&GatedSiluOp { intermediate_size })
}

/// Fused SiLU-Mul via Metal kernel.
///
/// Both `gate` and `up` must be contiguous tensors of the same shape.
/// Returns a tensor of the same shape with values `silu(gate[i]) * up[i]`.
pub fn silu_mul_fused(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    gate.apply_op2_no_bwd(up, &SiluMulOp)
}
