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
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, D, DType, InplaceOp1, Layout, MetalStorage,
    Result, Shape, Tensor, backend::BackendStorage,
};
use candle_metal_kernels::{
    SdpaDType,
    metal::{ComputeCommandEncoder, ComputePipeline},
};
use objc2_metal::{MTLDevice, MTLGPUFamily};
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
//   GatedSiluOp       — takes the combined matmul output [*, 2*N] produced by
//                       Fused/Packed gate_up projections and computes
//                       silu(gate)*up in a single pass, avoiding two extra
//                       encoder creations and an intermediate buffer.
//   SiluMulOp         — takes two separate contiguous tensors [*, N] (from the
//                       GGUF Separate path) and computes silu(gate)*up in-place.
//   GatedGeluTanhOp   — same shape contract as GatedSiluOp but applies the
//                       tanh approximation of GeLU; used by Gemma family FFNs.
//   GeluTanhMulOp     — same shape contract as SiluMulOp with GeLU-tanh.
//
// All kernels promote F16/BF16 arithmetic to F32 for the activation computation
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

// ─── GatedGeluTanhOp ─────────────────────────────────────────────────────────
//
// Same shape contract as GatedSiluOp: reads a single contiguous [*, 2*N] tensor
// and writes [*, N] = gelu_tanh(gate) * up.  Used by Gemma family FFNs.

struct GatedGeluTanhOp {
    intermediate_size: usize,
}

impl CustomOp1 for GatedGeluTanhOp {
    fn name(&self) -> &'static str {
        "metal-gated-gelu-tanh"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("GatedGeluTanhOp: Metal-only")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout) -> Result<(MetalStorage, Shape)> {
        if !lx.is_contiguous() {
            candle_core::bail!("GatedGeluTanhOp: input must be contiguous");
        }
        let rank = lx.shape().rank();
        let last_dim = lx.dims()[rank - 1];
        if last_dim != 2 * self.intermediate_size {
            candle_core::bail!(
                "GatedGeluTanhOp: last dim {last_dim} != 2*intermediate_size={}",
                2 * self.intermediate_size
            );
        }

        let out_elems = lx.shape().elem_count() / 2;
        let mut out_dims = lx.dims().to_vec();
        *out_dims.last_mut().unwrap() = self.intermediate_size;
        let out_shape = Shape::from_dims(&out_dims);

        let kernel_name: &'static str = match x.dtype() {
            DType::F32 => "gated_gelu_tanh_f32",
            DType::F16 => "gated_gelu_tanh_f16",
            DType::BF16 => "gated_gelu_tanh_bf16",
            other => candle_core::bail!("GatedGeluTanhOp: unsupported dtype {other:?}"),
        };

        let device = x.device();
        let output = device.new_buffer(out_elems, x.dtype(), "gated_gelu_tanh_out")?;
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

// ─── GeluTanhMulOp ───────────────────────────────────────────────────────────
//
// Same shape contract as SiluMulOp with GeLU-tanh as the gate activation.

struct GeluTanhMulOp;

impl CustomOp2 for GeluTanhMulOp {
    fn name(&self) -> &'static str {
        "metal-gelu-tanh-mul"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("GeluTanhMulOp: Metal-only")
    }

    fn metal_fwd(
        &self,
        gate: &MetalStorage,
        lg: &Layout,
        up: &MetalStorage,
        lu: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if !lg.is_contiguous() || !lu.is_contiguous() {
            candle_core::bail!("GeluTanhMulOp: both inputs must be contiguous");
        }
        if gate.dtype() != up.dtype() {
            candle_core::bail!(
                "GeluTanhMulOp: dtype mismatch {:?} vs {:?}",
                gate.dtype(),
                up.dtype()
            );
        }
        let elem_count = lg.shape().elem_count();
        if elem_count != lu.shape().elem_count() {
            candle_core::bail!("GeluTanhMulOp: shape mismatch");
        }

        let kernel_name: &'static str = match gate.dtype() {
            DType::F32 => "gelu_tanh_mul_f32",
            DType::F16 => "gelu_tanh_mul_f16",
            DType::BF16 => "gelu_tanh_mul_bf16",
            other => candle_core::bail!("GeluTanhMulOp: unsupported dtype {other:?}"),
        };

        let device = gate.device();
        let output = device.new_buffer(elem_count, gate.dtype(), "gelu_tanh_mul_out")?;
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

// ─── SoftcapOp ───────────────────────────────────────────────────────────────
//
// Element-wise out[i] = cap * tanh(x[i] / cap).  Replaces the 3-op fallback
// (`(x / cap).tanh() * cap`) with a single Metal pass.  Used by Gemma2
// attention scores (cap=50) and Gemma2/Gemma4 logits (cap=30).

struct SoftcapOp {
    softcap: f32,
}

impl CustomOp1 for SoftcapOp {
    fn name(&self) -> &'static str {
        "metal-softcap"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("SoftcapOp: Metal-only")
    }

    fn metal_fwd(&self, x: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        if !l.is_contiguous() {
            candle_core::bail!("SoftcapOp: input must be contiguous");
        }
        let kernel_name: &'static str = match x.dtype() {
            DType::F32 => "softcap_f32",
            DType::F16 => "softcap_f16",
            DType::BF16 => "softcap_bf16",
            other => candle_core::bail!("SoftcapOp: unsupported dtype {other:?}"),
        };

        let elem_count = l.shape().elem_count();
        let device = x.device();
        let output = device.new_buffer(elem_count, x.dtype(), "softcap_out")?;
        let pipeline = get_or_compile_ffn_pipeline(device.device(), kernel_name)?;
        let encoder = device.command_encoder()?;

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(
            0,
            Some(x.buffer()),
            l.start_offset() * x.dtype().size_in_bytes(),
        );
        encoder.set_buffer(1, Some(&*output), 0);
        encoder.set_bytes(2, &(elem_count as u32));
        encoder.set_bytes(3, &self.softcap);
        encoder.use_resource(x.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(&*output, MTLResourceUsage::Write);
        ffn_dispatch(&pipeline, &encoder, elem_count);

        Ok((
            MetalStorage::new(output, device.clone(), elem_count, x.dtype()),
            l.shape().clone(),
        ))
    }
}

// ─── Flash Attention prefill kernel ──────────────────────────────────────────
//
// Custom Metal FA2-style kernel for the prefill path (num_tokens > 1).
// Implements tiled QKᵀ → online softmax → PV with causal+prefix masking and
// GQA support, all in one dispatch.  The decode path (num_tokens == 1)
// continues to use candle's SDPA vector kernel.
//
// Algorithm derived from FlashAttention (Dao et al., BSD-3-Clause).  See the
// attribution header in `flash_attn.metal`.

const FA_METAL_SOURCE: &str = include_str!("flash_attn.metal");

static FA_PIPELINES: OnceLock<Mutex<HashMap<(u64, &'static str), ComputePipeline>>> =
    OnceLock::new();

fn get_or_compile_fa_pipeline(
    device: &candle_metal_kernels::metal::Device,
    kernel_name: &'static str,
) -> Result<ComputePipeline> {
    let cache = FA_PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.registry_id(), kernel_name);

    let mut guard = cache
        .lock()
        .map_err(|e| candle_core::Error::Msg(format!("FA pipeline cache poisoned: {e}")))?;

    if let Some(p) = guard.get(&key) {
        return Ok(p.clone());
    }

    let lib = device
        .new_library_with_source(FA_METAL_SOURCE, None)
        .map_err(|e| candle_core::Error::Msg(format!("FA Metal compile: {e}")))?;

    let func = lib
        .get_function(kernel_name, None)
        .map_err(|e| candle_core::Error::Msg(format!("FA kernel '{kernel_name}': {e}")))?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| candle_core::Error::Msg(format!("FA pipeline: {e}")))?;

    guard.insert(key, pipeline.clone());
    Ok(pipeline)
}

/// Tile sizes (Br, Bc) chosen to fit threadgroup memory budget on Apple GPUs
/// (~32 KB conservative).  Computed as:
///   (Br*D + 2*Br + Br*Bc)*4   (fp32 accumulators)
///   + (Br + Bc)*D*sizeof(T)   (Q tile + KV tile)
fn fa_tile_sizes(head_dim: usize) -> (usize, usize) {
    match head_dim {
        64 => (32, 32),
        128 => (16, 32),
        256 => (8, 16),
        _ => (16, 16), // conservative fallback
    }
}

/// Returns true if the device supports the hardware Matrix Multiply Accumulate
/// units present on Apple GPU family 8 and later (M3, M4, …).  On older Apple
/// silicon (M1/M2 = family 7) simdgroup_matrix is software-emulated and slower
/// than the scalar 4-way-unrolled kernel, so we explicitly opt-in here.
fn metal_supports_mma(device: &candle_metal_kernels::metal::Device) -> bool {
    // Cache the answer per device to avoid the Obj-C dispatch on every kernel call.
    static CACHE: OnceLock<Mutex<HashMap<u64, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = device.registry_id();
    if let Ok(map) = cache.lock()
        && let Some(&v) = map.get(&key)
    {
        return v;
    }
    let supported = device.as_ref().supportsFamily(MTLGPUFamily::Apple8);
    if let Ok(mut map) = cache.lock() {
        map.insert(key, supported);
    }
    supported
}

/// Layout (in fp32 equivalents — keep in sync with `flash_attn.metal`):
///   o_acc     : Br * D            fp32
///   m_row     : Br                fp32
///   l_row     : Br                fp32
///   s_scratch : Br * Bc           fp32
///   p_tile    : Br * Bc           T   (MMA variant only — for converted P)
///   q_tile    : Br * D            T
///   kv_tile   : Bc * (D + KV_PAD) T  (K and V share the same area)
///
/// KV_PAD = 4 / sizeof(T) adds exactly 1 word per row to break the 32-way
/// SIMD-group bank conflict on kv_tile reads in the QKᵀ inner loop.
///
/// The MMA variant needs `p_tile` for the bfloat/half P operand of the PV
/// MMA (the scalar kernel re-reads from float s_scratch directly).  Pass
/// `with_p_tile = true` for the MMA kernel allocation.
fn fa_threadgroup_bytes(
    br: usize,
    bc: usize,
    d: usize,
    dtype_bytes: usize,
    with_p_tile: bool,
) -> usize {
    let kv_pad = 4 / dtype_bytes; // 1 element for f32, 2 elements for f16/bf16
    let kv_stride = d + kv_pad;
    let fp32_bytes = (br * d + 2 * br + br * bc) * 4;
    let p_tile_bytes = if with_p_tile {
        br * bc * dtype_bytes
    } else {
        0
    };
    let tile_bytes = (br * d + bc * kv_stride) * dtype_bytes + p_tile_bytes;
    fp32_bytes + tile_bytes
}

/// Must match the `FlashAttnParams` struct in `flash_attn.metal`.
#[repr(C)]
struct FlashAttnParams {
    t_q: u32,
    t_kv: u32,
    h: u32,
    h_kv: u32,
    d: u32,
    br: u32,
    bc: u32,
    scale: f32,
    softcap: f32,
    prefix_len: u32,
}

struct FlashAttnPrefill {
    scale: f32,
    softcap: f32, // 0.0 = disabled
    prefix_len: usize,
    br: usize,
    bc: usize,
}

impl CustomOp3 for FlashAttnPrefill {
    fn name(&self) -> &'static str {
        "metal-flash-attn-prefill"
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
        candle_core::bail!("FlashAttnPrefill: Metal-only — use standard attention path on CPU")
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
        if !q_l.is_contiguous() {
            candle_core::bail!("FlashAttnPrefill: q must be contiguous");
        }
        if !k_l.is_contiguous() {
            candle_core::bail!("FlashAttnPrefill: k must be contiguous");
        }
        if !v_l.is_contiguous() {
            candle_core::bail!("FlashAttnPrefill: v must be contiguous");
        }
        if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
            candle_core::bail!("FlashAttnPrefill: q, k, v dtypes must match");
        }

        let q_dims = q_l.dims();
        let k_dims = k_l.dims();
        let v_dims = v_l.dims();
        if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
            candle_core::bail!("FlashAttnPrefill: q, k, v must be 4-D [B, H, T, D]");
        }
        let (b, h, t_q, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let h_kv = k_dims[1];
        let t_kv = k_dims[2];
        if k_dims[0] != b || v_dims[0] != b {
            candle_core::bail!("FlashAttnPrefill: batch dim mismatch");
        }
        if v_dims[1] != h_kv || v_dims[2] != t_kv {
            candle_core::bail!("FlashAttnPrefill: k and v must share head and seq dims");
        }
        if k_dims[3] != d || v_dims[3] != d {
            candle_core::bail!("FlashAttnPrefill: head_dim must match across q, k, v");
        }
        if h % h_kv != 0 {
            candle_core::bail!("FlashAttnPrefill: n_heads must be divisible by n_kv_heads");
        }
        if self.prefix_len + t_q != t_kv {
            candle_core::bail!(
                "FlashAttnPrefill: prefix_len ({}) + T_q ({t_q}) != T_kv ({t_kv})",
                self.prefix_len
            );
        }

        let dtype_bytes = q.dtype().size_in_bytes();
        let elem_count = b * h * t_q * d;
        let device = q.device();
        let use_mma = metal_supports_mma(device.device());
        let kernel_name: &'static str = match (q.dtype(), use_mma) {
            (DType::F32, true) => "flash_attention_prefill_mma_f32",
            (DType::F16, true) => "flash_attention_prefill_mma_f16",
            (DType::BF16, true) => "flash_attention_prefill_mma_bf16",
            (DType::F32, false) => "flash_attention_prefill_f32",
            (DType::F16, false) => "flash_attention_prefill_f16",
            (DType::BF16, false) => "flash_attention_prefill_bf16",
            (other, _) => candle_core::bail!("FlashAttnPrefill: unsupported dtype {other:?}"),
        };
        let output = device.new_buffer(elem_count, q.dtype(), "flash_attn_out")?;

        let params = FlashAttnParams {
            t_q: t_q as u32,
            t_kv: t_kv as u32,
            h: h as u32,
            h_kv: h_kv as u32,
            d: d as u32,
            br: self.br as u32,
            bc: self.bc as u32,
            scale: self.scale,
            softcap: self.softcap,
            prefix_len: self.prefix_len as u32,
        };

        let tg_bytes = fa_threadgroup_bytes(self.br, self.bc, d, dtype_bytes, use_mma);
        let n_q_blocks = t_q.div_ceil(self.br);
        let grid_x = b * h * n_q_blocks;
        let threads_per_group: usize = 128;

        let pipeline = get_or_compile_fa_pipeline(device.device(), kernel_name)?;
        // Cap threads at pipeline's reported maximum (typically 1024 on M-series).
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg_size = threads_per_group.min(max_threads);

        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(q.buffer()), q_l.start_offset() * dtype_bytes);
        encoder.set_buffer(1, Some(k.buffer()), k_l.start_offset() * dtype_bytes);
        encoder.set_buffer(2, Some(v.buffer()), v_l.start_offset() * dtype_bytes);
        encoder.set_buffer(3, Some(&*output), 0);
        encoder.set_bytes(4, &params);
        encoder.set_threadgroup_memory_length(0, tg_bytes);
        encoder.use_resource(q.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(k.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(v.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(&*output, MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(
            MTLSize {
                width: grid_x,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            },
        );

        Ok((
            MetalStorage::new(output, device.clone(), elem_count, q.dtype()),
            Shape::from_dims(&[b, h, t_q, d]),
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
    if !q.device().is_metal() || !k.device().is_metal() || !v.device().is_metal() {
        candle_core::bail!(
            "metal_ops::sdpa requires all of q, k, v on a Metal device (got q={:?}, k={:?}, v={:?})",
            q.device().location(),
            k.device().location(),
            v.device().location(),
        );
    }
    if let Some(m) = mask
        && !m.device().is_metal()
    {
        candle_core::bail!(
            "metal_ops::sdpa requires `mask` on a Metal device (got {:?})",
            m.device().location(),
        );
    }
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

/// Fused gated GeLU-tanh via Metal kernel.
///
/// Same shape contract as [`gated_silu_fused`] but uses the tanh approximation
/// of GeLU as the gate activation.  Used by Gemma family FFNs.
pub fn gated_gelu_tanh_fused(x: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    x.apply_op1_no_bwd(&GatedGeluTanhOp { intermediate_size })
}

/// Fused GeLU-tanh-Mul via Metal kernel.
///
/// Both `gate` and `up` must be contiguous tensors of the same shape.
/// Returns a tensor of the same shape with values `gelu_tanh(gate[i]) * up[i]`.
pub fn gelu_tanh_mul_fused(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    gate.apply_op2_no_bwd(up, &GeluTanhMulOp)
}

/// Fused softcap via Metal kernel: `out[i] = cap * tanh(x[i] / cap)`.
///
/// `x` must be contiguous.  Returns a tensor of the same shape and dtype.
pub fn softcap_fused(x: &Tensor, softcap: f32) -> Result<Tensor> {
    x.apply_op1_no_bwd(&SoftcapOp { softcap })
}

/// Flash Attention prefill via Metal custom kernel.
///
/// Computes scaled dot-product attention for the prefill path (multi-token Q)
/// using a tiled FA2-style kernel with online softmax — never materialises the
/// full QKᵀ matrix in unified memory.
///
/// Tensor layout (head counts are taken from the tensor shapes):
/// - `q`: `[B, H,    T_q,  D]`  contiguous
/// - `k`: `[B, H_kv, T_kv, D]`  contiguous
/// - `v`: `[B, H_kv, T_kv, D]`  contiguous
///
/// Where `T_kv = prefix_len + T_q` and `H % H_kv == 0` (GQA supported natively).
/// Causal masking is always applied; `softcap` of `Some(cap)` enables Gemma-style
/// score capping inline in the kernel.
pub fn flash_attention_metal_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    softcap: Option<f32>,
    prefix_len: usize,
) -> Result<Tensor> {
    if !q.device().is_metal() || !k.device().is_metal() || !v.device().is_metal() {
        candle_core::bail!("flash_attention_metal_prefill: q, k, v must all be on a Metal device");
    }
    let head_dim = q.dim(D::Minus1)?;
    let (br, bc) = fa_tile_sizes(head_dim);
    q.apply_op3_no_bwd(
        k,
        v,
        &FlashAttnPrefill {
            scale,
            softcap: softcap.unwrap_or(0.0),
            prefix_len,
            br,
            bc,
        },
    )
}

/// Check whether the custom Metal Flash Attention prefill kernel can handle
/// the given configuration.
///
/// Returns `true` for head dims 64, 128, 256 (the configurations with vetted
/// threadgroup memory budgets).  Other head dims fall back to the standard
/// attention path.
pub fn flash_attention_metal_available(head_dim: usize) -> bool {
    matches!(head_dim, 64 | 128 | 256)
}

// ─── W4A16 quantized matmul ──────────────────────────────────────────────────
//
// Fused dequantize + matmul for AWQ 4-bit weight-only checkpoints. The packed
// weight triplet (qweight/qzeros u32, scales T) stays resident; the unpack is
// done inline in the kernel. See `quant_kernels.metal` for the AWQ layout.

const QUANT_METAL_SOURCE: &str = include_str!("quant_kernels.metal");

static QUANT_PIPELINES: OnceLock<Mutex<HashMap<(u64, &'static str), ComputePipeline>>> =
    OnceLock::new();

fn get_or_compile_quant_pipeline(
    device: &candle_metal_kernels::metal::Device,
    kernel_name: &'static str,
) -> Result<ComputePipeline> {
    let cache = QUANT_PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (device.registry_id(), kernel_name);

    let mut guard = cache
        .lock()
        .map_err(|e| candle_core::Error::Msg(format!("quant pipeline cache poisoned: {e}")))?;

    if let Some(p) = guard.get(&key) {
        return Ok(p.clone());
    }

    let lib = device
        .new_library_with_source(QUANT_METAL_SOURCE, None)
        .map_err(|e| candle_core::Error::Msg(format!("quant Metal compile: {e}")))?;
    let func = lib
        .get_function(kernel_name, None)
        .map_err(|e| candle_core::Error::Msg(format!("quant kernel '{kernel_name}': {e}")))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| candle_core::Error::Msg(format!("quant pipeline: {e}")))?;

    guard.insert(key, pipeline.clone());
    Ok(pipeline)
}

/// Must match `W4A16Params` in `quant_kernels.metal`.
#[repr(C)]
struct W4A16Params {
    in_features: u32,
    out_features: u32,
    packed_out: u32,
    group_size: u32,
    group_shift: u32,
    k_splits: u32,
    chunk: u32,
}

/// Geometry of an AWQ weight triplet, validated once per dispatch.
struct AwqShape {
    in_features: usize,
    out_features: usize,
    group_size: usize,
    packed_out: usize,
}

impl AwqShape {
    fn new(
        in_features: usize,
        packed_out: usize,
        groups: usize,
        out_features: usize,
    ) -> Result<Self> {
        if packed_out * 8 != out_features {
            candle_core::bail!(
                "W4A16: qweight packed_out {packed_out} (×8) != scales out {out_features}"
            );
        }
        if groups == 0 || !in_features.is_multiple_of(groups) {
            candle_core::bail!("W4A16: in_features {in_features} not divisible by groups {groups}");
        }
        Ok(Self {
            in_features,
            out_features,
            group_size: in_features / groups,
            packed_out,
        })
    }

    /// `group_shift`/`k_splits`/`chunk` are gemv-only; pass 0 for the dequant kernel.
    fn params(&self, group_shift: usize, k_splits: usize, chunk: usize) -> W4A16Params {
        W4A16Params {
            in_features: self.in_features as u32,
            out_features: self.out_features as u32,
            packed_out: self.packed_out as u32,
            group_size: self.group_size as u32,
            group_shift: group_shift as u32,
            k_splits: k_splits as u32,
            chunk: chunk as u32,
        }
    }
}

/// Fused W4A16 GEMV. An `InplaceOp1` that atomically accumulates the split-K
/// partials into a pre-zeroed `[1, out]` F32 buffer; `x` and the AWQ triplet
/// are carried as fields (the trait passes only the in-place tensor).
struct W4A16Matmul {
    x: Tensor,
    qweight: Tensor,
    qzeros: Tensor,
    scales: Tensor,
}

impl InplaceOp1 for W4A16Matmul {
    fn name(&self) -> &'static str {
        "w4a16-matmul"
    }

    fn cpu_fwd(&self, _s: &mut CpuStorage, _l: &Layout) -> Result<()> {
        candle_core::bail!("W4A16Matmul: Metal-only — use the CPU dequantize_awq path")
    }

    fn metal_fwd(&self, out: &mut MetalStorage, out_l: &Layout) -> Result<()> {
        if out.dtype() != DType::F32 {
            candle_core::bail!(
                "W4A16Matmul: accumulator must be F32, got {:?}",
                out.dtype()
            );
        }

        let x_sl = self.x.storage_and_layout();
        let x_l = x_sl.1;
        if !x_l.is_contiguous() {
            candle_core::bail!("W4A16Matmul: x must be contiguous");
        }
        let x_dims = x_l.dims();
        if x_dims.len() != 2 || x_dims[0] != 1 {
            candle_core::bail!("W4A16Matmul: x must be [1, in] (M=1 decode path), got {x_dims:?}");
        }
        let in_features = x_dims[1];

        let (qw_in, packed_out) = self.qweight.dims2()?;
        let (groups, out_features) = self.scales.dims2()?;
        if self.qzeros.dims() != [groups, packed_out] {
            candle_core::bail!(
                "W4A16Matmul: qzeros shape {:?} != [{groups}, {packed_out}]",
                self.qzeros.dims()
            );
        }
        let shape = AwqShape::new(qw_in, packed_out, groups, out_features)?;
        if in_features != shape.in_features {
            candle_core::bail!(
                "W4A16Matmul: x in_features {in_features} != weight in_features {}",
                shape.in_features
            );
        }
        if out_l.dims() != [1, shape.out_features] {
            candle_core::bail!(
                "W4A16Matmul: accumulator shape {:?} != [1, {}]",
                out_l.dims(),
                shape.out_features
            );
        }
        if self.scales.dtype() != self.x.dtype() {
            candle_core::bail!(
                "W4A16Matmul: scales dtype {:?} must match x dtype {:?}",
                self.scales.dtype(),
                self.x.dtype()
            );
        }
        if !shape.group_size.is_power_of_two() {
            candle_core::bail!(
                "W4A16Matmul: group_size {} must be a power of two",
                shape.group_size
            );
        }

        let kernel_name = match self.x.dtype() {
            DType::F16 => "w4a16_gemv_f16",
            DType::BF16 => "w4a16_gemv_bf16",
            other => candle_core::bail!("W4A16Matmul: unsupported dtype {other:?}"),
        };

        // Split the in_features reduction so enough simdgroups are resident to
        // hide HBM latency. Chunks are contiguous, so each thread stays inside a
        // quant group and the scale/zero reload is amortised. The split-K
        // partials are accumulated straight into `out` with atomic adds — no
        // partial buffer, no separate reduction pass.
        const TARGET_THREADS: usize = 32768;
        let k_splits = (TARGET_THREADS / shape.packed_out.max(1))
            .clamp(1, 256)
            .min(shape.in_features.max(1));
        let chunk = shape.in_features.div_ceil(k_splits);
        let k_splits = shape.in_features.div_ceil(chunk);
        let group_shift = shape.group_size.trailing_zeros() as usize;
        let params = shape.params(group_shift, k_splits, chunk);

        let dtype_bytes = self.x.dtype().size_in_bytes();
        let device = out.device();

        let x_buf = match &*x_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("W4A16Matmul: x must be Metal-resident"),
        };
        let qw_sl = self.qweight.storage_and_layout();
        let qz_sl = self.qzeros.storage_and_layout();
        let sc_sl = self.scales.storage_and_layout();
        let qweight_buf = match &*qw_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("W4A16Matmul: qweight must be Metal-resident"),
        };
        let qzeros_buf = match &*qz_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("W4A16Matmul: qzeros must be Metal-resident"),
        };
        let scales_buf = match &*sc_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("W4A16Matmul: scales must be Metal-resident"),
        };

        let pipeline = get_or_compile_quant_pipeline(device.device(), kernel_name)?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), x_l.start_offset() * dtype_bytes);
        encoder.set_buffer(1, Some(qweight_buf), qw_sl.1.start_offset() * 4);
        encoder.set_buffer(2, Some(qzeros_buf), qz_sl.1.start_offset() * 4);
        encoder.set_buffer(3, Some(scales_buf), sc_sl.1.start_offset() * dtype_bytes);
        encoder.set_buffer(4, Some(out.buffer()), out_l.start_offset() * 4);
        encoder.set_bytes(5, &params);
        encoder.use_resource(x_buf, MTLResourceUsage::Read);
        encoder.use_resource(qweight_buf, MTLResourceUsage::Read);
        encoder.use_resource(qzeros_buf, MTLResourceUsage::Read);
        encoder.use_resource(scales_buf, MTLResourceUsage::Read);
        encoder.use_resource(out.buffer(), MTLResourceUsage::Write);

        const TG_WIDTH: usize = 64;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: shape.packed_out.div_ceil(TG_WIDTH),
                height: k_splits,
                depth: 1,
            },
            MTLSize {
                width: TG_WIDTH,
                height: 1,
                depth: 1,
            },
        );

        Ok(())
    }
}

/// Dequantize an AWQ triplet to a plain `[in, out]` weight. `CustomOp1` over
/// `qweight`; `qzeros`/`scales` carried as fields. Used for the prefill path
/// where a tuned GEMM on the dequantized weight beats a custom kernel.
struct DequantizeW4 {
    qzeros: Tensor,
    scales: Tensor,
}

impl CustomOp1 for DequantizeW4 {
    fn name(&self) -> &'static str {
        "dequantize-w4"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("DequantizeW4: Metal-only — use the CPU dequantize_awq path")
    }

    fn metal_fwd(&self, qweight: &MetalStorage, qw_l: &Layout) -> Result<(MetalStorage, Shape)> {
        if !qw_l.is_contiguous() {
            candle_core::bail!("DequantizeW4: qweight must be contiguous");
        }
        let qw_dims = qw_l.dims();
        if qw_dims.len() != 2 {
            candle_core::bail!("DequantizeW4: qweight must be 2-D [in, out/8]");
        }
        let (in_features, packed_out) = (qw_dims[0], qw_dims[1]);
        let (groups, out_features) = self.scales.dims2()?;
        if self.qzeros.dims() != [groups, packed_out] {
            candle_core::bail!(
                "DequantizeW4: qzeros shape {:?} != [{groups}, {packed_out}]",
                self.qzeros.dims()
            );
        }
        let shape = AwqShape::new(in_features, packed_out, groups, out_features)?;

        let out_dtype = self.scales.dtype();
        let kernel_name = match out_dtype {
            DType::F16 => "dequantize_w4_f16",
            DType::BF16 => "dequantize_w4_bf16",
            other => candle_core::bail!("DequantizeW4: unsupported dtype {other:?}"),
        };
        let dtype_bytes = out_dtype.size_in_bytes();
        let out_elems = in_features * out_features;
        let device = qweight.device();
        let output = device.new_buffer(out_elems, out_dtype, "dequant_w4")?;
        let params = shape.params(0, 0, 0);

        let qz_sl = self.qzeros.storage_and_layout();
        let sc_sl = self.scales.storage_and_layout();
        let qzeros_buf = match &*qz_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("DequantizeW4: qzeros must be Metal-resident"),
        };
        let scales_buf = match &*sc_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("DequantizeW4: scales must be Metal-resident"),
        };

        let pipeline = get_or_compile_quant_pipeline(device.device(), kernel_name)?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(qweight.buffer()), qw_l.start_offset() * 4);
        encoder.set_buffer(1, Some(qzeros_buf), qz_sl.1.start_offset() * 4);
        encoder.set_buffer(2, Some(scales_buf), sc_sl.1.start_offset() * dtype_bytes);
        encoder.set_buffer(3, Some(&*output), 0);
        encoder.set_bytes(4, &params);
        encoder.use_resource(qweight.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(qzeros_buf, MTLResourceUsage::Read);
        encoder.use_resource(scales_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*output, MTLResourceUsage::Write);

        const TG_WIDTH: usize = 64;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: in_features.div_ceil(TG_WIDTH),
                height: packed_out,
                depth: 1,
            },
            MTLSize {
                width: TG_WIDTH,
                height: 1,
                depth: 1,
            },
        );

        Ok((
            MetalStorage::new(output, device.clone(), out_elems, out_dtype),
            Shape::from_dims(&[in_features, out_features]),
        ))
    }
}

/// Fused W4A16 matmul: `x [M, in] @ dequant(AWQ)ᵀ → [M, out]`.
///
/// `x` must be 2-D, contiguous, F16/BF16 on a Metal device; the AWQ triplet
/// must be Metal-resident with `scales` in the same dtype as `x`.
pub fn w4a16_matmul(
    x: &Tensor,
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
) -> Result<Tensor> {
    // Zero-initialised F32 accumulator; the kernel atomic-adds the split-K
    // partials straight into it (no partial buffer, no reduction pass).
    let out_features = scales.dim(1)?;
    let out = Tensor::zeros((1, out_features), DType::F32, x.device())?;
    out.inplace_op1(&W4A16Matmul {
        x: x.clone(),
        qweight: qweight.clone(),
        qzeros: qzeros.clone(),
        scales: scales.clone(),
    })?;
    out.to_dtype(x.dtype())
}

/// Dequantize an AWQ triplet to a plain `[in, out]` weight in the scales' dtype.
pub fn dequantize_w4(qweight: &Tensor, qzeros: &Tensor, scales: &Tensor) -> Result<Tensor> {
    qweight.apply_op1_no_bwd(&DequantizeW4 {
        qzeros: qzeros.clone(),
        scales: scales.clone(),
    })
}

#[cfg(test)]
mod fused_kernel_parity_tests {
    //! Numerical parity between fused Metal kernels and the scalar reference
    //! path on a real Metal device.  Each test silently skips on machines
    //! without a Metal device available (e.g. CI Linux runners), so the
    //! suite stays useful both locally on macOS and in cross-platform CI.
    //!
    //! Tolerance is loose because F16/BF16 differ between fused (F32-promote
    //! intermediate) and scalar (F16/BF16-native) paths.

    use super::*;
    use crate::common::awq::{AWQ_PACK_ORDER, AwqRawTensors, dequantize_awq};
    use candle_core::{D, Device};

    fn metal_device_or_skip() -> Option<Device> {
        Device::new_metal(0).ok()
    }

    fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn sdpa_rejects_cpu_tensors_at_call_site() {
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 4, 4, 64), DType::F32, &dev).unwrap();
        let k = Tensor::zeros((1, 4, 4, 64), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 4, 4, 64), DType::F32, &dev).unwrap();

        let err = sdpa(&q, &k, &v, None, false, 1.0, 1.0).expect_err("must reject CPU tensors");
        let msg = format!("{err}");
        assert!(
            msg.contains("Metal"),
            "expected device-mismatch error, got: {msg}"
        );
    }

    #[test]
    fn gated_silu_matches_scalar_path_f32() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };

        // [batch=1, seq=4, 2*N=16] → output [1, 4, 8]
        let n = 8usize;
        let data: Vec<f32> = (0..1 * 4 * 2 * n)
            .map(|i| (i as f32 - 32.0) * 0.05)
            .collect();
        let x = Tensor::from_vec(data, (1, 4, 2 * n), &dev).unwrap();

        let fused = gated_silu_fused(&x, n).unwrap();

        let gate = x.narrow(D::Minus1, 0, n).unwrap();
        let up = x.narrow(D::Minus1, n, n).unwrap();
        let scalar = (gate.silu().unwrap() * up).unwrap();

        let f = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let diff = max_abs_diff_f32(&f, &s);
        assert!(diff < 1e-5, "max_abs_diff = {diff}");
    }

    #[test]
    fn gated_silu_matches_scalar_path_bf16() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };

        let n = 4usize;
        let data: Vec<f32> = (0..1 * 2 * 2 * n).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let x = Tensor::from_vec(data, (1, 2, 2 * n), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let fused = gated_silu_fused(&x, n).unwrap();

        let gate = x.narrow(D::Minus1, 0, n).unwrap();
        let up = x.narrow(D::Minus1, n, n).unwrap();
        let scalar = (gate.silu().unwrap() * up).unwrap();

        let f = fused
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let s = scalar
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let diff = max_abs_diff_f32(&f, &s);
        // BF16 precision: ~7 bits of mantissa → ~1e-2 absolute for activations near unity.
        assert!(diff < 0.05, "max_abs_diff (bf16) = {diff}");
    }

    #[test]
    fn silu_mul_matches_scalar_path_f32() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };

        let n = 16usize;
        let gate_data: Vec<f32> = (0..n).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let up_data: Vec<f32> = (0..n).map(|i| (i as f32 - 4.0) * 0.2).collect();
        let gate = Tensor::from_vec(gate_data, (1, n), &dev).unwrap();
        let up = Tensor::from_vec(up_data, (1, n), &dev).unwrap();

        let fused = silu_mul_fused(&gate, &up).unwrap();
        let scalar = (gate.silu().unwrap() * &up).unwrap();

        let f = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let diff = max_abs_diff_f32(&f, &s);
        assert!(diff < 1e-5, "max_abs_diff = {diff}");
    }

    #[test]
    fn gated_silu_rejects_wrong_last_dim() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        let x = Tensor::zeros((1, 2, 7), DType::F32, &dev).unwrap();
        // intermediate_size=4 → expects last dim = 8, but we have 7.
        let res = gated_silu_fused(&x, 4);
        assert!(res.is_err());
    }

    #[test]
    fn gated_gelu_tanh_matches_scalar_path_f32() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };

        let n = 8usize;
        let data: Vec<f32> = (0..1 * 4 * 2 * n)
            .map(|i| (i as f32 - 32.0) * 0.05)
            .collect();
        let x = Tensor::from_vec(data, (1, 4, 2 * n), &dev).unwrap();

        let fused = gated_gelu_tanh_fused(&x, n).unwrap();

        let gate = x.narrow(D::Minus1, 0, n).unwrap();
        let up = x.narrow(D::Minus1, n, n).unwrap();
        let scalar = (gate.gelu().unwrap() * up).unwrap();

        let f = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let diff = max_abs_diff_f32(&f, &s);
        assert!(diff < 1e-5, "max_abs_diff = {diff}");
    }

    #[test]
    fn gated_gelu_tanh_matches_scalar_path_bf16() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };

        let n = 4usize;
        let data: Vec<f32> = (0..1 * 2 * 2 * n).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let x = Tensor::from_vec(data, (1, 2, 2 * n), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let fused = gated_gelu_tanh_fused(&x, n).unwrap();

        let gate = x.narrow(D::Minus1, 0, n).unwrap();
        let up = x.narrow(D::Minus1, n, n).unwrap();
        let scalar = (gate.gelu().unwrap() * up).unwrap();

        let f = fused
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let s = scalar
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let diff = max_abs_diff_f32(&f, &s);
        assert!(diff < 0.05, "max_abs_diff (bf16) = {diff}");
    }

    #[test]
    fn gelu_tanh_mul_matches_scalar_path_f32() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };

        let n = 16usize;
        let gate_data: Vec<f32> = (0..n).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let up_data: Vec<f32> = (0..n).map(|i| (i as f32 - 4.0) * 0.2).collect();
        let gate = Tensor::from_vec(gate_data, (1, n), &dev).unwrap();
        let up = Tensor::from_vec(up_data, (1, n), &dev).unwrap();

        let fused = gelu_tanh_mul_fused(&gate, &up).unwrap();
        let scalar = (gate.gelu().unwrap() * &up).unwrap();

        let f = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let diff = max_abs_diff_f32(&f, &s);
        assert!(diff < 1e-5, "max_abs_diff = {diff}");
    }

    #[test]
    fn silu_mul_rejects_dtype_mismatch() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        let gate = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();
        let up = Tensor::zeros((1, 4), DType::F32, &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let res = silu_mul_fused(&gate, &up);
        assert!(res.is_err());
    }

    #[test]
    fn softcap_matches_scalar_path_f32() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        // Cover small, near-zero, and large-magnitude inputs.
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 5.0).collect();
        let x = Tensor::from_vec(data, (4, 16), &dev).unwrap();
        let cap = 30.0f32;

        let fused = softcap_fused(&x, cap).unwrap();
        let scalar = (x.affine(1.0 / cap as f64, 0.0).unwrap().tanh().unwrap())
            .affine(cap as f64, 0.0)
            .unwrap();

        let f = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let diff = max_abs_diff_f32(&f, &s);
        assert!(diff < 1e-5, "max_abs_diff = {diff}");
    }

    #[test]
    fn softcap_matches_scalar_path_bf16() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 2.0).collect();
        let x = Tensor::from_vec(data, (2, 16), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let cap = 50.0f32;

        let fused = softcap_fused(&x, cap).unwrap();
        let scalar = (x.affine(1.0 / cap as f64, 0.0).unwrap().tanh().unwrap())
            .affine(cap as f64, 0.0)
            .unwrap();

        let f = fused
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let s = scalar
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let diff = max_abs_diff_f32(&f, &s);
        assert!(diff < 0.5, "max_abs_diff (bf16) = {diff}");
    }

    // ── Flash Attention prefill parity tests ─────────────────────────────────
    //
    // Compare the custom Metal FA kernel against a naive `matmul + softmax +
    // matmul` reference computed on the same Metal device.  Both paths use
    // F32 internal accumulation so the numerical agreement should be tight
    // (<1e-3 for F32, ~1e-2 for BF16).

    fn naive_attention_reference(
        q: &Tensor, // [B, H, T_q, D]
        k: &Tensor, // [B, H_kv, T_kv, D]
        v: &Tensor, // [B, H_kv, T_kv, D]
        scale: f32,
        softcap: Option<f32>,
        prefix_len: usize,
    ) -> Result<Tensor> {
        let (b, h, t_q, _d) = q.dims4()?;
        let (_, h_kv, t_kv, _) = k.dims4()?;
        let n_rep = h / h_kv;

        // Expand KV to match Q heads if GQA
        let k_exp = if n_rep == 1 {
            k.clone()
        } else {
            k.unsqueeze(2)?
                .expand((b, h_kv, n_rep, t_kv, k.dim(3)?))?
                .reshape((b, h, t_kv, k.dim(3)?))?
        };
        let v_exp = if n_rep == 1 {
            v.clone()
        } else {
            v.unsqueeze(2)?
                .expand((b, h_kv, n_rep, t_kv, v.dim(3)?))?
                .reshape((b, h, t_kv, v.dim(3)?))?
        };

        // Compute scores: Q @ K^T * scale
        let mut scores = q
            .matmul(&k_exp.transpose(D::Minus1, D::Minus2)?)?
            .affine(scale as f64, 0.0)?;
        if let Some(cap) = softcap {
            scores = (scores / (cap as f64))?.tanh()?.affine(cap as f64, 0.0)?;
        }

        // Build causal mask shifted by prefix_len:
        //   mask[i, j] = -INF if j > prefix + i, else 0
        let device = q.device();
        let mut mask_data = vec![0.0f32; t_q * t_kv];
        for i in 0..t_q {
            for j in 0..t_kv {
                if j > prefix_len + i {
                    mask_data[i * t_kv + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask =
            Tensor::from_vec(mask_data, (1, 1, t_q, t_kv), device)?.to_dtype(scores.dtype())?;
        let masked = scores.broadcast_add(&mask)?;

        let attn = crate::common::linear::softmax_last_dim(&masked)?;
        attn.matmul(&v_exp)
    }

    /// Shape of a Flash Attention parity test case.
    struct FaCase {
        b: usize,
        h: usize,
        h_kv: usize,
        t_q: usize,
        t_kv: usize,
        d: usize,
    }

    fn run_fa_parity(
        dev: &Device,
        dtype: DType,
        case: FaCase,
        softcap: Option<f32>,
        tol: f32,
        label: &str,
    ) {
        let FaCase {
            b,
            h,
            h_kv,
            t_q,
            t_kv,
            d,
        } = case;
        let prefix = t_kv - t_q;
        let scale = 1.0 / (d as f32).sqrt();

        // Deterministic small values to keep softmax well-conditioned and BF16
        // accumulation stable across the two paths.
        let n_q = b * h * t_q * d;
        let n_kv = b * h_kv * t_kv * d;
        let q_data: Vec<f32> = (0..n_q).map(|i| ((i % 17) as f32 - 8.0) * 0.04).collect();
        let k_data: Vec<f32> = (0..n_kv).map(|i| ((i % 19) as f32 - 9.0) * 0.03).collect();
        let v_data: Vec<f32> = (0..n_kv).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
        let q = Tensor::from_vec(q_data, (b, h, t_q, d), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let k = Tensor::from_vec(k_data, (b, h_kv, t_kv, d), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let v = Tensor::from_vec(v_data, (b, h_kv, t_kv, d), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let out_fa = flash_attention_metal_prefill(&q, &k, &v, scale, softcap, prefix)
            .expect("FA kernel must not error");

        let out_ref = naive_attention_reference(&q, &k, &v, scale, softcap, prefix)
            .expect("naive reference must not error");

        let f = out_fa
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let r = out_ref
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(f.len(), r.len(), "{label}: size mismatch");
        let diff = max_abs_diff_f32(&f, &r);
        let all_finite_fa = f.iter().all(|x| x.is_finite());
        let all_finite_ref = r.iter().all(|x| x.is_finite());
        assert!(all_finite_fa, "{label}: FA output not finite");
        assert!(all_finite_ref, "{label}: ref output not finite");
        assert!(diff < tol, "{label}: max_abs_diff = {diff} (tol {tol})");
    }

    #[test]
    fn flash_attn_f32_d64_prefill_matches_naive() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_fa_parity(
            &dev,
            DType::F32,
            FaCase {
                b: 1,
                h: 2,
                h_kv: 2,
                t_q: 16,
                t_kv: 16,
                d: 64,
            },
            None,
            1e-4,
            "f32/d64/no-prefix",
        );
    }

    #[test]
    fn flash_attn_bf16_d64_prefill_matches_naive() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_fa_parity(
            &dev,
            DType::BF16,
            FaCase {
                b: 1,
                h: 4,
                h_kv: 4,
                t_q: 16,
                t_kv: 16,
                d: 64,
            },
            None,
            0.05,
            "bf16/d64/no-prefix",
        );
    }

    #[test]
    fn flash_attn_bf16_d128_prefill_matches_naive() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_fa_parity(
            &dev,
            DType::BF16,
            FaCase {
                b: 1,
                h: 4,
                h_kv: 4,
                t_q: 8,
                t_kv: 8,
                d: 128,
            },
            None,
            0.05,
            "bf16/d128/no-prefix",
        );
    }

    #[test]
    fn flash_attn_bf16_d64_gqa_matches_naive() {
        // n_heads=8, n_kv_heads=2 (GQA factor 4)
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_fa_parity(
            &dev,
            DType::BF16,
            FaCase {
                b: 1,
                h: 8,
                h_kv: 2,
                t_q: 8,
                t_kv: 8,
                d: 64,
            },
            None,
            0.05,
            "bf16/d64/gqa",
        );
    }

    #[test]
    fn flash_attn_bf16_d64_softcap_matches_naive() {
        // Gemma2-style softcap
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_fa_parity(
            &dev,
            DType::BF16,
            FaCase {
                b: 1,
                h: 4,
                h_kv: 4,
                t_q: 16,
                t_kv: 16,
                d: 64,
            },
            Some(30.0),
            0.05,
            "bf16/d64/softcap",
        );
    }

    #[test]
    fn flash_attn_bf16_d64_with_prefix_cache_matches_naive() {
        // prefix_len = 24, t_q = 8 → t_kv = 32 (simulates 24-token KV cache hit)
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_fa_parity(
            &dev,
            DType::BF16,
            FaCase {
                b: 1,
                h: 4,
                h_kv: 4,
                t_q: 8,
                t_kv: 32,
                d: 64,
            },
            None,
            0.05,
            "bf16/d64/prefix=24",
        );
    }

    #[test]
    fn flash_attn_f16_d64_prefill_matches_naive() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_fa_parity(
            &dev,
            DType::F16,
            FaCase {
                b: 1,
                h: 4,
                h_kv: 4,
                t_q: 16,
                t_kv: 16,
                d: 64,
            },
            None,
            0.02,
            "f16/d64/no-prefix",
        );
    }

    #[test]
    fn flash_attn_rejects_cpu_tensors() {
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 4, 4, 64), DType::F32, &dev).unwrap();
        let k = Tensor::zeros((1, 4, 4, 64), DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1, 4, 4, 64), DType::F32, &dev).unwrap();
        let err = flash_attention_metal_prefill(&q, &k, &v, 0.125, None, 0)
            .expect_err("must reject CPU tensors");
        assert!(format!("{err}").contains("Metal"));
    }

    #[test]
    fn flash_attn_available_for_supported_head_dims() {
        assert!(flash_attention_metal_available(64));
        assert!(flash_attention_metal_available(128));
        assert!(flash_attention_metal_available(256));
        assert!(!flash_attention_metal_available(80));
        assert!(!flash_attention_metal_available(96));
    }

    #[test]
    fn flash_attn_bf16_long_prefix_matches_naive() {
        // Stress test: short prefill, long prefix cache (decode-after-prefill scenario)
        // T_q=4, T_kv=128 (prefix=124), exercises the n_block_max optimization.
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_fa_parity(
            &dev,
            DType::BF16,
            FaCase {
                b: 1,
                h: 4,
                h_kv: 4,
                t_q: 4,
                t_kv: 128,
                d: 64,
            },
            None,
            0.05,
            "bf16/d64/long-prefix",
        );
    }

    // ── W4A16 quantized matmul parity ────────────────────────────────────────

    /// Pack a `[rows][out]` 4-bit-nibble matrix into AWQ `[rows, out/8]` words.
    fn pack_awq(matrix: &[Vec<u8>], rows: usize, out_features: usize) -> Vec<i32> {
        let packed_out = out_features / 8;
        let mut words = vec![0u32; rows * packed_out];
        for (i, row) in matrix.iter().enumerate().take(rows) {
            for j in 0..packed_out {
                let mut word = 0u32;
                for (k, &off) in AWQ_PACK_ORDER.iter().enumerate() {
                    word |= ((row[j * 8 + off] as u32) & 0xF) << (4 * k as u32);
                }
                words[i * packed_out + j] = word;
            }
        }
        words.into_iter().map(|w| w as i32).collect()
    }

    /// Deterministic synthetic AWQ triplet on `dev`, `scales` cast to `dtype`.
    fn build_awq_triplet(
        dev: &Device,
        dtype: DType,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> AwqRawTensors {
        let groups = in_features / group_size;
        let iweight: Vec<Vec<u8>> = (0..in_features)
            .map(|i| {
                (0..out_features)
                    .map(|j| ((i * 5 + j * 3 + 1) & 0xF) as u8)
                    .collect()
            })
            .collect();
        let izero: Vec<Vec<u8>> = (0..groups)
            .map(|g| {
                (0..out_features)
                    .map(|j| ((g * 7 + j + 2) & 0xF) as u8)
                    .collect()
            })
            .collect();
        let scales: Vec<f32> = (0..groups)
            .flat_map(|g| (0..out_features).map(move |j| 0.01 + 0.003 * ((g + j) % 5) as f32))
            .collect();

        let qweight = Tensor::from_vec(
            pack_awq(&iweight, in_features, out_features),
            (in_features, out_features / 8),
            dev,
        )
        .unwrap();
        let qzeros = Tensor::from_vec(
            pack_awq(&izero, groups, out_features),
            (groups, out_features / 8),
            dev,
        )
        .unwrap();
        let scales = Tensor::from_vec(scales, (groups, out_features), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        AwqRawTensors {
            qweight,
            qzeros,
            scales,
        }
    }

    /// Tolerance dominated by the kernel's single F16/BF16 output rounding.
    fn parity_tol(reference: &[f32]) -> f32 {
        let max_abs = reference.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        0.06 * max_abs + 1e-2
    }

    fn run_w4a16_gemv_parity(
        dev: &Device,
        dtype: DType,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        label: &str,
    ) {
        // The fused GEMV kernel is the M=1 (decode) path.
        let raw = build_awq_triplet(dev, dtype, in_features, out_features, group_size);
        let x_data: Vec<f32> = (0..in_features)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.03)
            .collect();
        let x = Tensor::from_vec(x_data, (1, in_features), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let y_kernel = w4a16_matmul(&x, &raw.qweight, &raw.qzeros, &raw.scales)
            .expect("w4a16 kernel must not error");

        // Golden: dequantize + matmul in F32 (x keeps its F16/BF16 precision).
        let w_ref = dequantize_awq(&raw, dev, DType::F32).unwrap(); // [out, in]
        let x_f32 = x.to_dtype(DType::F32).unwrap();
        let y_ref = x_f32
            .matmul(&w_ref.t().unwrap().contiguous().unwrap())
            .unwrap();

        assert_eq!(
            y_kernel.dims2().unwrap(),
            (1, out_features),
            "{label}: shape"
        );
        let f = y_kernel
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let r = y_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            f.iter().all(|v| v.is_finite()),
            "{label}: non-finite output"
        );
        let diff = max_abs_diff_f32(&f, &r);
        let tol = parity_tol(&r);
        assert!(diff < tol, "{label}: max_abs_diff = {diff} (tol {tol})");
    }

    fn run_dequant_w4_parity(
        dev: &Device,
        dtype: DType,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        label: &str,
    ) {
        let raw = build_awq_triplet(dev, dtype, in_features, out_features, group_size);
        let w_kernel = dequantize_w4(&raw.qweight, &raw.qzeros, &raw.scales)
            .expect("dequantize_w4 must not error"); // [in, out]
        let w_ref = dequantize_awq(&raw, dev, DType::F32).unwrap(); // [out, in]
        let w_ref_t = w_ref.t().unwrap().contiguous().unwrap(); // [in, out]

        assert_eq!(
            w_kernel.dims2().unwrap(),
            (in_features, out_features),
            "{label}: shape"
        );
        let f = w_kernel
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let r = w_ref_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let diff = max_abs_diff_f32(&f, &r);
        let tol = parity_tol(&r);
        assert!(diff < tol, "{label}: max_abs_diff = {diff} (tol {tol})");
    }

    #[test]
    fn w4a16_gemv_bf16_g128_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_w4a16_gemv_parity(&dev, DType::BF16, 256, 128, 128, "bf16/g128");
    }

    #[test]
    fn w4a16_gemv_bf16_g64_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_w4a16_gemv_parity(&dev, DType::BF16, 256, 128, 64, "bf16/g64");
    }

    #[test]
    fn w4a16_gemv_bf16_wide_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_w4a16_gemv_parity(&dev, DType::BF16, 512, 256, 128, "bf16/wide");
    }

    #[test]
    fn w4a16_gemv_f16_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_w4a16_gemv_parity(&dev, DType::F16, 256, 128, 64, "f16/g64");
    }

    #[test]
    fn dequantize_w4_bf16_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_dequant_w4_parity(&dev, DType::BF16, 256, 128, 128, "bf16/g128");
    }

    #[test]
    fn dequantize_w4_f16_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_dequant_w4_parity(&dev, DType::F16, 256, 128, 64, "f16/g64");
    }

    #[test]
    fn w4a16_matmul_rejects_cpu_tensors() {
        let dev = Device::Cpu;
        let raw = build_awq_triplet(&dev, DType::F32, 64, 64, 32);
        let x = Tensor::zeros((1, 64), DType::F32, &dev).unwrap();
        let err = w4a16_matmul(&x, &raw.qweight, &raw.qzeros, &raw.scales)
            .expect_err("must reject CPU tensors");
        assert!(
            format!("{err}").contains("Metal"),
            "unexpected error: {err}"
        );
    }

    /// Perf diagnostic (not a correctness gate): times the W4A16 matmul against
    /// the bf16 `matmul` it replaces, both as independent calls (the GPU can
    /// overlap them) and as a dependent chain (mirrors the decode loop, where
    /// each op feeds the next and the GPU cannot overlap). Run explicitly:
    ///   cargo test --release --bin oxydllm -- w4a16_decode_perf --ignored --nocapture
    #[test]
    #[ignore = "perf diagnostic"]
    fn w4a16_decode_perf_diagnostic() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        use std::time::Instant;

        let dtype = DType::BF16;
        let n = 2560; // square so W4A16 calls can be chained directly
        let group_size = 128;
        let iters = 144; // ~quantized matmuls per Qwen3-4B decode token

        let raw = build_awq_triplet(&dev, dtype, n, n, group_size);
        // bf16 [n, n] weight = exactly what the old dequant-at-load path used.
        let w_bf16 = dequantize_awq(&raw, &dev, dtype)
            .unwrap()
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        let x0 = Tensor::zeros((1, n), dtype, &dev).unwrap();

        for _ in 0..10 {
            let _ = w4a16_matmul(&x0, &raw.qweight, &raw.qzeros, &raw.scales).unwrap();
            let _ = x0.matmul(&w_bf16).unwrap();
        }
        x0.matmul(&w_bf16).unwrap().to_device(&Device::Cpu).unwrap();

        let per_call = |t: Instant| t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // Independent calls — the GPU may overlap successive dispatches.
        let t = Instant::now();
        let mut sink = Vec::with_capacity(iters);
        for _ in 0..iters {
            sink.push(w4a16_matmul(&x0, &raw.qweight, &raw.qzeros, &raw.scales).unwrap());
        }
        sink.last().unwrap().to_device(&Device::Cpu).unwrap();
        let w4_indep = per_call(t);

        let t = Instant::now();
        let mut sink = Vec::with_capacity(iters);
        for _ in 0..iters {
            sink.push(x0.matmul(&w_bf16).unwrap());
        }
        sink.last().unwrap().to_device(&Device::Cpu).unwrap();
        let bf16_indep = per_call(t);
        drop(sink);

        // Dependent chain — each call feeds the next, as in the decode loop.
        let t = Instant::now();
        let mut x = x0.clone();
        for _ in 0..iters {
            x = w4a16_matmul(&x, &raw.qweight, &raw.qzeros, &raw.scales).unwrap();
        }
        x.to_device(&Device::Cpu).unwrap();
        let w4_chain = per_call(t);

        let t = Instant::now();
        let mut x = x0.clone();
        for _ in 0..iters {
            x = x.matmul(&w_bf16).unwrap();
        }
        x.to_device(&Device::Cpu).unwrap();
        let bf16_chain = per_call(t);

        println!("[w4a16 {n}x{n}] per-call us  (iters={iters}):");
        println!("  independent : w4a16 {w4_indep:8.1}   bf16 {bf16_indep:8.1}");
        println!("  chained     : w4a16 {w4_chain:8.1}   bf16 {bf16_chain:8.1}");
    }
}

// ─── GGUF quantized matmul (Q5_0 first; Q4_K and Q2_K to follow) ────────────
//
// Bf16-aware GEMV kernels for GGUF-quantized weights. The packed block stream
// (e.g. 22 bytes per 32 elements for Q5_0) is carried as a `Tensor` of `U8`
// scalars Metal-resident — see `GgufFastPath` in `linear.rs` for the loader
// side. Algorithm is a port of `mul_vec_q_n_f32` from llama.cpp / candle
// (MIT) adapted to bf16 I/O (no host-side casts; bf16 read + bf16 write).
// Geometry: N_SIMDGROUP=2 simdgroups × N_DST=4 rows → 8 rows per threadgroup;
// each simdgroup loops over the row's K blocks with simd_sum at the end and
// one writer per row (no atomics, no split-K).
//
// Forward is M=1 only; the caller (QLinear) falls back to candle `QMatMul`
// for M>1 prefill.

#[repr(C)]
struct GgufParams {
    in_features: u32,
    out_features: u32,
}

const GGUF_N_DST: usize = 4;
const GGUF_N_SIMDGROUP: usize = 2;
const GGUF_N_SIMDWIDTH: usize = 32;
const Q5_0_BLOCK_BYTES: usize = 22;
const Q5_0_BLOCK_ELEMS: usize = 32;

struct GgufQ5_0Matmul {
    x: Tensor,
    weight_bytes: Tensor,
    in_features: usize,
    out_features: usize,
}

impl InplaceOp1 for GgufQ5_0Matmul {
    fn name(&self) -> &'static str {
        "gguf-q5_0-matmul"
    }

    fn cpu_fwd(&self, _s: &mut CpuStorage, _l: &Layout) -> Result<()> {
        candle_core::bail!("GgufQ5_0Matmul: Metal-only path")
    }

    fn metal_fwd(&self, out: &mut MetalStorage, out_l: &Layout) -> Result<()> {
        if out.dtype() != DType::BF16 {
            candle_core::bail!("GgufQ5_0Matmul: out must be BF16, got {:?}", out.dtype());
        }
        if self.x.dtype() != DType::BF16 {
            candle_core::bail!("GgufQ5_0Matmul: x must be BF16, got {:?}", self.x.dtype());
        }
        if self.weight_bytes.dtype() != DType::U8 {
            candle_core::bail!(
                "GgufQ5_0Matmul: weight_bytes must be U8, got {:?}",
                self.weight_bytes.dtype()
            );
        }
        if !self.in_features.is_multiple_of(Q5_0_BLOCK_ELEMS) {
            candle_core::bail!(
                "GgufQ5_0Matmul: in_features {} must be a multiple of {} (Q5_0 block size)",
                self.in_features,
                Q5_0_BLOCK_ELEMS,
            );
        }

        let x_sl = self.x.storage_and_layout();
        let x_l = x_sl.1;
        if !x_l.is_contiguous() {
            candle_core::bail!("GgufQ5_0Matmul: x must be contiguous");
        }
        let x_dims = x_l.dims();
        if x_dims.len() != 2 || x_dims[0] != 1 || x_dims[1] != self.in_features {
            candle_core::bail!(
                "GgufQ5_0Matmul: x shape {:?} != [1, {}]",
                x_dims,
                self.in_features
            );
        }
        if out_l.dims() != [1, self.out_features] {
            candle_core::bail!(
                "GgufQ5_0Matmul: output shape {:?} != [1, {}]",
                out_l.dims(),
                self.out_features
            );
        }
        let expected_bytes =
            self.out_features * (self.in_features / Q5_0_BLOCK_ELEMS) * Q5_0_BLOCK_BYTES;
        let w_dims = self.weight_bytes.dims();
        let w_elems: usize = w_dims.iter().product();
        if w_elems != expected_bytes {
            candle_core::bail!(
                "GgufQ5_0Matmul: weight_bytes has {} bytes, expected {} (out={} × blocks_per_row={} × {})",
                w_elems,
                expected_bytes,
                self.out_features,
                self.in_features / Q5_0_BLOCK_ELEMS,
                Q5_0_BLOCK_BYTES,
            );
        }

        let params = GgufParams {
            in_features: self.in_features as u32,
            out_features: self.out_features as u32,
        };

        let device = out.device();
        let x_buf = match &*x_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("GgufQ5_0Matmul: x must be Metal-resident"),
        };
        let w_sl = self.weight_bytes.storage_and_layout();
        let w_buf = match &*w_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("GgufQ5_0Matmul: weight_bytes must be Metal-resident"),
        };

        let pipeline = get_or_compile_quant_pipeline(device.device(), "gguf_q5_0_gemv_bf16")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), x_l.start_offset() * 2); // bf16 = 2 bytes
        encoder.set_buffer(1, Some(w_buf), w_sl.1.start_offset()); // u8 = 1 byte
        encoder.set_buffer(2, Some(out.buffer()), out_l.start_offset() * 2); // bf16 = 2 bytes
        encoder.set_bytes(3, &params);
        encoder.use_resource(x_buf, MTLResourceUsage::Read);
        encoder.use_resource(w_buf, MTLResourceUsage::Read);
        encoder.use_resource(out.buffer(), MTLResourceUsage::Write);

        // 1 threadgroup = N_SIMDGROUP simdgroups × N_SIMDWIDTH threads
        //               = 2 × 32 = 64 threads, covering N_SIMDGROUP × N_DST = 8 rows
        const TG_THREADS: usize = GGUF_N_SIMDGROUP * GGUF_N_SIMDWIDTH;
        const ROWS_PER_TG: usize = GGUF_N_SIMDGROUP * GGUF_N_DST;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.out_features.div_ceil(ROWS_PER_TG),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: TG_THREADS,
                height: 1,
                depth: 1,
            },
        );

        Ok(())
    }
}

/// Q5_0 GEMV: `out = x @ W.T`, where `W` is a GGUF Q5_0 quantized weight stored
/// as a raw byte stream in `weight_bytes` (shape `[N * K/32 * 22]`, dtype `U8`).
/// `x` must be BF16, shape `[1, K]`. Returns `[1, N]` in BF16.
pub fn gguf_q5_0_matmul(
    x: &Tensor,
    weight_bytes: &Tensor,
    in_features: usize,
    out_features: usize,
) -> Result<Tensor> {
    let device = x.device();
    let out = Tensor::zeros((1, out_features), DType::BF16, device)?;
    out.inplace_op1(&GgufQ5_0Matmul {
        x: x.clone(),
        weight_bytes: weight_bytes.clone(),
        in_features,
        out_features,
    })?;
    Ok(out)
}
