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

const FFN_METAL_SOURCE: &str = include_str!("ffn_kernels.metal");

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

// FlashAttention prefill kernel (Dao et al., BSD-3-Clause; see flash_attn.metal).
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

fn fa_tile_sizes(head_dim: usize) -> (usize, usize) {
    match head_dim {
        64 => (32, 32),
        128 => (16, 32),
        256 => (8, 16),
        _ => (16, 16),
    }
}

// Apple family 8+ (M3/M4) has hardware MMA; family 7 (M1/M2) emulates it.
fn metal_supports_mma(device: &candle_metal_kernels::metal::Device) -> bool {
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

// Threadgroup memory layout must match the kernel side in flash_attn.metal.
fn fa_threadgroup_bytes(
    br: usize,
    bc: usize,
    d: usize,
    dtype_bytes: usize,
    with_p_tile: bool,
) -> usize {
    let kv_pad = 4 / dtype_bytes;
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
    softcap: f32,
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

pub fn rms_norm_fused(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    x.apply_op2_no_bwd(weight, &RmsNormOp { eps })
}

pub fn softmax_fused(x: &Tensor) -> Result<Tensor> {
    x.apply_op1_no_bwd(&SoftmaxOp)
}

pub fn rope_fused(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    x.apply_op3_no_bwd(cos, sin, &RopeOp)
}

pub fn sdpa_available(tensor: &Tensor, head_dim: usize) -> bool {
    tensor.device().is_metal()
        && matches!(tensor.dtype(), DType::F16 | DType::BF16 | DType::F32)
        && matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256)
}

pub fn gated_silu_fused(x: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    x.apply_op1_no_bwd(&GatedSiluOp { intermediate_size })
}

pub fn silu_mul_fused(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    gate.apply_op2_no_bwd(up, &SiluMulOp)
}

pub fn gated_gelu_tanh_fused(x: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    x.apply_op1_no_bwd(&GatedGeluTanhOp { intermediate_size })
}

pub fn gelu_tanh_mul_fused(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    gate.apply_op2_no_bwd(up, &GeluTanhMulOp)
}

pub fn softcap_fused(x: &Tensor, softcap: f32) -> Result<Tensor> {
    x.apply_op1_no_bwd(&SoftcapOp { softcap })
}

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

pub fn flash_attention_metal_available(head_dim: usize) -> bool {
    matches!(head_dim, 64 | 128 | 256)
}

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

struct AwqShape {
    in_features: usize,
    out_features: usize,
    group_size: usize,
    packed_out: usize,
}

impl AwqShape {
    fn new_bits(
        in_features: usize,
        packed_out: usize,
        groups: usize,
        out_features: usize,
        bits: u32,
    ) -> Result<Self> {
        let pack_factor = (32 / bits) as usize;
        if packed_out * pack_factor != out_features {
            candle_core::bail!(
                "W{bits}A16: qweight packed_out {packed_out} (×{pack_factor}) != scales out {out_features}"
            );
        }
        if groups == 0 || !in_features.is_multiple_of(groups) {
            candle_core::bail!(
                "W{bits}A16: in_features {in_features} not divisible by groups {groups}"
            );
        }
        Ok(Self {
            in_features,
            out_features,
            group_size: in_features / groups,
            packed_out,
        })
    }

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

struct W4A16Matmul {
    x: Tensor,
    qweight: Tensor,
    qzeros: Tensor,
    scales: Tensor,
    bits: u32,
}

impl InplaceOp1 for W4A16Matmul {
    fn name(&self) -> &'static str {
        match self.bits {
            8 => "w8a16-matmul",
            _ => "w4a16-matmul",
        }
    }

    fn cpu_fwd(&self, _s: &mut CpuStorage, _l: &Layout) -> Result<()> {
        candle_core::bail!(
            "W{}A16Matmul: Metal-only — use the CPU dequantize_awq path",
            self.bits
        )
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
        let shape = AwqShape::new_bits(qw_in, packed_out, groups, out_features, self.bits)?;
        if in_features != shape.in_features {
            candle_core::bail!(
                "W{}A16Matmul: x in_features {in_features} != weight in_features {}",
                self.bits,
                shape.in_features
            );
        }
        if out_l.dims() != [1, shape.out_features] {
            candle_core::bail!(
                "W{}A16Matmul: accumulator shape {:?} != [1, {}]",
                self.bits,
                out_l.dims(),
                shape.out_features
            );
        }
        if self.scales.dtype() != self.x.dtype() {
            candle_core::bail!(
                "W{}A16Matmul: scales dtype {:?} must match x dtype {:?}",
                self.bits,
                self.scales.dtype(),
                self.x.dtype()
            );
        }
        if !shape.group_size.is_power_of_two() {
            candle_core::bail!(
                "W{}A16Matmul: group_size {} must be a power of two",
                self.bits,
                shape.group_size
            );
        }

        let kernel_name = match (self.bits, self.x.dtype()) {
            (4, DType::F16) => "w4a16_gemv_f16",
            (4, DType::BF16) => "w4a16_gemv_bf16",
            (8, DType::F16) => "w8a16_gemv_f16",
            (8, DType::BF16) => "w8a16_gemv_bf16",
            (bits, other) => candle_core::bail!(
                "W{bits}A16Matmul: unsupported (bits, dtype) combo ({bits}, {other:?})"
            ),
        };

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

struct DequantizeW4 {
    qzeros: Tensor,
    scales: Tensor,
    bits: u32,
}

impl CustomOp1 for DequantizeW4 {
    fn name(&self) -> &'static str {
        match self.bits {
            8 => "dequantize-w8",
            _ => "dequantize-w4",
        }
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!(
            "DequantizeW{}: Metal-only — use the CPU dequantize_awq path",
            self.bits
        )
    }

    fn metal_fwd(&self, qweight: &MetalStorage, qw_l: &Layout) -> Result<(MetalStorage, Shape)> {
        if !qw_l.is_contiguous() {
            candle_core::bail!("DequantizeW{}: qweight must be contiguous", self.bits);
        }
        let qw_dims = qw_l.dims();
        if qw_dims.len() != 2 {
            candle_core::bail!(
                "DequantizeW{}: qweight must be 2-D [in, out/pack_factor]",
                self.bits
            );
        }
        let (in_features, packed_out) = (qw_dims[0], qw_dims[1]);
        let (groups, out_features) = self.scales.dims2()?;
        if self.qzeros.dims() != [groups, packed_out] {
            candle_core::bail!(
                "DequantizeW{}: qzeros shape {:?} != [{groups}, {packed_out}]",
                self.bits,
                self.qzeros.dims()
            );
        }
        let shape = AwqShape::new_bits(in_features, packed_out, groups, out_features, self.bits)?;

        let out_dtype = self.scales.dtype();
        let kernel_name = match (self.bits, out_dtype) {
            (4, DType::F16) => "dequantize_w4_f16",
            (4, DType::BF16) => "dequantize_w4_bf16",
            (8, DType::F16) => "dequantize_w8_f16",
            (8, DType::BF16) => "dequantize_w8_bf16",
            (bits, other) => candle_core::bail!(
                "DequantizeW{bits}: unsupported (bits, dtype) combo ({bits}, {other:?})"
            ),
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

pub fn w4a16_matmul(
    x: &Tensor,
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
) -> Result<Tensor> {
    wna16_matmul_inner(x, qweight, qzeros, scales, 4)
}

pub fn w8a16_matmul(
    x: &Tensor,
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
) -> Result<Tensor> {
    wna16_matmul_inner(x, qweight, qzeros, scales, 8)
}

fn wna16_matmul_inner(
    x: &Tensor,
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    bits: u32,
) -> Result<Tensor> {
    let out_features = scales.dim(1)?;
    let out = Tensor::zeros((1, out_features), DType::F32, x.device())?;
    out.inplace_op1(&W4A16Matmul {
        x: x.clone(),
        qweight: qweight.clone(),
        qzeros: qzeros.clone(),
        scales: scales.clone(),
        bits,
    })?;
    out.to_dtype(x.dtype())
}

pub fn dequantize_w4(qweight: &Tensor, qzeros: &Tensor, scales: &Tensor) -> Result<Tensor> {
    qweight.apply_op1_no_bwd(&DequantizeW4 {
        qzeros: qzeros.clone(),
        scales: scales.clone(),
        bits: 4,
    })
}

pub fn dequantize_w8(qweight: &Tensor, qzeros: &Tensor, scales: &Tensor) -> Result<Tensor> {
    qweight.apply_op1_no_bwd(&DequantizeW4 {
        qzeros: qzeros.clone(),
        scales: scales.clone(),
        bits: 8,
    })
}

#[repr(C)]
struct GptqParams {
    in_features: u32,
    out_features: u32,
    group_size: u32,
    group_shift: u32,
    k_splits: u32,
    chunk: u32,
}

struct GptqShape {
    in_features: usize,
    out_features: usize,
    group_size: usize,
    packed_in: usize,
}

impl GptqShape {
    fn new_bits(packed_in: usize, out_features: usize, groups: usize, bits: u32) -> Result<Self> {
        let pack_factor = (32 / bits) as usize;
        let in_features = packed_in * pack_factor;
        if groups == 0 || !in_features.is_multiple_of(groups) {
            candle_core::bail!(
                "GPTQ-W{bits}A16: in_features {in_features} not divisible by groups {groups}"
            );
        }
        if !out_features.is_multiple_of(pack_factor) {
            candle_core::bail!(
                "GPTQ-W{bits}A16: out_features {out_features} not divisible by pack_factor {pack_factor}"
            );
        }
        Ok(Self {
            in_features,
            out_features,
            group_size: in_features / groups,
            packed_in,
        })
    }

    fn params(&self, group_shift: usize, k_splits: usize, chunk: usize) -> GptqParams {
        GptqParams {
            in_features: self.in_features as u32,
            out_features: self.out_features as u32,
            group_size: self.group_size as u32,
            group_shift: group_shift as u32,
            k_splits: k_splits as u32,
            chunk: chunk as u32,
        }
    }
}

struct GptqMatmul {
    x: Tensor,
    qweight: Tensor,
    qzeros: Tensor,
    scales: Tensor,
    bits: u32,
}

impl InplaceOp1 for GptqMatmul {
    fn name(&self) -> &'static str {
        match self.bits {
            8 => "gptq8-matmul",
            _ => "gptq4-matmul",
        }
    }

    fn cpu_fwd(&self, _s: &mut CpuStorage, _l: &Layout) -> Result<()> {
        candle_core::bail!(
            "GptqMatmul (bits={}): Metal-only — use the CPU dequantize_gptq path",
            self.bits
        )
    }

    fn metal_fwd(&self, out: &mut MetalStorage, out_l: &Layout) -> Result<()> {
        if out.dtype() != DType::F32 {
            candle_core::bail!("GptqMatmul: accumulator must be F32, got {:?}", out.dtype());
        }
        let pack_factor = (32 / self.bits) as usize;
        let x_sl = self.x.storage_and_layout();
        let x_l = x_sl.1;
        if !x_l.is_contiguous() {
            candle_core::bail!("GptqMatmul: x must be contiguous");
        }
        let x_dims = x_l.dims();
        if x_dims.len() != 2 || x_dims[0] != 1 {
            candle_core::bail!("GptqMatmul: x must be [1, in] (M=1 decode path), got {x_dims:?}");
        }
        let in_features = x_dims[1];

        let (packed_in, qw_out) = self.qweight.dims2()?;
        let (groups, out_features) = self.scales.dims2()?;
        if qw_out != out_features {
            candle_core::bail!(
                "GptqMatmul: qweight dim1 {qw_out} != scales out_features {out_features}"
            );
        }
        let qz_dims = self.qzeros.dims();
        if qz_dims.len() != 2 || qz_dims[0] != groups || qz_dims[1] * pack_factor != out_features {
            candle_core::bail!(
                "GptqMatmul: qzeros shape {:?} != [{groups}, {}]",
                qz_dims,
                out_features / pack_factor
            );
        }

        let shape = GptqShape::new_bits(packed_in, out_features, groups, self.bits)?;
        if in_features != shape.in_features {
            candle_core::bail!(
                "GptqMatmul: x in_features {in_features} != weight in_features {}",
                shape.in_features
            );
        }
        if out_l.dims() != [1, shape.out_features] {
            candle_core::bail!(
                "GptqMatmul: accumulator shape {:?} != [1, {}]",
                out_l.dims(),
                shape.out_features
            );
        }
        if self.scales.dtype() != self.x.dtype() {
            candle_core::bail!(
                "GptqMatmul: scales dtype {:?} must match x dtype {:?}",
                self.scales.dtype(),
                self.x.dtype()
            );
        }
        if !shape.group_size.is_power_of_two() {
            candle_core::bail!(
                "GptqMatmul: group_size {} must be a power of two",
                shape.group_size
            );
        }

        let kernel_name = match (self.bits, self.x.dtype()) {
            (4, DType::F16) => "gptq4_gemv_f16",
            (4, DType::BF16) => "gptq4_gemv_bf16",
            (8, DType::F16) => "gptq8_gemv_f16",
            (8, DType::BF16) => "gptq8_gemv_bf16",
            (bits, other) => candle_core::bail!(
                "GptqMatmul: unsupported (bits, dtype) combo ({bits}, {other:?})"
            ),
        };

        // `chunk` must be a multiple of pack_factor (kernel iterates whole words).
        const TARGET_THREADS: usize = 32768;
        let k_splits = (TARGET_THREADS / shape.out_features.max(1))
            .clamp(1, 256)
            .min(shape.in_features.max(1));
        let raw_chunk = shape.in_features.div_ceil(k_splits);
        let chunk = raw_chunk.div_ceil(pack_factor) * pack_factor;
        let k_splits = shape.in_features.div_ceil(chunk);
        let group_shift = shape.group_size.trailing_zeros() as usize;
        let params = shape.params(group_shift, k_splits, chunk);

        let dtype_bytes = self.x.dtype().size_in_bytes();
        let device = out.device();

        let x_buf = match &*x_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("GptqMatmul: x must be Metal-resident"),
        };
        let qw_sl = self.qweight.storage_and_layout();
        let qz_sl = self.qzeros.storage_and_layout();
        let sc_sl = self.scales.storage_and_layout();
        let qweight_buf = match &*qw_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("GptqMatmul: qweight must be Metal-resident"),
        };
        let qzeros_buf = match &*qz_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("GptqMatmul: qzeros must be Metal-resident"),
        };
        let scales_buf = match &*sc_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("GptqMatmul: scales must be Metal-resident"),
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
                width: shape.out_features.div_ceil(TG_WIDTH),
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

struct DequantizeGptqPacked {
    qzeros: Tensor,
    scales: Tensor,
    bits: u32,
}

impl CustomOp1 for DequantizeGptqPacked {
    fn name(&self) -> &'static str {
        match self.bits {
            8 => "dequantize-gptq8",
            _ => "dequantize-gptq4",
        }
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!(
            "DequantizeGptq{}: Metal-only — use the CPU dequantize_gptq path",
            self.bits
        )
    }

    fn metal_fwd(&self, qweight: &MetalStorage, qw_l: &Layout) -> Result<(MetalStorage, Shape)> {
        if !qw_l.is_contiguous() {
            candle_core::bail!("DequantizeGptq{}: qweight must be contiguous", self.bits);
        }
        let pack_factor = (32 / self.bits) as usize;
        let qw_dims = qw_l.dims();
        if qw_dims.len() != 2 {
            candle_core::bail!(
                "DequantizeGptq{}: qweight must be 2-D [in/pack_factor, out]",
                self.bits
            );
        }
        let (packed_in, out_features) = (qw_dims[0], qw_dims[1]);
        let (groups, sc_out) = self.scales.dims2()?;
        if sc_out != out_features {
            candle_core::bail!(
                "DequantizeGptq{}: scales dim1 {sc_out} != qweight dim1 {out_features}",
                self.bits
            );
        }
        let qz_dims = self.qzeros.dims();
        if qz_dims.len() != 2 || qz_dims[0] != groups || qz_dims[1] * pack_factor != out_features {
            candle_core::bail!(
                "DequantizeGptq{}: qzeros shape {:?} != [{groups}, {}]",
                self.bits,
                qz_dims,
                out_features / pack_factor
            );
        }
        let shape = GptqShape::new_bits(packed_in, out_features, groups, self.bits)?;

        let out_dtype = self.scales.dtype();
        let kernel_name = match (self.bits, out_dtype) {
            (4, DType::F16) => "dequantize_gptq4_f16",
            (4, DType::BF16) => "dequantize_gptq4_bf16",
            (8, DType::F16) => "dequantize_gptq8_f16",
            (8, DType::BF16) => "dequantize_gptq8_bf16",
            (bits, other) => candle_core::bail!(
                "DequantizeGptq{bits}: unsupported (bits, dtype) combo ({bits}, {other:?})"
            ),
        };
        let dtype_bytes = out_dtype.size_in_bytes();
        let out_elems = shape.in_features * shape.out_features;
        let device = qweight.device();
        let output = device.new_buffer(out_elems, out_dtype, "dequant_gptq")?;
        let params = shape.params(0, 0, 0);

        let qz_sl = self.qzeros.storage_and_layout();
        let sc_sl = self.scales.storage_and_layout();
        let qzeros_buf = match &*qz_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("DequantizeGptq: qzeros must be Metal-resident"),
        };
        let scales_buf = match &*sc_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("DequantizeGptq: scales must be Metal-resident"),
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
                width: shape.packed_in.div_ceil(TG_WIDTH),
                height: shape.out_features,
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
            Shape::from_dims(&[shape.in_features, shape.out_features]),
        ))
    }
}

pub fn gptq_matmul(
    x: &Tensor,
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    bits: u32,
) -> Result<Tensor> {
    let out_features = scales.dim(1)?;
    let out = Tensor::zeros((1, out_features), DType::F32, x.device())?;
    out.inplace_op1(&GptqMatmul {
        x: x.clone(),
        qweight: qweight.clone(),
        qzeros: qzeros.clone(),
        scales: scales.clone(),
        bits,
    })?;
    out.to_dtype(x.dtype())
}

pub fn dequantize_gptq_packed(
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    bits: u32,
) -> Result<Tensor> {
    qweight.apply_op1_no_bwd(&DequantizeGptqPacked {
        qzeros: qzeros.clone(),
        scales: scales.clone(),
        bits,
    })
}

#[cfg(test)]
mod fused_kernel_parity_tests {
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
        AwqRawTensors::new_awq(4, qweight, qzeros, scales)
    }

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
        let raw = build_awq_triplet(dev, dtype, in_features, out_features, group_size);
        let x_data: Vec<f32> = (0..in_features)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.03)
            .collect();
        let x = Tensor::from_vec(x_data, (1, in_features), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let y_kernel = w4a16_matmul(
            &x,
            &raw.qweight,
            raw.qzeros.as_ref().expect("AWQ qzeros"),
            &raw.scales,
        )
        .expect("w4a16 kernel must not error");

        let w_ref = dequantize_awq(&raw, dev, DType::F32).unwrap();
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
        let w_kernel = dequantize_w4(
            &raw.qweight,
            raw.qzeros.as_ref().expect("AWQ qzeros"),
            &raw.scales,
        )
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
        let err = w4a16_matmul(
            &x,
            &raw.qweight,
            raw.qzeros.as_ref().expect("AWQ qzeros"),
            &raw.scales,
        )
        .expect_err("must reject CPU tensors");
        assert!(
            format!("{err}").contains("Metal"),
            "unexpected error: {err}"
        );
    }

    fn pack_awq_8bit(matrix: &[Vec<u8>], rows: usize, out_features: usize) -> Vec<i32> {
        let packed_out = out_features / 4;
        let mut words = vec![0u32; rows * packed_out];
        for (i, row) in matrix.iter().enumerate().take(rows) {
            for j in 0..packed_out {
                let mut word = 0u32;
                for k in 0..4 {
                    word |= (row[j * 4 + k] as u32) << (8 * k as u32);
                }
                words[i * packed_out + j] = word;
            }
        }
        words.into_iter().map(|w| w as i32).collect()
    }

    fn build_w8a16_triplet(
        dev: &Device,
        dtype: DType,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let groups = in_features / group_size;
        let iweight: Vec<Vec<u8>> = (0..in_features)
            .map(|i| {
                (0..out_features)
                    .map(|j| ((i * 11 + j * 7) & 0xFF) as u8)
                    .collect()
            })
            .collect();
        let izero: Vec<Vec<u8>> = (0..groups)
            .map(|g| {
                (0..out_features)
                    .map(|j| ((g * 31 + j * 5 + 2) & 0xFF) as u8)
                    .collect()
            })
            .collect();
        let scales_f: Vec<f32> = (0..groups)
            .flat_map(|g| (0..out_features).map(move |j| 0.001 + 0.0007 * ((g + j) % 17) as f32))
            .collect();

        let mut ref_w = vec![0f32; out_features * in_features];
        for i in 0..in_features {
            let g = i / group_size;
            for j in 0..out_features {
                let v = (iweight[i][j] as i32 - izero[g][j] as i32) as f32
                    * scales_f[g * out_features + j];
                ref_w[j * in_features + i] = v;
            }
        }

        let qweight = Tensor::from_vec(
            pack_awq_8bit(&iweight, in_features, out_features),
            (in_features, out_features / 4),
            dev,
        )
        .unwrap();
        let qzeros = Tensor::from_vec(
            pack_awq_8bit(&izero, groups, out_features),
            (groups, out_features / 4),
            dev,
        )
        .unwrap();
        let scales = Tensor::from_vec(scales_f, (groups, out_features), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        (qweight, qzeros, scales, ref_w)
    }

    fn run_w8a16_gemv_parity(
        dev: &Device,
        dtype: DType,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        label: &str,
    ) {
        let (qw, qz, sc, ref_w) =
            build_w8a16_triplet(dev, dtype, in_features, out_features, group_size);
        let x_data: Vec<f32> = (0..in_features)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.03)
            .collect();
        let x = Tensor::from_vec(x_data.clone(), (1, in_features), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let y_kernel = w8a16_matmul(&x, &qw, &qz, &sc).expect("w8a16 kernel must not error");

        let w_ref = Tensor::from_vec(ref_w.clone(), (out_features, in_features), dev).unwrap();
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

    fn run_dequant_w8_parity(
        dev: &Device,
        dtype: DType,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        label: &str,
    ) {
        let (qw, qz, sc, ref_w) =
            build_w8a16_triplet(dev, dtype, in_features, out_features, group_size);
        let w_kernel = dequantize_w8(&qw, &qz, &sc).expect("dequantize_w8 must not error");
        let w_ref = Tensor::from_vec(ref_w, (out_features, in_features), dev).unwrap();
        let w_ref_t = w_ref.t().unwrap().contiguous().unwrap();

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
    fn w8a16_gemv_bf16_g128_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_w8a16_gemv_parity(&dev, DType::BF16, 256, 128, 128, "w8/bf16/g128");
    }

    #[test]
    fn w8a16_gemv_f16_g64_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_w8a16_gemv_parity(&dev, DType::F16, 256, 128, 64, "w8/f16/g64");
    }

    #[test]
    fn dequantize_w8_bf16_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_dequant_w8_parity(&dev, DType::BF16, 256, 128, 128, "w8/bf16/g128");
    }

    fn build_gptq_triplet(
        dev: &Device,
        dtype: DType,
        bits: u32,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> crate::common::awq::QuantWeight {
        let pf = 32usize / bits as usize;
        let groups = in_features / group_size;
        let mask = (1u32 << bits) - 1;

        let packed_in = in_features / pf;
        let mut qw = vec![0i32; packed_in * out_features];
        for iw in 0..packed_in {
            for o in 0..out_features {
                let mut w = 0u32;
                for k in 0..pf {
                    let i = iw * pf + k;
                    let v = ((i * 13 + o * 7) as u32) & mask;
                    w |= v << (bits as u32 * k as u32);
                }
                qw[iw * out_features + o] = w as i32;
            }
        }
        let packed_out = out_features / pf;
        let mut qz = vec![0i32; groups * packed_out];
        for g in 0..groups {
            for ow in 0..packed_out {
                let mut w = 0u32;
                for k in 0..pf {
                    let o = ow * pf + k;
                    let v = ((g * 31 + o * 17 + 1) as u32) & mask;
                    w |= v << (bits as u32 * k as u32);
                }
                qz[g * packed_out + ow] = w as i32;
            }
        }
        let scales: Vec<f32> = (0..groups)
            .flat_map(|g| (0..out_features).map(move |o| 0.002 + 0.0005 * ((g + o) % 11) as f32))
            .collect();

        let qweight = Tensor::from_vec(qw, (packed_in, out_features), dev).unwrap();
        let qzeros = Tensor::from_vec(qz, (groups, packed_out), dev).unwrap();
        let scales_t = Tensor::from_vec(scales, (groups, out_features), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        crate::common::awq::QuantWeight::new_gptq(bits, false, qweight, Some(qzeros), scales_t)
    }

    fn run_gptq_gemv_parity(
        dev: &Device,
        dtype: DType,
        bits: u32,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        label: &str,
    ) {
        let raw = build_gptq_triplet(dev, dtype, bits, in_features, out_features, group_size);
        let x_data: Vec<f32> = (0..in_features)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
            .collect();
        let x = Tensor::from_vec(x_data, (1, in_features), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let y_kernel = gptq_matmul(
            &x,
            &raw.qweight,
            raw.qzeros.as_ref().expect("GPTQ qzeros"),
            &raw.scales,
            bits,
        )
        .expect("gptq kernel must not error");

        let w_ref = crate::common::awq::dequantize_gptq(&raw, dev, DType::F32).unwrap();
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

    fn run_gptq_dequant_parity(
        dev: &Device,
        dtype: DType,
        bits: u32,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        label: &str,
    ) {
        let raw = build_gptq_triplet(dev, dtype, bits, in_features, out_features, group_size);
        let w_kernel = dequantize_gptq_packed(
            &raw.qweight,
            raw.qzeros.as_ref().expect("GPTQ qzeros"),
            &raw.scales,
            bits,
        )
        .expect("dequantize_gptq_packed must not error"); // [in, out]
        let w_ref = crate::common::awq::dequantize_gptq(&raw, dev, DType::F32).unwrap(); // [out, in]
        let w_ref_t = w_ref.t().unwrap().contiguous().unwrap();

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
    fn gptq4_gemv_bf16_g128_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_gptq_gemv_parity(&dev, DType::BF16, 4, 256, 128, 128, "gptq4/bf16/g128");
    }

    #[test]
    fn gptq8_gemv_bf16_g128_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_gptq_gemv_parity(&dev, DType::BF16, 8, 256, 128, 128, "gptq8/bf16/g128");
    }

    #[test]
    fn gptq4_gemv_f16_g64_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_gptq_gemv_parity(&dev, DType::F16, 4, 256, 128, 64, "gptq4/f16/g64");
    }

    #[test]
    fn gptq8_dequantize_bf16_matches_reference() {
        let Some(dev) = metal_device_or_skip() else {
            return;
        };
        run_gptq_dequant_parity(&dev, DType::BF16, 8, 256, 128, 128, "gptq8/bf16/g128");
    }

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
        let w_bf16 = dequantize_awq(&raw, &dev, dtype)
            .unwrap()
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        let x0 = Tensor::zeros((1, n), dtype, &dev).unwrap();

        for _ in 0..10 {
            let _ = w4a16_matmul(
                &x0,
                &raw.qweight,
                raw.qzeros.as_ref().expect("AWQ qzeros"),
                &raw.scales,
            )
            .unwrap();
            let _ = x0.matmul(&w_bf16).unwrap();
        }
        x0.matmul(&w_bf16).unwrap().to_device(&Device::Cpu).unwrap();

        let per_call = |t: Instant| t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let t = Instant::now();
        let mut sink = Vec::with_capacity(iters);
        for _ in 0..iters {
            sink.push(
                w4a16_matmul(
                    &x0,
                    &raw.qweight,
                    raw.qzeros.as_ref().expect("AWQ qzeros"),
                    &raw.scales,
                )
                .unwrap(),
            );
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

        let t = Instant::now();
        let mut x = x0.clone();
        for _ in 0..iters {
            x = w4a16_matmul(
                &x,
                &raw.qweight,
                raw.qzeros.as_ref().expect("AWQ qzeros"),
                &raw.scales,
            )
            .unwrap();
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

#[repr(C)]
struct GgufParams {
    in_features: u32,
    out_features: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GgufFastQuant {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl GgufFastQuant {
    pub fn block_layout(self) -> (usize, usize) {
        match self {
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
        }
    }

    fn gemv_kernel(self) -> &'static str {
        match self {
            Self::Q4_0 => "gguf_q4_0_gemv_bf16",
            Self::Q4_1 => "gguf_q4_1_gemv_bf16",
            Self::Q5_0 => "gguf_q5_0_gemv_bf16",
            Self::Q5_1 => "gguf_q5_1_gemv_bf16",
            Self::Q8_0 => "gguf_q8_0_gemv_bf16",
            Self::Q2K => "gguf_q2k_gemv_bf16",
            Self::Q3K => "gguf_q3k_gemv_bf16",
            Self::Q4K => "gguf_q4k_gemv_bf16",
            Self::Q5K => "gguf_q5k_gemv_bf16",
            Self::Q6K => "gguf_q6k_gemv_bf16",
        }
    }

    fn mul_mm_kernel(self) -> &'static str {
        match self {
            Self::Q4_0 => "gguf_q4_0_mul_mm_bf16",
            Self::Q4_1 => "gguf_q4_1_mul_mm_bf16",
            Self::Q5_0 => "gguf_q5_0_mul_mm_bf16",
            Self::Q5_1 => "gguf_q5_1_mul_mm_bf16",
            Self::Q8_0 => "gguf_q8_0_mul_mm_bf16",
            Self::Q2K => "gguf_q2k_mul_mm_bf16",
            Self::Q3K => "gguf_q3k_mul_mm_bf16",
            Self::Q4K => "gguf_q4k_mul_mm_bf16",
            Self::Q5K => "gguf_q5k_mul_mm_bf16",
            Self::Q6K => "gguf_q6k_mul_mm_bf16",
        }
    }

    // Batched-decode gemv (M activations share one weight read).
    fn batch_kernel(self) -> &'static str {
        match self {
            Self::Q4_0 => "gguf_q4_0_gemv_batch_bf16",
            Self::Q4_1 => "gguf_q4_1_gemv_batch_bf16",
            Self::Q5_0 => "gguf_q5_0_gemv_batch_bf16",
            Self::Q5_1 => "gguf_q5_1_gemv_batch_bf16",
            Self::Q8_0 => "gguf_q8_0_gemv_batch_bf16",
            Self::Q2K => "gguf_q2k_gemv_batch_bf16",
            Self::Q3K => "gguf_q3k_gemv_batch_bf16",
            Self::Q4K => "gguf_q4k_gemv_batch_bf16",
            Self::Q5K => "gguf_q5k_gemv_batch_bf16",
            Self::Q6K => "gguf_q6k_gemv_batch_bf16",
        }
    }

    fn op_name(self) -> &'static str {
        match self {
            Self::Q4_0 => "gguf-q4_0-matmul",
            Self::Q4_1 => "gguf-q4_1-matmul",
            Self::Q5_0 => "gguf-q5_0-matmul",
            Self::Q5_1 => "gguf-q5_1-matmul",
            Self::Q8_0 => "gguf-q8_0-matmul",
            Self::Q2K => "gguf-q2k-matmul",
            Self::Q3K => "gguf-q3k-matmul",
            Self::Q4K => "gguf-q4k-matmul",
            Self::Q5K => "gguf-q5k-matmul",
            Self::Q6K => "gguf-q6k-matmul",
        }
    }

    // (threads_per_TG, rows_per_TG) per quant — must match the kernel's geometry.
    fn dispatch_geometry(self) -> (usize, usize) {
        const SIMDWIDTH: usize = 32;
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q2K => {
                (2 * SIMDWIDTH, 2 * 4)
            }
            Self::Q3K => (2 * SIMDWIDTH, 2 * 2),
            Self::Q4K => (SIMDWIDTH, 4),
            Self::Q5K => (2 * SIMDWIDTH, 2 * 2),
            Self::Q6K => (2 * SIMDWIDTH, 2),
        }
    }
}

/// Max batch the batched-decode gemv kernels handle (register cap); above this,
/// the mul_mm prefill kernel is already efficient. Must match GGUF_BATCH_MAX in
/// quant_kernels.metal.
pub const GGUF_BATCH_MAX: usize = 8;

#[repr(C)]
struct GgufBatchParams {
    in_features: u32,
    out_features: u32,
    m_batch: u32,
}

// Decode matmul for 1 <= M <= GGUF_BATCH_MAX: M=1 runs the plain gemv kernel,
// M>=2 the batched gemv (M activation vectors share one weight read).
struct GgufQuantMatmul {
    x: Tensor,
    weight_bytes: Tensor,
    in_features: usize,
    out_features: usize,
    quant: GgufFastQuant,
    m: usize,
}

impl InplaceOp1 for GgufQuantMatmul {
    fn name(&self) -> &'static str {
        self.quant.op_name()
    }

    fn cpu_fwd(&self, _s: &mut CpuStorage, _l: &Layout) -> Result<()> {
        candle_core::bail!("GgufQuantMatmul ({:?}): Metal-only path", self.quant)
    }

    fn metal_fwd(&self, out: &mut MetalStorage, out_l: &Layout) -> Result<()> {
        if out.dtype() != DType::BF16 {
            candle_core::bail!(
                "GgufQuantMatmul ({:?}): out must be BF16, got {:?}",
                self.quant,
                out.dtype()
            );
        }
        if self.x.dtype() != DType::BF16 {
            candle_core::bail!(
                "GgufQuantMatmul ({:?}): x must be BF16, got {:?}",
                self.quant,
                self.x.dtype()
            );
        }
        if self.weight_bytes.dtype() != DType::U8 {
            candle_core::bail!(
                "GgufQuantMatmul ({:?}): weight_bytes must be U8, got {:?}",
                self.quant,
                self.weight_bytes.dtype()
            );
        }
        let (block_elems, block_bytes) = self.quant.block_layout();
        if !self.in_features.is_multiple_of(block_elems) {
            candle_core::bail!(
                "GgufQuantMatmul ({:?}): in_features {} must be a multiple of {}",
                self.quant,
                self.in_features,
                block_elems,
            );
        }
        if !(1..=GGUF_BATCH_MAX).contains(&self.m) {
            candle_core::bail!(
                "GgufQuantMatmul ({:?}): m {} out of [1, {GGUF_BATCH_MAX}]",
                self.quant,
                self.m
            );
        }

        let x_sl = self.x.storage_and_layout();
        let x_l = x_sl.1;
        if !x_l.is_contiguous() {
            candle_core::bail!("GgufQuantMatmul ({:?}): x must be contiguous", self.quant);
        }
        let x_dims = x_l.dims();
        if x_dims != [self.m, self.in_features] {
            candle_core::bail!(
                "GgufQuantMatmul ({:?}): x shape {:?} != [{}, {}]",
                self.quant,
                x_dims,
                self.m,
                self.in_features
            );
        }
        if out_l.dims() != [self.m, self.out_features] {
            candle_core::bail!(
                "GgufQuantMatmul ({:?}): output shape {:?} != [{}, {}]",
                self.quant,
                out_l.dims(),
                self.m,
                self.out_features
            );
        }
        let expected_bytes = self.out_features * (self.in_features / block_elems) * block_bytes;
        let w_elems: usize = self.weight_bytes.dims().iter().product();
        if w_elems != expected_bytes {
            candle_core::bail!(
                "GgufQuantMatmul ({:?}): weight_bytes has {} bytes, expected {} (out={} × blocks_per_row={} × {})",
                self.quant,
                w_elems,
                expected_bytes,
                self.out_features,
                self.in_features / block_elems,
                block_bytes,
            );
        }

        let device = out.device();
        let x_buf = match &*x_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!(
                "GgufQuantMatmul ({:?}): x must be Metal-resident",
                self.quant
            ),
        };
        let w_sl = self.weight_bytes.storage_and_layout();
        let w_buf = match &*w_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!(
                "GgufQuantMatmul ({:?}): weight_bytes must be Metal-resident",
                self.quant
            ),
        };

        let kernel = if self.m == 1 {
            self.quant.gemv_kernel()
        } else {
            self.quant.batch_kernel()
        };
        let pipeline = get_or_compile_quant_pipeline(device.device(), kernel)?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), x_l.start_offset() * 2);
        encoder.set_buffer(1, Some(w_buf), w_sl.1.start_offset());
        encoder.set_buffer(2, Some(out.buffer()), out_l.start_offset() * 2);
        if self.m == 1 {
            encoder.set_bytes(
                3,
                &GgufParams {
                    in_features: self.in_features as u32,
                    out_features: self.out_features as u32,
                },
            );
        } else {
            encoder.set_bytes(
                3,
                &GgufBatchParams {
                    in_features: self.in_features as u32,
                    out_features: self.out_features as u32,
                    m_batch: self.m as u32,
                },
            );
        }
        encoder.use_resource(x_buf, MTLResourceUsage::Read);
        encoder.use_resource(w_buf, MTLResourceUsage::Read);
        encoder.use_resource(out.buffer(), MTLResourceUsage::Write);

        // Batch kernels keep the gemv geometry (same threads/rows per TG).
        let (tg_threads, rows_per_tg) = self.quant.dispatch_geometry();
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.out_features.div_ceil(rows_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_threads,
                height: 1,
                depth: 1,
            },
        );

        Ok(())
    }
}

/// Decode matmul for x of shape [m, in_features] with 1 <= m <= GGUF_BATCH_MAX;
/// larger m belongs on gguf_quant_mul_mm.
pub fn gguf_quant_matmul(
    x: &Tensor,
    weight_bytes: &Tensor,
    in_features: usize,
    out_features: usize,
    quant: GgufFastQuant,
) -> Result<Tensor> {
    let m = x.dim(0)?;
    let out = Tensor::zeros((m, out_features), DType::BF16, x.device())?;
    out.inplace_op1(&GgufQuantMatmul {
        x: x.clone(),
        weight_bytes: weight_bytes.clone(),
        in_features,
        out_features,
        quant,
        m,
    })?;
    Ok(out)
}

const GGUF_MM_BM: usize = 16;
const GGUF_MM_BN: usize = 16;

#[repr(C)]
struct GgufMatmulParams {
    m_total: u32,
    n_total: u32,
    k_total: u32,
}

struct GgufQuantMulMM {
    weight_bytes: Tensor,
    n: usize,
    k: usize,
    quant: GgufFastQuant,
}

impl CustomOp1 for GgufQuantMulMM {
    fn name(&self) -> &'static str {
        self.quant.op_name()
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("GgufQuantMulMM ({:?}): Metal-only path", self.quant)
    }

    fn metal_fwd(&self, x: &MetalStorage, x_l: &Layout) -> Result<(MetalStorage, Shape)> {
        if x.dtype() != DType::BF16 {
            candle_core::bail!(
                "GgufQuantMulMM ({:?}): x must be BF16, got {:?}",
                self.quant,
                x.dtype()
            );
        }
        if !x_l.is_contiguous() {
            candle_core::bail!("GgufQuantMulMM ({:?}): x must be contiguous", self.quant);
        }
        let x_dims = x_l.dims();
        if x_dims.len() != 2 || x_dims[1] != self.k {
            candle_core::bail!(
                "GgufQuantMulMM ({:?}): x shape {:?} != [M, {}]",
                self.quant,
                x_dims,
                self.k
            );
        }
        let m = x_dims[0];
        let (block_elems, block_bytes) = self.quant.block_layout();
        if !self.k.is_multiple_of(block_elems) {
            candle_core::bail!(
                "GgufQuantMulMM ({:?}): K {} must be a multiple of {}",
                self.quant,
                self.k,
                block_elems,
            );
        }
        let w_sl = self.weight_bytes.storage_and_layout();
        if self.weight_bytes.dtype() != DType::U8 {
            candle_core::bail!(
                "GgufQuantMulMM ({:?}): weight_bytes must be U8, got {:?}",
                self.quant,
                self.weight_bytes.dtype()
            );
        }
        let w_elems: usize = w_sl.1.dims().iter().product();
        let expected_w = self.n * (self.k / block_elems) * block_bytes;
        if w_elems != expected_w {
            candle_core::bail!(
                "GgufQuantMulMM ({:?}): weight_bytes has {} bytes, expected {}",
                self.quant,
                w_elems,
                expected_w,
            );
        }

        let params = GgufMatmulParams {
            m_total: m as u32,
            n_total: self.n as u32,
            k_total: self.k as u32,
        };

        let device = x.device();
        let out_elems = m * self.n;
        let output = device.new_buffer(out_elems, DType::BF16, "gguf-mul-mm")?;

        let x_buf = x.buffer();
        let w_buf = match &*w_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!(
                "GgufQuantMulMM ({:?}): weight_bytes must be Metal-resident",
                self.quant
            ),
        };

        let pipeline = get_or_compile_quant_pipeline(device.device(), self.quant.mul_mm_kernel())?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), x_l.start_offset() * 2); // bf16 = 2 bytes
        encoder.set_buffer(1, Some(w_buf), w_sl.1.start_offset()); // u8 = 1 byte
        encoder.set_buffer(2, Some(&*output), 0);
        encoder.set_bytes(3, &params);
        encoder.use_resource(x_buf, MTLResourceUsage::Read);
        encoder.use_resource(w_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*output, MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.n.div_ceil(GGUF_MM_BN),
                height: m.div_ceil(GGUF_MM_BM),
                depth: 1,
            },
            MTLSize {
                width: GGUF_MM_BN,
                height: GGUF_MM_BM,
                depth: 1,
            },
        );

        let storage = MetalStorage::new(output, device.clone(), out_elems, DType::BF16);
        Ok((storage, Shape::from(vec![m, self.n])))
    }
}

pub fn gguf_quant_mul_mm(
    x: &Tensor,
    weight_bytes: &Tensor,
    in_features: usize,
    out_features: usize,
    quant: GgufFastQuant,
) -> Result<Tensor> {
    x.apply_op1(GgufQuantMulMM {
        weight_bytes: weight_bytes.clone(),
        n: out_features,
        k: in_features,
        quant,
    })
}

// ── MXFP4 (GPT-OSS experts): packed FP4 blocks + E8M0 scales ───────────────

struct Mxfp4Matmul {
    x: Tensor,
    blocks: Tensor,
    scales: Tensor,
    k: usize,
    n: usize,
    m: usize,
}

impl InplaceOp1 for Mxfp4Matmul {
    fn name(&self) -> &'static str {
        "mxfp4-matmul"
    }

    fn cpu_fwd(&self, _s: &mut CpuStorage, _l: &Layout) -> Result<()> {
        candle_core::bail!("Mxfp4Matmul: Metal-only path")
    }

    fn metal_fwd(&self, out: &mut MetalStorage, out_l: &Layout) -> Result<()> {
        const BLOCK: usize = 32;
        if out.dtype() != DType::BF16 || self.x.dtype() != DType::BF16 {
            candle_core::bail!(
                "Mxfp4Matmul: x/out must be BF16, got {:?}/{:?}",
                self.x.dtype(),
                out.dtype()
            );
        }
        if self.blocks.dtype() != DType::U8 || self.scales.dtype() != DType::U8 {
            candle_core::bail!("Mxfp4Matmul: blocks/scales must be U8");
        }
        if !self.k.is_multiple_of(BLOCK) {
            candle_core::bail!("Mxfp4Matmul: K {} must be a multiple of {BLOCK}", self.k);
        }
        if !(1..=GGUF_BATCH_MAX).contains(&self.m) {
            candle_core::bail!("Mxfp4Matmul: m {} out of [1, {GGUF_BATCH_MAX}]", self.m);
        }
        let nb = self.k / BLOCK;
        if self.blocks.elem_count() != self.n * nb * 16 || self.scales.elem_count() != self.n * nb {
            candle_core::bail!(
                "Mxfp4Matmul: blocks {} / scales {} elements don't match [{}, {}]",
                self.blocks.elem_count(),
                self.scales.elem_count(),
                self.n,
                self.k
            );
        }

        let x_sl = self.x.storage_and_layout();
        let x_l = x_sl.1;
        if !x_l.is_contiguous() {
            candle_core::bail!("Mxfp4Matmul: x must be contiguous");
        }
        if x_l.dims() != [self.m, self.k] {
            candle_core::bail!(
                "Mxfp4Matmul: x shape {:?} != [{}, {}]",
                x_l.dims(),
                self.m,
                self.k
            );
        }
        if out_l.dims() != [self.m, self.n] {
            candle_core::bail!(
                "Mxfp4Matmul: output shape {:?} != [{}, {}]",
                out_l.dims(),
                self.m,
                self.n
            );
        }

        let params = GgufBatchParams {
            in_features: self.k as u32,
            out_features: self.n as u32,
            m_batch: self.m as u32,
        };

        let device = out.device();
        let x_buf = match &*x_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("Mxfp4Matmul: x must be Metal-resident"),
        };
        let b_sl = self.blocks.storage_and_layout();
        let b_buf = match &*b_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("Mxfp4Matmul: blocks must be Metal-resident"),
        };
        let s_sl = self.scales.storage_and_layout();
        let s_buf = match &*s_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("Mxfp4Matmul: scales must be Metal-resident"),
        };

        let pipeline = get_or_compile_quant_pipeline(device.device(), "mxfp4_gemv_batch_bf16")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), x_l.start_offset() * 2);
        encoder.set_buffer(1, Some(b_buf), b_sl.1.start_offset());
        encoder.set_buffer(2, Some(s_buf), s_sl.1.start_offset());
        encoder.set_buffer(3, Some(out.buffer()), out_l.start_offset() * 2);
        encoder.set_bytes(4, &params);
        encoder.use_resource(x_buf, MTLResourceUsage::Read);
        encoder.use_resource(b_buf, MTLResourceUsage::Read);
        encoder.use_resource(s_buf, MTLResourceUsage::Read);
        encoder.use_resource(out.buffer(), MTLResourceUsage::Write);

        // 2 simdgroups x GGUF_N_DST(4) rows per TG — must match the kernel.
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.n.div_ceil(8),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );

        Ok(())
    }
}

struct Mxfp4MulMM {
    blocks: Tensor,
    scales: Tensor,
    k: usize,
    n: usize,
}

impl CustomOp1 for Mxfp4MulMM {
    fn name(&self) -> &'static str {
        "mxfp4-mul-mm"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Mxfp4MulMM: Metal-only path")
    }

    fn metal_fwd(&self, x: &MetalStorage, x_l: &Layout) -> Result<(MetalStorage, Shape)> {
        if x.dtype() != DType::BF16 {
            candle_core::bail!("Mxfp4MulMM: x must be BF16, got {:?}", x.dtype());
        }
        if !x_l.is_contiguous() {
            candle_core::bail!("Mxfp4MulMM: x must be contiguous");
        }
        let x_dims = x_l.dims();
        if x_dims.len() != 2 || x_dims[1] != self.k {
            candle_core::bail!("Mxfp4MulMM: x shape {:?} != [M, {}]", x_dims, self.k);
        }
        let m = x_dims[0];

        let params = GgufMatmulParams {
            m_total: m as u32,
            n_total: self.n as u32,
            k_total: self.k as u32,
        };

        let device = x.device();
        let out_elems = m * self.n;
        let output = device.new_buffer(out_elems, DType::BF16, "mxfp4-mul-mm")?;

        let x_buf = x.buffer();
        let b_sl = self.blocks.storage_and_layout();
        let b_buf = match &*b_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("Mxfp4MulMM: blocks must be Metal-resident"),
        };
        let s_sl = self.scales.storage_and_layout();
        let s_buf = match &*s_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("Mxfp4MulMM: scales must be Metal-resident"),
        };

        let pipeline = get_or_compile_quant_pipeline(device.device(), "mxfp4_mul_mm_bf16")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), x_l.start_offset() * 2);
        encoder.set_buffer(1, Some(b_buf), b_sl.1.start_offset());
        encoder.set_buffer(2, Some(s_buf), s_sl.1.start_offset());
        encoder.set_buffer(3, Some(&*output), 0);
        encoder.set_bytes(4, &params);
        encoder.use_resource(x_buf, MTLResourceUsage::Read);
        encoder.use_resource(b_buf, MTLResourceUsage::Read);
        encoder.use_resource(s_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*output, MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.n.div_ceil(GGUF_MM_BN),
                height: m.div_ceil(GGUF_MM_BM),
                depth: 1,
            },
            MTLSize {
                width: GGUF_MM_BN,
                height: GGUF_MM_BM,
                depth: 1,
            },
        );

        let storage = MetalStorage::new(output, device.clone(), out_elems, DType::BF16);
        Ok((storage, Shape::from(vec![m, self.n])))
    }
}

/// MXFP4 matmul for x `[m, k]`: batched gemv for m <= GGUF_BATCH_MAX,
/// tiled mul_mm above.
pub fn mxfp4_matmul(
    x: &Tensor,
    blocks: &Tensor,
    scales: &Tensor,
    k: usize,
    n: usize,
) -> Result<Tensor> {
    let m = x.dim(0)?;
    if m <= GGUF_BATCH_MAX {
        let out = Tensor::zeros((m, n), DType::BF16, x.device())?;
        out.inplace_op1(&Mxfp4Matmul {
            x: x.clone(),
            blocks: blocks.clone(),
            scales: scales.clone(),
            k,
            n,
            m,
        })?;
        return Ok(out);
    }
    x.apply_op1(Mxfp4MulMM {
        blocks: blocks.clone(),
        scales: scales.clone(),
        k,
        n,
    })
}

// ── Decode SDPA with attention sinks (GPT-OSS) ─────────────────────────────

#[repr(C)]
struct SdpaSinkParams {
    n_heads: u32,
    n_kv_heads: u32,
    kv_len: u32,
    head_dim: u32,
    scale: f32,
}

struct SdpaVectorSink {
    sinks: Tensor,
    scale: f32,
}

impl CustomOp3 for SdpaVectorSink {
    fn name(&self) -> &'static str {
        "sdpa-vector-sink"
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
        candle_core::bail!("SdpaVectorSink: Metal-only path")
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
        if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
            candle_core::bail!("SdpaVectorSink: q/k/v must be BF16");
        }
        if !q_l.is_contiguous() || !k_l.is_contiguous() || !v_l.is_contiguous() {
            candle_core::bail!("SdpaVectorSink: q/k/v must be contiguous");
        }
        let qd = q_l.dims();
        let kd = k_l.dims();
        let vd = v_l.dims();
        if qd.len() != 4 || qd[0] != 1 || qd[2] != 1 {
            candle_core::bail!("SdpaVectorSink: q shape {qd:?} != [1, H, 1, D]");
        }
        let (h, d) = (qd[1], qd[3]);
        if kd != vd || kd.len() != 4 || kd[0] != 1 || kd[3] != d {
            candle_core::bail!("SdpaVectorSink: k/v shape {kd:?} incompatible with q {qd:?}");
        }
        let (kvh, kv_len) = (kd[1], kd[2]);
        if kvh == 0 || !h.is_multiple_of(kvh) {
            candle_core::bail!("SdpaVectorSink: n_heads {h} not divisible by kv heads {kvh}");
        }
        if d > 128 {
            candle_core::bail!("SdpaVectorSink: head_dim {d} > 128");
        }
        if self.sinks.elem_count() != h || self.sinks.dtype() != DType::BF16 {
            candle_core::bail!(
                "SdpaVectorSink: sinks must be BF16 with {h} elements, got {} {:?}",
                self.sinks.elem_count(),
                self.sinks.dtype()
            );
        }

        let params = SdpaSinkParams {
            n_heads: h as u32,
            n_kv_heads: kvh as u32,
            kv_len: kv_len as u32,
            head_dim: d as u32,
            scale: self.scale,
        };

        let device = q.device();
        let out_elems = h * d;
        let output = device.new_buffer(out_elems, DType::BF16, "sdpa-vector-sink")?;

        let s_sl = self.sinks.storage_and_layout();
        let s_buf = match &*s_sl.0 {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => candle_core::bail!("SdpaVectorSink: sinks must be Metal-resident"),
        };

        let pipeline = get_or_compile_quant_pipeline(device.device(), "sdpa_vector_sink_bf16")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(q.buffer()), q_l.start_offset() * 2);
        encoder.set_buffer(1, Some(k.buffer()), k_l.start_offset() * 2);
        encoder.set_buffer(2, Some(v.buffer()), v_l.start_offset() * 2);
        encoder.set_buffer(3, Some(s_buf), s_sl.1.start_offset() * 2);
        encoder.set_buffer(4, Some(&*output), 0);
        encoder.set_bytes(5, &params);
        encoder.use_resource(q.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(k.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(v.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*output, MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(
            MTLSize {
                width: h,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );

        let storage = MetalStorage::new(output, device.clone(), out_elems, DType::BF16);
        Ok((storage, Shape::from(vec![1, h, 1, d])))
    }
}

/// Decode attention (q_len = 1) with a per-head sink logit in the softmax
/// denominator. K/V are the unrepeated per-kv-head tensors (GQA handled in
/// kernel). Shapes: q [1, H, 1, D], k/v [1, KVH, L, D], sinks H elements.
pub fn sdpa_vector_sink(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    q.apply_op3_no_bwd(
        k,
        v,
        &SdpaVectorSink {
            sinks: sinks.clone(),
            scale,
        },
    )
}

#[cfg(all(test, feature = "metal"))]
mod decode_batch_scaling_tests {
    use super::*;
    use candle_core::Device;
    use candle_core::quantized::{GgmlDType, QTensor};
    use std::time::Instant;

    // Contract: every batched decode gemv equals candle's dequant+matmul across
    // M=2,4,8 and shapes incl. N not a multiple of the geometry's rows-per-TG
    // (exercises the row-bound) and varying nb. Fails if the M-loop or dequant is
    // wrong. Covers each quant that has a batch_kernel.
    #[test]
    fn gguf_batch_matches_reference() {
        let Ok(dev) = Device::new_metal(0) else {
            return;
        };
        let quants = [
            (GgmlDType::Q4_0, GgufFastQuant::Q4_0, "Q4_0"),
            (GgmlDType::Q4_1, GgufFastQuant::Q4_1, "Q4_1"),
            (GgmlDType::Q5_0, GgufFastQuant::Q5_0, "Q5_0"),
            (GgmlDType::Q5_1, GgufFastQuant::Q5_1, "Q5_1"),
            (GgmlDType::Q8_0, GgufFastQuant::Q8_0, "Q8_0"),
            (GgmlDType::Q2K, GgufFastQuant::Q2K, "Q2K"),
            (GgmlDType::Q3K, GgufFastQuant::Q3K, "Q3K"),
            (GgmlDType::Q4K, GgufFastQuant::Q4K, "Q4K"),
            (GgmlDType::Q5K, GgufFastQuant::Q5K, "Q5K"),
            (GgmlDType::Q6K, GgufFastQuant::Q6K, "Q6K"),
        ];
        for (ggml, fast, tag) in quants {
            for (n, k) in [(128usize, 256usize), (100, 2560), (256, 512)] {
                let w_data: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
                let w_src = Tensor::from_vec(w_data, (n, k), &Device::Cpu).unwrap();
                let qt = QTensor::quantize(&w_src, ggml).unwrap();
                let bytes = qt.data().unwrap().into_owned();
                let blen = bytes.len();
                let wb = Tensor::from_vec(bytes, (blen,), &dev).unwrap();
                let w_ref = qt
                    .dequantize(&Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                let w_ref_t = w_ref.t().unwrap().contiguous().unwrap();

                for m in [2usize, 4, 8] {
                    let x_data: Vec<f32> =
                        (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.03).collect();
                    let x = Tensor::from_vec(x_data.clone(), (m, k), &dev)
                        .unwrap()
                        .to_dtype(DType::BF16)
                        .unwrap();
                    let y_kernel = gguf_quant_matmul(&x, &wb, k, n, fast).unwrap();
                    let x_cpu = Tensor::from_vec(x_data, (m, k), &Device::Cpu).unwrap();
                    let y_ref = x_cpu.matmul(&w_ref_t).unwrap();

                    assert_eq!(
                        y_kernel.dims2().unwrap(),
                        (m, n),
                        "{tag} M={m} n={n} k={k}: shape"
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
                        "{tag} M={m} n={n}: non-finite"
                    );
                    let max_abs = r.iter().fold(0f32, |a, &v| a.max(v.abs()));
                    let tol = 0.06 * max_abs + 1e-2;
                    let diff = f
                        .iter()
                        .zip(&r)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        diff < tol,
                        "{tag} M={m} n={n} k={k}: max_abs_diff {diff} (tol {tol})"
                    );
                }
            }
        }
    }

    // Contract: both MXFP4 kernels (batched gemv m<=8, tiled mul_mm above)
    // equal the CPU reference dequant + matmul, incl. unaligned N.
    #[test]
    fn mxfp4_matmul_matches_reference() {
        let Ok(dev) = Device::new_metal(0) else {
            return;
        };
        let (n, k) = (10usize, 64usize); // N unaligned to 8-row TGs; K = 2 blocks
        let nb = k / 32;
        let blocks: Vec<u8> = (0..n * nb * 16).map(|i| (i * 37 + 11) as u8).collect();
        // E8M0 around 127 so magnitudes stay ~1.
        let scales: Vec<u8> = (0..n * nb).map(|i| 125 + (i % 5) as u8).collect();
        let w_ref = crate::common::mxfp4::dequantize_mxfp4_f32(&blocks, &scales, n, k).unwrap();
        let w_ref_t = Tensor::from_vec(w_ref, (n, k), &Device::Cpu)
            .unwrap()
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        let blocks_t = Tensor::from_vec(blocks, n * nb * 16, &dev).unwrap();
        let scales_t = Tensor::from_vec(scales, n * nb, &dev).unwrap();

        for m in [1usize, 4, 8, 33] {
            let x_data: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
            let x = Tensor::from_vec(x_data.clone(), (m, k), &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let y = mxfp4_matmul(&x, &blocks_t, &scales_t, k, n).unwrap();
            let x_cpu = Tensor::from_vec(x_data, (m, k), &Device::Cpu).unwrap();
            let y_ref = x_cpu.matmul(&w_ref_t).unwrap();

            assert_eq!(y.dims2().unwrap(), (m, n), "M={m}: shape");
            let f = y
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let r = y_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(f.iter().all(|v| v.is_finite()), "M={m}: non-finite");
            let max_abs = r.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let tol = 0.06 * max_abs + 1e-2;
            let diff = f
                .iter()
                .zip(&r)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(diff < tol, "M={m}: max_abs_diff {diff} (tol {tol})");
        }
    }

    // Contract: sdpa_vector_sink equals the scalar reference — softmax over
    // [scores, sink] with the sink column dropped — including GQA head mapping
    // and kv lengths not aligned to the 32-thread stride.
    #[test]
    fn sdpa_vector_sink_matches_reference() {
        let Ok(dev) = Device::new_metal(0) else {
            return;
        };
        let (h, kvh, d, l) = (4usize, 2usize, 64usize, 37usize);
        let scale = 1.0f32 / (d as f32).sqrt();

        let mk = |n: usize, salt: u64| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let r = (i as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(salt);
                    ((r >> 33) & 0xFFFF) as f32 / 65535.0 - 0.5
                })
                .collect()
        };
        let q_data = mk(h * d, 1);
        let k_data = mk(kvh * l * d, 2);
        let v_data = mk(kvh * l * d, 3);
        let sink_data: Vec<f32> = (0..h).map(|i| (i as f32) * 0.7 - 1.0).collect();

        let to_bf16 = |data: &[f32], shape: Vec<usize>| -> Tensor {
            Tensor::from_vec(data.to_vec(), shape, &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let q = to_bf16(&q_data, vec![1, h, 1, d]);
        let k = to_bf16(&k_data, vec![1, kvh, l, d]);
        let v = to_bf16(&v_data, vec![1, kvh, l, d]);
        let sinks = to_bf16(&sink_data, vec![h]);

        let out = sdpa_vector_sink(&q, &k, &v, &sinks, scale)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Scalar reference (f32). BF16 inputs are exactly representable after
        // the round-trip, so only accumulation order differs.
        let bf = |x: f32| f32::from_bits(x.to_bits() & 0xFFFF_0000);
        for head in 0..h {
            let kv_head = head / (h / kvh);
            let scores: Vec<f32> = (0..l)
                .map(|j| {
                    (0..d)
                        .map(|x| bf(q_data[head * d + x]) * bf(k_data[(kv_head * l + j) * d + x]))
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let sink = bf(sink_data[head]);
            let m = scores.iter().fold(sink, |a, &b| a.max(b));
            let denom: f32 = scores.iter().map(|&s| (s - m).exp()).sum::<f32>() + (sink - m).exp();
            for x in 0..d {
                let expect: f32 = (0..l)
                    .map(|j| (scores[j] - m).exp() / denom * bf(v_data[(kv_head * l + j) * d + x]))
                    .sum();
                let got = out[head * d + x];
                assert!(
                    (got - expect).abs() < 0.02,
                    "head {head} dim {x}: got {got}, expected {expect}"
                );
            }
        }
    }

    // Perf contract: the batched decode gemv must beat mul_mm (the alternative
    // M>1 path) and its per-token cost should fall as M grows.
    #[test]
    fn gguf_decode_batch_scaling() {
        let Ok(dev) = Device::new_metal(0) else {
            return;
        };
        let (n, k) = (2560usize, 2560usize); // Qwen3-4B hidden, square (q/o/gate-ish)
        let iters = 200;
        let quants = [
            (GgmlDType::Q4_0, GgufFastQuant::Q4_0, "Q4_0"),
            (GgmlDType::Q5_0, GgufFastQuant::Q5_0, "Q5_0"),
            (GgmlDType::Q8_0, GgufFastQuant::Q8_0, "Q8_0"),
            (GgmlDType::Q2K, GgufFastQuant::Q2K, "Q2_K"),
            (GgmlDType::Q3K, GgufFastQuant::Q3K, "Q3_K"),
            (GgmlDType::Q4K, GgufFastQuant::Q4K, "Q4_K"),
            (GgmlDType::Q5K, GgufFastQuant::Q5K, "Q5_K"),
            (GgmlDType::Q6K, GgufFastQuant::Q6K, "Q6_K"),
        ];
        for (ggml, fast, tag) in quants {
            let w = Tensor::randn(0f32, 1f32, (n, k), &Device::Cpu).unwrap();
            let qt = QTensor::quantize(&w, ggml).unwrap();
            let bytes = qt.data().unwrap().into_owned();
            let blen = bytes.len();
            let wb = Tensor::from_vec(bytes, (blen,), &dev).unwrap();

            let timeit = |x: &Tensor, f: &dyn Fn(&Tensor) -> Tensor| -> f64 {
                for _ in 0..15 {
                    let _ = f(x);
                }
                let t = Instant::now();
                let mut sink = Vec::with_capacity(iters);
                for _ in 0..iters {
                    sink.push(f(x));
                }
                sink.last().unwrap().to_device(&Device::Cpu).unwrap();
                t.elapsed().as_secs_f64() * 1e6 / iters as f64
            };
            let x1 = Tensor::zeros((1, k), DType::BF16, &dev).unwrap();
            let gemv = timeit(&x1, &|x| gguf_quant_matmul(x, &wb, k, n, fast).unwrap());
            println!(
                "{tag} [{n}x{k}]  M=1 gemv {gemv:.1}us  (batch routed M=2..={GGUF_BATCH_MAX}, vs mul_mm):"
            );
            for m in [2usize, 4, 8] {
                let x = Tensor::zeros((m, k), DType::BF16, &dev).unwrap();
                let batch = timeit(&x, &|x| gguf_quant_matmul(x, &wb, k, n, fast).unwrap());
                let mm = timeit(&x, &|x| gguf_quant_mul_mm(x, &wb, k, n, fast).unwrap());
                println!(
                    "  M={m}: batch {batch:6.1}us ({:5.1}/tok)  mul_mm {mm:6.1}us ({:5.1}/tok)  speedup {:.2}x",
                    batch / m as f64,
                    mm / m as f64,
                    mm / batch
                );
            }
        }
    }
}
