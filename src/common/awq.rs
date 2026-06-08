//! Packed weight-only quantization formats: AWQ (active runtime path) and
//! GPTQ (loader path, dequantises at load to bf16).
//!
//! Both share a packed-int layout (qweight + scales + optional qzeros/g_idx),
//! but the byte arrangement differs:
//!
//! - **AWQ** (autoawq GEMM): `qweight` packs along **out_features** (8 nibbles
//!   per int32, interleaved via [`AWQ_PACK_ORDER`]). Zero-point convention:
//!   `val = (q - zero) * scale`.
//! - **GPTQ** (auto-gptq): `qweight` packs along **in_features** (pack_factor
//!   = 32/bits values per int32, **sequential**). qzeros store
//!   `zero_point - 1`; with `sym=True` the zero is fixed at `2^(bits-1)`.
//!
//! [`QuantWeight`] carries enough metadata for callers to dispatch the right
//! dequant. AWQ continues to use the resident W4A16 fused kernel on Metal;
//! GPTQ currently dequantises at load (no resident GPU kernel).
//!
//! `AwqRawTensors` remains a type alias for callers that only handle AWQ.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

pub const AWQ_PACK_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
pub const AWQ_PACK_FACTOR: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackDim {
    Out,
    In,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackOrder {
    AwqInterleaved,
    Sequential,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroPointMode {
    Signed,
    PlusOne,
    Symmetric,
}

#[derive(Clone)]
pub struct QuantWeight {
    pub bits: u32,
    pub pack_dim: PackDim,
    pub pack_order: PackOrder,
    pub zero_point: ZeroPointMode,
    pub qweight: Tensor,
    pub qzeros: Option<Tensor>,
    pub scales: Tensor,
}

pub type AwqRawTensors = QuantWeight;

impl QuantWeight {
    pub fn new_awq(bits: u32, qweight: Tensor, qzeros: Tensor, scales: Tensor) -> Self {
        debug_assert!(matches!(bits, 4 | 8), "AWQ bits must be 4 or 8, got {bits}");
        Self {
            bits,
            pack_dim: PackDim::Out,
            pack_order: PackOrder::AwqInterleaved,
            zero_point: ZeroPointMode::Signed,
            qweight,
            qzeros: Some(qzeros),
            scales,
        }
    }

    pub fn new_gptq(
        bits: u32,
        sym: bool,
        qweight: Tensor,
        qzeros: Option<Tensor>,
        scales: Tensor,
    ) -> Self {
        Self {
            bits,
            pack_dim: PackDim::In,
            pack_order: PackOrder::Sequential,
            zero_point: if sym {
                ZeroPointMode::Symmetric
            } else {
                ZeroPointMode::PlusOne
            },
            qweight,
            qzeros,
            scales,
        }
    }

    pub fn pack_factor(&self) -> usize {
        32 / self.bits as usize
    }

    pub fn in_features(&self) -> Result<usize> {
        let d0 = self.qweight.dim(0).context("qweight must be 2D")?;
        Ok(match self.pack_dim {
            PackDim::Out => d0,
            PackDim::In => d0 * self.pack_factor(),
        })
    }

    pub fn out_features(&self) -> Result<usize> {
        self.scales.dim(1).context("scales must be 2D")
    }

    pub fn group_size(&self) -> Result<usize> {
        let in_features = self.in_features()?;
        let groups = self.scales.dim(0).context("scales must be 2D")?;
        if groups == 0 {
            anyhow::bail!("quant scales has zero groups");
        }
        if in_features % groups != 0 {
            anyhow::bail!(
                "quant in_features ({in_features}) not divisible by scales groups ({groups})"
            );
        }
        Ok(in_features / groups)
    }

    #[cfg(feature = "metal")]
    pub fn runtime_size_bytes(&self) -> usize {
        let mut acc = self.qweight.dtype().size_in_bytes() * self.qweight.elem_count()
            + self.scales.dtype().size_in_bytes() * self.scales.elem_count();
        if let Some(qz) = &self.qzeros {
            acc += qz.dtype().size_in_bytes() * qz.elem_count();
        }
        acc
    }

    #[cfg(feature = "metal")]
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            bits: self.bits,
            pack_dim: self.pack_dim,
            pack_order: self.pack_order,
            zero_point: self.zero_point,
            qweight: self
                .qweight
                .to_device(device)
                .context("quant qweight → device")?,
            qzeros: self
                .qzeros
                .as_ref()
                .map(|t| t.to_device(device).context("quant qzeros → device"))
                .transpose()?,
            scales: self
                .scales
                .to_device(device)
                .context("quant scales → device")?,
        })
    }
}

fn read_packed_to_u32(t: &Tensor) -> Result<Vec<u32>> {
    let cpu = t
        .to_device(&Device::Cpu)
        .context("AWQ packed tensor → CPU")?;
    let flat = cpu.flatten_all().context("flatten AWQ packed tensor")?;
    match flat.dtype() {
        DType::I32 => {
            let v: Vec<i32> = flat.to_vec1().context("AWQ packed tensor to_vec1::<i32>")?;
            Ok(v.into_iter().map(|x| x as u32).collect())
        }
        DType::U32 => flat
            .to_vec1::<u32>()
            .context("AWQ packed tensor to_vec1::<u32>"),
        DType::I64 => {
            let v: Vec<i64> = flat.to_vec1().context("AWQ packed tensor to_vec1::<i64>")?;
            Ok(v.into_iter().map(|x| x as u32).collect())
        }
        other => {
            anyhow::bail!("Unsupported AWQ packed tensor dtype {other:?} (expected I32/U32/I64)")
        }
    }
}

pub fn concat_awq_along_out(parts: &[AwqRawTensors]) -> Result<AwqRawTensors> {
    if parts.is_empty() {
        anyhow::bail!("concat_awq_along_out: no parts");
    }
    let in_features = parts[0].in_features()?;
    let group_size = parts[0].group_size()?;
    let bits = parts[0].bits;
    let pack_factor = parts[0].pack_factor();
    for (i, p) in parts.iter().enumerate().skip(1) {
        if p.in_features()? != in_features {
            anyhow::bail!(
                "AWQ fuse: in_features mismatch at part {i}: expected {in_features}, got {}",
                p.in_features()?
            );
        }
        if p.group_size()? != group_size {
            anyhow::bail!(
                "AWQ fuse: group_size mismatch at part {i}: expected {group_size}, got {}",
                p.group_size()?
            );
        }
        if p.bits != bits {
            anyhow::bail!(
                "AWQ fuse: bits mismatch at part {i}: expected {bits}, got {}",
                p.bits
            );
        }
        if p.out_features()? % pack_factor != 0 {
            anyhow::bail!(
                "AWQ fuse: part {i} out_features {} not divisible by pack_factor {pack_factor}",
                p.out_features()?
            );
        }
    }

    // Concatenation must happen on CPU: candle Metal lacks copy2d for I32/U32.
    let cpu = Device::Cpu;
    let to_cpu = |t: &Tensor, name: &str| -> Result<Tensor> {
        t.to_device(&cpu)
            .with_context(|| format!("AWQ fuse: move {name} to CPU"))
    };

    let qweight_cpu: Vec<Tensor> = parts
        .iter()
        .map(|p| to_cpu(&p.qweight, "qweight"))
        .collect::<Result<_>>()?;
    let qzeros_cpu: Vec<Tensor> = parts
        .iter()
        .map(|p| match &p.qzeros {
            Some(t) => to_cpu(t, "qzeros"),
            None => anyhow::bail!("AWQ fuse: qzeros missing on a part — not an AWQ tensor"),
        })
        .collect::<Result<_>>()?;
    let scales_cpu: Vec<Tensor> = parts
        .iter()
        .map(|p| to_cpu(&p.scales, "scales"))
        .collect::<Result<_>>()?;

    let qweight_refs: Vec<&Tensor> = qweight_cpu.iter().collect();
    let qzeros_refs: Vec<&Tensor> = qzeros_cpu.iter().collect();
    let scales_refs: Vec<&Tensor> = scales_cpu.iter().collect();

    let qweight = Tensor::cat(&qweight_refs, 1).context("AWQ fuse: cat qweight")?;
    let qzeros = Tensor::cat(&qzeros_refs, 1).context("AWQ fuse: cat qzeros")?;
    let scales = Tensor::cat(&scales_refs, 1).context("AWQ fuse: cat scales")?;

    Ok(QuantWeight::new_awq(parts[0].bits, qweight, qzeros, scales))
}

pub fn dequantize_awq(raw: &AwqRawTensors, device: &Device, out_dtype: DType) -> Result<Tensor> {
    let in_features = raw.in_features()?;
    let out_features = raw.out_features()?;
    let group_size = raw.group_size()?;
    let groups = in_features / group_size;
    let packed_out = out_features / AWQ_PACK_FACTOR;

    if out_features % AWQ_PACK_FACTOR != 0 {
        anyhow::bail!(
            "AWQ out_features ({out_features}) not divisible by pack factor {AWQ_PACK_FACTOR}"
        );
    }
    if raw.qweight.dim(1)? != packed_out {
        anyhow::bail!(
            "AWQ qweight dim1 ({}) != out_features/8 ({packed_out})",
            raw.qweight.dim(1)?
        );
    }
    let qzeros_tensor = raw
        .qzeros
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("AWQ dequantize: qzeros tensor missing"))?;
    if qzeros_tensor.dims() != [groups, packed_out] {
        anyhow::bail!(
            "AWQ qzeros shape {:?} != [{groups}, {packed_out}]",
            qzeros_tensor.dims()
        );
    }
    if raw.scales.dims() != [groups, out_features] {
        anyhow::bail!(
            "AWQ scales shape {:?} != [{groups}, {out_features}]",
            raw.scales.dims()
        );
    }

    let qweight_vec = read_packed_to_u32(&raw.qweight)?;
    let qzeros_vec = read_packed_to_u32(qzeros_tensor)?;
    let scales_vec: Vec<f32> = raw
        .scales
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;

    if qweight_vec.len() != in_features * packed_out {
        anyhow::bail!(
            "qweight numel {} != in*packed_out {}",
            qweight_vec.len(),
            in_features * packed_out
        );
    }
    if qzeros_vec.len() != groups * packed_out {
        anyhow::bail!(
            "qzeros numel {} != groups*packed_out {}",
            qzeros_vec.len(),
            groups * packed_out
        );
    }
    if scales_vec.len() != groups * out_features {
        anyhow::bail!(
            "scales numel {} != groups*out_features {}",
            scales_vec.len(),
            groups * out_features
        );
    }

    let mut zeros = vec![0u8; groups * out_features];
    for g in 0..groups {
        for j in 0..packed_out {
            let packed = qzeros_vec[g * packed_out + j];
            for (k, &offset) in AWQ_PACK_ORDER.iter().enumerate() {
                let nibble = ((packed >> (4 * k as u32)) & 0xF) as u8;
                let out_idx = j * AWQ_PACK_FACTOR + offset;
                zeros[g * out_features + out_idx] = nibble;
            }
        }
    }

    let mut inv_pack_order = [0u32; AWQ_PACK_FACTOR];
    for (k, &offset) in AWQ_PACK_ORDER.iter().enumerate() {
        inv_pack_order[offset] = k as u32;
    }

    let plan = AwqDequantPlan {
        in_features,
        out_features,
        packed_out,
        group_size,
        qweight: &qweight_vec,
        zeros: &zeros,
        scales: &scales_vec,
        inv_pack_order,
    };
    let weight_cpu = match out_dtype {
        DType::F32 => {
            let mut w = vec![0f32; out_features * in_features];
            fill_awq_weight(&mut w, &plan, |x| x);
            Tensor::from_vec(w, (out_features, in_features), &Device::Cpu)?
        }
        DType::BF16 => {
            let mut w = vec![half::bf16::ZERO; out_features * in_features];
            fill_awq_weight(&mut w, &plan, half::bf16::from_f32);
            Tensor::from_vec(w, (out_features, in_features), &Device::Cpu)?
        }
        DType::F16 => {
            let mut w = vec![half::f16::ZERO; out_features * in_features];
            fill_awq_weight(&mut w, &plan, half::f16::from_f32);
            Tensor::from_vec(w, (out_features, in_features), &Device::Cpu)?
        }
        other => anyhow::bail!("AWQ dequantize: unsupported out_dtype {other:?}"),
    };

    weight_cpu
        .to_device(device)
        .context("AWQ dequantized weight → device")
}

pub fn dequantize_gptq(raw: &QuantWeight, device: &Device, out_dtype: DType) -> Result<Tensor> {
    if raw.pack_dim != PackDim::In || raw.pack_order != PackOrder::Sequential {
        anyhow::bail!(
            "dequantize_gptq: expected PackDim::In + Sequential, got {:?}/{:?}",
            raw.pack_dim,
            raw.pack_order
        );
    }
    let bits = raw.bits as usize;
    if bits != 4 && bits != 8 {
        anyhow::bail!("dequantize_gptq: only 4/8-bit supported, got {bits}");
    }
    let pack_factor = 32 / bits;
    let in_features = raw.in_features()?;
    let out_features = raw.out_features()?;
    let group_size = raw.group_size()?;
    let groups = in_features / group_size;
    let packed_in = in_features / pack_factor;
    let packed_out = out_features / pack_factor;
    let mask: u32 = (1u32 << bits) - 1;

    if raw.qweight.dims() != [packed_in, out_features] {
        anyhow::bail!(
            "GPTQ qweight shape {:?} != [{packed_in}, {out_features}]",
            raw.qweight.dims()
        );
    }
    if raw.scales.dims() != [groups, out_features] {
        anyhow::bail!(
            "GPTQ scales shape {:?} != [{groups}, {out_features}]",
            raw.scales.dims()
        );
    }
    if out_features % pack_factor != 0 {
        anyhow::bail!(
            "GPTQ out_features {out_features} not divisible by pack_factor {pack_factor}"
        );
    }

    let qweight_vec = read_packed_to_u32(&raw.qweight)?;
    if qweight_vec.len() != packed_in * out_features {
        anyhow::bail!(
            "GPTQ qweight numel {} != {packed_in}*{out_features}",
            qweight_vec.len()
        );
    }

    let qzeros_vec = match &raw.qzeros {
        Some(t) => {
            if t.dims() != [groups, packed_out] {
                anyhow::bail!(
                    "GPTQ qzeros shape {:?} != [{groups}, {packed_out}]",
                    t.dims()
                );
            }
            read_packed_to_u32(t)?
        }
        None => vec![],
    };

    let scales_vec: Vec<f32> = raw
        .scales
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;

    let plan = GptqDequantPlan {
        in_features,
        out_features,
        group_size,
        packed_out,
        pack_factor,
        bits,
        mask,
        sym: raw.qzeros.is_none() || raw.zero_point == ZeroPointMode::Symmetric,
        qweight: &qweight_vec,
        qzeros: &qzeros_vec,
        scales: &scales_vec,
    };

    let weight_cpu = match out_dtype {
        DType::F32 => {
            let mut w = vec![0f32; out_features * in_features];
            fill_gptq_weight(&mut w, &plan, |x| x);
            Tensor::from_vec(w, (out_features, in_features), &Device::Cpu)?
        }
        DType::BF16 => {
            let mut w = vec![half::bf16::ZERO; out_features * in_features];
            fill_gptq_weight(&mut w, &plan, half::bf16::from_f32);
            Tensor::from_vec(w, (out_features, in_features), &Device::Cpu)?
        }
        DType::F16 => {
            let mut w = vec![half::f16::ZERO; out_features * in_features];
            fill_gptq_weight(&mut w, &plan, half::f16::from_f32);
            Tensor::from_vec(w, (out_features, in_features), &Device::Cpu)?
        }
        other => anyhow::bail!("GPTQ dequantize: unsupported out_dtype {other:?}"),
    };

    weight_cpu
        .to_device(device)
        .context("GPTQ dequantized weight → device")
}

struct GptqDequantPlan<'a> {
    in_features: usize,
    out_features: usize,
    group_size: usize,
    packed_out: usize,
    pack_factor: usize,
    bits: usize,
    mask: u32,
    sym: bool,
    qweight: &'a [u32],
    qzeros: &'a [u32],
    scales: &'a [f32],
}

fn fill_gptq_weight<T: candle_core::WithDType + Send>(
    out: &mut [T],
    plan: &GptqDequantPlan,
    convert: impl Fn(f32) -> T + Sync,
) {
    use rayon::prelude::*;
    let sym_zero = (1u32 << (plan.bits - 1)).saturating_sub(1);
    out.par_chunks_mut(plan.in_features)
        .enumerate()
        .for_each(|(out_idx, row)| {
            let o_word = out_idx / plan.pack_factor;
            let o_slot = out_idx % plan.pack_factor;
            let o_shift = (o_slot * plan.bits) as u32;
            for (i, slot) in row.iter_mut().enumerate() {
                let g = i / plan.group_size;
                let i_word = i / plan.pack_factor;
                let i_slot = i % plan.pack_factor;
                let i_shift = (i_slot * plan.bits) as u32;
                let q = (plan.qweight[i_word * plan.out_features + out_idx] >> i_shift) & plan.mask;
                let zero = if plan.sym {
                    sym_zero
                } else {
                    (plan.qzeros[g * plan.packed_out + o_word] >> o_shift) & plan.mask
                };
                let val = (q as i32 - (zero as i32 + 1)) as f32
                    * plan.scales[g * plan.out_features + out_idx];
                *slot = convert(val);
            }
        });
}

pub fn dequantize_quant(raw: &QuantWeight, device: &Device, out_dtype: DType) -> Result<Tensor> {
    match raw.pack_dim {
        PackDim::Out => dequantize_awq(raw, device, out_dtype),
        PackDim::In => dequantize_gptq(raw, device, out_dtype),
    }
}

struct AwqDequantPlan<'a> {
    in_features: usize,
    out_features: usize,
    packed_out: usize,
    group_size: usize,
    qweight: &'a [u32],
    zeros: &'a [u8],
    scales: &'a [f32],
    inv_pack_order: [u32; AWQ_PACK_FACTOR],
}

// Parallelise over output rows: per-row writes stay sequential and don't
// thrash the memory controller (an earlier per-column variant did, badly).
fn fill_awq_weight<T: candle_core::WithDType + Send>(
    out: &mut [T],
    plan: &AwqDequantPlan,
    convert: impl Fn(f32) -> T + Sync,
) {
    use rayon::prelude::*;
    out.par_chunks_mut(plan.in_features)
        .enumerate()
        .for_each(|(out_idx, row)| {
            let j = out_idx / AWQ_PACK_FACTOR;
            let offset = out_idx % AWQ_PACK_FACTOR;
            let shift = 4 * plan.inv_pack_order[offset];
            for (i, slot) in row.iter_mut().enumerate() {
                let g = i / plan.group_size;
                let packed = plan.qweight[i * plan.packed_out + j];
                let nibble = ((packed >> shift) & 0xF) as i32;
                let zero = plan.zeros[g * plan.out_features + out_idx] as i32;
                let scale = plan.scales[g * plan.out_features + out_idx];
                *slot = convert((nibble - zero) as f32 * scale);
            }
        });
}

// RTN 4-bit group-wise AWQ quantization (autoawq GEMM-compatible).
#[cfg(any(feature = "metal", test))]
pub fn rtn_quantize_awq(weight: &Tensor, group_size: usize) -> Result<AwqRawTensors> {
    use rayon::prelude::*;

    let (out_features, in_features) = weight
        .dims2()
        .context("rtn_quantize_awq: weight must be 2-D [out, in]")?;
    if out_features % AWQ_PACK_FACTOR != 0 {
        anyhow::bail!(
            "rtn_quantize_awq: out_features ({out_features}) must be a multiple of {AWQ_PACK_FACTOR}"
        );
    }
    if group_size == 0 || in_features % group_size != 0 {
        anyhow::bail!(
            "rtn_quantize_awq: in_features ({in_features}) not divisible by group_size ({group_size})"
        );
    }
    let groups = in_features / group_size;
    let packed_out = out_features / AWQ_PACK_FACTOR;

    let w: Vec<f32> = weight
        .to_device(&Device::Cpu)
        .context("rtn_quantize_awq: weight → CPU")?
        .to_dtype(DType::F32)
        .context("rtn_quantize_awq: cast weight to F32")?
        .flatten_all()?
        .to_vec1()?;
    if w.len() != out_features * in_features {
        anyhow::bail!("rtn_quantize_awq: weight numel mismatch");
    }

    // Pass 1: per-row group quantization. qnib/znib/scl are independent per row.
    let mut qnib = vec![0u8; out_features * in_features];
    let mut znib = vec![0u8; out_features * groups];
    let mut scl = vec![0f32; out_features * groups];
    qnib.par_chunks_mut(in_features)
        .zip(znib.par_chunks_mut(groups))
        .zip(scl.par_chunks_mut(groups))
        .enumerate()
        .for_each(|(o, ((q_row, z_row), s_row))| {
            let w_row = &w[o * in_features..(o + 1) * in_features];
            for g in 0..groups {
                let grp = &w_row[g * group_size..(g + 1) * group_size];
                let mut xmin = grp[0];
                let mut xmax = grp[0];
                for &v in grp {
                    xmin = xmin.min(v);
                    xmax = xmax.max(v);
                }
                // Extend group range to include 0: AWQ's `(q - zero) * scale` with
                // q, zero ∈ [0, 15] always spans 0, so groups that don't straddle
                // 0 would otherwise clamp the zero-point and lose reconstruction.
                let xmin = xmin.min(0.0);
                let xmax = xmax.max(0.0);
                if xmax - xmin < 1e-12 {
                    s_row[g] = xmin;
                    z_row[g] = 0;
                    for q in &mut q_row[g * group_size..(g + 1) * group_size] {
                        *q = 1;
                    }
                } else {
                    let scale = (xmax - xmin) / 15.0;
                    let zero = (-xmin / scale).round().clamp(0.0, 15.0);
                    s_row[g] = scale;
                    z_row[g] = zero as u8;
                    for (idx, &v) in grp.iter().enumerate() {
                        q_row[g * group_size + idx] =
                            (v / scale + zero).round().clamp(0.0, 15.0) as u8;
                    }
                }
            }
        });
    drop(w);

    // Pass 2: pack into AWQ words (out-major buffers, then transposed to AWQ layout).
    let mut qw_om = vec![0i32; packed_out * in_features];
    let mut qz_om = vec![0i32; packed_out * groups];
    qw_om
        .par_chunks_mut(in_features)
        .zip(qz_om.par_chunks_mut(groups))
        .enumerate()
        .for_each(|(j, (qw_col, qz_col))| {
            for (k, &off) in AWQ_PACK_ORDER.iter().enumerate() {
                let o = j * AWQ_PACK_FACTOR + off;
                let shift = 4 * k as u32;
                for (i, qw) in qw_col.iter_mut().enumerate() {
                    *qw |= (qnib[o * in_features + i] as i32) << shift;
                }
                for (g, qz) in qz_col.iter_mut().enumerate() {
                    *qz |= (znib[o * groups + g] as i32) << shift;
                }
            }
        });

    let cpu = Device::Cpu;
    let qweight = Tensor::from_vec(qw_om, (packed_out, in_features), &cpu)?
        .t()?
        .contiguous()?;
    let qzeros = Tensor::from_vec(qz_om, (packed_out, groups), &cpu)?
        .t()?
        .contiguous()?;
    let scales = Tensor::from_vec(scl, (out_features, groups), &cpu)?
        .t()?
        .contiguous()?;

    Ok(QuantWeight::new_awq(4, qweight, qzeros, scales))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtn_quantize_awq_round_trips() -> Result<()> {
        let device = Device::Cpu;
        let (out_f, in_f, group) = (16usize, 64usize, 32usize);
        let data: Vec<f32> = (0..out_f * in_f)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let weight = Tensor::from_vec(data.clone(), (out_f, in_f), &device)?;

        let raw = rtn_quantize_awq(&weight, group)?;
        assert_eq!(raw.qweight.dims(), [in_f, out_f / 8]);
        assert_eq!(
            raw.qzeros.as_ref().expect("AWQ qzeros").dims(),
            [in_f / group, out_f / 8]
        );
        assert_eq!(raw.scales.dims(), [in_f / group, out_f]);

        let dq: Vec<f32> = dequantize_awq(&raw, &device, DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(dq.len(), data.len());
        let mut sq_err = 0f64;
        let mut sq = 0f64;
        for (q, w) in dq.iter().zip(data.iter()) {
            sq_err += ((q - w) as f64).powi(2);
            sq += (*w as f64).powi(2);
        }
        let rel = (sq_err / sq).sqrt();
        assert!(
            rel < 0.05,
            "RTN 4-bit relative error {rel} — packing likely wrong"
        );
        Ok(())
    }

    #[test]
    #[ignore = "A quality measurement — needs the Qwen3-4B-AWQ model on disk"]
    fn rtn_lm_head_quality_measurement() {
        let dir = std::path::Path::new("/Users/giovanni/.oxydllm/models/Qwen/Qwen3-4B-AWQ");
        if !dir.exists() {
            eprintln!("[rtn-measure] model dir not found — skipping");
            return;
        }
        let device = Device::Cpu;
        let paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect();
        let path_refs: Vec<&str> = paths.iter().map(|p| p.to_str().unwrap()).collect();
        let mmap =
            unsafe { candle_core::safetensors::MmapedSafetensors::multi(&path_refs).unwrap() };
        let w_ref = mmap
            .load("model.embed_tokens.weight", &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let (out_f, in_f) = w_ref.dims2().unwrap();
        eprintln!("[rtn-measure] embed_tokens [{out_f}, {in_f}]");

        let raw = rtn_quantize_awq(&w_ref, 128).unwrap();
        let w_q = dequantize_awq(&raw, &device, DType::F32).unwrap();

        let wv: Vec<f32> = w_ref.flatten_all().unwrap().to_vec1().unwrap();
        let qv: Vec<f32> = w_q.flatten_all().unwrap().to_vec1().unwrap();
        let (mut sq_err, mut sq, mut max_e) = (0f64, 0f64, 0f32);
        for (a, b) in wv.iter().zip(qv.iter()) {
            sq_err += ((a - b) as f64).powi(2);
            sq += (*a as f64).powi(2);
            max_e = max_e.max((a - b).abs());
        }
        eprintln!(
            "[rtn-measure] weight: rel-L2 {:.4}  max-abs {:.5}",
            (sq_err / sq).sqrt(),
            max_e
        );

        let n = 256usize;
        let xd: Vec<f32> = (0..n * in_f)
            .map(|i| {
                let h = (i as u64).wrapping_mul(2654435761) >> 8;
                (h & 0xffff) as f32 / 32768.0 - 1.0
            })
            .collect();
        let x = Tensor::from_vec(xd, (n, in_f), &device).unwrap();
        let rms = x.sqr().unwrap().mean_keepdim(1).unwrap().sqrt().unwrap();
        let x = x.broadcast_div(&rms).unwrap();

        let l_ref = x.matmul(&w_ref.t().unwrap().contiguous().unwrap()).unwrap();
        let l_q = x.matmul(&w_q.t().unwrap().contiguous().unwrap()).unwrap();
        let agree = l_ref
            .argmax(candle_core::D::Minus1)
            .unwrap()
            .eq(&l_q.argmax(candle_core::D::Minus1).unwrap())
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let max_dl = (l_ref - l_q)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!(
            "[rtn-measure] logits (n={n}): top-1 agreement {:.1}%  max-abs-err {:.4}",
            agree / n as f32 * 100.0,
            max_dl
        );
    }

    fn pack_awq_qweight(matrix: &[Vec<u8>], in_features: usize, out_features: usize) -> Vec<i32> {
        assert_eq!(out_features % AWQ_PACK_FACTOR, 0);
        let packed_out = out_features / AWQ_PACK_FACTOR;
        let mut out = vec![0u32; in_features * packed_out];
        for (i, row) in matrix.iter().enumerate().take(in_features) {
            for j in 0..packed_out {
                let mut word: u32 = 0;
                for (k, &offset) in AWQ_PACK_ORDER.iter().enumerate() {
                    let orig_col = j * AWQ_PACK_FACTOR + offset;
                    let nibble = row[orig_col] as u32 & 0xF;
                    word |= nibble << (4 * k as u32);
                }
                out[i * packed_out + j] = word;
            }
        }
        out.into_iter().map(|w| w as i32).collect()
    }

    fn pack_awq_qzeros(zeros: &[Vec<u8>], groups: usize, out_features: usize) -> Vec<i32> {
        let packed_out = out_features / AWQ_PACK_FACTOR;
        let mut out = vec![0u32; groups * packed_out];
        for (g, row) in zeros.iter().enumerate().take(groups) {
            for j in 0..packed_out {
                let mut word: u32 = 0;
                for (k, &offset) in AWQ_PACK_ORDER.iter().enumerate() {
                    let orig_col = j * AWQ_PACK_FACTOR + offset;
                    let nibble = row[orig_col] as u32 & 0xF;
                    word |= nibble << (4 * k as u32);
                }
                out[g * packed_out + j] = word;
            }
        }
        out.into_iter().map(|w| w as i32).collect()
    }

    #[test]
    fn dequantize_awq_recovers_original_matrix() -> Result<()> {
        let device = Device::Cpu;
        let in_features = 8;
        let out_features = 16;
        let group_size = 4;
        let groups = in_features / group_size;

        let mut iweight: Vec<Vec<u8>> = Vec::with_capacity(in_features);

        for i in 0..in_features {
            let mut row = Vec::with_capacity(out_features);
            for j in 0..out_features {
                row.push(((i * 3 + j * 5 + 1) & 0xF) as u8);
            }
            iweight.push(row);
        }
        let mut izero: Vec<Vec<u8>> = Vec::with_capacity(groups);
        for g in 0..groups {
            let mut row = Vec::with_capacity(out_features);
            for j in 0..out_features {
                row.push(((g * 7 + j) & 0xF) as u8);
            }
            izero.push(row);
        }
        let mut scales: Vec<f32> = Vec::with_capacity(groups * out_features);
        for g in 0..groups {
            for j in 0..out_features {
                scales.push(0.01 * (g + 1) as f32 + 0.001 * (j + 1) as f32);
            }
        }

        let mut expected = vec![0f32; out_features * in_features];
        for i in 0..in_features {
            let g = i / group_size;
            for j in 0..out_features {
                let w = iweight[i][j] as i32 - izero[g][j] as i32;
                let s = scales[g * out_features + j];
                expected[j * in_features + i] = w as f32 * s;
            }
        }

        let qweight_data = pack_awq_qweight(&iweight, in_features, out_features);
        let qzeros_data = pack_awq_qzeros(&izero, groups, out_features);
        let packed_out = out_features / AWQ_PACK_FACTOR;
        let qweight = Tensor::from_vec(qweight_data, (in_features, packed_out), &device)?;
        let qzeros = Tensor::from_vec(qzeros_data, (groups, packed_out), &device)?;
        let scales_t = Tensor::from_vec(scales, (groups, out_features), &device)?;

        let raw = QuantWeight::new_awq(4, qweight, qzeros, scales_t);
        let dequant = dequantize_awq(&raw, &device, DType::F32)?;
        assert_eq!(dequant.dims(), [out_features, in_features]);

        let got: Vec<f32> = dequant.flatten_all()?.to_vec1()?;
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got={g} expected={e}");
        }
        Ok(())
    }

    fn build_awq_triplet(
        iweight: &[Vec<u8>],
        izero: &[Vec<u8>],
        scales: &[f32],
        in_features: usize,
        out_features: usize,
        groups: usize,
        device: &Device,
    ) -> Result<AwqRawTensors> {
        let qweight_data = pack_awq_qweight(iweight, in_features, out_features);
        let qzeros_data = pack_awq_qzeros(izero, groups, out_features);
        let packed_out = out_features / AWQ_PACK_FACTOR;
        let qweight = Tensor::from_vec(qweight_data, (in_features, packed_out), device)?;
        let qzeros = Tensor::from_vec(qzeros_data, (groups, packed_out), device)?;
        let scales_t = Tensor::from_vec(scales.to_vec(), (groups, out_features), device)?;
        Ok(QuantWeight::new_awq(4, qweight, qzeros, scales_t))
    }

    #[test]
    fn concat_awq_along_out_matches_separate_dequant() -> Result<()> {
        let device = Device::Cpu;
        let in_features = 8;
        let group_size = 4;
        let groups = in_features / group_size;
        let parts_out = [16usize, 8, 8];

        let mut parts: Vec<AwqRawTensors> = Vec::new();
        let mut separate_dequants: Vec<Tensor> = Vec::new();

        for (idx, &out_features) in parts_out.iter().enumerate() {
            let iweight: Vec<Vec<u8>> = (0..in_features)
                .map(|i| {
                    (0..out_features)
                        .map(|j| ((i + j + idx * 3) & 0xF) as u8)
                        .collect()
                })
                .collect();
            let izero: Vec<Vec<u8>> = (0..groups)
                .map(|g| {
                    (0..out_features)
                        .map(|j| ((g * 2 + j + idx) & 0xF) as u8)
                        .collect()
                })
                .collect();
            let scales: Vec<f32> = (0..groups)
                .flat_map(|g| {
                    (0..out_features).map(move |j| {
                        0.02 * (g + 1) as f32 + 0.005 * (j + 1) as f32 + 0.001 * idx as f32
                    })
                })
                .collect();

            let raw = build_awq_triplet(
                &iweight,
                &izero,
                &scales,
                in_features,
                out_features,
                groups,
                &device,
            )?;
            separate_dequants.push(dequantize_awq(&raw, &device, DType::F32)?);
            parts.push(raw);
        }

        let fused = concat_awq_along_out(&parts)?;
        let total_out: usize = parts_out.iter().sum();
        assert_eq!(fused.scales.dims(), [groups, total_out]);

        let fused_w = dequantize_awq(&fused, &device, DType::F32)?;
        let separate_refs: Vec<&Tensor> = separate_dequants.iter().collect();
        let separate_w = Tensor::cat(&separate_refs, 0)?;

        let a: Vec<f32> = fused_w.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = separate_w.flatten_all()?.to_vec1()?;
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "fused={x} separate={y}");
        }
        Ok(())
    }

    #[test]
    fn concat_awq_rejects_in_features_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let mk = |in_f: usize| -> Result<AwqRawTensors> {
            let out = 8usize;
            let g = 1usize;
            let iw: Vec<Vec<u8>> = vec![vec![0u8; out]; in_f];
            let iz: Vec<Vec<u8>> = vec![vec![0u8; out]; g];
            let s: Vec<f32> = vec![0.0; g * out];
            build_awq_triplet(&iw, &iz, &s, in_f, out, g, &device)
        };
        let a = mk(4)?;
        let b = mk(8)?;
        assert!(concat_awq_along_out(&[a, b]).is_err());
        Ok(())
    }

    #[test]
    fn dequantize_awq_rejects_wrong_shapes() -> Result<()> {
        let device = Device::Cpu;
        let qweight = Tensor::zeros((8, 2), DType::I32, &device)?;
        let qzeros = Tensor::zeros((2, 2), DType::I32, &device)?;
        let scales = Tensor::zeros((2, 8), DType::F32, &device)?;
        let raw = QuantWeight::new_awq(4, qweight, qzeros, scales);
        assert!(dequantize_awq(&raw, &device, DType::F32).is_err());
        Ok(())
    }

    fn pack_gptq_qweight(
        iweight: &[Vec<u8>],
        in_features: usize,
        out_features: usize,
        bits: u32,
    ) -> Vec<i32> {
        let pf = 32 / bits as usize;
        let packed_in = in_features / pf;
        let mut out = vec![0i32; packed_in * out_features];
        for i_word in 0..packed_in {
            for j in 0..out_features {
                let mut word: u32 = 0;
                for k in 0..pf {
                    let i = i_word * pf + k;
                    let v = iweight[i][j] as u32;
                    word |= (v & ((1 << bits) - 1)) << (bits as u32 * k as u32);
                }
                out[i_word * out_features + j] = word as i32;
            }
        }
        out
    }

    fn pack_gptq_qzeros(
        izero: &[Vec<u8>],
        groups: usize,
        out_features: usize,
        bits: u32,
    ) -> Vec<i32> {
        let pf = 32 / bits as usize;
        let packed_out = out_features / pf;
        let mut out = vec![0i32; groups * packed_out];
        for g in 0..groups {
            for o_word in 0..packed_out {
                let mut word: u32 = 0;
                for k in 0..pf {
                    let o = o_word * pf + k;
                    let v = izero[g][o] as u32;
                    word |= (v & ((1 << bits) - 1)) << (bits as u32 * k as u32);
                }
                out[g * packed_out + o_word] = word as i32;
            }
        }
        out
    }

    #[test]
    fn dequantize_gptq_int8_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let (in_features, out_features, group_size, bits) = (32usize, 16usize, 16usize, 8u32);
        let groups = in_features / group_size;
        let pf = 32 / bits as usize;

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
                    .map(|j| ((g * 31 + j * 17 + 1) & 0xFF) as u8)
                    .collect()
            })
            .collect();
        let scales: Vec<f32> = (0..groups)
            .flat_map(|g| {
                (0..out_features).map(move |j| 0.02 * (g + 1) as f32 + 0.003 * (j + 1) as f32)
            })
            .collect();

        let mut expected = vec![0f32; out_features * in_features];
        for i in 0..in_features {
            let g = i / group_size;
            for j in 0..out_features {
                let q = iweight[i][j] as i32;
                let z = izero[g][j] as i32 + 1;
                let s = scales[g * out_features + j];
                expected[j * in_features + i] = (q - z) as f32 * s;
            }
        }

        let qw_data = pack_gptq_qweight(&iweight, in_features, out_features, bits);
        let qz_data = pack_gptq_qzeros(&izero, groups, out_features, bits);
        let packed_in = in_features / pf;
        let packed_out = out_features / pf;
        let qweight = Tensor::from_vec(qw_data, (packed_in, out_features), &device)?;
        let qzeros = Tensor::from_vec(qz_data, (groups, packed_out), &device)?;
        let scales_t = Tensor::from_vec(scales, (groups, out_features), &device)?;

        let raw = QuantWeight::new_gptq(bits, false, qweight, Some(qzeros), scales_t);
        let dq = dequantize_gptq(&raw, &device, DType::F32)?;
        assert_eq!(dq.dims(), [out_features, in_features]);

        let got: Vec<f32> = dq.flatten_all()?.to_vec1()?;
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "GPTQ-Int8 dequant: got={g} expected={e}"
            );
        }
        Ok(())
    }

    #[test]
    fn dequantize_gptq_int4_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let (in_features, out_features, group_size, bits) = (32usize, 16usize, 16usize, 4u32);
        let groups = in_features / group_size;
        let pf = 32 / bits as usize;

        let iweight: Vec<Vec<u8>> = (0..in_features)
            .map(|i| {
                (0..out_features)
                    .map(|j| ((i * 3 + j * 5) & 0xF) as u8)
                    .collect()
            })
            .collect();
        let izero: Vec<Vec<u8>> = (0..groups)
            .map(|g| {
                (0..out_features)
                    .map(|j| ((g * 2 + j) & 0xF) as u8)
                    .collect()
            })
            .collect();
        let scales: Vec<f32> = (0..groups)
            .flat_map(|g| {
                (0..out_features).map(move |j| 0.01 * (g + 1) as f32 + 0.002 * (j + 1) as f32)
            })
            .collect();

        let mut expected = vec![0f32; out_features * in_features];
        for i in 0..in_features {
            let g = i / group_size;
            for j in 0..out_features {
                let q = iweight[i][j] as i32;
                let z = izero[g][j] as i32 + 1;
                let s = scales[g * out_features + j];
                expected[j * in_features + i] = (q - z) as f32 * s;
            }
        }

        let qw_data = pack_gptq_qweight(&iweight, in_features, out_features, bits);
        let qz_data = pack_gptq_qzeros(&izero, groups, out_features, bits);
        let qweight = Tensor::from_vec(qw_data, (in_features / pf, out_features), &device)?;
        let qzeros = Tensor::from_vec(qz_data, (groups, out_features / pf), &device)?;
        let scales_t = Tensor::from_vec(scales, (groups, out_features), &device)?;

        let raw = QuantWeight::new_gptq(bits, false, qweight, Some(qzeros), scales_t);
        let dq = dequantize_gptq(&raw, &device, DType::F32)?;
        let got: Vec<f32> = dq.flatten_all()?.to_vec1()?;
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "GPTQ-Int4 dequant: got={g} expected={e}"
            );
        }
        Ok(())
    }
}
