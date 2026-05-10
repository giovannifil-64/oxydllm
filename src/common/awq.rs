//! AWQ (Activation-aware Weight Quantization) 4-bit linear layers.
//!
//! On-disk layout (autoawq, GEMM kernel):
//!   - qweight  i32 [in_features, out_features / 8]   8 nibbles per int32 along out dim
//!   - qzeros   i32 [in_features / group_size, out_features / 8]
//!   - scales   f16 [in_features / group_size, out_features]
//!
//! Each int32 word in qweight packs 8 4-bit values; the nibble at bit-offset
//! (4*k) corresponds to original output column `8j + AWQ_PACK_ORDER[k]`.
//!
//! v1 strategy: dequantize at load time on CPU into a standard fp16/bf16 weight
//! tensor on the target device. Memory savings of AWQ are sacrificed for
//! simplicity; the runtime path is then identical to a regular Linear and no
//! custom Metal kernel is required. A future optimization can keep the packed
//! tensors resident and dequantize on the fly in `forward()`.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

pub const AWQ_PACK_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
pub const AWQ_PACK_FACTOR: usize = 8;

#[derive(Clone)]
pub struct AwqRawTensors {
    pub qweight: Tensor,
    pub qzeros: Tensor,
    pub scales: Tensor,
}

impl AwqRawTensors {
    pub fn in_features(&self) -> Result<usize> {
        self.qweight.dim(0).context("qweight must be 2D")
    }

    pub fn out_features(&self) -> Result<usize> {
        self.scales.dim(1).context("scales must be 2D")
    }

    pub fn group_size(&self) -> Result<usize> {
        let in_features = self.in_features()?;
        let groups = self.scales.dim(0).context("scales must be 2D")?;
        if groups == 0 {
            anyhow::bail!("AWQ scales has zero groups");
        }
        if in_features % groups != 0 {
            anyhow::bail!(
                "AWQ in_features ({in_features}) not divisible by scales groups ({groups})"
            );
        }
        Ok(in_features / groups)
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

/// Concatenate AWQ tensor triplets along the output-features dimension.
///
/// All parts must share `in_features` and `group_size`. Each part's
/// `out_features` must be divisible by 8 (the AWQ pack factor) so that the
/// packed dim concatenation aligns to nibble boundaries — this holds for any
/// real model where each projection's output is a multiple of 8.
///
/// After fusion the resulting tensor describes a single linear layer whose
/// output is the concatenation of the inputs, equivalent to running each
/// `dequantize_awq` separately and then `Tensor::cat(..., dim=0)`.
pub fn concat_awq_along_out(parts: &[AwqRawTensors]) -> Result<AwqRawTensors> {
    if parts.is_empty() {
        anyhow::bail!("concat_awq_along_out: no parts");
    }
    let in_features = parts[0].in_features()?;
    let group_size = parts[0].group_size()?;
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
        if p.out_features()? % AWQ_PACK_FACTOR != 0 {
            anyhow::bail!(
                "AWQ fuse: part {i} out_features {} not divisible by {AWQ_PACK_FACTOR}",
                p.out_features()?
            );
        }
    }

    // Concatenation must happen on CPU because candle's Metal backend lacks
    // copy2d for integer dtypes (I32/U32) as of 0.10.2. The fused tensors are
    // immediately consumed by `dequantize_awq` — which already routes through
    // CPU — so there is no benefit to keeping them on the GPU here.
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
        .map(|p| to_cpu(&p.qzeros, "qzeros"))
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

    Ok(AwqRawTensors {
        qweight,
        qzeros,
        scales,
    })
}

/// Dequantize AWQ tensors into a `[out_features, in_features]` weight matrix
/// in `out_dtype`, ready to drive a standard `Linear`.
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
    if raw.qzeros.dims() != [groups, packed_out] {
        anyhow::bail!(
            "AWQ qzeros shape {:?} != [{groups}, {packed_out}]",
            raw.qzeros.dims()
        );
    }
    if raw.scales.dims() != [groups, out_features] {
        anyhow::bail!(
            "AWQ scales shape {:?} != [{groups}, {out_features}]",
            raw.scales.dims()
        );
    }

    let qweight_vec = read_packed_to_u32(&raw.qweight)?;
    let qzeros_vec = read_packed_to_u32(&raw.qzeros)?;
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

    let mut weight = vec![0f32; out_features * in_features];
    for i in 0..in_features {
        let g = i / group_size;
        for j in 0..packed_out {
            let packed = qweight_vec[i * packed_out + j];
            for (k, &offset) in AWQ_PACK_ORDER.iter().enumerate() {
                let nibble = ((packed >> (4 * k as u32)) & 0xF) as i32;
                let out_idx = j * AWQ_PACK_FACTOR + offset;
                let zero = zeros[g * out_features + out_idx] as i32;
                let scale = scales_vec[g * out_features + out_idx];
                let dequant = (nibble - zero) as f32 * scale;
                weight[out_idx * in_features + i] = dequant;
            }
        }
    }

    let weight_cpu = Tensor::from_vec(weight, (out_features, in_features), &Device::Cpu)?;
    weight_cpu
        .to_device(device)?
        .to_dtype(out_dtype)
        .context("AWQ dequantized weight dtype cast")
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let raw = AwqRawTensors {
            qweight,
            qzeros,
            scales: scales_t,
        };
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
        Ok(AwqRawTensors {
            qweight,
            qzeros,
            scales: scales_t,
        })
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
        let raw = AwqRawTensors {
            qweight,
            qzeros,
            scales,
        };
        assert!(dequantize_awq(&raw, &device, DType::F32).is_err());
        Ok(())
    }
}
