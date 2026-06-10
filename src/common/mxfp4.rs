//! MXFP4 (OCP Microscaling FP4) weights — GPT-OSS expert format.
//!
//! On-disk layout (per weight matrix, row-major over output rows):
//! - `blocks`: u8 `[rows, K/32, 16]` — 32 FP4 (E2M1) values per block, two per
//!   byte, low nibble first.
//! - `scales`: u8 `[rows, K/32]` — one E8M0 exponent per block: `2^(s - 127)`.
//!
//! Weights stay MXFP4-resident on Metal (dequantizing GPT-OSS-20B's experts to
//! BF16 would need ~38 GB); matmuls run fused dequant kernels.

use candle_core::{DType, Result, Tensor};

pub const MXFP4_BLOCK: usize = 32;

/// E2M1 magnitudes; full code is `(-1)^bit3 × MXFP4_LUT[code & 7]`.
const MXFP4_LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[inline]
fn mxfp4_decode(nibble: u8) -> f32 {
    let v = MXFP4_LUT[(nibble & 0b0111) as usize];
    if nibble & 0b1000 != 0 { -v } else { v }
}

#[inline]
fn e8m0_scale(scale_byte: u8) -> f32 {
    // 0xFF is the E8M0 NaN encoding per the OCP spec.
    if scale_byte == 0xFF {
        return f32::NAN;
    }
    (scale_byte as f32 - 127.0).exp2()
}

/// CPU reference dequantization to row-major `[rows, cols]` f32.
/// Ground truth for the Metal kernel parity tests.
pub fn dequantize_mxfp4_f32(
    blocks: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    if !cols.is_multiple_of(MXFP4_BLOCK) {
        candle_core::bail!("MXFP4: cols {cols} must be a multiple of {MXFP4_BLOCK}");
    }
    let nb = cols / MXFP4_BLOCK;
    if blocks.len() != rows * nb * (MXFP4_BLOCK / 2) || scales.len() != rows * nb {
        candle_core::bail!(
            "MXFP4: buffer sizes (blocks {}, scales {}) don't match [{rows}, {cols}]",
            blocks.len(),
            scales.len()
        );
    }
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for b in 0..nb {
            let scale = e8m0_scale(scales[r * nb + b]);
            let bytes = &blocks[(r * nb + b) * 16..(r * nb + b) * 16 + 16];
            for byte in bytes {
                out.push(mxfp4_decode(byte & 0xF) * scale);
                out.push(mxfp4_decode(byte >> 4) * scale);
            }
        }
    }
    Ok(out)
}

/// MXFP4-resident linear layer: `y = x @ W^T + bias` with W kept packed.
pub struct Mxfp4Linear {
    /// Flattened `[out_features × in_features/2]` u8 (16 bytes per block).
    blocks: Tensor,
    /// Flattened `[out_features × in_features/32]` u8 (E8M0).
    scales: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl Mxfp4Linear {
    /// `blocks`/`scales` as loaded (any shape with matching element counts);
    /// `bias` must already be in the activation dtype.
    pub fn new(
        blocks: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        if blocks.dtype() != DType::U8 || scales.dtype() != DType::U8 {
            candle_core::bail!(
                "Mxfp4Linear: blocks/scales must be U8, got {:?}/{:?}",
                blocks.dtype(),
                scales.dtype()
            );
        }
        if !in_features.is_multiple_of(MXFP4_BLOCK) {
            candle_core::bail!(
                "Mxfp4Linear: in_features {in_features} must be a multiple of {MXFP4_BLOCK}"
            );
        }
        let nb = in_features / MXFP4_BLOCK;
        if blocks.elem_count() != out_features * nb * (MXFP4_BLOCK / 2)
            || scales.elem_count() != out_features * nb
        {
            candle_core::bail!(
                "Mxfp4Linear: blocks {} / scales {} elements don't match [{out_features}, {in_features}]",
                blocks.elem_count(),
                scales.elem_count()
            );
        }
        let blocks = blocks.reshape(out_features * nb * (MXFP4_BLOCK / 2))?;
        let scales = scales.reshape(out_features * nb)?;
        Ok(Self {
            blocks,
            scales,
            bias,
            in_features,
            out_features,
        })
    }

    /// `x`: `[..., in_features]` BF16 → `[..., out_features]` BF16.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_dims = x.dims().to_vec();
        let in_features = *original_dims.last().unwrap();
        if in_features != self.in_features {
            candle_core::bail!(
                "Mxfp4Linear: x last dim {} != in_features {}",
                in_features,
                self.in_features
            );
        }
        let m: usize = original_dims[..original_dims.len() - 1].iter().product();
        let x_2d = x.reshape((m, in_features))?.contiguous()?;

        #[cfg(feature = "metal")]
        let y_2d = if x.device().is_metal() && x.dtype() == DType::BF16 {
            super::metal_ops::mxfp4_matmul(
                &x_2d,
                &self.blocks,
                &self.scales,
                self.in_features,
                self.out_features,
            )?
        } else {
            self.forward_cpu_reference(&x_2d, m)?
        };
        #[cfg(not(feature = "metal"))]
        let y_2d = self.forward_cpu_reference(&x_2d, m)?;

        let mut out_dims = original_dims;
        *out_dims.last_mut().unwrap() = self.out_features;
        let y = y_2d.reshape(out_dims)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    /// Dequant + dense matmul via the CPU reference. Correctness fallback and
    /// parity baseline; not a serving path.
    fn forward_cpu_reference(&self, x_2d: &Tensor, m: usize) -> Result<Tensor> {
        let blocks = self.blocks.to_vec1::<u8>()?;
        let scales = self.scales.to_vec1::<u8>()?;
        let w = dequantize_mxfp4_f32(&blocks, &scales, self.out_features, self.in_features)?;
        let w_t = Tensor::from_vec(w, (self.out_features, self.in_features), x_2d.device())?
            .to_dtype(x_2d.dtype())?
            .t()?
            .contiguous()?;
        let _ = m;
        x_2d.matmul(&w_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // Contract: FP4 E2M1 decode table, low-nibble-first packing, and E8M0
    // scaling match the OCP/GPT-OSS spec.
    #[test]
    fn dequantize_known_values() {
        // One row, one block: codes 0..15 as low/high nibble pairs, then zeros.
        let mut bytes = Vec::new();
        for i in 0..8u8 {
            bytes.push((2 * i) | ((2 * i + 1) << 4));
        }
        bytes.extend([0u8; 8]);
        let scales = vec![128u8]; // 2^(128-127) = ×2
        let out = dequantize_mxfp4_f32(&bytes, &scales, 1, 32).unwrap();
        let expected_first16 = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, // codes 0..7
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // codes 8..15
        ];
        for (i, &e) in expected_first16.iter().enumerate() {
            assert_eq!(out[i], e * 2.0, "element {i}");
        }
        assert!(out[16..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn linear_cpu_matches_manual_dot() {
        let dev = Device::Cpu;
        // 2 rows, K=32. Row 0: all code 2 (=1.0) scale ×1; row 1: all code 4 (=2.0) scale ×0.5.
        let blocks_r0 = [0x22u8; 16];
        let blocks_r1 = [0x44u8; 16];
        let blocks: Vec<u8> = blocks_r0.iter().chain(blocks_r1.iter()).copied().collect();
        let scales = vec![127u8, 126u8];
        let blocks_t = Tensor::from_vec(blocks, 32, &dev).unwrap();
        let scales_t = Tensor::from_vec(scales, 2, &dev).unwrap();
        let lin = Mxfp4Linear::new(blocks_t, scales_t, None, 32, 2).unwrap();

        let x = Tensor::ones((1, 32), DType::F32, &dev).unwrap();
        let y = lin.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        // Row 0: 32 × 1.0 × 1 = 32; row 1: 32 × 2.0 × 0.5 = 32.
        assert_eq!(y[0], vec![32.0, 32.0]);
    }
}
