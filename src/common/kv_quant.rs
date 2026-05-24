use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuantMode {
    Off,
    Lossless,
    Balanced,
    Aggressive,
}

impl KvQuantMode {
    pub fn bit_width(self) -> u8 {
        match self {
            Self::Off => 0,
            Self::Lossless => 4,
            Self::Balanced => 3,
            Self::Aggressive => 2,
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "lossless" => Ok(Self::Lossless),
            "balanced" => Ok(Self::Balanced),
            "aggressive" => Ok(Self::Aggressive),
            other => Err(format!(
                "Unknown --kv-quant mode '{}'. Use: off, lossless, balanced, aggressive",
                other
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Lossless => "lossless (4-bit)",
            Self::Balanced => "balanced (3-bit)",
            Self::Aggressive => "aggressive (2-bit)",
        }
    }
}

// Lloyd-Max optimal centroids for N(0,1).
const CENTROIDS_1BIT: &[f32] = &[-0.7979, 0.7979];
const CENTROIDS_2BIT: &[f32] = &[-1.5104, -0.4528, 0.4528, 1.5104];
const CENTROIDS_3BIT: &[f32] = &[
    -2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520,
];
const CENTROIDS_4BIT: &[f32] = &[
    -2.7326, -2.0691, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284, 0.1284, 0.3881, 0.6568,
    0.9423, 1.2562, 1.6180, 2.0691, 2.7326,
];

fn compute_boundaries(centroids: &[f32]) -> Vec<f32> {
    (0..centroids.len() - 1)
        .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
        .collect()
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn generate_signs_with_seed(head_dim: usize, seed: u64) -> Vec<f32> {
    let mut state: u64 = seed ^ (head_dim as u64).wrapping_mul(2654435761);
    if state == 0 {
        state = 1;
    }
    (0..head_dim)
        .map(|_| {
            if xorshift64(&mut state) & 1 == 0 {
                -1.0f32
            } else {
                1.0f32
            }
        })
        .collect()
}

fn generate_signs(head_dim: usize) -> Vec<f32> {
    generate_signs_with_seed(head_dim, 0x5DEECE66D_u64)
}

fn generate_qjl_signs(head_dim: usize) -> Vec<f32> {
    generate_signs_with_seed(head_dim, 0x9E3779B97F4A7C15_u64)
}

/// Unnormalized, requires power-of-2 length.
fn wht_inplace(x: &mut [f32]) {
    let d = x.len();
    debug_assert!(d.is_power_of_two(), "WHT requires power-of-2, got {d}");
    let mut half = 1;
    while half < d {
        for i in (0..d).step_by(half * 2) {
            for j in i..i + half {
                let a = x[j];
                let b = x[j + half];
                x[j] = a + b;
                x[j + half] = a - b;
            }
        }
        half *= 2;
    }
}

/// Values: Stage-1 MSE at `bit_width`. Keys: Stage-2 MSE at `bit_width - 1`
/// + 1-bit QJL residual sign + residual norm scalar.
pub struct KvQuantizer {
    bit_width: u8,
    key_mse_bit_width: u8,
    qjl_quantization: bool,
    use_hadamard: bool,
    head_dim: usize,
    mse_signs: Vec<f32>,
    qjl_signs: Vec<f32>,
    value_scaled_centroids: Vec<f32>,
    value_scaled_boundaries: Vec<f32>,
    key_scaled_centroids: Vec<f32>,
    key_scaled_boundaries: Vec<f32>,
    inv_sqrt_d: f32,
    qjl_scale: f32,
}

impl KvQuantizer {
    pub fn new_with_qjl(bit_width: u8, head_dim: usize, qjl_quantization: bool) -> Self {
        assert!(
            (2..=4).contains(&bit_width),
            "Unsupported KV quant bit_width: {bit_width}"
        );

        let use_hadamard = head_dim.is_power_of_two();
        if !use_hadamard {
            tracing::warn!(
                head_dim,
                "head_dim is not power-of-two; using sign-only rotation fallback"
            );
        }

        let value_centroids: &[f32] = match bit_width {
            2 => CENTROIDS_2BIT,
            3 => CENTROIDS_3BIT,
            4 => CENTROIDS_4BIT,
            _ => panic!("Unsupported KV quant bit_width: {bit_width}"),
        };
        let key_mse_bit_width = if qjl_quantization {
            bit_width - 1
        } else {
            bit_width
        };
        let key_centroids: &[f32] = match key_mse_bit_width {
            1 => CENTROIDS_1BIT,
            2 => CENTROIDS_2BIT,
            3 => CENTROIDS_3BIT,
            4 => CENTROIDS_4BIT,
            _ => panic!("Unsupported Stage-2 key MSE bit_width: {key_mse_bit_width}"),
        };

        let value_boundaries = compute_boundaries(value_centroids);
        let key_boundaries = compute_boundaries(key_centroids);
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

        Self {
            bit_width,
            key_mse_bit_width,
            qjl_quantization,
            use_hadamard,
            head_dim,
            mse_signs: generate_signs(head_dim),
            qjl_signs: generate_qjl_signs(head_dim),
            value_scaled_centroids: value_centroids.iter().map(|c| c * inv_sqrt_d).collect(),
            value_scaled_boundaries: value_boundaries.iter().map(|b| b * inv_sqrt_d).collect(),
            key_scaled_centroids: key_centroids.iter().map(|c| c * inv_sqrt_d).collect(),
            key_scaled_boundaries: key_boundaries.iter().map(|b| b * inv_sqrt_d).collect(),
            inv_sqrt_d,
            qjl_scale: (PI / 2.0).sqrt() / head_dim as f32,
        }
    }

    pub fn qjl_quantization_enabled(&self) -> bool {
        self.qjl_quantization
    }

    pub fn key_packed_bytes(&self) -> usize {
        if self.qjl_quantization {
            self.packed_bytes_for_bits(self.key_mse_bit_width) + self.packed_bytes_for_bits(1)
        } else {
            self.packed_bytes_for_bits(self.bit_width)
        }
    }

    pub fn value_packed_bytes(&self) -> usize {
        self.packed_bytes_for_bits(self.bit_width)
    }

    pub fn quantize(&self, x: &[f32]) -> (Vec<u8>, f32) {
        self.quantize_mse(
            x,
            self.bit_width,
            &self.value_scaled_boundaries,
            self.value_packed_bytes(),
        )
    }

    pub fn dequantize(&self, packed: &[u8], norm: f32) -> Vec<f32> {
        self.dequantize_mse(
            packed,
            norm,
            self.bit_width,
            &self.value_scaled_centroids,
            self.value_packed_bytes(),
        )
    }

    /// Returns `(packed_key, mse_norm, residual_norm)` where packed_key is
    /// `[mse_indices | qjl_sign_bits]`.
    pub fn quantize_key(&self, x: &[f32]) -> (Vec<u8>, f32, f32) {
        if !self.qjl_quantization {
            let (packed, norm) = self.quantize(x);
            return (packed, norm, 0.0);
        }

        let (mse_packed, mse_norm) = self.quantize_mse(
            x,
            self.key_mse_bit_width,
            &self.key_scaled_boundaries,
            self.packed_bytes_for_bits(self.key_mse_bit_width),
        );

        let mut packed = Vec::with_capacity(self.key_packed_bytes());
        packed.extend_from_slice(&mse_packed);

        if mse_norm < 1e-10 {
            packed.resize(self.key_packed_bytes(), 0u8);
            return (packed, 0.0, 0.0);
        }

        let x_mse = self.dequantize_mse(
            &mse_packed,
            mse_norm,
            self.key_mse_bit_width,
            &self.key_scaled_centroids,
            self.packed_bytes_for_bits(self.key_mse_bit_width),
        );

        let mut residual = vec![0.0f32; self.head_dim];
        let mut residual_sq = 0.0f32;
        for (r_slot, (&xv, &x_msev)) in residual.iter_mut().zip(x.iter().zip(x_mse.iter())) {
            let r = xv - x_msev;
            *r_slot = r;
            residual_sq += r * r;
        }
        let residual_norm = residual_sq.sqrt();

        if residual_norm < 1e-10 {
            packed.resize(self.key_packed_bytes(), 0u8);
            return (packed, mse_norm, 0.0);
        }

        // sign(S r), S approximated via randomized Hadamard.
        for (r, &sign) in residual.iter_mut().zip(self.qjl_signs.iter()) {
            *r *= sign;
        }
        if self.use_hadamard {
            wht_inplace(&mut residual);
        }

        let mut qjl_bits = vec![0u8; self.head_dim];
        for (bit, &res) in qjl_bits.iter_mut().zip(residual.iter()) {
            *bit = if res >= 0.0 { 1 } else { 0 };
        }
        let qjl_packed = self.pack_indices_bits(&qjl_bits, 1);
        packed.extend_from_slice(&qjl_packed);

        (packed, mse_norm, residual_norm)
    }

    pub fn dequantize_key(&self, packed: &[u8], mse_norm: f32, residual_norm: f32) -> Vec<f32> {
        if !self.qjl_quantization {
            return self.dequantize(packed, mse_norm);
        }

        let mse_bytes = self.packed_bytes_for_bits(self.key_mse_bit_width);
        let qjl_bytes = self.packed_bytes_for_bits(1);
        debug_assert_eq!(packed.len(), mse_bytes + qjl_bytes);

        if packed.len() < mse_bytes + qjl_bytes {
            return vec![0.0f32; self.head_dim];
        }

        let mut out = self.dequantize_mse(
            &packed[..mse_bytes],
            mse_norm,
            self.key_mse_bit_width,
            &self.key_scaled_centroids,
            mse_bytes,
        );

        if residual_norm < 1e-10 {
            return out;
        }

        let qjl_indices = self.unpack_indices_bits(&packed[mse_bytes..mse_bytes + qjl_bytes], 1);
        let mut qjl = vec![0.0f32; self.head_dim];
        for (q, &idx) in qjl.iter_mut().zip(qjl_indices.iter()) {
            *q = if idx == 0 { -1.0 } else { 1.0 };
        }

        // x_qjl = gamma * sqrt(pi/2) / d * S^T sign(Sr).
        if self.use_hadamard {
            wht_inplace(&mut qjl);
        }
        let scale = residual_norm * self.qjl_scale;
        for ((out_i, &q), &sign) in out.iter_mut().zip(qjl.iter()).zip(self.qjl_signs.iter()) {
            *out_i += scale * q * sign;
        }

        out
    }

    fn quantize_mse(
        &self,
        x: &[f32],
        bit_width: u8,
        boundaries: &[f32],
        packed_bytes: usize,
    ) -> (Vec<u8>, f32) {
        debug_assert_eq!(x.len(), self.head_dim);

        let norm = {
            let mut s = 0.0f32;
            for &v in x {
                s += v * v;
            }
            s.sqrt()
        };

        if norm < 1e-10 {
            return (vec![0u8; packed_bytes], 0.0);
        }

        let inv_norm = 1.0 / norm;
        let mut y = Vec::with_capacity(self.head_dim);
        for &v in x {
            y.push(v * inv_norm);
        }

        // y = (1/sqrt(d)) * H * D * x_hat
        for (yv, &sign) in y.iter_mut().zip(self.mse_signs.iter()) {
            *yv *= sign;
        }
        if self.use_hadamard {
            wht_inplace(&mut y);
            for v in y.iter_mut() {
                *v *= self.inv_sqrt_d;
            }
        }

        let mut indices = vec![0u8; self.head_dim];
        for (idx, &yv) in indices.iter_mut().zip(y.iter()) {
            *idx = Self::find_nearest(boundaries, yv);
        }

        (self.pack_indices_bits(&indices, bit_width), norm)
    }

    fn dequantize_mse(
        &self,
        packed: &[u8],
        norm: f32,
        bit_width: u8,
        centroids: &[f32],
        packed_bytes: usize,
    ) -> Vec<f32> {
        if norm < 1e-10 {
            return vec![0.0f32; self.head_dim];
        }

        if packed.len() < packed_bytes {
            return vec![0.0f32; self.head_dim];
        }

        let indices = self.unpack_indices_bits(&packed[..packed_bytes], bit_width);

        let mut y = Vec::with_capacity(self.head_dim);
        for j in 0..self.head_dim {
            y.push(centroids[indices[j] as usize]);
        }

        if self.use_hadamard {
            wht_inplace(&mut y);
            for (yv, &sign) in y.iter_mut().zip(self.mse_signs.iter()) {
                *yv *= self.inv_sqrt_d * sign * norm;
            }
        } else {
            for (yv, &sign) in y.iter_mut().zip(self.mse_signs.iter()) {
                *yv *= sign * norm;
            }
        }

        y
    }

    fn find_nearest(boundaries: &[f32], val: f32) -> u8 {
        let mut lo = 0usize;
        let mut hi = boundaries.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if val <= boundaries[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo as u8
    }

    fn packed_bytes_for_bits(&self, bit_width: u8) -> usize {
        (self.head_dim * bit_width as usize).div_ceil(8)
    }

    fn pack_indices_bits(&self, indices: &[u8], bit_width: u8) -> Vec<u8> {
        match bit_width {
            1 => self.pack_1bit(indices),
            4 => self.pack_4bit(indices),
            3 => self.pack_3bit(indices),
            2 => self.pack_2bit(indices),
            _ => unreachable!(),
        }
    }

    fn unpack_indices_bits(&self, packed: &[u8], bit_width: u8) -> Vec<u8> {
        match bit_width {
            1 => self.unpack_1bit(packed),
            4 => self.unpack_4bit(packed),
            3 => self.unpack_3bit(packed),
            2 => self.unpack_2bit(packed),
            _ => unreachable!(),
        }
    }

    fn pack_1bit(&self, indices: &[u8]) -> Vec<u8> {
        let n = self.head_dim;
        let mut packed = vec![0u8; n.div_ceil(8)];
        for (i, &idx) in indices.iter().enumerate().take(n) {
            if (idx & 1) != 0 {
                packed[i / 8] |= 1 << (i % 8);
            }
        }
        packed
    }

    fn unpack_1bit(&self, packed: &[u8]) -> Vec<u8> {
        let n = self.head_dim;
        let mut indices = vec![0u8; n];
        for (i, idx) in indices.iter_mut().enumerate().take(n) {
            *idx = (packed[i / 8] >> (i % 8)) & 1;
        }
        indices
    }

    fn pack_4bit(&self, indices: &[u8]) -> Vec<u8> {
        let n = self.head_dim;
        let mut packed = vec![0u8; n.div_ceil(2)];
        for i in (0..n).step_by(2) {
            let lo = indices[i] & 0x0F;
            let hi = if i + 1 < n { indices[i + 1] & 0x0F } else { 0 };
            packed[i / 2] = lo | (hi << 4);
        }
        packed
    }

    fn unpack_4bit(&self, packed: &[u8]) -> Vec<u8> {
        let n = self.head_dim;
        let mut indices = vec![0u8; n];
        for i in (0..n).step_by(2) {
            indices[i] = packed[i / 2] & 0x0F;
            if i + 1 < n {
                indices[i + 1] = (packed[i / 2] >> 4) & 0x0F;
            }
        }
        indices
    }

    fn pack_3bit(&self, indices: &[u8]) -> Vec<u8> {
        let n = self.head_dim;
        let total_bytes = (n * 3).div_ceil(8);
        let mut packed = vec![0u8; total_bytes];
        for (i, &idx) in indices.iter().enumerate().take(n) {
            let val = idx & 0x07;
            let bit_offset = i * 3;
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;
            packed[byte_idx] |= val << bit_idx;
            if bit_idx > 5 && byte_idx + 1 < total_bytes {
                packed[byte_idx + 1] |= val >> (8 - bit_idx);
            }
        }
        packed
    }

    fn unpack_3bit(&self, packed: &[u8]) -> Vec<u8> {
        let n = self.head_dim;
        let mut indices = vec![0u8; n];
        for (i, idx) in indices.iter_mut().enumerate().take(n) {
            let bit_offset = i * 3;
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;
            let mut val = (packed[byte_idx] >> bit_idx) & 0x07;
            if bit_idx > 5 && byte_idx + 1 < packed.len() {
                val |= (packed[byte_idx + 1] << (8 - bit_idx)) & 0x07;
            }
            *idx = val;
        }
        indices
    }

    fn pack_2bit(&self, indices: &[u8]) -> Vec<u8> {
        let n = self.head_dim;
        let mut packed = vec![0u8; n.div_ceil(4)];
        for i in (0..n).step_by(4) {
            let mut byte = 0u8;
            for j in 0..4 {
                if i + j < n {
                    byte |= (indices[i + j] & 0x03) << (j * 2);
                }
            }
            packed[i / 4] = byte;
        }
        packed
    }

    fn unpack_2bit(&self, packed: &[u8]) -> Vec<u8> {
        let n = self.head_dim;
        let mut indices = vec![0u8; n];
        for i in (0..n).step_by(4) {
            let byte = packed[i / 4];
            for j in 0..4 {
                if i + j < n {
                    indices[i + j] = (byte >> (j * 2)) & 0x03;
                }
            }
        }
        indices
    }
}

pub fn quantized_value_bytes_per_head(head_dim: usize, bit_width: u8) -> usize {
    (head_dim * bit_width as usize).div_ceil(8) + 4
}

pub fn quantized_key_bytes_per_head_with_qjl(
    head_dim: usize,
    bit_width: u8,
    qjl_quantization: bool,
) -> usize {
    if qjl_quantization {
        let key_mse_bits = bit_width.saturating_sub(1);
        (head_dim * key_mse_bits as usize).div_ceil(8) + head_dim.div_ceil(8) + 8
    } else {
        (head_dim * bit_width as usize).div_ceil(8) + 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wht_roundtrip() {
        let d = 128;
        let signs = generate_signs(d);
        let inv = 1.0 / (d as f32).sqrt();
        let original: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

        let mut y = original.clone();
        for i in 0..d {
            y[i] *= signs[i];
        }
        wht_inplace(&mut y);
        for v in y.iter_mut() {
            *v *= inv;
        }

        wht_inplace(&mut y);
        for i in 0..d {
            y[i] *= inv * signs[i];
        }

        let diff: f32 = original
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-3, "WHT roundtrip error: {diff}");
    }

    #[test]
    fn quantize_dequantize_4bit_d128() {
        let q = KvQuantizer::new_with_qjl(4, 128, true);
        let x: Vec<f32> = (0..128)
            .map(|i| (i as f32 * 0.3 + 0.5).sin() * 2.0)
            .collect();
        let (packed, norm) = q.quantize(&x);
        let y = q.dequantize(&packed, norm);
        let mse: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 128.0;
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 128.0;
        let nmse = mse / var;
        assert!(nmse < 0.05, "4-bit d=128 NMSE too high: {nmse}");
    }

    #[test]
    fn quantize_dequantize_4bit_d64() {
        let q = KvQuantizer::new_with_qjl(4, 64, true);
        let x: Vec<f32> = (0..64)
            .map(|i| (i as f32 * 0.3 + 0.5).sin() * 2.0)
            .collect();
        let (packed, norm) = q.quantize(&x);
        let y = q.dequantize(&packed, norm);
        let mse: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 64.0;
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 64.0;
        let nmse = mse / var;
        assert!(nmse < 0.08, "4-bit d=64 NMSE too high: {nmse}");
    }

    #[test]
    fn quantize_dequantize_4bit_d96_non_power_of_two() {
        // Phi-family models use head_dim=96 (not power-of-2).
        let q = KvQuantizer::new_with_qjl(4, 96, false);
        let x: Vec<f32> = (0..96)
            .map(|i| (i as f32 * 0.27 + 0.4).sin() * 1.9)
            .collect();
        let (packed, norm) = q.quantize(&x);
        let y = q.dequantize(&packed, norm);
        let mse: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 96.0;
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 96.0;
        let nmse = mse / var;
        assert!(nmse < 0.35, "4-bit d=96 NMSE too high: {nmse}");
    }

    #[test]
    fn quantize_dequantize_3bit() {
        let q = KvQuantizer::new_with_qjl(3, 128, true);
        let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.3).sin() * 1.5).collect();
        let (packed, norm) = q.quantize(&x);
        let y = q.dequantize(&packed, norm);
        let mse: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 128.0;
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 128.0;
        let nmse = mse / var;
        assert!(nmse < 0.15, "3-bit NMSE too high: {nmse}");
    }

    #[test]
    fn quantize_dequantize_2bit() {
        let q = KvQuantizer::new_with_qjl(2, 128, true);
        let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.3).sin() * 1.5).collect();
        let (packed, norm) = q.quantize(&x);
        let y = q.dequantize(&packed, norm);
        let mse: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 128.0;
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 128.0;
        let nmse = mse / var;
        assert!(nmse < 0.35, "2-bit NMSE too high: {nmse}");
    }

    #[test]
    fn key_quantize_dequantize_4bit() {
        let q = KvQuantizer::new_with_qjl(4, 128, true);
        let x: Vec<f32> = (0..128)
            .map(|i| (i as f32 * 0.21 + 0.7).cos() * 1.7)
            .collect();
        let (packed, norm, residual_norm) = q.quantize_key(&x);
        let y = q.dequantize_key(&packed, norm, residual_norm);
        let mse: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 128.0;
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 128.0;
        let nmse = mse / var;
        assert!(nmse < 0.25, "Stage-2 key 4-bit NMSE too high: {nmse}");
    }

    #[test]
    fn key_quantization_without_qjl_matches_stage1() {
        let q = KvQuantizer::new_with_qjl(4, 128, false);
        let x: Vec<f32> = (0..128)
            .map(|i| (i as f32 * 0.11 + 0.3).sin() * 1.3)
            .collect();

        let (k_packed, k_norm, k_residual_norm) = q.quantize_key(&x);
        let (v_packed, v_norm) = q.quantize(&x);

        assert_eq!(k_packed, v_packed);
        assert!((k_norm - v_norm).abs() < 1e-8);
        assert!(k_residual_norm.abs() < 1e-8);

        let yk = q.dequantize_key(&k_packed, k_norm, k_residual_norm);
        let yv = q.dequantize(&v_packed, v_norm);
        let diff: f32 = yk.iter().zip(yv.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff < 1e-6,
            "QJL-off key path diverged from Stage-1: {diff}"
        );
    }

    #[test]
    fn zero_vector() {
        let q = KvQuantizer::new_with_qjl(4, 128, true);
        let x = vec![0.0f32; 128];
        let (packed, norm) = q.quantize(&x);
        assert!(norm < 1e-10);
        let y = q.dequantize(&packed, norm);
        assert!(y.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn pack_unpack_4bit() {
        let q = KvQuantizer::new_with_qjl(4, 128, true);
        let idx: Vec<u8> = (0..128).map(|i| (i % 16) as u8).collect();
        assert_eq!(idx, q.unpack_4bit(&q.pack_4bit(&idx)));
    }

    #[test]
    fn pack_unpack_3bit() {
        let q = KvQuantizer::new_with_qjl(3, 128, true);
        let idx: Vec<u8> = (0..128).map(|i| (i % 8) as u8).collect();
        assert_eq!(idx, q.unpack_3bit(&q.pack_3bit(&idx)));
    }

    #[test]
    fn pack_unpack_2bit() {
        let q = KvQuantizer::new_with_qjl(2, 128, true);
        let idx: Vec<u8> = (0..128).map(|i| (i % 4) as u8).collect();
        assert_eq!(idx, q.unpack_2bit(&q.pack_2bit(&idx)));
    }

    #[test]
    fn pack_unpack_4bit_d64() {
        let q = KvQuantizer::new_with_qjl(4, 64, true);
        let idx: Vec<u8> = (0..64).map(|i| (i % 16) as u8).collect();
        assert_eq!(idx, q.unpack_4bit(&q.pack_4bit(&idx)));
    }

    #[test]
    fn pack_unpack_3bit_d64() {
        let q = KvQuantizer::new_with_qjl(3, 64, true);
        let idx: Vec<u8> = (0..64).map(|i| (i % 8) as u8).collect();
        assert_eq!(idx, q.unpack_3bit(&q.pack_3bit(&idx)));
    }

    #[test]
    fn bytes_per_head_matches() {
        assert_eq!(quantized_value_bytes_per_head(128, 4), 68);
        assert_eq!(quantized_value_bytes_per_head(128, 3), 52);
        assert_eq!(quantized_value_bytes_per_head(128, 2), 36);
        assert_eq!(quantized_value_bytes_per_head(64, 4), 36);

        assert_eq!(quantized_key_bytes_per_head_with_qjl(128, 4, true), 72);
        assert_eq!(quantized_key_bytes_per_head_with_qjl(128, 3, true), 56);
        assert_eq!(quantized_key_bytes_per_head_with_qjl(128, 2, true), 40);

        assert_eq!(
            quantized_key_bytes_per_head_with_qjl(128, 4, true)
                + quantized_value_bytes_per_head(128, 4),
            140
        );
        assert_eq!(
            quantized_key_bytes_per_head_with_qjl(128, 3, true)
                + quantized_value_bytes_per_head(128, 3),
            108
        );
        assert_eq!(
            quantized_key_bytes_per_head_with_qjl(128, 2, true)
                + quantized_value_bytes_per_head(128, 2),
            76
        );

        assert_eq!(quantized_key_bytes_per_head_with_qjl(128, 4, false), 68);
        assert_eq!(quantized_key_bytes_per_head_with_qjl(128, 3, false), 52);
        assert_eq!(quantized_key_bytes_per_head_with_qjl(128, 2, false), 36);
    }
}
