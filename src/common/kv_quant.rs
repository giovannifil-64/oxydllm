/// KV cache quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuantMode {
    Off,
    Lossless,   // 4-bit MSE, quality-neutral
    Balanced,   // 3-bit MSE, near-identical quality
    Aggressive, // 2-bit MSE, maximum compression
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
const CENTROIDS_2BIT: &[f32] = &[-1.5104, -0.4528, 0.4528, 1.5104];
const CENTROIDS_3BIT: &[f32] = &[
    -2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520,
];
const CENTROIDS_4BIT: &[f32] = &[
    -2.7326, -2.0691, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
    0.1284, 0.3881, 0.6568, 0.9423, 1.2562, 1.6180, 2.0691, 2.7326,
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

fn generate_signs(head_dim: usize) -> Vec<f32> {
    let mut state: u64 = 0x5DEECE66D_u64 ^ (head_dim as u64).wrapping_mul(2654435761);
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

/// In-place Walsh-Hadamard transform (unnormalized). Requires power-of-2 length.
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

/// TurboQuant MSE quantizer for a given (bit_width, head_dim) pair.
pub struct KvQuantizer {
    bit_width: u8,
    head_dim: usize,
    signs: Vec<f32>,
    scaled_centroids: Vec<f32>,
    scaled_boundaries: Vec<f32>,
    inv_sqrt_d: f32,
}

impl KvQuantizer {
    pub fn new(bit_width: u8, head_dim: usize) -> Self {
        assert!(
            head_dim.is_power_of_two(),
            "KV quantization requires power-of-2 head_dim, got {head_dim}"
        );
        let raw_centroids: &[f32] = match bit_width {
            2 => CENTROIDS_2BIT,
            3 => CENTROIDS_3BIT,
            4 => CENTROIDS_4BIT,
            _ => panic!("Unsupported KV quant bit_width: {bit_width}"),
        };
        let raw_boundaries = compute_boundaries(raw_centroids);
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

        Self {
            bit_width,
            head_dim,
            signs: generate_signs(head_dim),
            scaled_centroids: raw_centroids.iter().map(|c| c * inv_sqrt_d).collect(),
            scaled_boundaries: raw_boundaries.iter().map(|b| b * inv_sqrt_d).collect(),
            inv_sqrt_d,
        }
    }

    /// Packed index bytes per head per token.
    pub fn packed_index_bytes(&self) -> usize {
        (self.head_dim * self.bit_width as usize + 7) / 8
    }

    /// Total bytes per head per token (indices + f32 norm).
    pub fn bytes_per_head(&self) -> usize {
        self.packed_index_bytes() + 4
    }

    /// Quantize a single f32 vector of length head_dim.
    pub fn quantize(&self, x: &[f32]) -> (Vec<u8>, f32) {
        debug_assert_eq!(x.len(), self.head_dim);

        let norm = {
            let mut s = 0.0f32;
            for &v in x {
                s += v * v;
            }
            s.sqrt()
        };

        if norm < 1e-10 {
            return (vec![0u8; self.packed_index_bytes()], 0.0);
        }

        let inv_norm = 1.0 / norm;
        let mut y = Vec::with_capacity(self.head_dim);
        for &v in x {
            y.push(v * inv_norm);
        }

        // Forward rotation: y = (1/sqrt(d)) * H * D * x_hat
        for i in 0..self.head_dim {
            y[i] *= self.signs[i];
        }
        wht_inplace(&mut y);
        for v in y.iter_mut() {
            *v *= self.inv_sqrt_d;
        }

        // Quantize each coordinate using scaled boundaries
        let mut indices = vec![0u8; self.head_dim];
        for j in 0..self.head_dim {
            indices[j] = self.find_nearest(y[j]);
        }

        (self.pack_indices(&indices), norm)
    }

    /// Dequantize packed indices + norm back to f32 vector of length head_dim.
    pub fn dequantize(&self, packed: &[u8], norm: f32) -> Vec<f32> {
        if norm < 1e-10 {
            return vec![0.0f32; self.head_dim];
        }

        let indices = self.unpack_indices(packed);

        // Reconstruct rotated coordinates
        let mut y = Vec::with_capacity(self.head_dim);
        for j in 0..self.head_dim {
            y.push(self.scaled_centroids[indices[j] as usize]);
        }

        // Inverse rotation: x_hat = D * (1/sqrt(d)) * H * y_tilde
        wht_inplace(&mut y);
        for i in 0..self.head_dim {
            y[i] *= self.inv_sqrt_d * self.signs[i] * norm;
        }

        y
    }

    fn find_nearest(&self, val: f32) -> u8 {
        let mut lo = 0usize;
        let mut hi = self.scaled_boundaries.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if val <= self.scaled_boundaries[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo as u8
    }

    fn pack_indices(&self, indices: &[u8]) -> Vec<u8> {
        match self.bit_width {
            4 => self.pack_4bit(indices),
            3 => self.pack_3bit(indices),
            2 => self.pack_2bit(indices),
            _ => unreachable!(),
        }
    }

    fn unpack_indices(&self, packed: &[u8]) -> Vec<u8> {
        match self.bit_width {
            4 => self.unpack_4bit(packed),
            3 => self.unpack_3bit(packed),
            2 => self.unpack_2bit(packed),
            _ => unreachable!(),
        }
    }

    fn pack_4bit(&self, indices: &[u8]) -> Vec<u8> {
        let n = self.head_dim;
        let mut packed = vec![0u8; (n + 1) / 2];
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
        let total_bytes = (n * 3 + 7) / 8;
        let mut packed = vec![0u8; total_bytes];
        for i in 0..n {
            let val = indices[i] & 0x07;
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
        for i in 0..n {
            let bit_offset = i * 3;
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;
            let mut val = (packed[byte_idx] >> bit_idx) & 0x07;
            if bit_idx > 5 && byte_idx + 1 < packed.len() {
                val |= (packed[byte_idx + 1] << (8 - bit_idx)) & 0x07;
            }
            indices[i] = val;
        }
        indices
    }

    fn pack_2bit(&self, indices: &[u8]) -> Vec<u8> {
        let n = self.head_dim;
        let mut packed = vec![0u8; (n + 3) / 4];
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

/// Compute bytes_per_head for a given (head_dim, bit_width) without creating a full quantizer.
pub fn quantized_bytes_per_head(head_dim: usize, bit_width: u8) -> usize {
    (head_dim * bit_width as usize + 7) / 8 + 4 // packed indices + f32 norm
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
        for i in 0..d { y[i] *= signs[i]; }
        wht_inplace(&mut y);
        for v in y.iter_mut() { *v *= inv; }

        wht_inplace(&mut y);
        for i in 0..d { y[i] *= inv * signs[i]; }

        let diff: f32 = original.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff < 1e-3, "WHT roundtrip error: {diff}");
    }

    #[test]
    fn quantize_dequantize_4bit_d128() {
        let q = KvQuantizer::new(4, 128);
        let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.3 + 0.5).sin() * 2.0).collect();
        let (packed, norm) = q.quantize(&x);
        let y = q.dequantize(&packed, norm);
        let mse: f32 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / 128.0;
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 128.0;
        let nmse = mse / var;
        assert!(nmse < 0.05, "4-bit d=128 NMSE too high: {nmse}");
    }

    #[test]
    fn quantize_dequantize_4bit_d64() {
        let q = KvQuantizer::new(4, 64);
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3 + 0.5).sin() * 2.0).collect();
        let (packed, norm) = q.quantize(&x);
        let y = q.dequantize(&packed, norm);
        let mse: f32 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / 64.0;
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 64.0;
        let nmse = mse / var;
        assert!(nmse < 0.08, "4-bit d=64 NMSE too high: {nmse}");
    }

    #[test]
    fn quantize_dequantize_3bit() {
        let q = KvQuantizer::new(3, 128);
        let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.3).sin() * 1.5).collect();
        let (packed, norm) = q.quantize(&x);
        let y = q.dequantize(&packed, norm);
        let mse: f32 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / 128.0;
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 128.0;
        let nmse = mse / var;
        assert!(nmse < 0.15, "3-bit NMSE too high: {nmse}");
    }

    #[test]
    fn quantize_dequantize_2bit() {
        let q = KvQuantizer::new(2, 128);
        let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.3).sin() * 1.5).collect();
        let (packed, norm) = q.quantize(&x);
        let y = q.dequantize(&packed, norm);
        let mse: f32 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / 128.0;
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 128.0;
        let nmse = mse / var;
        assert!(nmse < 0.35, "2-bit NMSE too high: {nmse}");
    }

    #[test]
    fn zero_vector() {
        let q = KvQuantizer::new(4, 128);
        let x = vec![0.0f32; 128];
        let (packed, norm) = q.quantize(&x);
        assert!(norm < 1e-10);
        let y = q.dequantize(&packed, norm);
        assert!(y.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn pack_unpack_4bit() {
        let q = KvQuantizer::new(4, 128);
        let idx: Vec<u8> = (0..128).map(|i| (i % 16) as u8).collect();
        assert_eq!(idx, q.unpack_4bit(&q.pack_4bit(&idx)));
    }

    #[test]
    fn pack_unpack_3bit() {
        let q = KvQuantizer::new(3, 128);
        let idx: Vec<u8> = (0..128).map(|i| (i % 8) as u8).collect();
        assert_eq!(idx, q.unpack_3bit(&q.pack_3bit(&idx)));
    }

    #[test]
    fn pack_unpack_2bit() {
        let q = KvQuantizer::new(2, 128);
        let idx: Vec<u8> = (0..128).map(|i| (i % 4) as u8).collect();
        assert_eq!(idx, q.unpack_2bit(&q.pack_2bit(&idx)));
    }

    #[test]
    fn pack_unpack_4bit_d64() {
        let q = KvQuantizer::new(4, 64);
        let idx: Vec<u8> = (0..64).map(|i| (i % 16) as u8).collect();
        assert_eq!(idx, q.unpack_4bit(&q.pack_4bit(&idx)));
    }

    #[test]
    fn pack_unpack_3bit_d64() {
        let q = KvQuantizer::new(3, 64);
        let idx: Vec<u8> = (0..64).map(|i| (i % 8) as u8).collect();
        assert_eq!(idx, q.unpack_3bit(&q.pack_3bit(&idx)));
    }

    #[test]
    fn bytes_per_head_matches() {
        // 4-bit, d=128: 64 bytes indices + 4 bytes norm = 68
        assert_eq!(quantized_bytes_per_head(128, 4), 68);
        // 3-bit, d=128: 48 bytes indices + 4 bytes norm = 52
        assert_eq!(quantized_bytes_per_head(128, 3), 52);
        // 2-bit, d=128: 32 bytes indices + 4 bytes norm = 36
        assert_eq!(quantized_bytes_per_head(128, 2), 36);
        // 4-bit, d=64: 32 bytes indices + 4 bytes norm = 36
        assert_eq!(quantized_bytes_per_head(64, 4), 36);
    }
}
