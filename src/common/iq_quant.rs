//! The importance-quantized GGUF block types, decoded on the CPU exactly as
//! ggml decodes them.
//!
//! candle has no notion of these types, so a weight in one of them is owned
//! by this crate from the loader down, and the GPU kernels that read it have
//! no reference to be checked against except this module: every function here
//! is a line-by-line translation of the matching `dequantize_row_iq*` in
//! `ggml-quants.c`, and the tables are copied from `ggml-common.h` verbatim.
//! The fixtures beside it are blocks cut out of a published file with a range
//! request, so the kernels are also checked on bit patterns a real quantizer
//! produced rather than on ones invented here.

/// Elements per block of the 256-wide types.
pub const QK_K: usize = 256;

/// Bytes of one block, and elements per block, of each type this module reads.
pub const IQ4_NL_BLOCK_BYTES: usize = 18;
pub const IQ4_NL_BLOCK_ELEMS: usize = 32;
pub const IQ4_XS_BLOCK_BYTES: usize = 136;
pub const IQ3_S_BLOCK_BYTES: usize = 110;

/// The sixteen values a four-bit IQ4 code stands for, `kvalues_iq4nl`.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// The bit each of eight sign positions occupies in a sign byte, `kmask_iq2xs`.
pub const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// The 512 four-weight entries an IQ3_S code selects, `iq3s_grid`; the four
/// bytes of each entry are the four weights, low byte first.
pub const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // A subnormal: renormalise into the F32 exponent range.
            let mut e: i32 = 0;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            (sign << 31) | (((127 - 15 + 1 + e) as u32) << 23) | (f << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

fn half_at(bytes: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// Decodes `bytes`, whole IQ4_NL blocks, into `out`, 32 weights per block.
///
/// ## Panics
/// When `bytes` is not whole blocks or `out` is not their weights' worth.
pub fn dequantize_iq4_nl(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(bytes.len() % IQ4_NL_BLOCK_BYTES, 0);
    let nb = bytes.len() / IQ4_NL_BLOCK_BYTES;
    assert_eq!(out.len(), nb * IQ4_NL_BLOCK_ELEMS);
    for i in 0..nb {
        let b = &bytes[i * IQ4_NL_BLOCK_BYTES..(i + 1) * IQ4_NL_BLOCK_BYTES];
        let d = half_at(b, 0);
        let qs = &b[2..18];
        let y = &mut out[i * IQ4_NL_BLOCK_ELEMS..(i + 1) * IQ4_NL_BLOCK_ELEMS];
        for j in 0..16 {
            y[j] = d * f32::from(KVALUES_IQ4NL[(qs[j] & 0xf) as usize]);
            y[j + 16] = d * f32::from(KVALUES_IQ4NL[(qs[j] >> 4) as usize]);
        }
    }
}

/// Decodes `bytes`, whole IQ4_XS blocks, into `out`, 256 weights per block.
///
/// ## Panics
/// When `bytes` is not whole blocks or `out` is not their weights' worth.
pub fn dequantize_iq4_xs(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(bytes.len() % IQ4_XS_BLOCK_BYTES, 0);
    let nb = bytes.len() / IQ4_XS_BLOCK_BYTES;
    assert_eq!(out.len(), nb * QK_K);
    for i in 0..nb {
        let b = &bytes[i * IQ4_XS_BLOCK_BYTES..(i + 1) * IQ4_XS_BLOCK_BYTES];
        let d = half_at(b, 0);
        let scales_h = u16::from_le_bytes([b[2], b[3]]);
        let scales_l = &b[4..8];
        let qs = &b[8..136];
        let y = &mut out[i * QK_K..(i + 1) * QK_K];
        for ib in 0..8 {
            let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32
                | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
            let dl = d * (ls - 32) as f32;
            let q = &qs[ib * 16..(ib + 1) * 16];
            for j in 0..16 {
                y[ib * 32 + j] = dl * f32::from(KVALUES_IQ4NL[(q[j] & 0xf) as usize]);
                y[ib * 32 + j + 16] = dl * f32::from(KVALUES_IQ4NL[(q[j] >> 4) as usize]);
            }
        }
    }
}

/// Decodes `bytes`, whole IQ3_S blocks, into `out`, 256 weights per block.
///
/// ## Panics
/// When `bytes` is not whole blocks or `out` is not their weights' worth.
pub fn dequantize_iq3_s(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(bytes.len() % IQ3_S_BLOCK_BYTES, 0);
    let nb = bytes.len() / IQ3_S_BLOCK_BYTES;
    assert_eq!(out.len(), nb * QK_K);
    for i in 0..nb {
        let b = &bytes[i * IQ3_S_BLOCK_BYTES..(i + 1) * IQ3_S_BLOCK_BYTES];
        let d = half_at(b, 0);
        let qs_all = &b[2..66];
        let qh_all = &b[66..74];
        let signs_all = &b[74..106];
        let scales = &b[106..110];
        let y = &mut out[i * QK_K..(i + 1) * QK_K];
        let mut pos = 0usize;
        let mut qs = 0usize;
        let mut signs = 0usize;
        let mut qh = 0usize;
        let grid_bytes = |idx: usize| -> [f32; 4] {
            let g = IQ3S_GRID[idx];
            [
                (g & 0xff) as f32,
                ((g >> 8) & 0xff) as f32,
                ((g >> 16) & 0xff) as f32,
                ((g >> 24) & 0xff) as f32,
            ]
        };
        let mut ib32 = 0;
        while ib32 < QK_K / 32 {
            let db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf) as i32) as f32;
            let db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4) as i32) as f32;
            for (db, qh_byte) in [(db1, qh_all[qh]), (db2, qh_all[qh + 1])] {
                for l in 0..4 {
                    let i1 =
                        qs_all[qs + 2 * l] as usize | (((qh_byte as usize) << (8 - 2 * l)) & 256);
                    let i2 = qs_all[qs + 2 * l + 1] as usize
                        | (((qh_byte as usize) << (7 - 2 * l)) & 256);
                    let g1 = grid_bytes(i1);
                    let g2 = grid_bytes(i2);
                    let s = signs_all[signs + l];
                    for j in 0..4 {
                        let neg1 = s & KMASK_IQ2XS[j] != 0;
                        let neg2 = s & KMASK_IQ2XS[j + 4] != 0;
                        y[pos + j] = db * g1[j] * if neg1 { -1.0 } else { 1.0 };
                        y[pos + j + 4] = db * g2[j] * if neg2 { -1.0 } else { 1.0 };
                    }
                    pos += 8;
                }
                qs += 8;
                signs += 4;
            }
            qh += 2;
            ib32 += 2;
        }
    }
}

/// Blocks cut from `unsloth/Qwen3.8-27B-GGUF`'s `Qwen3.8-27B-UD-Q4_K_M.gguf`
/// with a range request, sixteen of each type, from `blk.0.ffn_down`
/// (IQ4_XS), `blk.1.ffn_down` (IQ4_NL) and `blk.11.ffn_gate` (IQ3_S).
pub const FIXTURE_IQ4_XS: &[u8] = include_bytes!("iq_fixtures/iq4_xs.bin");
pub const FIXTURE_IQ4_NL: &[u8] = include_bytes!("iq_fixtures/iq4_nl.bin");
pub const FIXTURE_IQ3_S: &[u8] = include_bytes!("iq_fixtures/iq3_s.bin");

#[cfg(test)]
mod tests {
    use super::*;

    /// Contract: the tables are ggml's, checked at their ends and at a few
    /// places inside, so a copy that dropped or reordered an entry fails here.
    #[test]
    fn the_tables_are_ggml_s() {
        assert_eq!(KVALUES_IQ4NL[0], -127);
        assert_eq!(KVALUES_IQ4NL[7], -10);
        assert_eq!(KVALUES_IQ4NL[8], 1);
        assert_eq!(KVALUES_IQ4NL[15], 113);
        assert_eq!(KMASK_IQ2XS, [1, 2, 4, 8, 16, 32, 64, 128]);
        assert_eq!(IQ3S_GRID[0], 0x0101_0101);
        assert_eq!(IQ3S_GRID[1], 0x0101_0103);
        assert_eq!(IQ3S_GRID[511], 252_641_537);
        assert!(
            IQ3S_GRID
                .iter()
                .all(|g| g.to_le_bytes().iter().all(|b| b % 2 == 1 && *b <= 15)),
            "every grid weight is an odd value from 1 to 15"
        );
    }

    /// Contract: the half-precision decoder agrees with the IEEE definition on
    /// the cases that matter: zero, one, a subnormal, a negative, infinity.
    #[test]
    fn half_precision_decodes_by_the_book() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
    }

    /// Contract: real blocks decode to finite weights of a weight's size, and
    /// every block's scale is a small non-zero number.
    ///
    /// Not a proof of the bit mapping, which nothing on this machine can give
    /// short of running the model; a wrong mapping tends to show as weights
    /// far outside the range a trained matrix has, and that is what this
    /// catches. The scales themselves are tiny, since a weight is the scale
    /// times a sub-block scale up to 31 times a code value up to 127: on this
    /// file the IQ4_XS scales are half-precision subnormals of either sign,
    /// which is also why the half decoder here handles subnormals.
    #[test]
    fn real_blocks_decode_to_weights_of_a_weight_s_size() {
        let mut xs = vec![0f32; 16 * QK_K];
        dequantize_iq4_xs(FIXTURE_IQ4_XS, &mut xs);
        let mut nl = vec![0f32; 16 * IQ4_NL_BLOCK_ELEMS];
        dequantize_iq4_nl(FIXTURE_IQ4_NL, &mut nl);
        let mut s3 = vec![0f32; 16 * QK_K];
        dequantize_iq3_s(FIXTURE_IQ3_S, &mut s3);
        for (name, v) in [("IQ4_XS", &xs), ("IQ4_NL", &nl), ("IQ3_S", &s3)] {
            assert!(v.iter().all(|x| x.is_finite()), "{name}: not finite");
            let peak = v.iter().fold(0f32, |m, x| m.max(x.abs()));
            assert!(
                peak > 1e-4 && peak < 1.0,
                "{name}: peak {peak} is not a weight's size"
            );
            let nonzero = v.iter().filter(|x| **x != 0.0).count();
            assert!(nonzero > v.len() / 2, "{name}: mostly zeros");
        }
        for i in 0..16 {
            let d = half_at(FIXTURE_IQ4_XS, i * IQ4_XS_BLOCK_BYTES);
            assert!(d != 0.0 && d.abs() < 0.01, "IQ4_XS block {i}: scale {d}");
            let d = half_at(FIXTURE_IQ3_S, i * IQ3_S_BLOCK_BYTES);
            assert!(d != 0.0 && d.abs() < 0.01, "IQ3_S block {i}: scale {d}");
            let d = half_at(FIXTURE_IQ4_NL, i * IQ4_NL_BLOCK_BYTES);
            assert!(d != 0.0 && d.abs() < 0.01, "IQ4_NL block {i}: scale {d}");
        }
    }
}
