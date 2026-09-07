// TensorOps (Metal Performance Primitives) GEMM, prefill fast path using the
// M5 neural accelerator. Requires Metal 4 (macOS 26+); compiled at runtime
// with MTLLanguageVersion 4.0 and gated behind a compile-once availability
// check (older OS / non-M5 falls back to the candle GEMM).
//
// Measured on M5 vs candle BF16 GEMM (2560x9728, see mpp_gemm_perf_probe):
// 1.96x at M=64, 3.6x at M=256, 2.3x at M=1024.
//
// Layout convention (MPP guide §1.2/§2.1, row-major operands):
//   A [M, K]  -> tensor extents {K, M},  strides {1, lda}
//   B [K, N]  -> tensor extents {N, K},  strides {1, ldb}   (nn kernel)
//   B [N, K]  -> transpose_right descriptor                  (nt kernel)
//   D [M, N]  -> tensor extents {N, M},  strides {1, ldd}
//
// NB: no `const` on a/b, the TensorOps dispatch matches value types without
// stripping cv-qualifiers, so `const bfloat` falls through to a static_assert.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

struct MppGemmParams {
    int m;
    int n;
    int k;
};

constant constexpr int TM = 64;
constant constexpr int TN = 64;

kernel void mpp_gemm_bf16_nn(
    device bfloat*        a [[buffer(0)]],
    device bfloat*        b [[buffer(1)]],
    device bfloat*        d [[buffer(2)]],
    constant MppGemmParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(TM, TN);
    matmul2d<desc, execution_simdgroups<4>> op;

    int row0 = int(tgid.y) * TM;
    int col0 = int(tgid.x) * TN;
    int tm = min(TM, p.m - row0);
    int tn = min(TN, p.n - col0);
    if (tm <= 0 || tn <= 0) {
        return;
    }

    auto tA = tensor(a + row0 * p.k, dextents<int, 2>{p.k, tm}, array<int, 2>{1, p.k});
    auto tB = tensor(b + col0, dextents<int, 2>{tn, p.k}, array<int, 2>{1, p.n});
    auto tD = tensor(d + row0 * p.n + col0, dextents<int, 2>{tn, tm}, array<int, 2>{1, p.n});

    op.run(tA, tB, tD);
}

kernel void mpp_gemm_bf16_nt(
    device bfloat*        a [[buffer(0)]],
    device bfloat*        b [[buffer(1)]],
    device bfloat*        d [[buffer(2)]],
    constant MppGemmParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(metal::dynamic_extent),
                                              false, /*transpose_right=*/true);
    matmul2d<desc, execution_simdgroups<4>> op;

    int row0 = int(tgid.y) * TM;
    int col0 = int(tgid.x) * TN;
    int tm = min(TM, p.m - row0);
    int tn = min(TN, p.n - col0);
    if (tm <= 0 || tn <= 0) {
        return;
    }

    auto tA = tensor(a + row0 * p.k, dextents<int, 2>{p.k, tm}, array<int, 2>{1, p.k});
    auto tB = tensor(b + col0 * p.k, dextents<int, 2>{p.k, tn}, array<int, 2>{1, p.k});
    auto tD = tensor(d + row0 * p.n + col0, dextents<int, 2>{tn, tm}, array<int, 2>{1, p.n});

    op.run(tA, tB, tD);
}

// ── Staged packed-quant GEMM ────────────────────────────────────────────────
//
// Prefill matmul for packed-quant weights without materializing the dense
// weight: each K-iteration dequantizes a [BK × TN] tile of B into threadgroup
// memory and feeds it to matmul2d in multiply_accumulate mode, with a float
// cooperative-tensor accumulator stored to D at the end. Packing layouts and
// dequant math mirror quant_kernels.metal (AWQ: word packs PACK_FACTOR output
// columns for one k, 4-bit interleaved by AWQ_PACK_ORDER; GPTQ: word packs
// PACK_FACTOR k-positions for one column, zero stored as z-1).

struct MppQuantGemmParams {
    int m;
    int n;
    int k;
    int group_shift;
};

constant constexpr int QBK = 64;

constant uint MPP_AWQ_PACK_ORDER[8] = {0u, 2u, 4u, 6u, 1u, 3u, 5u, 7u};

template<uint BITS>
inline uint mpp_unpack(uint word, uint s) {
    return (word >> (BITS * s)) & ((1u << BITS) - 1u);
}

template<uint BITS>
inline uint mpp_pack_position(uint s) {
    return (BITS == 4u) ? MPP_AWQ_PACK_ORDER[s] : s;
}

template<uint BITS>
inline void awq_stage_tile(
    device uint*    qweight,
    device uint*    qzeros,
    device bfloat*  scales,
    threadgroup bfloat* sB,
    constant MppQuantGemmParams& p,
    int k0, int col0, int bk, int tn, uint lid)
{
    constexpr uint PF = 32u / BITS;
    uint packed_n = uint(p.n) / PF;
    uint word0 = uint(col0) / PF;
    uint words_per_row = uint(TN) / PF;
    uint total = uint(bk) * words_per_row;
    for (uint w = lid; w < total; w += 128u) {
        uint kk = w / words_per_row;
        uint wj = w % words_per_row;
        uint k = uint(k0) + kk;
        uint g = k >> p.group_shift;
        uint ww = qweight[k * packed_n + word0 + wj];
        uint zw = qzeros[g * packed_n + word0 + wj];
        for (uint s = 0; s < PF; ++s) {
            uint o = wj * PF + mpp_pack_position<BITS>(s);
            if (int(o) >= tn) {
                continue;
            }
            float scale = float(scales[g * uint(p.n) + uint(col0) + o]);
            float v = (float(mpp_unpack<BITS>(ww, s)) - float(mpp_unpack<BITS>(zw, s))) * scale;
            sB[kk * uint(TN) + o] = bfloat(v);
        }
    }
}

template<uint BITS>
inline void gptq_stage_tile(
    device uint*    qweight,
    device uint*    qzeros,
    device bfloat*  scales,
    threadgroup bfloat* sB,
    constant MppQuantGemmParams& p,
    int k0, int col0, int bk, int tn, uint lid)
{
    constexpr uint PF = 32u / BITS;
    constexpr uint MASK = (1u << BITS) - 1u;
    uint qzeros_inner = uint(p.n) / PF;
    uint word_rows = (uint(bk) + PF - 1u) / PF;
    uint total = word_rows * uint(TN);
    for (uint w = lid; w < total; w += 128u) {
        uint kw = w / uint(TN);
        uint col = w % uint(TN);
        if (int(col) >= tn) {
            continue;
        }
        uint o = uint(col0) + col;
        uint k_base = uint(k0) + kw * PF;
        uint g = k_base >> p.group_shift;
        uint ww = qweight[(k_base / PF) * uint(p.n) + o];
        uint zw = qzeros[g * qzeros_inner + o / PF];
        float zp1 = float((zw >> (BITS * (o % PF))) & MASK) + 1.0f;
        float scale = float(scales[g * uint(p.n) + o]);
        for (uint s = 0; s < PF; ++s) {
            uint kk = kw * PF + s;
            if (int(kk) >= bk) {
                break;
            }
            float v = (float((ww >> (BITS * s)) & MASK) - zp1) * scale;
            sB[kk * uint(TN) + col] = bfloat(v);
        }
    }
}

// ── Staged GGUF k-quant GEMM ───────────────────────────────────────────────
//
// Same shape as the packed-quant path above, for the block layout GGUF uses:
// one buffer of 256-element superblocks per output row, each carrying a scale,
// a minimum, eight pairs of six-bit sub-scales and 128 bytes of nibbles. The
// block maths mirrors ggml's dequantize_row_q4_K.

#define QK_K 256

// A taller tile than the packed-quant path uses: each staged tile of the weight
// then serves twice as many rows of the activation, which is the ratio that
// decides how much the dequantisation costs per multiply.
constant constexpr int Q4K_TM = 128;
constant constexpr int Q4K_BK = 64;
constant constexpr int Q6K_BK = 128;
constant constexpr int Q4K_TN = 64;

typedef struct {
    half     d;
    half     dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

typedef struct {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    half     d;
} block_q6_K;

inline void q4k_scale_min(device const uint8_t* scales, uint j,
                          thread float& sc, thread float& m);

// The remaining GGUF block layouts, as ggml lays them out. The blocks of 32
// carry one scale, and for the `_1` variants one offset, ahead of their
// quants; the K blocks of 256 carry sub-block scales in their own packings.
// Several are not a multiple of four bytes long, so every read below that is
// not known to be aligned goes through packed bytes.
typedef struct { half d; uint8_t qs[16]; } block_q4_0;
typedef struct { half d; half m; uint8_t qs[16]; } block_q4_1;
typedef struct { half d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0;
typedef struct { half d; half m; uint8_t qh[4]; uint8_t qs[16]; } block_q5_1;
typedef struct { half d; int8_t qs[32]; } block_q8_0;
typedef struct { uint8_t scales[16]; uint8_t qs[64]; half d; half dmin; } block_q2_K;
typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; half d; } block_q3_K;
typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;

inline constexpr uint block_elems(device const block_q4_0*) { return 32u; }
inline constexpr uint block_elems(device const block_q4_1*) { return 32u; }
inline constexpr uint block_elems(device const block_q5_0*) { return 32u; }
inline constexpr uint block_elems(device const block_q5_1*) { return 32u; }
inline constexpr uint block_elems(device const block_q8_0*) { return 32u; }
inline constexpr uint block_elems(device const block_q2_K*) { return QK_K; }
inline constexpr uint block_elems(device const block_q3_K*) { return QK_K; }
inline constexpr uint block_elems(device const block_q5_K*) { return QK_K; }

inline uint le_word(device const uint8_t* b) {
    return uint(b[0]) | (uint(b[1]) << 8) | (uint(b[2]) << 16) | (uint(b[3]) << 24);
}

inline uchar4 ld4u(device const uint8_t* p) {
    return uchar4(*(device const packed_uchar4*)p);
}

inline char4 ld4s(device const int8_t* p) {
    return char4(*(device const packed_char4*)p);
}

// Each `dequant32` writes the weights `j0 .. j0 + limit` of one block to
// `out[i * stride]`, which is one column of the staged tile. The bytes come in
// four at a time through packed loads, which take any address: several of
// these blocks are 18, 22, 34 or 110 bytes long, so a four-byte load at a
// block boundary is unaligned every other block. A block of 32 is one span,
// so `j0` is zero for those and the span is the block.
inline void dequant32(device const block_q4_0* blk, uint j0, threadgroup bfloat* out, uint stride, uint limit) {
    const float d = float(blk->d);
    for (uint c = 0; c < 4u; ++c) {
        const uchar4 b = ld4u(blk->qs + 4u * c);
        for (uint i = 0; i < 4u; ++i) {
            const uint lo = 4u * c + i;
            const uint hi = lo + 16u;
            if (lo < limit) {
                out[lo * stride] = bfloat(d * (float(b[i] & 0xFu) - 8.0f));
            }
            if (hi < limit) {
                out[hi * stride] = bfloat(d * (float(b[i] >> 4) - 8.0f));
            }
        }
    }
}

inline void dequant32(device const block_q4_1* blk, uint j0, threadgroup bfloat* out, uint stride, uint limit) {
    const float d = float(blk->d);
    const float m = float(blk->m);
    for (uint c = 0; c < 4u; ++c) {
        const uchar4 b = ld4u(blk->qs + 4u * c);
        for (uint i = 0; i < 4u; ++i) {
            const uint lo = 4u * c + i;
            const uint hi = lo + 16u;
            if (lo < limit) {
                out[lo * stride] = bfloat(d * float(b[i] & 0xFu) + m);
            }
            if (hi < limit) {
                out[hi * stride] = bfloat(d * float(b[i] >> 4) + m);
            }
        }
    }
}

inline void dequant32(device const block_q5_0* blk, uint j0, threadgroup bfloat* out, uint stride, uint limit) {
    const float d = float(blk->d);
    const uint qh = le_word(blk->qh);
    for (uint c = 0; c < 4u; ++c) {
        const uchar4 b = ld4u(blk->qs + 4u * c);
        for (uint i = 0; i < 4u; ++i) {
            const uint lo = 4u * c + i;
            const uint hi = lo + 16u;
            if (lo < limit) {
                const uint q = (b[i] & 0xFu) | (((qh >> lo) & 1u) << 4);
                out[lo * stride] = bfloat(d * (float(q) - 16.0f));
            }
            if (hi < limit) {
                const uint q = (b[i] >> 4) | (((qh >> hi) & 1u) << 4);
                out[hi * stride] = bfloat(d * (float(q) - 16.0f));
            }
        }
    }
}

inline void dequant32(device const block_q5_1* blk, uint j0, threadgroup bfloat* out, uint stride, uint limit) {
    const float d = float(blk->d);
    const float m = float(blk->m);
    const uint qh = le_word(blk->qh);
    for (uint c = 0; c < 4u; ++c) {
        const uchar4 b = ld4u(blk->qs + 4u * c);
        for (uint i = 0; i < 4u; ++i) {
            const uint lo = 4u * c + i;
            const uint hi = lo + 16u;
            if (lo < limit) {
                const uint q = (b[i] & 0xFu) | (((qh >> lo) & 1u) << 4);
                out[lo * stride] = bfloat(d * float(q) + m);
            }
            if (hi < limit) {
                const uint q = (b[i] >> 4) | (((qh >> hi) & 1u) << 4);
                out[hi * stride] = bfloat(d * float(q) + m);
            }
        }
    }
}

inline void dequant32(device const block_q8_0* blk, uint j0, threadgroup bfloat* out, uint stride, uint limit) {
    const float d = float(blk->d);
    for (uint c = 0; c < 8u; ++c) {
        const char4 b = ld4s(blk->qs + 4u * c);
        for (uint i = 0; i < 4u; ++i) {
            const uint e = 4u * c + i;
            if (e < limit) {
                out[e * stride] = bfloat(d * float(b[i]));
            }
        }
    }
}

// Q2_K: two bits per weight, four groups of thirty-two per half of the
// block sharing a byte with a shift of two per group; a four-bit scale and a
// four-bit offset per sixteen weights.
inline void dequant32(device const block_q2_K* blk, uint j0, threadgroup bfloat* out, uint stride, uint limit) {
    const float d = float(blk->d);
    const float dmin = float(blk->dmin);
    const uint h = j0 / 128u;
    const uint jj = (j0 % 128u) / 32u;
    const uint shift = 2u * jj;
    const uint sc0 = uint(blk->scales[h * 8u + 2u * jj]);
    const uint sc1 = uint(blk->scales[h * 8u + 2u * jj + 1u]);
    const float dl0 = d * float(sc0 & 0xFu);
    const float ml0 = dmin * float(sc0 >> 4);
    const float dl1 = d * float(sc1 & 0xFu);
    const float ml1 = dmin * float(sc1 >> 4);
    device const uint8_t* q = blk->qs + h * 32u;
    for (uint c = 0; c < 8u; ++c) {
        const uchar4 b = ld4u(q + 4u * c);
        const bool first = c < 4u;
        const float dl = first ? dl0 : dl1;
        const float ml = first ? ml0 : ml1;
        for (uint i = 0; i < 4u; ++i) {
            const uint e = 4u * c + i;
            if (e < limit) {
                out[e * stride] = bfloat(dl * float((b[i] >> shift) & 3u) - ml);
            }
        }
    }
}

// Q3_K: two low bits in `qs` as Q2_K lays them, a third bit in `hmask` at
// bit `4 * half + group` of the weight's byte, and sixteen six-bit scales
// packed into twelve bytes the way ggml unpacks them.
inline void dequant32(device const block_q3_K* blk, uint j0, threadgroup bfloat* out, uint stride, uint limit) {
    const float d = float(blk->d);
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    uint aux0 = le_word(blk->scales);
    uint aux1 = le_word(blk->scales + 4);
    const uint tmp = le_word(blk->scales + 8);
    const uint aux2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    const uint aux3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux0 = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const uint aux[4] = {aux0, aux1, aux2, aux3};
    const uint h = j0 / 128u;
    const uint jj = (j0 % 128u) / 32u;
    const uint shift = 2u * jj;
    const uint m = 1u << (h * 4u + jj);
    const uint is0 = h * 8u + 2u * jj;
    const float s0 = d * float(int((aux[is0 / 4u] >> (8u * (is0 % 4u))) & 0xFFu) - 32);
    const float s1 = d * float(int((aux[(is0 + 1u) / 4u] >> (8u * ((is0 + 1u) % 4u))) & 0xFFu) - 32);
    device const uint8_t* q = blk->qs + h * 32u;
    for (uint c = 0; c < 8u; ++c) {
        const uchar4 b = ld4u(q + 4u * c);
        const uchar4 hm = ld4u(blk->hmask + 4u * c);
        const float sc = c < 4u ? s0 : s1;
        for (uint i = 0; i < 4u; ++i) {
            const uint e = 4u * c + i;
            if (e < limit) {
                const int low = int((b[i] >> shift) & 3u);
                const int val = low - ((hm[i] & m) ? 0 : 4);
                out[e * stride] = bfloat(sc * float(val));
            }
        }
    }
}

// Q5_K: Q4_K's nibbles and scales, with a fifth bit per weight in `qh` at bit
// `2 * span + half` of the weight's byte.
inline void dequant32(device const block_q5_K* blk, uint j0, threadgroup bfloat* out, uint stride, uint limit) {
    const uint g = j0 / 64u;
    const uint half_ = (j0 % 64u) / 32u;
    float sc, mn;
    q4k_scale_min(blk->scales, 2u * g + half_, sc, mn);
    const float dl = float(blk->d) * sc;
    const float ml = float(blk->dmin) * mn;
    const uint u = 1u << (2u * g + half_);
    const uint shift = half_ * 4u;
    device const uint8_t* ql = blk->qs + g * 32u;
    for (uint c = 0; c < 8u; ++c) {
        const uchar4 b = ld4u(ql + 4u * c);
        const uchar4 hq = ld4u(blk->qh + 4u * c);
        for (uint i = 0; i < 4u; ++i) {
            const uint e = 4u * c + i;
            if (e < limit) {
                const uint q = ((b[i] >> shift) & 0xFu) + ((hq[i] & u) ? 16u : 0u);
                out[e * stride] = bfloat(dl * float(q) - ml);
            }
        }
    }
}

// The importance-quantized types, decoded as ggml's `dequantize_row_iq*` do.
// candle has none of them, so the CPU reference in `iq_quant.rs` is what the
// kernels are checked against, and the tables below are copied from
// `ggml-common.h` verbatim: the sixteen values a four-bit IQ4 code stands
// for, the bit each sign position occupies, and the 512 four-weight entries
// an IQ3_S code selects, four bytes per entry, low byte first.
typedef struct { half d; uint8_t qs[16]; } block_iq4_nl;
typedef struct { half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; } block_iq4_xs;
typedef struct { half d; uint8_t qs[64]; uint8_t qh[8]; uint8_t signs[32]; uint8_t scales[4]; } block_iq3_s;

inline constexpr uint block_elems(device const block_iq4_nl*) { return 32u; }
inline constexpr uint block_elems(device const block_iq4_xs*) { return QK_K; }
inline constexpr uint block_elems(device const block_iq3_s*) { return QK_K; }

constant int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

constant uint8_t kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128,
};

constant uint32_t iq3s_grid[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu, 0x0101010fu, 0x01010301u,
    0x01010303u, 0x01010305u, 0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u,
    0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u, 0x0101090bu, 0x0101090fu,
    0x01010b03u, 0x01010b07u, 0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u,
    0x01030303u, 0x0103030bu, 0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u,
    0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu, 0x01030f05u, 0x01050101u,
    0x01050103u, 0x0105010bu, 0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090bu,
    0x0105090fu, 0x01050b03u, 0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u,
    0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u, 0x01070703u, 0x01070707u,
    0x0107070du, 0x01070909u, 0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu, 0x01090503u, 0x01090509u,
    0x01090705u, 0x01090901u, 0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u,
    0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du, 0x010b0707u, 0x010b0903u,
    0x010b090bu, 0x010b090fu, 0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u, 0x010f0101u, 0x010f0105u,
    0x010f0109u, 0x010f0501u, 0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u,
    0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u, 0x03010109u, 0x03010301u,
    0x03010303u, 0x03010307u, 0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u, 0x03010b0du, 0x03010d03u,
    0x03010f05u, 0x03030101u, 0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u,
    0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u, 0x03030903u, 0x03030b01u,
    0x03030b05u, 0x03030f01u, 0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u,
    0x03050b0bu, 0x03050d01u, 0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu,
    0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu, 0x03070701u, 0x03070709u,
    0x03070903u, 0x03070d05u, 0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090du, 0x03090b01u,
    0x03090b09u, 0x030b0103u, 0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u,
    0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u, 0x030d050fu, 0x030d0909u,
    0x030d090du, 0x030f0103u, 0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u, 0x05010101u, 0x05010103u,
    0x05010107u, 0x0501010bu, 0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u,
    0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu, 0x05010701u, 0x05010705u,
    0x05010903u, 0x05010907u, 0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u, 0x05030105u, 0x05030301u,
    0x05030307u, 0x0503030fu, 0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u,
    0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u, 0x0505010fu, 0x05050503u,
    0x05050507u, 0x05050701u, 0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u, 0x0507010bu, 0x05070303u,
    0x05070505u, 0x05070509u, 0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u,
    0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u, 0x05090507u, 0x05090705u,
    0x0509070bu, 0x05090903u, 0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u, 0x050b0f01u, 0x050d0101u,
    0x050d0105u, 0x050d010fu, 0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu,
    0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u, 0x050f0b01u, 0x07010105u,
    0x07010303u, 0x07010307u, 0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u, 0x0701090fu, 0x07010b03u,
    0x07010d07u, 0x07010f03u, 0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u,
    0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u, 0x07030f05u, 0x07030f0du,
    0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050fu,
    0x07070701u, 0x07070903u, 0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u,
    0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u, 0x07090703u, 0x07090b05u,
    0x07090d01u, 0x07090d09u, 0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u, 0x070d030du, 0x070d0903u,
    0x070f0103u, 0x070f0107u, 0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u,
    0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u, 0x0901050fu, 0x09010705u,
    0x09010903u, 0x09010b01u, 0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu, 0x09030907u, 0x09030b03u,
    0x09030b0bu, 0x09050103u, 0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u,
    0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u, 0x09050f01u, 0x09070109u,
    0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu, 0x09090901u, 0x09090f03u,
    0x090b010bu, 0x090b010fu, 0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u,
    0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u, 0x090f0907u, 0x090f0b03u,
    0x0b010105u, 0x0b010301u, 0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u, 0x0b030103u, 0x0b030107u,
    0x0b03010bu, 0x0b030305u, 0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u,
    0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du, 0x0b050b07u, 0x0b070105u,
    0x0b07010fu, 0x0b070301u, 0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u, 0x0b090705u, 0x0b09090du,
    0x0b0b0305u, 0x0b0b050du, 0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u,
    0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u, 0x0d01030bu, 0x0d010703u,
    0x0d010707u, 0x0d010d01u, 0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu, 0x0d050d05u, 0x0d050f01u,
    0x0d070101u, 0x0d070309u, 0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u,
    0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u, 0x0d0b0d01u, 0x0d0d010bu,
    0x0d0d0901u, 0x0d0f0303u, 0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u, 0x0f010b09u, 0x0f010d05u,
    0x0f030105u, 0x0f030303u, 0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u,
    0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u, 0x0f050701u, 0x0f050b03u,
    0x0f070105u, 0x0f070705u, 0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u, 0x0f0b0905u, 0x0f0d0105u,
    0x0f0d0703u, 0x0f0f0101u,
};

// Where a decoded weight lands: a staged tile in threadgroup memory, or the
// registers of a lane doing a dot product.
inline void put(threadgroup bfloat* out, float v) { *out = bfloat(v); }
inline void put(thread float* out, float v) { *out = v; }

template <typename Out>
inline void dequant32(device const block_iq4_nl* blk, uint j0, Out out, uint stride, uint limit) {
    const float d = float(blk->d);
    for (uint c = 0; c < 4u; ++c) {
        const uchar4 b = ld4u(blk->qs + 4u * c);
        for (uint i = 0; i < 4u; ++i) {
            const uint lo = 4u * c + i;
            const uint hi = lo + 16u;
            if (lo < limit) {
                put(out + lo * stride, d * float(kvalues_iq4nl[b[i] & 0xFu]));
            }
            if (hi < limit) {
                put(out + hi * stride, d * float(kvalues_iq4nl[b[i] >> 4]));
            }
        }
    }
}

template <typename Out>
inline void dequant32(device const block_iq4_xs* blk, uint j0, Out out, uint stride, uint limit) {
    const float d = float(blk->d);
    const uint ib = j0 / 32u;
    const uint scales_h = uint(blk->scales_h);
    const int ls = int((uint(blk->scales_l[ib / 2u]) >> (4u * (ib % 2u))) & 0xFu)
                 | int(((scales_h >> (2u * ib)) & 3u) << 4);
    const float dl = d * float(ls - 32);
    device const uint8_t* qs = blk->qs + ib * 16u;
    for (uint c = 0; c < 4u; ++c) {
        const uchar4 b = ld4u(qs + 4u * c);
        for (uint i = 0; i < 4u; ++i) {
            const uint lo = 4u * c + i;
            const uint hi = lo + 16u;
            if (lo < limit) {
                put(out + lo * stride, dl * float(kvalues_iq4nl[b[i] & 0xFu]));
            }
            if (hi < limit) {
                put(out + hi * stride, dl * float(kvalues_iq4nl[b[i] >> 4]));
            }
        }
    }
}

// IQ3_S: a span of 32 weights is four groups of eight; each group takes two
// bytes of `qs`, each extended to nine bits by one bit of the span's `qh`
// byte, into the grid, with the signs of its eight weights in one byte.
template <typename Out>
inline void dequant32(device const block_iq3_s* blk, uint j0, Out out, uint stride, uint limit) {
    const float d = float(blk->d);
    const uint ib32 = j0 / 32u;
    const uint sc = uint(blk->scales[ib32 / 2u]);
    const float db = d * float(1 + 2 * int((ib32 % 2u) == 0u ? (sc & 0xFu) : (sc >> 4)));
    const uint qh = uint(blk->qh[ib32]);
    device const uint8_t* qs = blk->qs + ib32 * 8u;
    device const uint8_t* signs = blk->signs + ib32 * 4u;
    for (uint l = 0; l < 4u; ++l) {
        const uint i1 = uint(qs[2u * l]) | ((qh << (8u - 2u * l)) & 256u);
        const uint i2 = uint(qs[2u * l + 1u]) | ((qh << (7u - 2u * l)) & 256u);
        const uint g1 = iq3s_grid[i1];
        const uint g2 = iq3s_grid[i2];
        const uint s = uint(signs[l]);
        for (uint j = 0; j < 4u; ++j) {
            const uint e1 = 8u * l + j;
            const uint e2 = e1 + 4u;
            if (e1 < limit) {
                const float w = float((g1 >> (8u * j)) & 0xFFu);
                put(out + e1 * stride, db * ((s & kmask_iq2xs[j]) ? -w : w));
            }
            if (e2 < limit) {
                const float w = float((g2 >> (8u * j)) & 0xFFu);
                put(out + e2 * stride, db * ((s & kmask_iq2xs[j + 4u]) ? -w : w));
            }
        }
    }
}

// The stager every block type without a hand-tuned one shares: one thread
// per column and thirty-two-weight span of the tile, as the Q4_K and Q6_K
// stagers do, with the block's own `dequant32` filling the span.
template <typename Block>
inline void stage_tile(
    device Block* weight,
    threadgroup bfloat* sB,
    constant MppQuantGemmParams& p,
    int k0, int col0, int bk, int tn, int tn_stride, uint lid)
{
    const uint be = block_elems(weight);
    const uint blocks_per_row = uint(p.k) / be;
    const uint subs = (uint(bk) + 31u) / 32u;
    const uint total = uint(tn) * subs;
    for (uint t = lid; t < total; t += 128u) {
        const uint col = t / subs;
        const uint sub = t % subs;
        const uint k_start = uint(k0) + sub * 32u;
        device const Block* blk = weight + (uint(col0) + col) * blocks_per_row + k_start / be;
        const uint j0 = k_start % be;
        const int limit = min(32, bk - int(sub * 32u));
        dequant32(blk, j0, sB + sub * 32u * uint(tn_stride) + col, uint(tn_stride), uint(limit));
    }
}

inline void q4k_scale_min(device const uint8_t* scales, uint j,
                          thread float& sc, thread float& m)
{
    if (j < 4u) {
        sc = float(scales[j] & 63u);
        m  = float(scales[j + 4u] & 63u);
    } else {
        sc = float((scales[j + 4u] & 0xFu) | ((scales[j - 4u] >> 6) << 4));
        m  = float((scales[j + 4u] >> 4) | ((scales[j] >> 6) << 4));
    }
}

// Each thread of the threadgroup dequantizes one column of the tile over 32
// consecutive k, which is the span one Q4_K scale covers and one quarter of
// the span a Q6_K half-block interleaves, so a thread never crosses a scale.
inline void stage_tile(
    device block_q4_K* weight,
    threadgroup bfloat* sB,
    constant MppQuantGemmParams& p,
    int k0, int col0, int bk, int tn, int tn_stride, uint lid)
{
    uint blocks_per_row = uint(p.k) / QK_K;
    uint subs = (uint(bk) + 31u) / 32u;
    uint total = uint(tn) * subs;
    for (uint t = lid; t < total; t += 128u) {
        uint col = t / subs;
        uint sub = t % subs;
        uint k_start = uint(k0) + sub * 32u;
        device const block_q4_K* blk =
            weight + (uint(col0) + col) * blocks_per_row + k_start / QK_K;
        uint j0 = k_start % QK_K;

        float sc, m;
        q4k_scale_min(blk->scales, j0 / 32u, sc, m);
        float dl = float(blk->d) * sc;
        float ml = float(blk->dmin) * m;

        device const uint* qs = (device const uint*)(blk->qs + (j0 / 64u) * 32u);
        const uint shift = ((j0 % 64u) / 32u) * 4u;
        for (uint l = 0; l < 32u; l += 4u) {
            uint kk = sub * 32u + l;
            if (int(kk) >= bk) {
                break;
            }
            const uint word = qs[l / 4u];
            threadgroup bfloat* out = sB + kk * uint(tn_stride) + col;
            out[0] = bfloat(dl * float((word >> shift) & 0xFu) - ml);
            out[tn_stride] = bfloat(dl * float((word >> (shift + 8u)) & 0xFu) - ml);
            out[2 * tn_stride] = bfloat(dl * float((word >> (shift + 16u)) & 0xFu) - ml);
            out[3 * tn_stride] = bfloat(dl * float((word >> (shift + 24u)) & 0xFu) - ml);
        }
    }
}

// A Q6_K block holds two halves of 128 weights. Within a half, weight
// g * 32 + l takes its low four bits from ql[(g & 1) * 32 + l], high or low
// nibble by g >> 1, its top two bits from qh[l] shifted by 2 * g, and its
// scale from scales[2 * g + l / 16], all offset by 32.
inline void stage_tile(
    device block_q6_K* weight,
    threadgroup bfloat* sB,
    constant MppQuantGemmParams& p,
    int k0, int col0, int bk, int tn, int tn_stride, uint lid)
{
    uint blocks_per_row = uint(p.k) / QK_K;
    uint subs = (uint(bk) + 31u) / 32u;
    uint total = uint(tn) * subs;
    for (uint t = lid; t < total; t += 128u) {
        uint col = t / subs;
        uint sub = t % subs;
        uint k_start = uint(k0) + sub * 32u;
        device const block_q6_K* blk =
            weight + (uint(col0) + col) * blocks_per_row + k_start / QK_K;
        uint j0 = k_start % QK_K;
        uint h = j0 / 128u;
        uint g = (j0 % 128u) / 32u;

        float d = float(blk->d);
        // Packed loads: a block is 210 bytes, so every other one sits at an
        // address a four-byte load cannot take.
        device const packed_uchar4* ql = (device const packed_uchar4*)(blk->ql + h * 64u + (g & 1u) * 32u);
        device const packed_uchar4* qh = (device const packed_uchar4*)(blk->qh + h * 32u);
        device const int8_t* sc = blk->scales + h * 8u + 2u * g;
        uint shift_l = (g >> 1) * 4u;
        uint shift_h = 2u * g;
        for (uint l = 0; l < 32u; l += 4u) {
            uint kk = sub * 32u + l;
            if (int(kk) >= bk) {
                break;
            }
            const packed_uchar4 lo = ql[l / 4u];
            const packed_uchar4 hi = qh[l / 4u];
            const float ds = d * float(sc[l / 16u]);
            threadgroup bfloat* out = sB + kk * uint(tn_stride) + col;
            const uchar4 lo4 = uchar4(lo);
            const uchar4 hi4 = uchar4(hi);
            for (uint i = 0; i < 4u; ++i) {
                uint low = (uint(lo4[i]) >> shift_l) & 0xFu;
                uint top = (uint(hi4[i]) >> shift_h) & 3u;
                out[i * tn_stride] = bfloat(ds * float(int(low | (top << 4)) - 32));
            }
        }
    }
}

template <typename Block, int TNS, int BKS>
inline void gemm_staged(
    device bfloat* a,
    device Block* weight,
    device bfloat* d,
    constant MppQuantGemmParams& p,
    threadgroup bfloat* sB0,
    threadgroup bfloat* sB1,
    uint2 tgid, uint lid)
{
    constexpr auto desc = matmul2d_descriptor(
        Q4K_TM, TNS, BKS, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    int row0 = int(tgid.y) * Q4K_TM;
    int col0 = int(tgid.x) * TNS;
    int tm = min(Q4K_TM, p.m - row0);
    int tn = min(TNS, p.n - col0);
    if (tm <= 0 || tn <= 0) {
        return;
    }

    auto cT = op.template get_destination_cooperative_tensor<
        tensor<device bfloat, dextents<int, 2>, tensor_inline>,
        tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline>,
        float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    stage_tile(weight, sB0, p, 0, col0, min(BKS, p.k), tn, TNS, lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int slot = 0;
    for (int k0 = 0; k0 < p.k; k0 += BKS) {
        int bk = min(BKS, p.k - k0);
        int k1 = k0 + BKS;
        threadgroup bfloat* cur = slot == 0 ? sB0 : sB1;
        threadgroup bfloat* nxt = slot == 0 ? sB1 : sB0;
        if (k1 < p.k) {
            stage_tile(weight, nxt, p, k1, col0, min(BKS, p.k - k1), tn, TNS, lid);
        }

        auto tA = tensor(a + row0 * p.k + k0, dextents<int, 2>{bk, tm}, array<int, 2>{1, p.k});
        auto tB = tensor(cur, dextents<int, 2>{tn, bk}, array<int, 2>{1, TNS});
        op.run(tA, tB, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        slot = 1 - slot;
    }

#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            int nn = int(idx[0]);
            int mm = int(idx[1]);
            if (nn < tn && mm < tm) {
                d[(row0 + mm) * p.n + col0 + nn] = bfloat(cT[i]);
            }
        }
    }
}

kernel void mpp_gemm_q4k_staged(
    device bfloat*      a       [[buffer(0)]],
    device block_q4_K*  weight  [[buffer(1)]],
    device bfloat*      d       [[buffer(2)]],
    constant MppQuantGemmParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
    threadgroup bfloat sB[2][Q4K_BK * Q4K_TN];
    gemm_staged<block_q4_K, Q4K_TN, Q4K_BK>(a, weight, d, p, sB[0], sB[1], tgid, lid);
}

#define MPP_STAGED_KERNEL(NAME, BLOCK, BK)                                      \
kernel void NAME(                                                              \
    device bfloat*      a       [[buffer(0)]],                                 \
    device BLOCK*       weight  [[buffer(1)]],                                 \
    device bfloat*      d       [[buffer(2)]],                                 \
    constant MppQuantGemmParams& p [[buffer(3)]],                              \
    uint2 tgid [[threadgroup_position_in_grid]],                               \
    uint  lid  [[thread_index_in_threadgroup]])                                \
{                                                                              \
    threadgroup bfloat sB[2][BK * Q4K_TN];                                     \
    gemm_staged<BLOCK, Q4K_TN, BK>(a, weight, d, p, sB[0], sB[1], tgid, lid);  \
}

MPP_STAGED_KERNEL(mpp_gemm_q4_0_staged, block_q4_0, Q4K_BK)
MPP_STAGED_KERNEL(mpp_gemm_q4_1_staged, block_q4_1, Q4K_BK)
MPP_STAGED_KERNEL(mpp_gemm_q5_0_staged, block_q5_0, Q4K_BK)
MPP_STAGED_KERNEL(mpp_gemm_q5_1_staged, block_q5_1, Q4K_BK)
MPP_STAGED_KERNEL(mpp_gemm_q8_0_staged, block_q8_0, Q4K_BK)
MPP_STAGED_KERNEL(mpp_gemm_q2k_staged, block_q2_K, Q4K_BK)
MPP_STAGED_KERNEL(mpp_gemm_q3k_staged, block_q3_K, Q4K_BK)
MPP_STAGED_KERNEL(mpp_gemm_q5k_staged, block_q5_K, Q4K_BK)
MPP_STAGED_KERNEL(mpp_gemm_iq4_nl_staged, block_iq4_nl, Q4K_BK)
MPP_STAGED_KERNEL(mpp_gemm_iq4_xs_staged, block_iq4_xs, Q4K_BK)
MPP_STAGED_KERNEL(mpp_gemm_iq3_s_staged, block_iq3_s, Q4K_BK)

kernel void mpp_gemm_q6k_staged(
    device bfloat*      a       [[buffer(0)]],
    device block_q6_K*  weight  [[buffer(1)]],
    device bfloat*      d       [[buffer(2)]],
    constant MppQuantGemmParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
    threadgroup bfloat sB[2][Q6K_BK * Q4K_TN];
    gemm_staged<block_q6_K, Q4K_TN, Q6K_BK>(a, weight, d, p, sB[0], sB[1], tgid, lid);
}

// Activation rows the matvec takes at once, and output rows per threadgroup.
#define MV_MAX_M 8
#define MV_ROWS 4

// One simdgroup per output row for a weight candle cannot serve: each lane
// takes a span of thirty-two weights, decodes it into registers with the
// block's own `dequant32`, and dots it with the same span of every activation
// row; the lanes' partial sums meet in a simd reduction. It reads the weight
// once, which is what a matvec is bound by.
template <typename Block>
inline void matvec_owned(
    device const bfloat* a,
    device const Block* weight,
    device bfloat* d,
    constant MppQuantGemmParams& p,
    uint tg, uint sg, uint lane)
{
    const uint row = tg * MV_ROWS + sg;
    if (row >= uint(p.n)) {
        return;
    }
    constexpr uint BE = block_elems((device const Block*)0);
    constexpr uint SPB = BE / 32u;
    const uint k = uint(p.k);
    const uint m = uint(p.m);
    const uint spans = k / 32u;
    device const Block* wrow = weight + row * (k / BE);
    float acc[MV_MAX_M];
    for (uint r = 0; r < MV_MAX_M; ++r) {
        acc[r] = 0.0f;
    }
    for (uint s = lane; s < spans; s += 32u) {
        float w[32];
        dequant32(wrow + s / SPB, (s % SPB) * 32u, w, 1u, 32u);
        device const bfloat* xa = a + s * 32u;
        for (uint r = 0; r < m; ++r) {
            device const bfloat* xr = xa + r * k;
            float sum = 0.0f;
            for (uint i = 0; i < 32u; ++i) {
                sum += w[i] * float(xr[i]);
            }
            acc[r] += sum;
        }
    }
    for (uint r = 0; r < m; ++r) {
        const float v = simd_sum(acc[r]);
        if (lane == 0u) {
            d[r * uint(p.n) + row] = bfloat(v);
        }
    }
}

#define MPP_MATVEC_KERNEL(NAME, BLOCK)                                         \
kernel void NAME(                                                              \
    device const bfloat* a       [[buffer(0)]],                                \
    device const BLOCK*  weight  [[buffer(1)]],                                \
    device bfloat*       d       [[buffer(2)]],                                \
    constant MppQuantGemmParams& p [[buffer(3)]],                              \
    uint tg   [[threadgroup_position_in_grid]],                                \
    uint sg   [[simdgroup_index_in_threadgroup]],                              \
    uint lane [[thread_index_in_simdgroup]])                                   \
{                                                                              \
    matvec_owned<BLOCK>(a, weight, d, p, tg, sg, lane);                        \
}

MPP_MATVEC_KERNEL(mpp_mv_iq4_nl, block_iq4_nl)
MPP_MATVEC_KERNEL(mpp_mv_iq4_xs, block_iq4_xs)
MPP_MATVEC_KERNEL(mpp_mv_iq3_s, block_iq3_s)

template<uint BITS, bool GPTQ>
inline void mpp_gemm_quant_impl(
    device bfloat*  a,
    device uint*    qweight,
    device uint*    qzeros,
    device bfloat*  scales,
    device bfloat*  d,
    constant MppQuantGemmParams& p,
    threadgroup bfloat* sB,
    uint2 tgid, uint lid)
{
    constexpr auto desc = matmul2d_descriptor(
        TM, TN, QBK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    int row0 = int(tgid.y) * TM;
    int col0 = int(tgid.x) * TN;
    int tm = min(TM, p.m - row0);
    int tn = min(TN, p.n - col0);
    if (tm <= 0 || tn <= 0) {
        return;
    }

    auto cT = op.get_destination_cooperative_tensor<
        tensor<device bfloat, dextents<int, 2>, tensor_inline>,
        tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline>,
        float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < p.k; k0 += QBK) {
        int bk = min(QBK, p.k - k0);
        if (GPTQ) {
            gptq_stage_tile<BITS>(qweight, qzeros, scales, sB, p, k0, col0, bk, tn, lid);
        } else {
            awq_stage_tile<BITS>(qweight, qzeros, scales, sB, p, k0, col0, bk, tn, lid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tA = tensor(a + row0 * p.k + k0, dextents<int, 2>{bk, tm}, array<int, 2>{1, p.k});
        auto tB = tensor(sB, dextents<int, 2>{tn, bk}, array<int, 2>{1, TN});
        op.run(tA, tB, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            int nn = int(idx[0]);
            int mm = int(idx[1]);
            if (nn < tn && mm < tm) {
                d[(row0 + mm) * p.n + col0 + nn] = bfloat(cT[i]);
            }
        }
    }
}

#define MPP_QUANT_KERNEL(NAME, BITS, GPTQ)                                     \
kernel void NAME(                                                              \
    device bfloat*  a       [[buffer(0)]],                                     \
    device uint*    qweight [[buffer(1)]],                                     \
    device uint*    qzeros  [[buffer(2)]],                                     \
    device bfloat*  scales  [[buffer(3)]],                                     \
    device bfloat*  d       [[buffer(4)]],                                     \
    constant MppQuantGemmParams& p [[buffer(5)]],                              \
    uint2 tgid [[threadgroup_position_in_grid]],                               \
    uint  lid  [[thread_index_in_threadgroup]])                                \
{                                                                              \
    threadgroup bfloat sB[QBK * TN];                                           \
    mpp_gemm_quant_impl<BITS, GPTQ>(a, qweight, qzeros, scales, d, p, sB, tgid, lid); \
}

MPP_QUANT_KERNEL(mpp_gemm_w4_staged, 4, false)
MPP_QUANT_KERNEL(mpp_gemm_w8_staged, 8, false)
MPP_QUANT_KERNEL(mpp_gemm_gptq4_staged, 4, true)
MPP_QUANT_KERNEL(mpp_gemm_gptq8_staged, 8, true)

// ── FlashAttention prefill ──────────────────────────────────────────────────
//
// One simdgroup per (batch·head, 32-row Q block). S = Q·Kᵀ runs on matmul2d
// with a float cooperative-tensor destination; the online softmax keeps the
// running row max/denominator in threadgroup scratch (indexed by the element
// coordinates of the cooperative tensor); P and the V tile are staged in
// threadgroup memory zero-padded to the static-K tile so P·V can accumulate
// into the output cooperative tensor. Causal mask shifted by prefix_len,
// GQA-native (no KV repeat).

// `kv_head_stride` is how far apart two KV heads sit, in elements. It is
// t_kv * D for a packed tensor and larger for a view into a cache that holds
// more tokens than this call reads, which is what lets the cache hand its
// window over without copying it.
struct MppFaParams {
    int t_q;
    int t_kv;
    int h;
    int h_kv;
    float scale;
    int prefix_len;
    int window;
    int kv_head_stride;
};

constant constexpr int FA_BR = 32;
constant constexpr int FA_BC = 32;

template<int D>
inline void mpp_fa_impl(
    device bfloat* q,
    device bfloat* k,
    device bfloat* v,
    device bfloat* o,
    constant MppFaParams& p,
    threadgroup float* tg_m,
    threadgroup float* tg_l,
    threadgroup float* tg_a,
    threadgroup float* tg_r,
    threadgroup bfloat* tg_p,
    threadgroup bfloat* tg_v,
    uint2 tgid,
    uint lane)
{
    constexpr auto desc_s = matmul2d_descriptor(FA_BR, FA_BC, D, false, true);
    matmul2d<desc_s, execution_simdgroup> op_s;
    constexpr auto desc_o = matmul2d_descriptor(
        FA_BR, D, FA_BC, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_o, execution_simdgroup> op_o;

    int q0 = int(tgid.x) * FA_BR;
    if (q0 >= p.t_q) {
        return;
    }
    int br = min(FA_BR, p.t_q - q0);
    int bh = int(tgid.y);
    int b = bh / p.h;
    int hh = bh % p.h;
    int hkv = hh / (p.h / p.h_kv);

    device bfloat* qp = q + ((size_t(b) * p.h + hh) * p.t_q + q0) * D;
    size_t kv_stride = size_t(p.kv_head_stride);
    device bfloat* kb = k + (size_t(b) * p.h_kv + hkv) * kv_stride;
    device bfloat* vb = v + (size_t(b) * p.h_kv + hkv) * kv_stride;

    auto tQ = tensor(qp, dextents<int, 2>{D, br}, array<int, 2>{1, D});
    auto tP = tensor(tg_p, dextents<int, 2>{FA_BC, FA_BR}, array<int, 2>{1, FA_BC});
    auto tV = tensor(tg_v, dextents<int, 2>{D, FA_BC}, array<int, 2>{1, D});

    auto sT = op_s.template get_destination_cooperative_tensor<decltype(tQ), decltype(tQ), float>();
    auto oT = op_o.template get_destination_cooperative_tensor<decltype(tP), decltype(tV), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oT.get_capacity(); ++i) {
        if (oT.is_valid_element(i)) {
            oT[i] = 0.0f;
        }
    }
    for (uint r = lane; r < uint(FA_BR); r += 32u) {
        tg_m[r] = -INFINITY;
        tg_l[r] = 0.0f;
    }

    int kv_max = min(p.t_kv, p.prefix_len + q0 + br);
    int kv_min = 0;
    if (p.window > 0) {
        int oldest = p.prefix_len + q0 - p.window + 1;
        if (oldest > 0) {
            kv_min = (oldest / FA_BC) * FA_BC;
        }
    }
    for (int kv0 = kv_min; kv0 < kv_max; kv0 += FA_BC) {
        int bc = min(FA_BC, p.t_kv - kv0);

#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sT.get_capacity(); ++i) {
            if (sT.is_valid_element(i)) {
                sT[i] = 0.0f;
            }
        }
        auto tK = tensor(kb + size_t(kv0) * D, dextents<int, 2>{D, bc}, array<int, 2>{1, D});
        op_s.run(tQ, tK, sT);

#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sT.get_capacity(); ++i) {
            if (sT.is_valid_element(i)) {
                auto idx = sT.get_multidimensional_index(i);
                int n = int(idx[0]);
                int m = int(idx[1]);
                float val = sT[i] * p.scale;
                int q_pos = p.prefix_len + q0 + m;
                int kv_pos = kv0 + n;
                if (kv_pos > q_pos || (p.window > 0 && q_pos >= kv_pos + p.window)) {
                    val = -INFINITY;
                }
                sT[i] = val;
            }
        }

        auto rT = op_s.template get_row_reduction_destination_cooperative_tensor<
            decltype(tQ), decltype(tQ), float>();
        reduce_rows(sT, rT, reduction_operation::max,
                    reduction_operation_identity<float>::max_identity);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < rT.get_capacity(); ++i) {
            if (rT.is_valid_element(i)) {
                auto idx = rT.get_multidimensional_index(i);
                tg_r[idx[0]] = rT[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = lane; r < uint(FA_BR); r += 32u) {
            float m_new = max(tg_m[r], tg_r[r]);
            tg_a[r] = (tg_m[r] == -INFINITY) ? 0.0f : exp(tg_m[r] - m_new);
            tg_m[r] = m_new;
        }
        for (uint i = lane; i < uint(FA_BR * FA_BC); i += 32u) {
            tg_p[i] = bfloat(0.0f);
        }
        for (uint i = lane; i < uint(FA_BC * D); i += 32u) {
            uint row = i / uint(D);
            tg_v[i] = (int(row) < bc) ? vb[size_t(kv0) * D + i] : bfloat(0.0f);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sT.get_capacity(); ++i) {
            if (sT.is_valid_element(i)) {
                auto idx = sT.get_multidimensional_index(i);
                int n = int(idx[0]);
                int m = int(idx[1]);
                float pv = (sT[i] == -INFINITY) ? 0.0f : exp(sT[i] - tg_m[m]);
                sT[i] = pv;
                tg_p[m * FA_BC + n] = bfloat(pv);
            }
        }

        auto rsT = op_s.template get_row_reduction_destination_cooperative_tensor<
            decltype(tQ), decltype(tQ), float>();
        reduce_rows(sT, rsT, reduction_operation::sum, 0.0f);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < rsT.get_capacity(); ++i) {
            if (rsT.is_valid_element(i)) {
                auto idx = rsT.get_multidimensional_index(i);
                tg_r[idx[0]] = rsT[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = lane; r < uint(FA_BR); r += 32u) {
            tg_l[r] = tg_l[r] * tg_a[r] + tg_r[r];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < oT.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) {
                auto idx = oT.get_multidimensional_index(i);
                oT[i] *= tg_a[idx[1]];
            }
        }
        op_o.run(tP, tV, oT);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oT.get_capacity(); ++i) {
        if (oT.is_valid_element(i)) {
            auto idx = oT.get_multidimensional_index(i);
            int dd = int(idx[0]);
            int m = int(idx[1]);
            if (m < br) {
                float denom = tg_l[m];
                float val = (denom > 0.0f) ? oT[i] / denom : 0.0f;
                o[((size_t(b) * p.h + hh) * p.t_q + q0 + m) * D + dd] = bfloat(val);
            }
        }
    }
}

// A head too wide for one pass: the output accumulator of a 32-query block at
// 512 wide is 64 KB of registers per simdgroup and the V tile 32 KB of
// threadgroup memory, twice what either can take. So the block is 16 queries,
// the head is walked in two halves of 256, and the two halves of the output
// accumulate side by side: the scores need the whole width and are summed
// over the halves, Q is staged once since every key block multiplies it, and
// K and V are read at half width where they lie through strided views. The
// key block is 128 wide, which measured best of 64, 128 and 256 on the shape
// of Gemma 4's global layers, the reason this exists: 512 wide, one sixth of
// the layers, and until now on the path that materialises every score, which
// this beats by 3.3x there.
template<int D, int BR, int BC>
inline void mpp_fa_wide_impl(
    device bfloat* q,
    device bfloat* k,
    device bfloat* v,
    device bfloat* o,
    constant MppFaParams& p,
    threadgroup float* tg_m,
    threadgroup float* tg_l,
    threadgroup float* tg_a,
    threadgroup float* tg_r,
    threadgroup bfloat* tg_p,
    threadgroup bfloat* tg_q,
    uint2 tgid,
    uint lane)
{
    constexpr int DC = D / 2;
    constexpr auto desc_s = matmul2d_descriptor(
        BR, BC, DC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_s, execution_simdgroup> op_s;
    constexpr auto desc_o = matmul2d_descriptor(
        BR, DC, BC, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_o, execution_simdgroup> op_o;

    int q0 = int(tgid.x) * BR;
    if (q0 >= p.t_q) {
        return;
    }
    int br = min(BR, p.t_q - q0);
    int bh = int(tgid.y);
    int b = bh / p.h;
    int hh = bh % p.h;
    int hkv = hh / (p.h / p.h_kv);

    device bfloat* qp = q + ((size_t(b) * p.h + hh) * p.t_q + q0) * D;
    size_t kv_stride = size_t(p.kv_head_stride);
    device bfloat* kb = k + (size_t(b) * p.h_kv + hkv) * kv_stride;
    device bfloat* vb = v + (size_t(b) * p.h_kv + hkv) * kv_stride;

    // Q is read once into threadgroup memory: every key block multiplies it.
    for (uint i = lane; i < uint(BR * D); i += 32u) {
        uint row = i / uint(D);
        tg_q[i] = (int(row) < br) ? qp[i] : bfloat(0.0f);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto tQ0 = tensor(tg_q, dextents<int, 2>{DC, br}, array<int, 2>{1, D});
    auto tQ1 = tensor(tg_q + DC, dextents<int, 2>{DC, br}, array<int, 2>{1, D});
    auto tP = tensor(tg_p, dextents<int, 2>{BC, BR}, array<int, 2>{1, BC});
    auto tVshape = tensor(vb, dextents<int, 2>{DC, BC}, array<int, 2>{1, D});

    auto sT = op_s.template get_destination_cooperative_tensor<decltype(tQ0), decltype(tQ0), float>();
    auto oT0 = op_o.template get_destination_cooperative_tensor<decltype(tP), decltype(tVshape), float>();
    auto oT1 = op_o.template get_destination_cooperative_tensor<decltype(tP), decltype(tVshape), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oT0.get_capacity(); ++i) {
        if (oT0.is_valid_element(i)) {
            oT0[i] = 0.0f;
            oT1[i] = 0.0f;
        }
    }
    for (uint r = lane; r < uint(BR); r += 32u) {
        tg_m[r] = -INFINITY;
        tg_l[r] = 0.0f;
    }

    int kv_max = min(p.t_kv, p.prefix_len + q0 + br);
    int kv_min = 0;
    if (p.window > 0) {
        int oldest = p.prefix_len + q0 - p.window + 1;
        if (oldest > 0) {
            kv_min = (oldest / BC) * BC;
        }
    }
    for (int kv0 = kv_min; kv0 < kv_max; kv0 += BC) {
        int bc = min(BC, p.t_kv - kv0);

#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sT.get_capacity(); ++i) {
            if (sT.is_valid_element(i)) {
                sT[i] = 0.0f;
            }
        }
        auto tK0 = tensor(kb + size_t(kv0) * D, dextents<int, 2>{DC, bc}, array<int, 2>{1, D});
        auto tK1 = tensor(kb + size_t(kv0) * D + DC, dextents<int, 2>{DC, bc}, array<int, 2>{1, D});
        op_s.run(tQ0, tK0, sT);
        op_s.run(tQ1, tK1, sT);

#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sT.get_capacity(); ++i) {
            if (sT.is_valid_element(i)) {
                auto idx = sT.get_multidimensional_index(i);
                int n = int(idx[0]);
                int m = int(idx[1]);
                float val = sT[i] * p.scale;
                int q_pos = p.prefix_len + q0 + m;
                int kv_pos = kv0 + n;
                // A key past the end of the cache is not a key: its column
                // of P must be zero, since V is read at that extent and not
                // zero-filled beyond it.
                if (n >= bc || kv_pos > q_pos || (p.window > 0 && q_pos >= kv_pos + p.window)) {
                    val = -INFINITY;
                }
                sT[i] = val;
            }
        }

        auto rT = op_s.template get_row_reduction_destination_cooperative_tensor<
            decltype(tQ0), decltype(tQ0), float>();
        reduce_rows(sT, rT, reduction_operation::max,
                    reduction_operation_identity<float>::max_identity);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < rT.get_capacity(); ++i) {
            if (rT.is_valid_element(i)) {
                auto idx = rT.get_multidimensional_index(i);
                tg_r[idx[0]] = rT[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = lane; r < uint(BR); r += 32u) {
            float m_new = max(tg_m[r], tg_r[r]);
            tg_a[r] = (tg_m[r] == -INFINITY) ? 0.0f : exp(tg_m[r] - m_new);
            tg_m[r] = m_new;
        }
        for (uint i = lane; i < uint(BR * BC); i += 32u) {
            tg_p[i] = bfloat(0.0f);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sT.get_capacity(); ++i) {
            if (sT.is_valid_element(i)) {
                auto idx = sT.get_multidimensional_index(i);
                int n = int(idx[0]);
                int m = int(idx[1]);
                float pv = (sT[i] == -INFINITY) ? 0.0f : exp(sT[i] - tg_m[m]);
                sT[i] = pv;
                tg_p[m * BC + n] = bfloat(pv);
            }
        }

        auto rsT = op_s.template get_row_reduction_destination_cooperative_tensor<
            decltype(tQ0), decltype(tQ0), float>();
        reduce_rows(sT, rsT, reduction_operation::sum, 0.0f);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < rsT.get_capacity(); ++i) {
            if (rsT.is_valid_element(i)) {
                auto idx = rsT.get_multidimensional_index(i);
                tg_r[idx[0]] = rsT[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = lane; r < uint(BR); r += 32u) {
            tg_l[r] = tg_l[r] * tg_a[r] + tg_r[r];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < oT0.get_capacity(); ++i) {
            if (oT0.is_valid_element(i)) {
                auto idx = oT0.get_multidimensional_index(i);
                oT0[i] *= tg_a[idx[1]];
                oT1[i] *= tg_a[idx[1]];
            }
        }
        // V is read where it lies, a half at a time, through the same kind
        // of strided view K is: staging it would copy the whole head's V
        // through threadgroup memory once per query block.
        auto tV0 = tensor(vb + size_t(kv0) * D, dextents<int, 2>{DC, bc}, array<int, 2>{1, D});
        auto tV1 = tensor(vb + size_t(kv0) * D + DC, dextents<int, 2>{DC, bc}, array<int, 2>{1, D});
        op_o.run(tP, tV0, oT0);
        op_o.run(tP, tV1, oT1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oT0.get_capacity(); ++i) {
        if (oT0.is_valid_element(i)) {
            auto idx = oT0.get_multidimensional_index(i);
            int dd = int(idx[0]);
            int m = int(idx[1]);
            if (m < br) {
                float denom = tg_l[m];
                float inv = (denom > 0.0f) ? 1.0f / denom : 0.0f;
                size_t at = ((size_t(b) * p.h + hh) * p.t_q + q0 + m) * D + dd;
                o[at] = bfloat(oT0[i] * inv);
                o[at + DC] = bfloat(oT1[i] * inv);
            }
        }
    }
}

#define MPP_FA_WIDE_KERNEL(NAME, D, BR, BC)                                    \
kernel void NAME(                                                              \
    device bfloat* q [[buffer(0)]],                                            \
    device bfloat* k [[buffer(1)]],                                            \
    device bfloat* v [[buffer(2)]],                                            \
    device bfloat* o [[buffer(3)]],                                            \
    constant MppFaParams& p [[buffer(4)]],                                     \
    uint2 tgid [[threadgroup_position_in_grid]],                               \
    uint lane [[thread_index_in_threadgroup]])                                 \
{                                                                              \
    threadgroup float tg_m[BR];                                                \
    threadgroup float tg_l[BR];                                                \
    threadgroup float tg_a[BR];                                                \
    threadgroup float tg_r[BR];                                                \
    threadgroup bfloat tg_p[BR * BC];                                          \
    threadgroup bfloat tg_q[BR * D];                                           \
    mpp_fa_wide_impl<D, BR, BC>(q, k, v, o, p, tg_m, tg_l, tg_a, tg_r, tg_p, tg_q, tgid, lane); \
}

MPP_FA_WIDE_KERNEL(mpp_fa_bf16_d512, 512, 16, 128)

#define MPP_FA_KERNEL(NAME, D)                                                 \
kernel void NAME(                                                              \
    device bfloat* q [[buffer(0)]],                                            \
    device bfloat* k [[buffer(1)]],                                            \
    device bfloat* v [[buffer(2)]],                                            \
    device bfloat* o [[buffer(3)]],                                            \
    constant MppFaParams& p [[buffer(4)]],                                     \
    uint2 tgid [[threadgroup_position_in_grid]],                               \
    uint lane [[thread_index_in_threadgroup]])                                 \
{                                                                              \
    threadgroup float tg_m[FA_BR];                                             \
    threadgroup float tg_l[FA_BR];                                             \
    threadgroup float tg_a[FA_BR];                                             \
    threadgroup float tg_r[FA_BR];                                             \
    threadgroup bfloat tg_p[FA_BR * FA_BC];                                    \
    threadgroup bfloat tg_v[FA_BC * D];                                        \
    mpp_fa_impl<D>(q, k, v, o, p, tg_m, tg_l, tg_a, tg_r, tg_p, tg_v, tgid, lane); \
}

MPP_FA_KERNEL(mpp_fa_bf16_d64, 64)
MPP_FA_KERNEL(mpp_fa_bf16_d128, 128)
MPP_FA_KERNEL(mpp_fa_bf16_d256, 256)
