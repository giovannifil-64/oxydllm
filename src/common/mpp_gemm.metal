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
