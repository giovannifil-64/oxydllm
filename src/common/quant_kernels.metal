// W4A16 (and future W8A16) quantized matmul kernels for Apple Metal.
//
// Fused dequantization + matmul for AWQ-style weight-only quantization. The
// packed weights stay resident (no bf16 expansion at load): decode reads the
// packed data straight from HBM, cutting weight bandwidth ~`32/BITS`×.
//
// On-disk AWQ layout (autoawq, GEMM kernel):
//   qweight  u32 [in_features, out_features / pack_factor]
//   qzeros   u32 [in_features / group_size, out_features / pack_factor]
//   scales   T   [in_features / group_size, out_features]
// where `pack_factor = 32 / BITS` (8 for 4-bit, 4 for 8-bit). Slot k of a word
// (bits BITS*k..BITS*(k+1)) maps to output column `pack_factor*j + pack_pos(k)`.
// For 4-bit AWQ `pack_pos` is the interleave `AWQ_PACK_ORDER`; for 8-bit it is
// identity. Dequant: W[o][i] = (qw_slot - qz_slot) * scale[group(i)][o].
//
// Per-BITS public kernels are thin instantiations of the templates below; only
// the 4-bit symbols ship today (`w4a16_*`, `dequantize_w4_*`). The 8-bit
// symbols are intentionally absent: Phase 2 (GPTQ) will add `w8a16_*` /
// `dequantize_w8_*` plus their host-side wiring.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Must stay in sync with `W4A16Params` in metal_ops.rs.
struct W4A16Params {
    uint in_features;
    uint out_features;
    uint packed_out;    // out_features / pack_factor
    uint group_size;    // dequant kernel only
    uint group_shift;   // log2(group_size) — gemv kernel only
    uint k_splits;      // in_features partitions — gemv kernel only
    uint chunk;         // ceil(in_features / k_splits) — gemv kernel only
};

constant uint AWQ_PACK_ORDER[8] = {0u, 2u, 4u, 6u, 1u, 3u, 5u, 7u};

// Extract slot k (BITS bits at offset BITS*k) from a 32-bit packed word.
template<uint BITS>
inline uint unpack(uint word, uint k) {
    return (word >> (BITS * k)) & ((1u << BITS) - 1u);
}

// Map slot index k → its output-column offset within a packed word. AWQ's
// 4-bit kernel uses an interleave; 8-bit is sequential. The compiler folds
// this away per template instantiation.
template<uint BITS>
inline uint pack_position(uint k) {
    return (BITS == 4u) ? AWQ_PACK_ORDER[k] : k;
}

// ── Fused WxA16 GEMV (M=1), split-K: out += x @ dequant(W)ᵀ ──────────────────
//
// Grid: (packed_out, k_splits). A naive one-thread-per-output GEMV launches
// only `packed_out` threads (~hundreds) and leaves the GPU idle with HBM load
// latency fully exposed. Here the reduction is split `k_splits` ways,
// multiplying the thread count so enough simdgroups are resident to hide
// latency.
//
// Each thread owns one packed-out column `j` and a CONTIGUOUS in_features
// chunk `[ks*chunk, (ks+1)*chunk)`. Contiguous (not strided) is essential: a
// thread stays inside a quant group for `group_size` steps, so the per-group
// scale/zero reload is amortised. Reads stay coalesced — consecutive threads
// (consecutive j) read consecutive qweight words. The `k_splits` partial sums
// are combined straight into `out` with relaxed atomic adds: this avoids
// materialising a `[k_splits, out]` partial buffer and a separate (strided,
// slow) reduction. `out` must be zero-initialised by the host.
template<typename T, uint BITS>
inline void awq_gemv_impl(
    device const T*       x,
    device const uint*    qweight,
    device const uint*    qzeros,
    device const T*       scales,
    device atomic_float*  out,
    constant W4A16Params& p,
    uint2 gid)
{
    constexpr uint PACK_FACTOR = 32u / BITS;

    uint j = gid.x;
    if (j >= p.packed_out) {
        return;
    }
    uint ks = gid.y;
    if (ks >= p.k_splits) {
        return;
    }

    uint i_start = ks * p.chunk;
    uint i_end = min(i_start + p.chunk, p.in_features);
    if (i_start >= i_end) {
        return;
    }

    float acc[PACK_FACTOR];
    for (uint k = 0; k < PACK_FACTOR; ++k) {
        acc[k] = 0.0f;
    }

    // Zero-point and scale depend only on the group → cached, refreshed on change.
    float zero_slot[PACK_FACTOR];
    float scale_v[PACK_FACTOR];
    uint last_g = 0xFFFFFFFFu;

    for (uint i = i_start; i < i_end; ++i) {
        uint g = i >> p.group_shift;
        if (g != last_g) {
            uint zw = qzeros[g * p.packed_out + j];
            for (uint k = 0; k < PACK_FACTOR; ++k) {
                zero_slot[k] = float(unpack<BITS>(zw, k));
                scale_v[k] = float(scales[g * p.out_features + j * PACK_FACTOR + pack_position<BITS>(k)]);
            }
            last_g = g;
        }

        uint ww = qweight[i * p.packed_out + j];
        float xv = float(x[i]);
        for (uint k = 0; k < PACK_FACTOR; ++k) {
            float w = (float(unpack<BITS>(ww, k)) - zero_slot[k]) * scale_v[k];
            acc[k] += xv * w;
        }
    }

    for (uint k = 0; k < PACK_FACTOR; ++k) {
        uint o = j * PACK_FACTOR + pack_position<BITS>(k);
        atomic_fetch_add_explicit(&out[o], acc[k], memory_order_relaxed);
    }
}

// ── Dequantize only: weight[in, out] = dequant(packed) ───────────────────────
//
// Grid: (in_features, packed_out). Produces a plain row-major [in, out] weight
// ready for a standard matmul — used for the prefill / batched (M>1) path, where
// a tuned GEMM beats a custom kernel.
template<typename T, uint BITS>
inline void awq_dequantize_impl(
    device const uint*    qweight,
    device const uint*    qzeros,
    device const T*       scales,
    device       T*       weight,
    constant W4A16Params& p,
    uint2 gid)
{
    constexpr uint PACK_FACTOR = 32u / BITS;

    uint i = gid.x;
    uint j = gid.y;
    if (i >= p.in_features || j >= p.packed_out) {
        return;
    }

    uint g = i / p.group_size;
    uint ww = qweight[i * p.packed_out + j];
    uint zw = qzeros[g * p.packed_out + j];

    for (uint k = 0; k < PACK_FACTOR; ++k) {
        uint o = j * PACK_FACTOR + pack_position<BITS>(k);
        float val = (float(unpack<BITS>(ww, k)) - float(unpack<BITS>(zw, k)))
            * float(scales[g * p.out_features + o]);
        weight[i * p.out_features + o] = T(val);
    }
}

kernel void w4a16_gemv_f16(
    device const half*    x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const half*    scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_gemv_impl<half, 4>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void w4a16_gemv_bf16(
    device const bfloat*  x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const bfloat*  scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_gemv_impl<bfloat, 4>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void dequantize_w4_f16(
    device const uint*    qweight  [[buffer(0)]],
    device const uint*    qzeros   [[buffer(1)]],
    device const half*    scales   [[buffer(2)]],
    device       half*    weight   [[buffer(3)]],
    constant W4A16Params& p        [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_dequantize_impl<half, 4>(qweight, qzeros, scales, weight, p, gid);
}

kernel void dequantize_w4_bf16(
    device const uint*    qweight  [[buffer(0)]],
    device const uint*    qzeros   [[buffer(1)]],
    device const bfloat*  scales   [[buffer(2)]],
    device       bfloat*  weight   [[buffer(3)]],
    constant W4A16Params& p        [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_dequantize_impl<bfloat, 4>(qweight, qzeros, scales, weight, p, gid);
}

// =============================================================================
// GGUF quantized GEMV kernels (Q5_0 first; Q4_K and Q2_K to follow).
//
// Bf16-aware port of llama.cpp's `mul_vec_q_n_f32` template (MIT) via
// candle-metal-kernels (MIT). The candle path runs in F32 only — the host
// wrapper has to cast bf16 activations to f32 and the f32 output back to bf16
// per call. Here we read bf16 activations directly, do the reduction in
// register float, and write bf16 directly to dst — eliminating both cast
// kernels.
//
// Algorithm (per simdgroup of N_SIMDWIDTH=32 threads):
//   • Each simdgroup owns N_DST=4 consecutive output rows.
//   • Each threadgroup contains N_SIMDGROUP=2 simdgroups → 8 rows/TG.
//   • Within a simdgroup, threads `tiisg` (0..31) cooperate over a row's
//     `nb = K/32` blocks: thread `tiisg` handles block `tiisg/2`, half `il`
//     (0 or 8), striding by `N_SIMDWIDTH/2 = 16` blocks per pass.
//   • Per-thread partials accumulate into `sumf[N_DST]`, then `simd_sum`
//     reduces across the simdgroup; thread 0 writes one bf16 per row.
//   • No atomics, no split-K — one writer per output row.
// =============================================================================

#define QK5_0 32
#define GGUF_N_SIMDWIDTH 32
#define GGUF_N_DST 4
#define GGUF_N_SIMDGROUP 2

struct GgufParams {
    uint in_features;   // K — must be a multiple of block_elems
    uint out_features;  // N
};

typedef struct {
    half     d;             // delta (offset 0,  2 bytes)
    uint8_t  qh[4];         // 5-th bit per element (offset 2, 4 bytes)
    uint8_t  qs[QK5_0/2];   // low 4 bits, 2 elements per byte (offset 6, 16 bytes)
} block_q5_0;

// Inner product of one Q5_0 block-half with 16 pre-scaled activations.
// `yl[]` were pre-scaled in the caller so that the qs[] bits can be ANDed at
// their native positions without further shifting (1, 1/16, 1/256, 1/4096).
// `sumy` is the sum of the 16 raw activations, used for the -16 offset.
inline float gguf_q5_0_dot_y(device const block_q5_0 *qb,
                              float sumy,
                              thread float *yl,
                              int il)
{
    float d = qb->d;
    float2 acc = 0.f;
    device const uint16_t *qs = ((device const uint16_t *)qb + 3 + il/2);
    const uint32_t qh = *((device const uint32_t *)qb->qh);

    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i+0] * ((qs[i/2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010))
                + yl[i+1] * ((qs[i/2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[1] += yl[i+8] * ((qs[i/2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100))
                + yl[i+9] * ((qs[i/2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }
    return d * (sumy * -16.f + acc[0] + acc[1]);
}

// Q5_0 GEMV bf16: M=1 decode path. Weight is laid out as a contiguous block
// stream `[N * (K/32) * 22]` bytes (the GGUF on-disk format); we cast to
// `block_q5_0*` for typed access.
//
// Tried during F2 verticale (2026-05-22), all rolled back:
//   • `gguf_q5_0_gemv_bf16_cached` — threadgroup memory caching of the K
//     activations. Cooperative-load + barrier overhead supersedes the
//     savings (Apple L2 already does the work). Net: −5%.
//   • bfloat4-vectorised activation reads in the inner loop. Net: within
//     noise (the GPU coalesces scalar bf16 reads already).
// The verbatim port of the candle template wins, +6% over the F32 path
// (entirely the elimination of the bf16↔f32 host-side casts).
kernel void gguf_q5_0_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],   // [K]
    device const void         *weight   [[buffer(1)]],   // [N * K/32 * 22] raw bytes
    device       bfloat       *out      [[buffer(2)]],   // [N]
    constant     GgufParams   &p        [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint nb = K / QK5_0;

    const uint r0 = tgpig.x;
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * GGUF_N_DST;
    if (first_row >= N) return;

    device const block_q5_0 *x_blocks =
        (device const block_q5_0 *)weight + (ulong)first_row * (ulong)nb;

    float yl[16];
    float sumf[GGUF_N_DST] = {0.f};

    const uint ix = (tiisg / 2);          // which block (0..15)
    const uint il = (tiisg % 2) * 8;      // 0 or 8 — which half of the block

    device const bfloat *yb = x + ix * QK5_0 + il;

    for (uint ib = ix; ib < nb; ib += GGUF_N_SIMDWIDTH/2) {
        float sumy = 0.f;
        for (int i = 0; i < 8; i += 2) {
            const float a0  = (float)yb[i+0];
            const float a1  = (float)yb[i+1];
            const float a16 = (float)yb[i+16];
            const float a17 = (float)yb[i+17];
            sumy   += a0 + a1 + a16 + a17;
            yl[i+0] = a0;
            yl[i+1] = a1  / 256.f;
            yl[i+8] = a16 / 16.f;
            yl[i+9] = a17 / 4096.f;
        }
        for (uint row = 0; row < GGUF_N_DST; ++row) {
            if (first_row + row < N) {
                sumf[row] += gguf_q5_0_dot_y(x_blocks + ib + row * nb, sumy, yl, il);
            }
        }
        yb += (ulong)QK5_0 * (ulong)(GGUF_N_SIMDWIDTH / 2);
    }

    for (uint row = 0; row < GGUF_N_DST; ++row) {
        const float tot = simd_sum(sumf[row]);
        const uint r = first_row + row;
        if (tiisg == 0 && r < N) {
            out[r] = (bfloat)tot;
        }
    }
}

// =============================================================================
// Q8_0 GEMV bf16 — port of llama.cpp / candle `kernel_mul_mv_q8_0_f32_impl`.
// Block layout (34 bytes, 32 elements):
//   half  d;       // global scale
//   int8  qs[32];  // signed quants (no offset, no bit packing — simplest)
// Dequantized value: q * d.
// Geometry: same as Q5_0 (N_SIMDGROUP=2 × N_DST=4 = 8 rows/TG).
// =============================================================================

#define QK8_0 32
#define NB_Q8_0 8        // 8 quants per thread per pass

typedef struct {
    half    d;
    int8_t  qs[QK8_0];
} block_q8_0;

kernel void gguf_q8_0_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],   // [K]
    device const void         *weight   [[buffer(1)]],   // [N * K/32 * 34] raw bytes
    device       bfloat       *out      [[buffer(2)]],   // [N]
    constant     GgufParams   &p        [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint nb = K / QK8_0;

    const uint r0 = tgpig.x;
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * GGUF_N_DST;
    if (first_row >= N) return;

    device const block_q8_0 *x_blocks =
        (device const block_q8_0 *)weight + (ulong)first_row * (ulong)nb;

    float yl[NB_Q8_0];
    float sumf[GGUF_N_DST] = {0.f};

    const uint ix = tiisg / 4;         // which block-index stride (0..7)
    const uint il = tiisg % 4;         // 0..3 — which 8-element chunk

    device const bfloat *yb = x + ix * QK8_0 + NB_Q8_0 * il;

    // Each thread handles NB_Q8_0=8 quants from a block; 4 threads cover the
    // full QK8_0=32-element block. ib steps by GGUF_N_SIMDWIDTH/4 = 8.
    for (uint ib = ix; ib < nb; ib += GGUF_N_SIMDWIDTH/4) {
        for (uint i = 0; i < NB_Q8_0; ++i) {
            yl[i] = (float)yb[i];
        }
        for (uint row = 0; row < GGUF_N_DST; ++row) {
            if (first_row + row < N) {
                device const int8_t *qs =
                    x_blocks[ib + row * nb].qs + NB_Q8_0 * il;
                float sumq = 0.f;
                for (uint iq = 0; iq < NB_Q8_0; ++iq) {
                    sumq += (float)qs[iq] * yl[iq];
                }
                sumf[row] += sumq * (float)x_blocks[ib + row * nb].d;
            }
        }
        yb += (ulong)NB_Q8_0 * (ulong)GGUF_N_SIMDWIDTH;
    }

    for (uint row = 0; row < GGUF_N_DST; ++row) {
        const float tot = simd_sum(sumf[row]);
        const uint r = first_row + row;
        if (tiisg == 0 && r < N) {
            out[r] = (bfloat)tot;
        }
    }
}

// =============================================================================
// Q4_K GEMV bf16 — port of llama.cpp / candle `kernel_mul_mv_q4_K_f32_impl`.
// Block layout (144 bytes, 256 elements):
//   half  d;            // global scale
//   half  dmin;         // global min
//   u8    scales[12];   // 8 sub-block scales + 8 sub-block mins, 6-bit packed
//   u8    qs[128];      // 256 4-bit quants
// Sub-block decoding: kmask1/kmask2/kmask3 are the bit-tricks from llama.cpp
// (split the 6-bit sc/min nibbles across bytes [0,2,4] / [1,3,5]).
//
// Geometry: 1 simdgroup × N_DST=4 rows per TG (the working set per thread —
// 16 yl + 16 yh + 8 acc + 4 sc16 — keeps register pressure high enough that
// adding a second simdgroup per TG would spill; this matches candle's
// hard-coded `N_DST` use without `N_SIMDGROUP`).
// =============================================================================

#define QK_K 256
#define Q4_K_SCALE_SIZE 12

typedef struct {
    half     d;
    half     dmin;
    uint8_t  scales[Q4_K_SCALE_SIZE];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

kernel void gguf_q4k_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],   // [K]
    device const void         *weight   [[buffer(1)]],   // [N * K/256 * 144] raw bytes
    device       bfloat       *out      [[buffer(2)]],   // [N]
    constant     GgufParams   &p        [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const uint ix = tiisg / 8;          // 0..3 — which super-block of the 4-stride
    const uint it = tiisg % 8;          // 0..7
    const uint iq = it / 4;             // 0 or 1 — which 128-element half
    const uint ir = it % 4;             // 0..3 — which 8-element chunk

    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint nb = K / QK_K;

    const uint r0 = tgpig.x;
    const uint first_row = r0 * GGUF_N_DST;
    if (first_row >= N) return;

    device const block_q4_K *x_base = (device const block_q4_K *)weight;
    device const block_q4_K *x_row0 = x_base + (ulong)first_row * (ulong)nb;
    device const bfloat *y_row = x;

    float yl[16];
    float yh[16];
    float sumf[GGUF_N_DST] = {0.f};

    device const bfloat *y4 = y_row + ix * QK_K + 64 * iq + 8 * ir;
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = (float)y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = (float)y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = (float)y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = (float)y4[i+160]; sumy[3] += yh[i+8];
        }

        for (uint row = 0; row < GGUF_N_DST; ++row) {
            if (first_row + row >= N) break;
            device const block_q4_K *xb = x_row0 + ib + row * nb;
            device const uint16_t *sc = (device const uint16_t *)xb->scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)xb->qs + 16 * iq + 4 * ir;
            device const half     *dh = &xb->d;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t *q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+0] * (q1[i/2] & 0x000F);
                acc1[1] += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1[2] += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1[3] += yl[i+9] * (q1[i/2] & 0xF000);
                acc2[0] += yh[i+0] * (q2[i/2] & 0x000F);
                acc2[1] += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2[2] += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2[3] += yh[i+9] * (q2[i/2] & 0xF000);
            }

            float dall = (float)dh[0];
            float dmin = (float)dh[1];
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                 (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                 (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                 sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * QK_K;
    }

    for (uint row = 0; row < GGUF_N_DST; ++row) {
        const float tot = simd_sum(sumf[row]);
        const uint r = first_row + row;
        if (tiisg == 0 && r < N) {
            out[r] = (bfloat)tot;
        }
    }
}

// =============================================================================
// Fused mul_mm_q*_bf16 — GEMM for M>1 (prefill).
//
// `out[M,N] = x[M,K] @ W[K,N].T` where W is GGUF-quantized. The weight is
// dequantized **inline** in the inner loop — no intermediate bf16 weight
// tensor is ever materialised (which is what made the dequant+matmul rollback
// regress: a single `gate_up_proj` would allocate ~192 MB per call).
//
// Algorithm: each threadgroup owns a `BM × BN` tile of the output. The K
// dimension is walked one quant block at a time; the BN weight blocks for the
// current K-strip are cooperatively dequantized into threadgroup memory once
// per K-tick (a 32-element row each), and every thread of the TG accumulates
// its row's dot-product against that shared row using the cached `x` tile.
// =============================================================================

struct GgufMatmulParams {
    uint m_total;        // batch / sequence length
    uint n_total;        // output features
    uint k_total;        // input features
};

constant uint GGUF_MM_BM = 16;   // rows of output per TG
constant uint GGUF_MM_BN = 16;   // cols of output per TG
constant uint GGUF_MM_TG = GGUF_MM_BM * GGUF_MM_BN;  // 256 threads per TG

// Cooperative dequantize of `BN` weight blocks (one block = QK5_0=32 elements)
// for the BN output columns of the current tile, written into `w_tile`.
// `n_base` is the global column index of the tile's first column; `b` is the
// current K-block index. Each TG-thread dequantizes BN*QK5_0/GGUF_MM_TG = 2
// elements.
inline void gguf_q5_0_dequant_strip_into(
    threadgroup bfloat (&w_tile)[GGUF_MM_BN][QK5_0],
    device const block_q5_0 *w_blocks_base,
    uint n_base,
    uint n_total,
    uint nb,
    uint b,
    uint tg_tid)
{
    for (uint kk = tg_tid; kk < GGUF_MM_BN * QK5_0; kk += GGUF_MM_TG) {
        const uint c = kk / QK5_0;
        const uint j = kk % QK5_0;
        const uint n = n_base + c;
        if (n >= n_total) { continue; }
        device const block_q5_0 *blk = w_blocks_base + (ulong)n * (ulong)nb + (ulong)b;
        const float d  = (float)blk->d;
        const uint  qh = *((device const uint *)blk->qh);
        const uint  hi = j / (QK5_0 / 2);
        const uint  jj = j % (QK5_0 / 2);
        const uint  xh = ((qh >> (jj + hi * 16)) & 1u) << 4;
        const uint  q_packed = (hi == 0) ? (blk->qs[jj] & 0x0Fu)
                                         : (blk->qs[jj] >> 4);
        const int   q = (int)(q_packed | xh) - 16;
        w_tile[c][j] = (bfloat)((float)q * d);
    }
}

inline void gguf_q8_0_dequant_strip_into(
    threadgroup bfloat (&w_tile)[GGUF_MM_BN][QK8_0],
    device const block_q8_0 *w_blocks_base,
    uint n_base,
    uint n_total,
    uint nb,
    uint b,
    uint tg_tid)
{
    for (uint kk = tg_tid; kk < GGUF_MM_BN * QK8_0; kk += GGUF_MM_TG) {
        const uint c = kk / QK8_0;
        const uint j = kk % QK8_0;
        const uint n = n_base + c;
        if (n >= n_total) { continue; }
        device const block_q8_0 *blk = w_blocks_base + (ulong)n * (ulong)nb + (ulong)b;
        const float d = (float)blk->d;
        w_tile[c][j] = (bfloat)((float)blk->qs[j] * d);
    }
}

// Get scale/min byte for Q4_K sub-block `j` (0..7) from the 12-byte packed
// scales array. Matches `get_scale_min_k4` from candle (MIT) / ggml.
inline void gguf_q4k_get_scale_min(uint j,
                                    device const uint8_t *q,
                                    thread uint &d_out,
                                    thread uint &m_out)
{
    if (j < 4) {
        d_out = (uint)(q[j])     & 63u;
        m_out = (uint)(q[j + 4]) & 63u;
    } else {
        d_out = ((uint)q[j + 4] & 0x0Fu) | (((uint)q[j - 4] >> 6) << 4);
        m_out = ((uint)q[j + 4] >>   4 ) | (((uint)q[j    ] >> 6) << 4);
    }
}

inline void gguf_q4k_dequant_strip_into(
    threadgroup bfloat (&w_tile)[GGUF_MM_BN][QK_K],
    device const block_q4_K *w_blocks_base,
    uint n_base,
    uint n_total,
    uint nb,
    uint b,
    uint tg_tid)
{
    for (uint kk = tg_tid; kk < GGUF_MM_BN * QK_K; kk += GGUF_MM_TG) {
        const uint c = kk / QK_K;
        const uint j = kk % QK_K;
        const uint n = n_base + c;
        if (n >= n_total) { continue; }
        device const block_q4_K *blk = w_blocks_base + (ulong)n * (ulong)nb + (ulong)b;
        const float d    = (float)blk->d;
        const float dmin = (float)blk->dmin;
        // j in [0, 256): which 32-element half (8 halves), which element.
        const uint half_idx = j / 32;          // 0..7
        const uint elem     = j % 32;          // 0..31
        uint sc_byte, m_byte;
        gguf_q4k_get_scale_min(half_idx, blk->scales, sc_byte, m_byte);
        const float sc_v = d    * (float)sc_byte;
        const float m_v  = dmin * (float)m_byte;

        const uint iq = half_idx / 2;
        const uint lo = half_idx % 2;
        const uint nibble = (lo == 0) ? (uint)(blk->qs[iq * 32 + elem] & 0x0Fu)
                                       : (uint)(blk->qs[iq * 32 + elem] >> 4);
        w_tile[c][j] = (bfloat)(sc_v * (float)nibble - m_v);
    }
}

// Common GEMM kernel body: each thread accumulates one output element
// `out[m, n]`. The weight is provided as a TG-shared `w_tile[BN][BLOCK]`
// dequantized strip; the x activation is fetched directly from HBM (cached
// per-row by L2 across the BN threads sharing the same row).
//
// `dequant_fn` is the per-quant lambda that fills `w_tile` for a given K-block.
// We instantiate one kernel per quant type because Metal lacks first-class
// function templates over `inline` lambdas — see the kernel bodies below.
//
// Layout: dispatch grid is (ceil(N/BN), ceil(M/BM), 1); TG is (BN, BM, 1).

kernel void gguf_q5_0_mul_mm_bf16(
    device const bfloat            *x        [[buffer(0)]],  // [M, K]
    device const void              *weight   [[buffer(1)]],  // packed block stream
    device       bfloat            *out      [[buffer(2)]],  // [M, N]
    constant     GgufMatmulParams  &p        [[buffer(3)]],
    uint3  tg_tid_xy [[thread_position_in_threadgroup]],
    uint3  tgpig     [[threadgroup_position_in_grid]])
{
    threadgroup bfloat w_tile[GGUF_MM_BN][QK5_0];

    const uint M  = p.m_total;
    const uint N  = p.n_total;
    const uint K  = p.k_total;
    const uint nb = K / QK5_0;

    const uint n_base = tgpig.x * GGUF_MM_BN;
    const uint m_base = tgpig.y * GGUF_MM_BM;
    const uint n_off  = tg_tid_xy.x;
    const uint m_off  = tg_tid_xy.y;
    const uint n      = n_base + n_off;
    const uint m      = m_base + m_off;
    const uint tg_tid = m_off * GGUF_MM_BN + n_off;

    device const block_q5_0 *w_base = (device const block_q5_0 *)weight;

    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        gguf_q5_0_dequant_strip_into(w_tile, w_base, n_base, N, nb, b, tg_tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (m < M && n < N) {
            device const bfloat *x_chunk = x + (ulong)m * (ulong)K + (ulong)b * (ulong)QK5_0;
            float partial = 0.0f;
            for (uint j = 0; j < QK5_0; ++j) {
                partial += (float)x_chunk[j] * (float)w_tile[n_off][j];
            }
            acc += partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m < M && n < N) {
        out[(ulong)m * (ulong)N + (ulong)n] = (bfloat)acc;
    }
}

kernel void gguf_q8_0_mul_mm_bf16(
    device const bfloat            *x        [[buffer(0)]],
    device const void              *weight   [[buffer(1)]],
    device       bfloat            *out      [[buffer(2)]],
    constant     GgufMatmulParams  &p        [[buffer(3)]],
    uint3  tg_tid_xy [[thread_position_in_threadgroup]],
    uint3  tgpig     [[threadgroup_position_in_grid]])
{
    threadgroup bfloat w_tile[GGUF_MM_BN][QK8_0];

    const uint M  = p.m_total;
    const uint N  = p.n_total;
    const uint K  = p.k_total;
    const uint nb = K / QK8_0;

    const uint n_base = tgpig.x * GGUF_MM_BN;
    const uint m_base = tgpig.y * GGUF_MM_BM;
    const uint n_off  = tg_tid_xy.x;
    const uint m_off  = tg_tid_xy.y;
    const uint n      = n_base + n_off;
    const uint m      = m_base + m_off;
    const uint tg_tid = m_off * GGUF_MM_BN + n_off;

    device const block_q8_0 *w_base = (device const block_q8_0 *)weight;

    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        gguf_q8_0_dequant_strip_into(w_tile, w_base, n_base, N, nb, b, tg_tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (m < M && n < N) {
            device const bfloat *x_chunk = x + (ulong)m * (ulong)K + (ulong)b * (ulong)QK8_0;
            float partial = 0.0f;
            for (uint j = 0; j < QK8_0; ++j) {
                partial += (float)x_chunk[j] * (float)w_tile[n_off][j];
            }
            acc += partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m < M && n < N) {
        out[(ulong)m * (ulong)N + (ulong)n] = (bfloat)acc;
    }
}

kernel void gguf_q4k_mul_mm_bf16(
    device const bfloat            *x        [[buffer(0)]],
    device const void              *weight   [[buffer(1)]],
    device       bfloat            *out      [[buffer(2)]],
    constant     GgufMatmulParams  &p        [[buffer(3)]],
    uint3  tg_tid_xy [[thread_position_in_threadgroup]],
    uint3  tgpig     [[threadgroup_position_in_grid]])
{
    threadgroup bfloat w_tile[GGUF_MM_BN][QK_K];

    const uint M  = p.m_total;
    const uint N  = p.n_total;
    const uint K  = p.k_total;
    const uint nb = K / QK_K;

    const uint n_base = tgpig.x * GGUF_MM_BN;
    const uint m_base = tgpig.y * GGUF_MM_BM;
    const uint n_off  = tg_tid_xy.x;
    const uint m_off  = tg_tid_xy.y;
    const uint n      = n_base + n_off;
    const uint m      = m_base + m_off;
    const uint tg_tid = m_off * GGUF_MM_BN + n_off;

    device const block_q4_K *w_base = (device const block_q4_K *)weight;

    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        gguf_q4k_dequant_strip_into(w_tile, w_base, n_base, N, nb, b, tg_tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (m < M && n < N) {
            device const bfloat *x_chunk = x + (ulong)m * (ulong)K + (ulong)b * (ulong)QK_K;
            float partial = 0.0f;
            for (uint j = 0; j < QK_K; ++j) {
                partial += (float)x_chunk[j] * (float)w_tile[n_off][j];
            }
            acc += partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m < M && n < N) {
        out[(ulong)m * (ulong)N + (ulong)n] = (bfloat)acc;
    }
}
