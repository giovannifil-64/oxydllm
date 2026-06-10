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

struct W4A16Params {
    uint in_features;
    uint out_features;
    uint packed_out;
    uint group_size;
    uint group_shift;
    uint k_splits;
    uint chunk;
};

constant uint AWQ_PACK_ORDER[8] = {0u, 2u, 4u, 6u, 1u, 3u, 5u, 7u};

template<uint BITS>
inline uint unpack(uint word, uint k) {
    return (word >> (BITS * k)) & ((1u << BITS) - 1u);
}

template<uint BITS>
inline uint pack_position(uint k) {
    return (BITS == 4u) ? AWQ_PACK_ORDER[k] : k;
}

// Split-K AWQ GEMV: `out` must be host-zeroed (atomic_fetch_add accumulates).
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

kernel void w8a16_gemv_f16(
    device const half*    x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const half*    scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_gemv_impl<half, 8>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void w8a16_gemv_bf16(
    device const bfloat*  x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const bfloat*  scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_gemv_impl<bfloat, 8>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void dequantize_w8_f16(
    device const uint*    qweight  [[buffer(0)]],
    device const uint*    qzeros   [[buffer(1)]],
    device const half*    scales   [[buffer(2)]],
    device       half*    weight   [[buffer(3)]],
    constant W4A16Params& p        [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_dequantize_impl<half, 8>(qweight, qzeros, scales, weight, p, gid);
}

kernel void dequantize_w8_bf16(
    device const uint*    qweight  [[buffer(0)]],
    device const uint*    qzeros   [[buffer(1)]],
    device const bfloat*  scales   [[buffer(2)]],
    device       bfloat*  weight   [[buffer(3)]],
    constant W4A16Params& p        [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_dequantize_impl<bfloat, 8>(qweight, qzeros, scales, weight, p, gid);
}

// GPTQ (auto-gptq): qweight packed along IN, zero stored as (z-1), grid is
// (out_features, k_splits). Caller pre-aligns `chunk` to PACK_FACTOR.

struct GptqParams {
    uint in_features;
    uint out_features;
    uint group_size;
    uint group_shift;
    uint k_splits;
    uint chunk;
};

template<typename T, uint BITS>
inline void gptq_gemv_impl(
    device const T*       x,
    device const uint*    qweight,
    device const uint*    qzeros,
    device const T*       scales,
    device atomic_float*  out,
    constant GptqParams&  p,
    uint2 gid)
{
    constexpr uint PACK_FACTOR = 32u / BITS;
    constexpr uint MASK = (1u << BITS) - 1u;

    uint o = gid.x;
    if (o >= p.out_features) {
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
    uint w_start = i_start / PACK_FACTOR;
    uint w_end = (i_end + PACK_FACTOR - 1u) / PACK_FACTOR;

    uint o_word = o / PACK_FACTOR;
    uint o_slot = o % PACK_FACTOR;
    uint o_shift = BITS * o_slot;
    uint qzeros_inner = p.out_features / PACK_FACTOR;

    float acc = 0.0f;
    float scale_v = 0.0f;
    float zp1 = 0.0f;
    uint last_g = 0xFFFFFFFFu;

    for (uint iw = w_start; iw < w_end; ++iw) {
        uint i_base = iw * PACK_FACTOR;
        uint g = i_base >> p.group_shift;
        if (g != last_g) {
            uint zw = qzeros[g * qzeros_inner + o_word];
            uint z = (zw >> o_shift) & MASK;
            zp1 = float(z) + 1.0f;
            scale_v = float(scales[g * p.out_features + o]);
            last_g = g;
        }

        uint ww = qweight[iw * p.out_features + o];
        for (uint k = 0; k < PACK_FACTOR; ++k) {
            uint q = (ww >> (BITS * k)) & MASK;
            uint i = i_base + k;
            acc += float(x[i]) * scale_v * (float(q) - zp1);
        }
    }

    atomic_fetch_add_explicit(&out[o], acc, memory_order_relaxed);
}

template<typename T, uint BITS>
inline void gptq_dequantize_impl(
    device const uint*    qweight,
    device const uint*    qzeros,
    device const T*       scales,
    device       T*       weight,
    constant GptqParams&  p,
    uint2 gid)
{
    constexpr uint PACK_FACTOR = 32u / BITS;
    constexpr uint MASK = (1u << BITS) - 1u;

    uint iw = gid.x;
    uint o = gid.y;
    uint packed_in = p.in_features / PACK_FACTOR;
    if (iw >= packed_in || o >= p.out_features) {
        return;
    }

    uint i_base = iw * PACK_FACTOR;
    uint g = i_base / p.group_size;

    uint o_word = o / PACK_FACTOR;
    uint o_slot = o % PACK_FACTOR;
    uint o_shift = BITS * o_slot;
    uint qzeros_inner = p.out_features / PACK_FACTOR;

    uint ww = qweight[iw * p.out_features + o];
    uint zw = qzeros[g * qzeros_inner + o_word];
    float zp1 = float((zw >> o_shift) & MASK) + 1.0f;
    float scale_v = float(scales[g * p.out_features + o]);

    for (uint k = 0; k < PACK_FACTOR; ++k) {
        uint q = (ww >> (BITS * k)) & MASK;
        uint i = i_base + k;
        weight[i * p.out_features + o] = T(scale_v * (float(q) - zp1));
    }
}

kernel void gptq4_gemv_f16(
    device const half*    x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const half*    scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_gemv_impl<half, 4>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void gptq4_gemv_bf16(
    device const bfloat*  x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const bfloat*  scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_gemv_impl<bfloat, 4>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void gptq8_gemv_f16(
    device const half*    x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const half*    scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_gemv_impl<half, 8>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void gptq8_gemv_bf16(
    device const bfloat*  x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const bfloat*  scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_gemv_impl<bfloat, 8>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void dequantize_gptq4_f16(
    device const uint*    qweight  [[buffer(0)]],
    device const uint*    qzeros   [[buffer(1)]],
    device const half*    scales   [[buffer(2)]],
    device       half*    weight   [[buffer(3)]],
    constant GptqParams&  p        [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_dequantize_impl<half, 4>(qweight, qzeros, scales, weight, p, gid);
}

kernel void dequantize_gptq4_bf16(
    device const uint*    qweight  [[buffer(0)]],
    device const uint*    qzeros   [[buffer(1)]],
    device const bfloat*  scales   [[buffer(2)]],
    device       bfloat*  weight   [[buffer(3)]],
    constant GptqParams&  p        [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_dequantize_impl<bfloat, 4>(qweight, qzeros, scales, weight, p, gid);
}

kernel void dequantize_gptq8_f16(
    device const uint*    qweight  [[buffer(0)]],
    device const uint*    qzeros   [[buffer(1)]],
    device const half*    scales   [[buffer(2)]],
    device       half*    weight   [[buffer(3)]],
    constant GptqParams&  p        [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_dequantize_impl<half, 8>(qweight, qzeros, scales, weight, p, gid);
}

kernel void dequantize_gptq8_bf16(
    device const uint*    qweight  [[buffer(0)]],
    device const uint*    qzeros   [[buffer(1)]],
    device const bfloat*  scales   [[buffer(2)]],
    device       bfloat*  weight   [[buffer(3)]],
    constant GptqParams&  p        [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_dequantize_impl<bfloat, 8>(qweight, qzeros, scales, weight, p, gid);
}

// GGUF GEMV kernels: bf16-aware ports of llama.cpp `mul_vec_q_n_f32` (MIT).
// One simdgroup writes N_DST consecutive rows; no atomics.

#define QK5_0 32
#define GGUF_N_SIMDWIDTH 32
#define GGUF_N_DST 4
#define GGUF_N_SIMDGROUP 2

struct GgufParams {
    uint in_features;
    uint out_features;
};

// Batched-decode kernels: M activation vectors share one weight read (M capped at
// GGUF_BATCH_MAX for register pressure; above the cap, batched decode uses mul_mm).
#define GGUF_BATCH_MAX 8
struct GgufBatchParams {
    uint in_features;
    uint out_features;
    uint m_batch;
};

typedef struct {
    half     d;
    uint8_t  qh[4];
    uint8_t  qs[QK5_0/2];
} block_q5_0;

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

kernel void gguf_q5_0_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],
    device const void         *weight   [[buffer(1)]],
    device       bfloat       *out      [[buffer(2)]],
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

    const uint ix = (tiisg / 2);
    const uint il = (tiisg % 2) * 8;

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

typedef struct {
    half     d;
    uint8_t  qs[QK5_0 / 2];
} block_q4_0;

inline float gguf_q4_0_dot_y(device const block_q4_0 *qb,
                              float sumy,
                              thread float *yl,
                              int il)
{
    const float d = qb->d;
    float2 acc = 0.f;
    device const uint16_t *qs = ((device const uint16_t *)qb + 1 + il/2);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i+0] * (qs[i/2] & 0x000F)
                + yl[i+1] * (qs[i/2] & 0x0F00);
        acc[1] += yl[i+8] * (qs[i/2] & 0x00F0)
                + yl[i+9] * (qs[i/2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1]);
}

kernel void gguf_q4_0_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],
    device const void         *weight   [[buffer(1)]],
    device       bfloat       *out      [[buffer(2)]],
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

    device const block_q4_0 *x_blocks =
        (device const block_q4_0 *)weight + (ulong)first_row * (ulong)nb;

    float yl[16];
    float sumf[GGUF_N_DST] = {0.f};

    const uint ix = (tiisg / 2);
    const uint il = (tiisg % 2) * 8;

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
                sumf[row] += gguf_q4_0_dot_y(x_blocks + ib + row * nb, sumy, yl, il);
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

typedef struct {
    half     d;
    half     m;
    uint8_t  qs[QK5_0 / 2];
} block_q4_1;

inline float gguf_q4_1_dot_y(device const block_q4_1 *qb,
                              float sumy,
                              thread float *yl,
                              int il)
{
    const float d = qb->d;
    const float m = qb->m;
    float2 acc = 0.f;
    device const uint16_t *qs = ((device const uint16_t *)qb + 2 + il/2);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i+0] * (qs[i/2] & 0x000F)
                + yl[i+1] * (qs[i/2] & 0x0F00);
        acc[1] += yl[i+8] * (qs[i/2] & 0x00F0)
                + yl[i+9] * (qs[i/2] & 0xF000);
    }
    return d * (acc[0] + acc[1]) + sumy * m;
}

kernel void gguf_q4_1_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],
    device const void         *weight   [[buffer(1)]],
    device       bfloat       *out      [[buffer(2)]],
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

    device const block_q4_1 *x_blocks =
        (device const block_q4_1 *)weight + (ulong)first_row * (ulong)nb;

    float yl[16];
    float sumf[GGUF_N_DST] = {0.f};

    const uint ix = (tiisg / 2);
    const uint il = (tiisg % 2) * 8;

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
                sumf[row] += gguf_q4_1_dot_y(x_blocks + ib + row * nb, sumy, yl, il);
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

typedef struct {
    half     d;
    half     m;
    uint8_t  qh[4];
    uint8_t  qs[QK5_0 / 2];
} block_q5_1;

inline float gguf_q5_1_dot_y(device const block_q5_1 *qb,
                              float sumy,
                              thread float *yl,
                              int il)
{
    const float d = qb->d;
    const float m = qb->m;
    float2 acc = 0.f;
    device const uint16_t *qs = ((device const uint16_t *)qb + 4 + il/2);
    const uint32_t qh = *((device const uint32_t *)qb->qh);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i+0] * ((qs[i/2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010))
                + yl[i+1] * ((qs[i/2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[1] += yl[i+8] * ((qs[i/2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100))
                + yl[i+9] * ((qs[i/2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }
    return d * (acc[0] + acc[1]) + sumy * m;
}

kernel void gguf_q5_1_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],
    device const void         *weight   [[buffer(1)]],
    device       bfloat       *out      [[buffer(2)]],
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

    device const block_q5_1 *x_blocks =
        (device const block_q5_1 *)weight + (ulong)first_row * (ulong)nb;

    float yl[16];
    float sumf[GGUF_N_DST] = {0.f};

    const uint ix = (tiisg / 2);
    const uint il = (tiisg % 2) * 8;

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
                sumf[row] += gguf_q5_1_dot_y(x_blocks + ib + row * nb, sumy, yl, il);
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

// Batched legacy-quant decode GEMV (2 simdgroups x GGUF_N_DST rows), one
// instantiation per quant: only the block type and dot_y differ. Weights are
// read once per (row, block) and L2-amortized across the inner M-loop.
#define GGUF_LEGACY_GEMV_BATCH(NAME, BLOCK, DOT_Y)                                      \
kernel void NAME(                                                                       \
    device const bfloat        *x        [[buffer(0)]],                                 \
    device const void          *weight   [[buffer(1)]],                                 \
    device       bfloat        *out      [[buffer(2)]],                                 \
    constant     GgufBatchParams &p      [[buffer(3)]],                                 \
    uint3 tgpig [[threadgroup_position_in_grid]],                                       \
    uint tiisg  [[thread_index_in_simdgroup]],                                          \
    uint sgitg  [[simdgroup_index_in_threadgroup]])                                     \
{                                                                                       \
    const uint N = p.out_features, K = p.in_features;                                   \
    const uint M = min(p.m_batch, (uint)GGUF_BATCH_MAX);                                \
    const uint nb = K / QK5_0;                                                          \
    const uint r0 = tgpig.x;                                                            \
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * GGUF_N_DST;                \
    if (first_row >= N) return;                                                         \
    device const BLOCK *x_blocks =                                                      \
        (device const BLOCK *)weight + (ulong)first_row * (ulong)nb;                    \
    const uint ix = (tiisg / 2);                                                        \
    const uint il = (tiisg % 2) * 8;                                                    \
    float sumf[GGUF_N_DST][GGUF_BATCH_MAX];                                             \
    for (uint row = 0; row < GGUF_N_DST; ++row)                                         \
        for (uint m = 0; m < GGUF_BATCH_MAX; ++m) sumf[row][m] = 0.f;                   \
    for (uint ib = ix; ib < nb; ib += GGUF_N_SIMDWIDTH / 2) {                           \
        for (uint row = 0; row < GGUF_N_DST; ++row) {                                   \
            if (first_row + row >= N) continue;                                         \
            device const BLOCK *blk = x_blocks + ib + row * nb;                         \
            for (uint m = 0; m < M; ++m) {                                              \
                device const bfloat *yb =                                               \
                    x + (ulong)m * (ulong)K + (ulong)ib * QK5_0 + il;                   \
                float yl[16]; float sumy = 0.f;                                         \
                for (int i = 0; i < 8; i += 2) {                                        \
                    const float a0=(float)yb[i+0], a1=(float)yb[i+1];                   \
                    const float a16=(float)yb[i+16], a17=(float)yb[i+17];               \
                    sumy += a0 + a1 + a16 + a17;                                        \
                    yl[i+0]=a0; yl[i+1]=a1/256.f;                                       \
                    yl[i+8]=a16/16.f; yl[i+9]=a17/4096.f;                               \
                }                                                                       \
                sumf[row][m] += DOT_Y(blk, sumy, yl, il);                               \
            }                                                                           \
        }                                                                               \
    }                                                                                   \
    for (uint row = 0; row < GGUF_N_DST; ++row) {                                       \
        const uint r = first_row + row;                                                 \
        for (uint m = 0; m < M; ++m) {                                                  \
            const float tot = simd_sum(sumf[row][m]);                                   \
            if (tiisg == 0 && r < N) out[(ulong)m * (ulong)N + r] = (bfloat)tot;        \
        }                                                                               \
    }                                                                                   \
}

GGUF_LEGACY_GEMV_BATCH(gguf_q5_0_gemv_batch_bf16, block_q5_0, gguf_q5_0_dot_y)
GGUF_LEGACY_GEMV_BATCH(gguf_q4_0_gemv_batch_bf16, block_q4_0, gguf_q4_0_dot_y)
GGUF_LEGACY_GEMV_BATCH(gguf_q4_1_gemv_batch_bf16, block_q4_1, gguf_q4_1_dot_y)
GGUF_LEGACY_GEMV_BATCH(gguf_q5_1_gemv_batch_bf16, block_q5_1, gguf_q5_1_dot_y)

#define QK8_0 32
#define NB_Q8_0 8

typedef struct {
    half    d;
    int8_t  qs[QK8_0];
} block_q8_0;

kernel void gguf_q8_0_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],
    device const void         *weight   [[buffer(1)]],
    device       bfloat       *out      [[buffer(2)]],
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

    const uint ix = tiisg / 4;
    const uint il = tiisg % 4;

    device const bfloat *yb = x + ix * QK8_0 + NB_Q8_0 * il;

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

// Batched Q8_0 decode GEMV (2 simdgroups x GGUF_N_DST rows per TG).
kernel void gguf_q8_0_gemv_batch_bf16(
    device const bfloat        *x        [[buffer(0)]],   // [M, K]
    device const void          *weight   [[buffer(1)]],
    device       bfloat        *out      [[buffer(2)]],   // [M, N]
    constant     GgufBatchParams &p      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint M  = min(p.m_batch, (uint)GGUF_BATCH_MAX);
    const uint nb = K / QK8_0;

    const uint r0 = tgpig.x;
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * GGUF_N_DST;
    if (first_row >= N) return;

    device const block_q8_0 *x_blocks =
        (device const block_q8_0 *)weight + (ulong)first_row * (ulong)nb;

    const uint ix = tiisg / 4;
    const uint il = tiisg % 4;
    const uint y_off = NB_Q8_0 * il;

    float sumf[GGUF_N_DST][GGUF_BATCH_MAX];
    for (uint row = 0; row < GGUF_N_DST; ++row) {
        for (uint m = 0; m < GGUF_BATCH_MAX; ++m) {
            sumf[row][m] = 0.f;
        }
    }

    for (uint ib = ix; ib < nb; ib += GGUF_N_SIMDWIDTH / 4) {
        for (uint row = 0; row < GGUF_N_DST; ++row) {
            if (first_row + row >= N) {
                continue;
            }
            device const int8_t *qs = x_blocks[ib + row * nb].qs + NB_Q8_0 * il;
            const float d = (float)x_blocks[ib + row * nb].d;
            for (uint m = 0; m < M; ++m) {
                device const bfloat *yb =
                    x + (ulong)m * (ulong)K + (ulong)ib * QK8_0 + y_off;
                float sumq = 0.f;
                for (uint iq = 0; iq < NB_Q8_0; ++iq) {
                    sumq += (float)qs[iq] * (float)yb[iq];
                }
                sumf[row][m] += sumq * d;
            }
        }
    }

    for (uint row = 0; row < GGUF_N_DST; ++row) {
        const uint r = first_row + row;
        for (uint m = 0; m < M; ++m) {
            const float tot = simd_sum(sumf[row][m]);
            if (tiisg == 0 && r < N) {
                out[(ulong)m * (ulong)N + r] = (bfloat)tot;
            }
        }
    }
}

// Q4_K geometry: 1 simdgroup × N_DST=4 rows per TG (register pressure forbids 2).

#define QK_K 256
#define Q4_K_SCALE_SIZE 12

typedef struct {
    half     d;
    half     dmin;
    uint8_t  scales[Q4_K_SCALE_SIZE];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

kernel void gguf_q4k_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],
    device const void         *weight   [[buffer(1)]],
    device       bfloat       *out      [[buffer(2)]],
    constant     GgufParams   &p        [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const uint ix = tiisg / 8;
    const uint it = tiisg % 8;
    const uint iq = it / 4;
    const uint ir = it % 4;

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

// Batched Q4_K decode GEMV (1 simdgroup x GGUF_N_DST rows, same geometry as the
// gemv): weights read once per (row, super-block), reused across the inner M-loop.
kernel void gguf_q4k_gemv_batch_bf16(
    device const bfloat        *x        [[buffer(0)]],   // [M, K]
    device const void          *weight   [[buffer(1)]],
    device       bfloat        *out      [[buffer(2)]],   // [M, N]
    constant     GgufBatchParams &p      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const uint ix = tiisg / 8;
    const uint it = tiisg % 8;
    const uint iq = it / 4;
    const uint ir = it % 4;

    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint M  = min(p.m_batch, (uint)GGUF_BATCH_MAX);
    const uint nb = K / QK_K;

    const uint r0 = tgpig.x;
    const uint first_row = r0 * GGUF_N_DST;
    if (first_row >= N) return;

    device const block_q4_K *x_base = (device const block_q4_K *)weight;
    device const block_q4_K *x_row0 = x_base + (ulong)first_row * (ulong)nb;

    const uint y_off = 64 * iq + 8 * ir;

    float sumf[GGUF_N_DST][GGUF_BATCH_MAX];
    for (uint row = 0; row < GGUF_N_DST; ++row) {
        for (uint m = 0; m < GGUF_BATCH_MAX; ++m) {
            sumf[row][m] = 0.f;
        }
    }

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
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
            const float dall = (float)dh[0];
            const float dmin = (float)dh[1];

            for (uint m = 0; m < M; ++m) {
                device const bfloat *y4 =
                    x + (ulong)m * (ulong)K + (ulong)ib * QK_K + y_off;
                float yl[16], yh[16];
                float4 sumy = {0.f, 0.f, 0.f, 0.f};
                for (int i = 0; i < 8; ++i) {
                    yl[i+0] = (float)y4[i+  0]; sumy[0] += yl[i+0];
                    yl[i+8] = (float)y4[i+ 32]; sumy[1] += yl[i+8];
                    yh[i+0] = (float)y4[i+128]; sumy[2] += yh[i+0];
                    yh[i+8] = (float)y4[i+160]; sumy[3] += yh[i+8];
                }

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

                sumf[row][m] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                        (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                        (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                        (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                                dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                        sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            }
        }
    }

    for (uint row = 0; row < GGUF_N_DST; ++row) {
        const uint r = first_row + row;
        for (uint m = 0; m < M; ++m) {
            const float tot = simd_sum(sumf[row][m]);
            if (tiisg == 0 && r < N) {
                out[(ulong)m * (ulong)N + r] = (bfloat)tot;
            }
        }
    }
}

// Q5_K geometry: 2 simdgroups × 2 rows per TG.
typedef struct {
    half     d;
    half     dmin;
    uint8_t  scales[Q4_K_SCALE_SIZE];
    uint8_t  qh[QK_K / 8];
    uint8_t  qs[QK_K / 2];
} block_q5_K;

kernel void gguf_q5k_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],
    device const void         *weight   [[buffer(1)]],
    device       bfloat       *out      [[buffer(2)]],
    constant     GgufParams   &p        [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint nb = K / QK_K;

    const uint r0 = tgpig.x;
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * 2;
    if (first_row >= N) return;

    device const block_q5_K *x_blocks =
        (device const block_q5_K *)weight + (ulong)first_row * (ulong)nb;

    float sumf[2] = {0.f, 0.f};

    const uint tid = tiisg / 4;
    const uint ix  = tiisg % 4;
    const uint iq  = tid / 4;
    const uint ir  = tid % 4;
    const uint nn  = 8;

    const uint l0 = nn * ir;
    const uint q_offset = 32 * iq + l0;
    const uint y_offset = 64 * iq + l0;

    const uint8_t hm1 = 1u << (2 * iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    float yl[16], yh[16];
    device const bfloat *y1 = x + ix * QK_K + y_offset;

    const uint step = (uint)sizeof(block_q5_K) * nb;

    for (uint i = ix; i < nb; i += 4) {
        device const block_q5_K *xb = x_blocks + i;
        device const uint8_t *q1 = xb->qs + q_offset;
        device const uint8_t *qh = xb->qh + l0;
        device const half    *dh = &xb->d;
        device const uint16_t *a = (device const uint16_t *)xb->scales + iq;

        device const bfloat *y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (uint l = 0; l < 8; ++l) {
            yl[l+0] = (float)y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = (float)y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = (float)y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = (float)y2[l+32]; sumy[3] += yh[l+8];
        }

        for (uint row = 0; row < 2; ++row) {
            device const uint8_t *q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (uint l = 0; l < nn; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
                acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
                acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
                acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
            }
            const float dall = (float)dh[0];
            const float dmin = (float)dh[1];
            sumf[row] += dall * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                 sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                 sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                 sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dmin * (sumy[0]*sc8[2] + sumy[1]*sc8[3] +
                                 sumy[2]*sc8[6] + sumy[3]*sc8[7]);

            q1 = (device const uint8_t *)((device const uint8_t *)q1 + step);
            qh = (device const uint8_t *)((device const uint8_t *)qh + step);
            dh = (device const half *)   ((device const uint8_t *)dh + step);
            a  = (device const uint16_t *)((device const uint8_t *)a + step);
        }

        y1 += 4 * QK_K;
    }

    for (uint row = 0; row < 2; ++row) {
        const float tot = simd_sum(sumf[row]);
        const uint r = first_row + row;
        if (tiisg == 0 && r < N) {
            out[r] = (bfloat)tot;
        }
    }
}

// Batched Q5_K decode GEMV (2 simdgroups x 2 rows per TG, step-advance per row).
kernel void gguf_q5k_gemv_batch_bf16(
    device const bfloat        *x        [[buffer(0)]],   // [M, K]
    device const void          *weight   [[buffer(1)]],
    device       bfloat        *out      [[buffer(2)]],   // [M, N]
    constant     GgufBatchParams &p      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint M  = min(p.m_batch, (uint)GGUF_BATCH_MAX);
    const uint nb = K / QK_K;

    const uint r0 = tgpig.x;
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * 2;
    if (first_row >= N) return;

    device const block_q5_K *x_blocks =
        (device const block_q5_K *)weight + (ulong)first_row * (ulong)nb;

    const uint tid = tiisg / 4;
    const uint ix  = tiisg % 4;
    const uint iq  = tid / 4;
    const uint ir  = tid % 4;
    const uint nn  = 8;

    const uint l0 = nn * ir;
    const uint q_offset = 32 * iq + l0;
    const uint y_offset = 64 * iq + l0;

    const uint8_t hm1 = 1u << (2 * iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    float sumf[2][GGUF_BATCH_MAX];
    for (uint row = 0; row < 2; ++row) {
        for (uint m = 0; m < GGUF_BATCH_MAX; ++m) {
            sumf[row][m] = 0.f;
        }
    }

    const uint step = (uint)sizeof(block_q5_K) * nb;

    for (uint i = ix; i < nb; i += 4) {
        device const block_q5_K *xb = x_blocks + i;
        device const uint8_t *q1 = xb->qs + q_offset;
        device const uint8_t *qh = xb->qh + l0;
        device const half    *dh = &xb->d;
        device const uint16_t *a = (device const uint16_t *)xb->scales + iq;

        for (uint row = 0; row < 2; ++row) {
            device const uint8_t *q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            const float dall = (float)dh[0];
            const float dmin = (float)dh[1];

            // Dequantize ONCE into registers, affine fold: w = d*sc*q5 - dmin*msc
            // (the q5 value is (4-bit nibble) + 16*high-bit; the original kernel's
            // /16 and ×16 tricks are absorbed here). m-loop below is pure FMA.
            const float d0 = dall * sc8[0];
            const float d1 = dall * sc8[1];
            const float d4 = dall * sc8[4];
            const float d5 = dall * sc8[5];
            const float mn2 = dmin * sc8[2];
            const float mn3 = dmin * sc8[3];
            const float mn6 = dmin * sc8[6];
            const float mn7 = dmin * sc8[7];
            float w0[8], w1[8], w2[8], w3[8];
            for (uint l = 0; l < nn; ++l) {
                const uint8_t h = qh[l];
                w0[l] = d0 * (float)((q1[l] & 0x0F)        + ((h & hm1) ? 16 : 0)) - mn2;
                w1[l] = d1 * (float)(((q1[l] & 0xF0) >> 4) + ((h & hm2) ? 16 : 0)) - mn3;
                w2[l] = d4 * (float)((q2[l] & 0x0F)        + ((h & hm3) ? 16 : 0)) - mn6;
                w3[l] = d5 * (float)(((q2[l] & 0xF0) >> 4) + ((h & hm4) ? 16 : 0)) - mn7;
            }

            for (uint m = 0; m < M; ++m) {
                device const bfloat *y1 = x + (ulong)m * (ulong)K + (ulong)i * QK_K + y_offset;
                device const bfloat *y2 = y1 + 128;
                float acc = 0.f;
                for (uint l = 0; l < nn; ++l) {
                    acc += (float)y1[l+ 0] * w0[l] + (float)y1[l+32] * w1[l] +
                           (float)y2[l+ 0] * w2[l] + (float)y2[l+32] * w3[l];
                }
                sumf[row][m] += acc;
            }

            q1 = (device const uint8_t *)((device const uint8_t *)q1 + step);
            qh = (device const uint8_t *)((device const uint8_t *)qh + step);
            dh = (device const half *)   ((device const uint8_t *)dh + step);
            a  = (device const uint16_t *)((device const uint8_t *)a + step);
        }
    }

    for (uint row = 0; row < 2; ++row) {
        const uint r = first_row + row;
        for (uint m = 0; m < M; ++m) {
            const float tot = simd_sum(sumf[row][m]);
            if (tiisg == 0 && r < N) {
                out[(ulong)m * (ulong)N + r] = (bfloat)tot;
            }
        }
    }
}

// Q6_K geometry: 2 simdgroups × 1 row per TG.
typedef struct {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    half     d;
} block_q6_K;

kernel void gguf_q6k_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],
    device const void         *weight   [[buffer(1)]],
    device       bfloat       *out      [[buffer(2)]],
    constant     GgufParams   &p        [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint nb = K / QK_K;

    const uint r0 = tgpig.x;
    const uint row = 2 * r0 + sgitg;
    if (row >= N) return;

    device const block_q6_K *x_blocks =
        (device const block_q6_K *)weight + (ulong)row * (ulong)nb;

    const uint tid = tiisg / 2;
    const uint ix  = tiisg % 2;
    const uint ip  = tid / 8;
    const uint il  = tid % 8;
    const uint nn  = 4;
    const uint l0  = nn * il;
    const uint is  = 8 * ip + l0 / 16;

    const uint y_offset    = 128 * ip + l0;
    const uint q_offset_l  = 64  * ip + l0;
    const uint q_offset_h  = 32  * ip + l0;

    float sumf = 0.f;
    for (uint i = ix; i < nb; i += 2) {
        device const block_q6_K *xb = x_blocks + i;
        device const uint8_t *q1 = xb->ql + q_offset_l;
        device const uint8_t *q2 = q1 + 32;
        device const uint8_t *qh = xb->qh + q_offset_h;
        device const int8_t  *sc = xb->scales + is;
        device const bfloat  *y  = x + i * QK_K + y_offset;
        const float dall = (float)xb->d;

        float4 sums = {0.f, 0.f, 0.f, 0.f};
        for (uint l = 0; l < nn; ++l) {
            sums[0] += (float)y[l+ 0] *
                       (float)((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            sums[1] += (float)y[l+32] *
                       (float)((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            sums[2] += (float)y[l+64] *
                       (float)((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
            sums[3] += (float)y[l+96] *
                       (float)((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
        }
        sumf += dall * (sums[0]*sc[0] + sums[1]*sc[2] +
                        sums[2]*sc[4] + sums[3]*sc[6]);
    }

    const float tot = simd_sum(sumf);
    if (tiisg == 0 && row < N) {
        out[row] = (bfloat)tot;
    }
}

// Batched Q6_K decode GEMV (2 simdgroups x 1 row per TG).
kernel void gguf_q6k_gemv_batch_bf16(
    device const bfloat        *x        [[buffer(0)]],   // [M, K]
    device const void          *weight   [[buffer(1)]],
    device       bfloat        *out      [[buffer(2)]],   // [M, N]
    constant     GgufBatchParams &p      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint M  = min(p.m_batch, (uint)GGUF_BATCH_MAX);
    const uint nb = K / QK_K;

    const uint r0 = tgpig.x;
    const uint row = 2 * r0 + sgitg;
    if (row >= N) return;

    device const block_q6_K *x_blocks =
        (device const block_q6_K *)weight + (ulong)row * (ulong)nb;

    const uint tid = tiisg / 2;
    const uint ix  = tiisg % 2;
    const uint ip  = tid / 8;
    const uint il  = tid % 8;
    const uint nn  = 4;
    const uint l0  = nn * il;
    const uint is  = 8 * ip + l0 / 16;

    const uint y_offset    = 128 * ip + l0;
    const uint q_offset_l  = 64  * ip + l0;
    const uint q_offset_h  = 32  * ip + l0;

    float sumf[GGUF_BATCH_MAX];
    for (uint m = 0; m < GGUF_BATCH_MAX; ++m) {
        sumf[m] = 0.f;
    }

    for (uint i = ix; i < nb; i += 2) {
        device const block_q6_K *xb = x_blocks + i;
        device const uint8_t *q1 = xb->ql + q_offset_l;
        device const uint8_t *q2 = q1 + 32;
        device const uint8_t *qh = xb->qh + q_offset_h;
        device const int8_t  *sc = xb->scales + is;
        const float dall = (float)xb->d;

        // Dequantize once into registers, group scales folded in; m-loop is pure FMA.
        const float ds0 = dall * sc[0];
        const float ds2 = dall * sc[2];
        const float ds4 = dall * sc[4];
        const float ds6 = dall * sc[6];
        float w0[4], w1[4], w2[4], w3[4];
        for (uint l = 0; l < nn; ++l) {
            w0[l] = ds0 * (float)((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            w1[l] = ds2 * (float)((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            w2[l] = ds4 * (float)((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
            w3[l] = ds6 * (float)((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
        }

        for (uint m = 0; m < M; ++m) {
            device const bfloat *y = x + (ulong)m * (ulong)K + (ulong)i * QK_K + y_offset;
            float acc = 0.f;
            for (uint l = 0; l < nn; ++l) {
                acc += (float)y[l+ 0] * w0[l] + (float)y[l+32] * w1[l] +
                       (float)y[l+64] * w2[l] + (float)y[l+96] * w3[l];
            }
            sumf[m] += acc;
        }
    }

    for (uint m = 0; m < M; ++m) {
        const float tot = simd_sum(sumf[m]);
        if (tiisg == 0 && row < N) {
            out[(ulong)m * (ulong)N + row] = (bfloat)tot;
        }
    }
}

// Q2_K geometry: 2 simdgroups × 4 rows per TG.
typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    half     d;
    half     dmin;
} block_q2_K;

kernel void gguf_q2k_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],
    device const void         *weight   [[buffer(1)]],
    device       bfloat       *out      [[buffer(2)]],
    constant     GgufParams   &p        [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint nb = K / QK_K;

    const uint r0 = tgpig.x;
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * GGUF_N_DST;
    if (first_row >= N) return;

    device const block_q2_K *x_blocks =
        (device const block_q2_K *)weight + (ulong)first_row * (ulong)nb;

    float yl[32];
    float sumf[GGUF_N_DST] = {0.f};

    const uint ix = tiisg / 8;
    const uint it = tiisg % 8;
    const uint iq = it / 4;
    const uint ir = it % 4;
    const uint is = (8 * ir) / 16;

    device const bfloat *y4 = x + ix * QK_K + 128 * iq + 8 * ir;
    const uint step = (uint)sizeof(block_q2_K) * nb;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = (float)y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = (float)y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = (float)y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = (float)y4[i+96]; sumy[3] += yl[i+24];
        }

        device const block_q2_K *xb0 = x_blocks + ib;
        device const uint8_t  *sc = (device const uint8_t  *)xb0->scales + 8 * iq + is;
        device const uint16_t *qs = (device const uint16_t *)xb0->qs + 16 * iq + 4 * ir;
        device const half     *dh = &xb0->d;

        for (uint row = 0; row < GGUF_N_DST; ++row) {
            if (first_row + row >= N) break;
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            const float dall = (float)dh[0];
            const float dmin = (float)dh[1] * (1.f / 16.f);
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * (1.f /  1.f) +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * (1.f /  4.f) +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * (1.f / 16.f) +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * (1.f / 64.f)) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                 sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs = (device const uint16_t *)((device const uint8_t *)qs + step);
            sc = (device const uint8_t  *)((device const uint8_t *)sc + step);
            dh = (device const half     *)((device const uint8_t *)dh + step);
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

// Batched Q2_K decode GEMV (2 simdgroups x GGUF_N_DST rows per TG, step-advance).
kernel void gguf_q2k_gemv_batch_bf16(
    device const bfloat        *x        [[buffer(0)]],   // [M, K]
    device const void          *weight   [[buffer(1)]],
    device       bfloat        *out      [[buffer(2)]],   // [M, N]
    constant     GgufBatchParams &p      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint M  = min(p.m_batch, (uint)GGUF_BATCH_MAX);
    const uint nb = K / QK_K;

    const uint r0 = tgpig.x;
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * GGUF_N_DST;
    if (first_row >= N) return;

    device const block_q2_K *x_blocks =
        (device const block_q2_K *)weight + (ulong)first_row * (ulong)nb;

    const uint ix = tiisg / 8;
    const uint it = tiisg % 8;
    const uint iq = it / 4;
    const uint ir = it % 4;
    const uint is = (8 * ir) / 16;
    const uint y_off = 128 * iq + 8 * ir;

    float sumf[GGUF_N_DST][GGUF_BATCH_MAX];
    for (uint row = 0; row < GGUF_N_DST; ++row) {
        for (uint m = 0; m < GGUF_BATCH_MAX; ++m) {
            sumf[row][m] = 0.f;
        }
    }

    const uint step = (uint)sizeof(block_q2_K) * nb;

    for (uint ib = ix; ib < nb; ib += 4) {
        device const block_q2_K *xb0 = x_blocks + ib;
        device const uint8_t  *sc = (device const uint8_t  *)xb0->scales + 8 * iq + is;
        device const uint16_t *qs = (device const uint16_t *)xb0->qs + 16 * iq + 4 * ir;
        device const half     *dh = &xb0->d;

        for (uint row = 0; row < GGUF_N_DST; ++row) {
            if (first_row + row >= N) break;
            const float dall = (float)dh[0];
            const float dmin = (float)dh[1] * (1.f / 16.f);

            for (uint m = 0; m < M; ++m) {
                device const bfloat *y4 = x + (ulong)m * (ulong)K + (ulong)ib * QK_K + y_off;
                float yl[32];
                float4 sumy = {0.f, 0.f, 0.f, 0.f};
                for (int i = 0; i < 8; ++i) {
                    yl[i+ 0] = (float)y4[i+ 0]; sumy[0] += yl[i+ 0];
                    yl[i+ 8] = (float)y4[i+32]; sumy[1] += yl[i+ 8];
                    yl[i+16] = (float)y4[i+64]; sumy[2] += yl[i+16];
                    yl[i+24] = (float)y4[i+96]; sumy[3] += yl[i+24];
                }

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                for (int i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                    acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                    acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                    acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                    acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                    acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                    acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                    acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                }
                sumf[row][m] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * (1.f /  1.f) +
                                        (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * (1.f /  4.f) +
                                        (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * (1.f / 16.f) +
                                        (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * (1.f / 64.f)) -
                                dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                        sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
            }

            qs = (device const uint16_t *)((device const uint8_t *)qs + step);
            sc = (device const uint8_t  *)((device const uint8_t *)sc + step);
            dh = (device const half     *)((device const uint8_t *)dh + step);
        }
    }

    for (uint row = 0; row < GGUF_N_DST; ++row) {
        const uint r = first_row + row;
        for (uint m = 0; m < M; ++m) {
            const float tot = simd_sum(sumf[row][m]);
            if (tiisg == 0 && r < N) {
                out[(ulong)m * (ulong)N + r] = (bfloat)tot;
            }
        }
    }
}

// Q3_K geometry: 2 simdgroups × 2 rows per TG.
typedef struct {
    uint8_t  hmask[QK_K / 8];
    uint8_t  qs[QK_K / 4];
    uint8_t  scales[12];
    half     d;
} block_q3_K;

kernel void gguf_q3k_gemv_bf16(
    device const bfloat       *x        [[buffer(0)]],
    device const void         *weight   [[buffer(1)]],
    device       bfloat       *out      [[buffer(2)]],
    constant     GgufParams   &p        [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint nb = K / QK_K;

    const uint r0 = tgpig.x;
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * 2;
    if (first_row >= N) return;

    device const block_q3_K *x_blocks =
        (device const block_q3_K *)weight + (ulong)first_row * (ulong)nb;

    float yl[32];

    const uint tid = tiisg / 4;
    const uint ix  = tiisg % 4;
    const uint ip  = tid / 4;
    const uint il  = 2u * ((tid % 4u) / 2u);
    const uint ir  = tid % 2;
    const uint n   = 8;
    const uint l0  = n * ir;

    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},
                           {0x0004, 0x0400, 0x0008, 0x0800},
                           {0x0010, 0x1000, 0x0020, 0x2000},
                           {0x0040, 0x4000, 0x0080, 0x8000}};
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00},
                        {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2u * ip + il / 2u];

    const uint shift = 2u * il;
    const float v1 = (il == 0u) ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    const uint16_t s_shift1 = 4u * (uint16_t)ip;
    const uint16_t s_shift2 = s_shift1 + (uint16_t)il;

    const uint q_offset = 32u * ip + l0;
    const uint y_offset = 128u * ip + 32u * il + l0;

    const uint step_bytes = (uint)sizeof(block_q3_K) * nb;

    device const bfloat *y1 = x + ix * QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t   *scales16 = (thread uint16_t   *)&scales32;
    thread const int8_t *scales = (thread const int8_t *)&scales32;

    float sumf1[2] = {0.f, 0.f};
    float sumf2[2] = {0.f, 0.f};

    for (uint i = ix; i < nb; i += 4) {
        for (uint l = 0; l < 8; ++l) {
            yl[l +  0] = (float)y1[l +  0];
            yl[l +  8] = (float)y1[l + 16];
            yl[l + 16] = (float)y1[l + 32];
            yl[l + 24] = (float)y1[l + 48];
        }

        device const uint16_t *q = (device const uint16_t *)(x_blocks[i].qs    + q_offset);
        device const uint16_t *h = (device const uint16_t *)(x_blocks[i].hmask + l0);
        device const uint16_t *a = (device const uint16_t *)(x_blocks[i].scales);
        device const half     *dh = &x_blocks[i].d;

        for (uint row = 0; row < 2; ++row) {
            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030u;
            scales16[0] = a[il + 0u];
            scales16[1] = a[il + 1u];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0fu) | aux32;

            float s1 = 0.f, s2 = 0.f, s3 = 0.f, s4 = 0.f, s5 = 0.f, s6 = 0.f;
            for (uint l = 0; l < n; l += 2) {
                const int32_t qs = (int32_t)q[l / 2];
                s1 += yl[l + 0] * (float)(qs & qm[il / 2u][0]);
                s2 += yl[l + 1] * (float)(qs & qm[il / 2u][1]);
                s3 += ((h[l / 2] & hm[0]) ? 0.f : yl[l + 0]) +
                      ((h[l / 2] & hm[1]) ? 0.f : yl[l + 1]);
                s4 += yl[l + 16] * (float)(qs & qm[il / 2u][2]);
                s5 += yl[l + 17] * (float)(qs & qm[il / 2u][3]);
                s6 += ((h[l / 2] & hm[2]) ? 0.f : yl[l + 16]) +
                      ((h[l / 2] & hm[3]) ? 0.f : yl[l + 17]);
            }
            float d1 = d_all * (s1 + 1.f / 256.f * s2 - s3 * v1);
            float d2 = d_all * (s4 + 1.f / 256.f * s5 - s6 * v2);
            sumf1[row] += d1 * (float)(scales[0] - 32);
            sumf2[row] += d2 * (float)(scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0.f;
            for (uint l = 0; l < n; l += 2) {
                const int32_t qs = (int32_t)q[l / 2 + 8];
                s1 += yl[l + 8] * (float)(qs & qm[il / 2u][0]);
                s2 += yl[l + 9] * (float)(qs & qm[il / 2u][1]);
                s3 += ((h[l / 2 + 8] & hm[0]) ? 0.f : yl[l +  8]) +
                      ((h[l / 2 + 8] & hm[1]) ? 0.f : yl[l +  9]);
                s4 += yl[l + 24] * (float)(qs & qm[il / 2u][2]);
                s5 += yl[l + 25] * (float)(qs & qm[il / 2u][3]);
                s6 += ((h[l / 2 + 8] & hm[2]) ? 0.f : yl[l + 24]) +
                      ((h[l / 2 + 8] & hm[3]) ? 0.f : yl[l + 25]);
            }
            d1 = d_all * (s1 + 1.f / 256.f * s2 - s3 * v1);
            d2 = d_all * (s4 + 1.f / 256.f * s5 - s6 * v2);
            sumf1[row] += d1 * (float)(scales[1] - 32);
            sumf2[row] += d2 * (float)(scales[3] - 32);

            q  = (device const uint16_t *)((device const uint8_t *)q  + step_bytes);
            h  = (device const uint16_t *)((device const uint8_t *)h  + step_bytes);
            a  = (device const uint16_t *)((device const uint8_t *)a  + step_bytes);
            dh = (device const half     *)((device const uint8_t *)dh + step_bytes);
        }

        y1 += 4 * QK_K;
    }

    for (uint row = 0; row < 2; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (float)(1u << shift);
        const float tot = simd_sum(sumf);
        const uint r = first_row + row;
        if (tiisg == 0 && r < N) {
            out[r] = (bfloat)tot;
        }
    }
}

// Batched Q3_K decode GEMV (2 simdgroups x 2 rows per TG, step-advance): weights
// dequantized once per (row, super-block), reused across the inner M-loop.
kernel void gguf_q3k_gemv_batch_bf16(
    device const bfloat        *x        [[buffer(0)]],   // [M, K]
    device const void          *weight   [[buffer(1)]],
    device       bfloat        *out      [[buffer(2)]],   // [M, N]
    constant     GgufBatchParams &p      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint M  = min(p.m_batch, (uint)GGUF_BATCH_MAX);
    const uint nb = K / QK_K;

    const uint r0 = tgpig.x;
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * 2;
    if (first_row >= N) return;

    device const block_q3_K *x_blocks =
        (device const block_q3_K *)weight + (ulong)first_row * (ulong)nb;

    const uint tid = tiisg / 4;
    const uint ix  = tiisg % 4;
    const uint ip  = tid / 4;
    const uint il  = 2u * ((tid % 4u) / 2u);
    const uint ir  = tid % 2;
    const uint n   = 8;
    const uint l0  = n * ir;

    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},
                           {0x0004, 0x0400, 0x0008, 0x0800},
                           {0x0010, 0x1000, 0x0020, 0x2000},
                           {0x0040, 0x4000, 0x0080, 0x8000}};

    const ushort4 hm = mm[2u * ip + il / 2u];

    const uint shift = 2u * il;

    const uint16_t s_shift1 = 4u * (uint16_t)ip;
    const uint16_t s_shift2 = s_shift1 + (uint16_t)il;

    const uint q_offset = 32u * ip + l0;
    const uint y_offset = 128u * ip + 32u * il + l0;

    const uint step_bytes = (uint)sizeof(block_q3_K) * nb;

    uint32_t scales32, aux32;
    thread uint16_t   *scales16 = (thread uint16_t   *)&scales32;
    thread const int8_t *scales = (thread const int8_t *)&scales32;

    const ushort hmv[4] = {hm[0], hm[1], hm[2], hm[3]};

    float sumf[2][GGUF_BATCH_MAX];
    for (uint row = 0; row < 2; ++row) {
        for (uint m = 0; m < GGUF_BATCH_MAX; ++m) {
            sumf[row][m] = 0.f;
        }
    }

    for (uint i = ix; i < nb; i += 4) {
        device const uint16_t *q = (device const uint16_t *)(x_blocks[i].qs    + q_offset);
        device const uint16_t *h = (device const uint16_t *)(x_blocks[i].hmask + l0);
        device const uint16_t *a = (device const uint16_t *)(x_blocks[i].scales);
        device const half     *dh = &x_blocks[i].d;

        for (uint row = 0; row < 2; ++row) {
            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030u;
            scales16[0] = a[il + 0u];
            scales16[1] = a[il + 1u];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0fu) | aux32;

            // Dequantize ONCE into registers. True per-element value:
            // d*(sc-32)*(q2bit - (hbit ? 0 : 4)) — extracting the 2-bit values
            // directly cancels the template's in-place <<shift, /256 odd-byte and
            // 0.25*sumf2 normalizations, so no final rescale is needed.
            const float dsA = d_all * (float)(scales[0] - 32);
            const float dsB = d_all * (float)(scales[1] - 32);
            const float dsC = d_all * (float)(scales[2] - 32);
            const float dsD = d_all * (float)(scales[3] - 32);
            float wA[8], wB[8], wC[8], wD[8];
            for (uint l = 0; l < n; ++l) {
                const uint sh  = shift + 8u * (l & 1u);
                const uint qlo = (uint)q[l / 2];
                const uint qhi = (uint)q[l / 2 + 8];
                const uint hlo = (uint)h[l / 2];
                const uint hhi = (uint)h[l / 2 + 8];
                const ushort hm01 = hmv[l & 1u];
                const ushort hm23 = hmv[2u + (l & 1u)];
                wA[l] = dsA * ((float)((qlo >> sh) & 3u)        - ((hlo & hm01) ? 0.f : 4.f));
                wB[l] = dsB * ((float)((qhi >> sh) & 3u)        - ((hhi & hm01) ? 0.f : 4.f));
                wC[l] = dsC * ((float)((qlo >> (sh + 2u)) & 3u) - ((hlo & hm23) ? 0.f : 4.f));
                wD[l] = dsD * ((float)((qhi >> (sh + 2u)) & 3u) - ((hhi & hm23) ? 0.f : 4.f));
            }

            for (uint m = 0; m < M; ++m) {
                device const bfloat *y1 = x + (ulong)m * (ulong)K + (ulong)i * QK_K + y_offset;
                float acc = 0.f;
                for (uint l = 0; l < n; ++l) {
                    acc += (float)y1[l +  0] * wA[l] + (float)y1[l + 16] * wB[l] +
                           (float)y1[l + 32] * wC[l] + (float)y1[l + 48] * wD[l];
                }
                sumf[row][m] += acc;
            }

            q  = (device const uint16_t *)((device const uint8_t *)q  + step_bytes);
            h  = (device const uint16_t *)((device const uint8_t *)h  + step_bytes);
            a  = (device const uint16_t *)((device const uint8_t *)a  + step_bytes);
            dh = (device const half     *)((device const uint8_t *)dh + step_bytes);
        }
    }

    for (uint row = 0; row < 2; ++row) {
        const uint r = first_row + row;
        for (uint m = 0; m < M; ++m) {
            const float tot = simd_sum(sumf[row][m]);
            if (tiisg == 0 && r < N) {
                out[(ulong)m * (ulong)N + r] = (bfloat)tot;
            }
        }
    }
}

// Fused mul_mm_q*_bf16: BM×BN tile of out, cooperative inline dequant per K-block.

struct GgufMatmulParams {
    uint m_total;
    uint n_total;
    uint k_total;
};

constant uint GGUF_MM_BM = 16;
constant uint GGUF_MM_BN = 16;
constant uint GGUF_MM_TG = GGUF_MM_BM * GGUF_MM_BN;

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

inline void gguf_q4_0_dequant_strip_into(
    threadgroup bfloat (&w_tile)[GGUF_MM_BN][QK5_0],
    device const block_q4_0 *w_blocks_base,
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
        device const block_q4_0 *blk = w_blocks_base + (ulong)n * (ulong)nb + (ulong)b;
        const float d  = (float)blk->d;
        const uint hi  = j / (QK5_0 / 2);
        const uint jj  = j % (QK5_0 / 2);
        const uint q_packed = (hi == 0) ? (blk->qs[jj] & 0x0Fu) : (blk->qs[jj] >> 4);
        const int  q = (int)q_packed - 8;
        w_tile[c][j] = (bfloat)((float)q * d);
    }
}

inline void gguf_q4_1_dequant_strip_into(
    threadgroup bfloat (&w_tile)[GGUF_MM_BN][QK5_0],
    device const block_q4_1 *w_blocks_base,
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
        device const block_q4_1 *blk = w_blocks_base + (ulong)n * (ulong)nb + (ulong)b;
        const float d = (float)blk->d;
        const float m = (float)blk->m;
        const uint hi = j / (QK5_0 / 2);
        const uint jj = j % (QK5_0 / 2);
        const uint q_packed = (hi == 0) ? (blk->qs[jj] & 0x0Fu) : (blk->qs[jj] >> 4);
        w_tile[c][j] = (bfloat)((float)q_packed * d + m);
    }
}

inline void gguf_q5_1_dequant_strip_into(
    threadgroup bfloat (&w_tile)[GGUF_MM_BN][QK5_0],
    device const block_q5_1 *w_blocks_base,
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
        device const block_q5_1 *blk = w_blocks_base + (ulong)n * (ulong)nb + (ulong)b;
        const float d  = (float)blk->d;
        const float m  = (float)blk->m;
        const uint  qh = *((device const uint *)blk->qh);
        const uint  hi = j / (QK5_0 / 2);
        const uint  jj = j % (QK5_0 / 2);
        const uint  xh = ((qh >> (jj + hi * 16)) & 1u) << 4;
        const uint  q_packed = (hi == 0) ? (blk->qs[jj] & 0x0Fu) : (blk->qs[jj] >> 4);
        const uint  q = q_packed | xh;
        w_tile[c][j] = (bfloat)((float)q * d + m);
    }
}

// Q4_K sub-block (0..7) scale+min from the 12-byte packed scales (get_scale_min_k4).
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
        const uint half_idx = j / 32;
        const uint elem     = j % 32;
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

inline void gguf_q5k_dequant_strip_into(
    threadgroup bfloat (&w_tile)[GGUF_MM_BN][QK_K],
    device const block_q5_K *w_blocks_base,
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
        device const block_q5_K *blk = w_blocks_base + (ulong)n * (ulong)nb + (ulong)b;
        const float d    = (float)blk->d;
        const float dmin = (float)blk->dmin;
        const uint half_idx = j / 32;
        const uint elem     = j % 32;
        uint sc_byte, m_byte;
        gguf_q4k_get_scale_min(half_idx, blk->scales, sc_byte, m_byte);
        const float sc_v = d    * (float)sc_byte;
        const float m_v  = dmin * (float)m_byte;

        const uint iq = half_idx / 2;
        const uint lo = half_idx % 2;
        const uint ql_byte = blk->qs[iq * 32 + elem];
        const uint low4    = (lo == 0) ? (ql_byte & 0x0Fu) : (ql_byte >> 4);
        const uint qh_bit  = (uint)(blk->qh[elem] >> (2u * iq + lo)) & 1u;
        const uint qv      = low4 | (qh_bit << 4);
        w_tile[c][j] = (bfloat)(sc_v * (float)qv - m_v);
    }
}

inline void gguf_q6k_dequant_strip_into(
    threadgroup bfloat (&w_tile)[GGUF_MM_BN][QK_K],
    device const block_q6_K *w_blocks_base,
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
        device const block_q6_K *blk = w_blocks_base + (ulong)n * (ulong)nb + (ulong)b;
        const float d = (float)blk->d;

        const uint idx_half = j / 128;
        const uint pos      = j % 128;
        const uint quadrant = pos / 32;
        const uint l        = pos % 32;

        device const uint8_t *ql_base = blk->ql + 64u * idx_half;
        device const uint8_t *qh_base = blk->qh + 32u * idx_half;

        const uint ql_byte_idx = (quadrant & 1u) ? (l + 32u) : l;
        const uint shift_ql    = (quadrant < 2u) ? 0u : 4u;
        const uint shift_qh    = 2u * quadrant;
        const uint ql_nibble   = (ql_base[ql_byte_idx] >> shift_ql) & 0x0Fu;
        const uint qh_bits     = (qh_base[l] >> shift_qh) & 0x03u;
        const int  qv          = (int)(ql_nibble | (qh_bits << 4)) - 32;

        const uint scale_idx = 8u * idx_half + 2u * quadrant + (l / 16u);
        const float sc_v = d * (float)blk->scales[scale_idx];
        w_tile[c][j] = (bfloat)(sc_v * (float)qv);
    }
}

inline void gguf_q2k_dequant_strip_into(
    threadgroup bfloat (&w_tile)[GGUF_MM_BN][QK_K],
    device const block_q2_K *w_blocks_base,
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
        device const block_q2_K *blk = w_blocks_base + (ulong)n * (ulong)nb + (ulong)b;
        const float d    = (float)blk->d;
        const float dmin = (float)blk->dmin;

        const uint idx_half = j / 128;
        const uint pos      = j % 128;
        const uint sub      = pos / 32;
        const uint pos_in_sub = pos % 32;
        const uint second_half = pos_in_sub / 16;
        const uint elem     = pos_in_sub % 16;
        const uint shift    = sub * 2u;

        const uint qs_byte_idx = idx_half * 32u + second_half * 16u + elem;
        const uint q = (uint)(blk->qs[qs_byte_idx] >> shift) & 3u;

        const uint scale_idx = idx_half * 8u + sub * 2u + second_half;
        const uint sc = (uint)blk->scales[scale_idx];
        const float dl = d    * (float)(sc & 0x0Fu);
        const float ml = dmin * (float)((sc >> 4) & 0x0Fu);

        w_tile[c][j] = (bfloat)(dl * (float)q - ml);
    }
}

// Q3_K signed 6-bit scale for sub-block 0..15 (offset −32 applied).
inline int gguf_q3k_get_scale(uint sub_idx, device const uint8_t *scales) {
    const uint k          = sub_idx % 4u;
    const uint group      = sub_idx / 4u;
    const uint low_byte   = (group < 2u) ? (k + 4u * group) : (k + 4u * (group - 2u));
    const uint low_shift  = (group < 2u) ? 0u : 4u;
    const uint high_shift = 2u * group;
    const uint low  = (uint)(scales[low_byte] >> low_shift) & 0x0Fu;
    const uint high = (uint)(scales[8u + k]   >> high_shift) & 0x03u;
    return (int)(low | (high << 4)) - 32;
}

inline void gguf_q3k_dequant_strip_into(
    threadgroup bfloat (&w_tile)[GGUF_MM_BN][QK_K],
    device const block_q3_K *w_blocks_base,
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
        device const block_q3_K *blk = w_blocks_base + (ulong)n * (ulong)nb + (ulong)b;
        const float d = (float)blk->d;

        const uint block_half = j / 128u;
        const uint shift_idx  = (j % 128u) / 32u;
        const uint elem_in_32 = j % 32u;

        const uint qs_byte_idx = block_half * 32u + elem_in_32;
        const uint shift       = shift_idx * 2u;
        const uint low_2       = (uint)(blk->qs[qs_byte_idx] >> shift) & 0x03u;

        const uint hbit_pos = block_half * 4u + shift_idx;
        const uint hbit     = (uint)(blk->hmask[elem_in_32] >> hbit_pos) & 0x01u;
        const int  q        = (int)low_2 - ((hbit != 0u) ? 0 : 4);

        const uint sub_idx     = j / 16u;
        const int  scale_signed = gguf_q3k_get_scale(sub_idx, blk->scales);

        w_tile[c][j] = (bfloat)(d * (float)scale_signed * (float)q);
    }
}

// Per-quant mul_mm kernels: grid (ceil(N/BN), ceil(M/BM)), TG (BN, BM).

kernel void gguf_q5_0_mul_mm_bf16(
    device const bfloat            *x        [[buffer(0)]],
    device const void              *weight   [[buffer(1)]],
    device       bfloat            *out      [[buffer(2)]],
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

kernel void gguf_q5k_mul_mm_bf16(
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

    device const block_q5_K *w_base = (device const block_q5_K *)weight;

    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        gguf_q5k_dequant_strip_into(w_tile, w_base, n_base, N, nb, b, tg_tid);
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

kernel void gguf_q6k_mul_mm_bf16(
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

    device const block_q6_K *w_base = (device const block_q6_K *)weight;

    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        gguf_q6k_dequant_strip_into(w_tile, w_base, n_base, N, nb, b, tg_tid);
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

kernel void gguf_q4_0_mul_mm_bf16(
    device const bfloat            *x        [[buffer(0)]],
    device const void              *weight   [[buffer(1)]],
    device       bfloat            *out      [[buffer(2)]],
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

    device const block_q4_0 *w_base = (device const block_q4_0 *)weight;

    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        gguf_q4_0_dequant_strip_into(w_tile, w_base, n_base, N, nb, b, tg_tid);
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

kernel void gguf_q4_1_mul_mm_bf16(
    device const bfloat            *x        [[buffer(0)]],
    device const void              *weight   [[buffer(1)]],
    device       bfloat            *out      [[buffer(2)]],
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

    device const block_q4_1 *w_base = (device const block_q4_1 *)weight;

    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        gguf_q4_1_dequant_strip_into(w_tile, w_base, n_base, N, nb, b, tg_tid);
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

kernel void gguf_q5_1_mul_mm_bf16(
    device const bfloat            *x        [[buffer(0)]],
    device const void              *weight   [[buffer(1)]],
    device       bfloat            *out      [[buffer(2)]],
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

    device const block_q5_1 *w_base = (device const block_q5_1 *)weight;

    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        gguf_q5_1_dequant_strip_into(w_tile, w_base, n_base, N, nb, b, tg_tid);
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

kernel void gguf_q2k_mul_mm_bf16(
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

    device const block_q2_K *w_base = (device const block_q2_K *)weight;

    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        gguf_q2k_dequant_strip_into(w_tile, w_base, n_base, N, nb, b, tg_tid);
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

kernel void gguf_q3k_mul_mm_bf16(
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

    device const block_q3_K *w_base = (device const block_q3_K *)weight;

    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        gguf_q3k_dequant_strip_into(w_tile, w_base, n_base, N, nb, b, tg_tid);
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
