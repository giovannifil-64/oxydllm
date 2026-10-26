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
// Reassociation is off so that two instantiations of one source, one row and
// a batch, reduce in the same order and give a row the same bits; the rest of
// fast math, contraction into fused multiply-adds included, stays for speed.
#pragma METAL fp contract(fast)
#pragma clang fp reassociate(off)


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
    uint m;
    uint out_f32;
};

constant uint AWQ_BATCH_MAX = 8u;

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
inline void awq_gemv_atomic_impl(
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

// Batched split-K AWQ GEMV for M = 2..8 decode rows: identical geometry to
// awq_gemv_atomic_impl, but the weight word is unpacked ONCE and reused for all M
// activation rows (the GGUF batch-kernel design, concurrent decode shares
// one weight read instead of falling onto the dequant+GEMM prefill path).
// `out` must be host-zeroed (atomic accumulation).
template<typename T, uint BITS>
inline void awq_gemv_batch_atomic_impl(
    device const T*       x,        // [m, in_features]
    device const uint*    qweight,
    device const uint*    qzeros,
    device const T*       scales,
    device atomic_float*  out,      // [m, out_features]
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
    uint m_rows = min(p.m, AWQ_BATCH_MAX);

    uint i_start = ks * p.chunk;
    uint i_end = min(i_start + p.chunk, p.in_features);
    if (i_start >= i_end) {
        return;
    }

    float acc[8][PACK_FACTOR];
    for (uint m = 0; m < m_rows; ++m) {
        for (uint k = 0; k < PACK_FACTOR; ++k) {
            acc[m][k] = 0.0f;
        }
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
        float w[PACK_FACTOR];
        for (uint k = 0; k < PACK_FACTOR; ++k) {
            w[k] = (float(unpack<BITS>(ww, k)) - zero_slot[k]) * scale_v[k];
        }
        for (uint m = 0; m < m_rows; ++m) {
            float xv = float(x[m * p.in_features + i]);
            for (uint k = 0; k < PACK_FACTOR; ++k) {
                acc[m][k] += xv * w[k];
            }
        }
    }

    for (uint m = 0; m < m_rows; ++m) {
        for (uint k = 0; k < PACK_FACTOR; ++k) {
            uint o = j * PACK_FACTOR + pack_position<BITS>(k);
            atomic_fetch_add_explicit(&out[m * p.out_features + o], acc[m][k], memory_order_relaxed);
        }
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

kernel void w4a16_gemv_f16_atomic(
    device const half*    x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const half*    scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_gemv_atomic_impl<half, 4>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void w4a16_gemv_bf16_atomic(
    device const bfloat*  x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const bfloat*  scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_gemv_atomic_impl<bfloat, 4>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void w4a16_gemv_batch_f16_atomic(
    device const half*    x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const half*    scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_gemv_batch_atomic_impl<half, 4>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void w4a16_gemv_batch_bf16_atomic(
    device const bfloat*  x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const bfloat*  scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_gemv_batch_atomic_impl<bfloat, 4>(x, qweight, qzeros, scales, out, p, gid);
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

kernel void w8a16_gemv_f16_atomic(
    device const half*    x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const half*    scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_gemv_atomic_impl<half, 8>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void w8a16_gemv_bf16_atomic(
    device const bfloat*  x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const bfloat*  scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_gemv_atomic_impl<bfloat, 8>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void w8a16_gemv_batch_f16_atomic(
    device const half*    x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const half*    scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_gemv_batch_atomic_impl<half, 8>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void w8a16_gemv_batch_bf16_atomic(
    device const bfloat*  x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const bfloat*  scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    awq_gemv_batch_atomic_impl<bfloat, 8>(x, qweight, qzeros, scales, out, p, gid);
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
    uint m;
    uint out_f32;
};

template<typename T, uint BITS>
inline void gptq_gemv_atomic_impl(
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
inline void gptq_gemv_batch_atomic_impl(
    device const T*       x,        // [m, in_features]
    device const uint*    qweight,
    device const uint*    qzeros,
    device const T*       scales,
    device atomic_float*  out,      // [m, out_features]
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
    uint m_rows = min(p.m, AWQ_BATCH_MAX);

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

    float acc[AWQ_BATCH_MAX];
    for (uint m = 0; m < m_rows; ++m) {
        acc[m] = 0.0f;
    }
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
        float w[PACK_FACTOR];
        for (uint k = 0; k < PACK_FACTOR; ++k) {
            uint q = (ww >> (BITS * k)) & MASK;
            w[k] = scale_v * (float(q) - zp1);
        }
        for (uint m = 0; m < m_rows; ++m) {
            device const T* xr = x + (m * p.in_features + i_base);
            float a = 0.0f;
            for (uint k = 0; k < PACK_FACTOR; ++k) {
                a += float(xr[k]) * w[k];
            }
            acc[m] += a;
        }
    }

    for (uint m = 0; m < m_rows; ++m) {
        atomic_fetch_add_explicit(&out[m * p.out_features + o], acc[m], memory_order_relaxed);
    }
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

kernel void gptq4_gemv_f16_atomic(
    device const half*    x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const half*    scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_gemv_atomic_impl<half, 4>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void gptq4_gemv_bf16_atomic(
    device const bfloat*  x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const bfloat*  scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_gemv_atomic_impl<bfloat, 4>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void gptq8_gemv_f16_atomic(
    device const half*    x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const half*    scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_gemv_atomic_impl<half, 8>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void gptq8_gemv_bf16_atomic(
    device const bfloat*  x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const bfloat*  scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_gemv_atomic_impl<bfloat, 8>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void gptq4_gemv_batch_f16_atomic(
    device const half*    x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const half*    scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_gemv_batch_atomic_impl<half, 4>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void gptq4_gemv_batch_bf16_atomic(
    device const bfloat*  x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const bfloat*  scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_gemv_batch_atomic_impl<bfloat, 4>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void gptq8_gemv_batch_f16_atomic(
    device const half*    x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const half*    scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_gemv_batch_atomic_impl<half, 8>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void gptq8_gemv_batch_bf16_atomic(
    device const bfloat*  x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const bfloat*  scales   [[buffer(3)]],
    device atomic_float*  out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    gptq_gemv_batch_atomic_impl<bfloat, 8>(x, qweight, qzeros, scales, out, p, gid);
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
// One simdgroup writes N_DST consecutive rows; no atomics. One kernel per
// quant serves one row and a batch of up to GGUF_BATCH_MAX rows alike, so a
// row gets the same bits whatever it is batched with.

#define QK5_0 32
#define GGUF_N_SIMDWIDTH 32
#define GGUF_N_DST 4
#define GGUF_N_SIMDGROUP 2

// The gemv kernels accumulate in float and store in the dtype the layer wants:
// BF16 for the hidden states, F32 for the logits.
inline void gguf_store(device void *out, ulong idx, float v, uint as_f32) {
    if (as_f32) {
        ((device float *)out)[idx] = v;
    } else {
        ((device bfloat *)out)[idx] = (bfloat)v;
    }
}

// Decode kernels: M activation vectors share one weight read, M capped at
// GGUF_BATCH_MAX for register pressure. Each quant is one template kernel
// instantiated twice, under its host names, for one row and for a batch, so
// the two are the same arithmetic and, with reassociation off, the same bits.
#define GGUF_BATCH_MAX 8
struct GgufBatchParams {
    uint in_features;
    uint out_features;
    uint m_batch;
    uint out_f32;
};

typedef struct {
    half     d;
    uint8_t  qh[4];
    uint8_t  qs[QK5_0/2];
} block_q5_0;

inline float gguf_dot_y(device const block_q5_0 *qb,
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


typedef struct {
    half     d;
    uint8_t  qs[QK5_0 / 2];
} block_q4_0;

inline float gguf_dot_y(device const block_q4_0 *qb,
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


typedef struct {
    half     d;
    half     m;
    uint8_t  qs[QK5_0 / 2];
} block_q4_1;

inline float gguf_dot_y(device const block_q4_1 *qb,
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


typedef struct {
    half     d;
    half     m;
    uint8_t  qh[4];
    uint8_t  qs[QK5_0 / 2];
} block_q5_1;

inline float gguf_dot_y(device const block_q5_1 *qb,
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


// Batched legacy-quant decode GEMV (2 simdgroups x GGUF_N_DST rows), one
// instantiation per quant: only the block type and dot_y differ. Weights are
// read once per (row, block) and L2-amortized across the inner M-loop.
template <typename BLOCK, int MAXM>
kernel void gguf_legacy_gemv_kernel(
    device const bfloat        *x        [[buffer(0)]],
    device const void          *weight   [[buffer(1)]],
    device       void          *out      [[buffer(2)]],
    constant     GgufBatchParams &p      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N = p.out_features, K = p.in_features;
    const uint M = min(p.m_batch, (uint)MAXM);
    const uint nb = K / QK5_0;
    const uint r0 = tgpig.x;
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * GGUF_N_DST;
    if (first_row >= N) return;
    device const BLOCK *x_blocks =
        (device const BLOCK *)weight + (ulong)first_row * (ulong)nb;
    const uint ix = (tiisg / 2);
    const uint il = (tiisg % 2) * 8;
    float sumf[GGUF_N_DST][MAXM];
    for (uint row = 0; row < GGUF_N_DST; ++row)
        for (uint m = 0; m < MAXM; ++m) sumf[row][m] = 0.f;
    for (uint ib = ix; ib < nb; ib += GGUF_N_SIMDWIDTH / 2) {
        for (uint row = 0; row < GGUF_N_DST; ++row) {
            if (first_row + row >= N) continue;
            device const BLOCK *blk = x_blocks + ib + row * nb;
            for (uint m = 0; m < M; ++m) {
                device const bfloat *yb =
                    x + (ulong)m * (ulong)K + (ulong)ib * QK5_0 + il;
                float yl[16]; float sumy = 0.f;
                for (int i = 0; i < 8; i += 2) {
                    const float a0=(float)yb[i+0], a1=(float)yb[i+1];
                    const float a16=(float)yb[i+16], a17=(float)yb[i+17];
                    sumy += a0 + a1 + a16 + a17;
                    yl[i+0]=a0; yl[i+1]=a1/256.f;
                    yl[i+8]=a16/16.f; yl[i+9]=a17/4096.f;
                }
                sumf[row][m] += gguf_dot_y(blk, sumy, yl, il);
            }
        }
    }
    for (uint row = 0; row < GGUF_N_DST; ++row) {
        const uint r = first_row + row;
        for (uint m = 0; m < M; ++m) {
            const float tot = simd_sum(sumf[row][m]);
            if (tiisg == 0 && r < N) gguf_store(out, (ulong)m * (ulong)N + r, tot, p.out_f32);
        }
    }
}

template [[host_name("gguf_q5_0_gemv_bf16")]] kernel void gguf_legacy_gemv_kernel<block_q5_0, 1>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q4_0_gemv_bf16")]] kernel void gguf_legacy_gemv_kernel<block_q4_0, 1>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q4_1_gemv_bf16")]] kernel void gguf_legacy_gemv_kernel<block_q4_1, 1>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q5_1_gemv_bf16")]] kernel void gguf_legacy_gemv_kernel<block_q5_1, 1>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q5_0_gemv_batch_bf16")]] kernel void gguf_legacy_gemv_kernel<block_q5_0, GGUF_BATCH_MAX>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q4_0_gemv_batch_bf16")]] kernel void gguf_legacy_gemv_kernel<block_q4_0, GGUF_BATCH_MAX>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q4_1_gemv_batch_bf16")]] kernel void gguf_legacy_gemv_kernel<block_q4_1, GGUF_BATCH_MAX>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q5_1_gemv_batch_bf16")]] kernel void gguf_legacy_gemv_kernel<block_q5_1, GGUF_BATCH_MAX>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);

#define QK8_0 32
#define NB_Q8_0 8

typedef struct {
    half    d;
    int8_t  qs[QK8_0];
} block_q8_0;


// Batched Q8_0 decode GEMV (2 simdgroups x GGUF_N_DST rows per TG).
template <int MAXM>
kernel void gguf_q8_0_gemv_kernel(
    device const bfloat        *x        [[buffer(0)]],
    device const void          *weight   [[buffer(1)]],
    device       void          *out      [[buffer(2)]],
    constant     GgufBatchParams &p      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint M  = min(p.m_batch, (uint)MAXM);
    const uint nb = K / QK8_0;

    const uint r0 = tgpig.x;
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * GGUF_N_DST;
    if (first_row >= N) return;

    device const block_q8_0 *x_blocks =
        (device const block_q8_0 *)weight + (ulong)first_row * (ulong)nb;

    const uint ix = tiisg / 4;
    const uint il = tiisg % 4;
    const uint y_off = NB_Q8_0 * il;

    float sumf[GGUF_N_DST][MAXM];
    for (uint row = 0; row < GGUF_N_DST; ++row) {
        for (uint m = 0; m < MAXM; ++m) {
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
                gguf_store(out, (ulong)m * (ulong)N + r, tot, p.out_f32);
            }
        }
    }
}
template [[host_name("gguf_q8_0_gemv_bf16")]] kernel void gguf_q8_0_gemv_kernel<1>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q8_0_gemv_batch_bf16")]] kernel void gguf_q8_0_gemv_kernel<GGUF_BATCH_MAX>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);

// Q4_K geometry: 1 simdgroup × N_DST=4 rows per TG (register pressure forbids 2).

#define QK_K 256
#define Q4_K_SCALE_SIZE 12

typedef struct {
    half     d;
    half     dmin;
    uint8_t  scales[Q4_K_SCALE_SIZE];
    uint8_t  qs[QK_K / 2];
} block_q4_K;


// Batched Q4_K decode GEMV (1 simdgroup x GGUF_N_DST rows, same geometry as the
// gemv): weights read once per (row, super-block), reused across the inner M-loop.
template <int MAXM>
kernel void gguf_q4k_gemv_kernel(
    device const bfloat        *x        [[buffer(0)]],
    device const void          *weight   [[buffer(1)]],
    device       void          *out      [[buffer(2)]],
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
    const uint M  = min(p.m_batch, (uint)MAXM);
    const uint nb = K / QK_K;

    const uint r0 = tgpig.x;
    const uint first_row = r0 * GGUF_N_DST;
    if (first_row >= N) return;

    device const block_q4_K *x_base = (device const block_q4_K *)weight;
    device const block_q4_K *x_row0 = x_base + (ulong)first_row * (ulong)nb;

    const uint y_off = 64 * iq + 8 * ir;

    float sumf[GGUF_N_DST][MAXM];
    for (uint row = 0; row < GGUF_N_DST; ++row) {
        for (uint m = 0; m < MAXM; ++m) {
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
                gguf_store(out, (ulong)m * (ulong)N + r, tot, p.out_f32);
            }
        }
    }
}
template [[host_name("gguf_q4k_gemv_bf16")]] kernel void gguf_q4k_gemv_kernel<1>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q4k_gemv_batch_bf16")]] kernel void gguf_q4k_gemv_kernel<GGUF_BATCH_MAX>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);

// Q5_K geometry: 2 simdgroups × 2 rows per TG.
typedef struct {
    half     d;
    half     dmin;
    uint8_t  scales[Q4_K_SCALE_SIZE];
    uint8_t  qh[QK_K / 8];
    uint8_t  qs[QK_K / 2];
} block_q5_K;


// Batched Q5_K decode GEMV (2 simdgroups x 2 rows per TG): the single-row
// kernel's arithmetic, run for each of the M rows in turn, so a row gets the
// bits it gets alone.
template <int MAXM>
kernel void gguf_q5k_gemv_kernel(
    device const bfloat        *x        [[buffer(0)]],
    device const void          *weight   [[buffer(1)]],
    device       void          *out      [[buffer(2)]],
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
    const uint M  = min(p.m_batch, (uint)MAXM);
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

    float yl[16], yh[16];

    float sumf[2][MAXM];
    for (uint row = 0; row < 2; ++row) {
        for (uint m = 0; m < MAXM; ++m) {
            sumf[row][m] = 0.f;
        }
    }

    for (uint i = ix; i < nb; i += 4) {
        for (uint m = 0; m < M; ++m) {
            device const bfloat *y1 = x + (ulong)m * (ulong)K + (ulong)i * QK_K + y_offset;
            device const bfloat *y2 = y1 + 128;
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (uint l = 0; l < 8; ++l) {
                yl[l+0] = (float)y1[l+ 0]; sumy[0] += yl[l+0];
                yl[l+8] = (float)y1[l+32]; sumy[1] += yl[l+8];
                yh[l+0] = (float)y2[l+ 0]; sumy[2] += yh[l+0];
                yh[l+8] = (float)y2[l+32]; sumy[3] += yh[l+8];
            }

            for (uint row = 0; row < 2; ++row) {
                device const block_q5_K *xb = x_blocks + i + row * nb;
                device const uint8_t *q1 = xb->qs + q_offset;
                device const uint8_t *q2 = q1 + 64;
                device const uint8_t *qh = xb->qh + l0;
                device const half    *dh = &xb->d;
                device const uint16_t *a = (device const uint16_t *)xb->scales + iq;

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
                sumf[row][m] += dall * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                        sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                        sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                        sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                                dmin * (sumy[0]*sc8[2] + sumy[1]*sc8[3] +
                                        sumy[2]*sc8[6] + sumy[3]*sc8[7]);
            }
        }
    }

    for (uint row = 0; row < 2; ++row) {
        const uint r = first_row + row;
        for (uint m = 0; m < M; ++m) {
            const float tot = simd_sum(sumf[row][m]);
            if (tiisg == 0 && r < N) {
                gguf_store(out, (ulong)m * (ulong)N + r, tot, p.out_f32);
            }
        }
    }
}
template [[host_name("gguf_q5k_gemv_bf16")]] kernel void gguf_q5k_gemv_kernel<1>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q5k_gemv_batch_bf16")]] kernel void gguf_q5k_gemv_kernel<GGUF_BATCH_MAX>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);

// Q6_K geometry: 2 simdgroups × 1 row per TG.
typedef struct {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    half     d;
} block_q6_K;


// Batched Q6_K decode GEMV (2 simdgroups x 1 row per TG): the single-row
// kernel's arithmetic, run for each of the M rows in turn, so a row gets the
// bits it gets alone.
template <int MAXM>
kernel void gguf_q6k_gemv_kernel(
    device const bfloat        *x        [[buffer(0)]],
    device const void          *weight   [[buffer(1)]],
    device       void          *out      [[buffer(2)]],
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
    const uint M  = min(p.m_batch, (uint)MAXM);
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

    float sumf[MAXM];
    for (uint m = 0; m < MAXM; ++m) {
        sumf[m] = 0.f;
    }

    for (uint i = ix; i < nb; i += 2) {
        device const block_q6_K *xb = x_blocks + i;
        device const uint8_t *q1 = xb->ql + q_offset_l;
        device const uint8_t *q2 = q1 + 32;
        device const uint8_t *qh = xb->qh + q_offset_h;
        device const int8_t  *sc = xb->scales + is;
        const float dall = (float)xb->d;

        for (uint m = 0; m < M; ++m) {
            device const bfloat *y = x + (ulong)m * (ulong)K + (ulong)i * QK_K + y_offset;
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
            sumf[m] += dall * (sums[0]*sc[0] + sums[1]*sc[2] +
                               sums[2]*sc[4] + sums[3]*sc[6]);
        }
    }

    for (uint m = 0; m < M; ++m) {
        const float tot = simd_sum(sumf[m]);
        if (tiisg == 0 && row < N) {
            gguf_store(out, (ulong)m * (ulong)N + row, tot, p.out_f32);
        }
    }
}
template [[host_name("gguf_q6k_gemv_bf16")]] kernel void gguf_q6k_gemv_kernel<1>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q6k_gemv_batch_bf16")]] kernel void gguf_q6k_gemv_kernel<GGUF_BATCH_MAX>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);

// Q2_K geometry: 2 simdgroups × 4 rows per TG.
typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    half     d;
    half     dmin;
} block_q2_K;


// Batched Q2_K decode GEMV (2 simdgroups x GGUF_N_DST rows per TG, step-advance).
template <int MAXM>
kernel void gguf_q2k_gemv_kernel(
    device const bfloat        *x        [[buffer(0)]],
    device const void          *weight   [[buffer(1)]],
    device       void          *out      [[buffer(2)]],
    constant     GgufBatchParams &p      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint M  = min(p.m_batch, (uint)MAXM);
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

    float sumf[GGUF_N_DST][MAXM];
    for (uint row = 0; row < GGUF_N_DST; ++row) {
        for (uint m = 0; m < MAXM; ++m) {
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
                gguf_store(out, (ulong)m * (ulong)N + r, tot, p.out_f32);
            }
        }
    }
}
template [[host_name("gguf_q2k_gemv_bf16")]] kernel void gguf_q2k_gemv_kernel<1>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q2k_gemv_batch_bf16")]] kernel void gguf_q2k_gemv_kernel<GGUF_BATCH_MAX>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);

// Q3_K geometry: 2 simdgroups × 2 rows per TG.
typedef struct {
    uint8_t  hmask[QK_K / 8];
    uint8_t  qs[QK_K / 4];
    uint8_t  scales[12];
    half     d;
} block_q3_K;


// Batched Q3_K decode GEMV (2 simdgroups x 2 rows per TG, step-advance): the
// single-row kernel's arithmetic, run for each of the M rows in turn, so a row
// gets the bits it gets alone; weights and scales are read once per (row,
// super-block) and reused across the M-loop.
template <int MAXM>
kernel void gguf_q3k_gemv_kernel(
    device const bfloat        *x        [[buffer(0)]],
    device const void          *weight   [[buffer(1)]],
    device       void          *out      [[buffer(2)]],
    constant     GgufBatchParams &p      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint M  = min(p.m_batch, (uint)MAXM);
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

    uint32_t scales32, aux32;
    thread uint16_t   *scales16 = (thread uint16_t   *)&scales32;
    thread const int8_t *scales = (thread const int8_t *)&scales32;

    float sumf1[2][MAXM];
    float sumf2[2][MAXM];
    for (uint row = 0; row < 2; ++row) {
        for (uint m = 0; m < MAXM; ++m) {
            sumf1[row][m] = 0.f;
            sumf2[row][m] = 0.f;
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

            for (uint m = 0; m < M; ++m) {
                device const bfloat *y1 = x + (ulong)m * (ulong)K + (ulong)i * QK_K + y_offset;
                for (uint l = 0; l < 8; ++l) {
                    yl[l +  0] = (float)y1[l +  0];
                    yl[l +  8] = (float)y1[l + 16];
                    yl[l + 16] = (float)y1[l + 32];
                    yl[l + 24] = (float)y1[l + 48];
                }

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
                sumf1[row][m] += d1 * (float)(scales[0] - 32);
                sumf2[row][m] += d2 * (float)(scales[2] - 32);

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
                sumf1[row][m] += d1 * (float)(scales[1] - 32);
                sumf2[row][m] += d2 * (float)(scales[3] - 32);
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
            const float sumf = (sumf1[row][m] + 0.25f * sumf2[row][m]) / (float)(1u << shift);
            const float tot = simd_sum(sumf);
            if (tiisg == 0 && r < N) {
                gguf_store(out, (ulong)m * (ulong)N + r, tot, p.out_f32);
            }
        }
    }
}
template [[host_name("gguf_q3k_gemv_bf16")]] kernel void gguf_q3k_gemv_kernel<1>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("gguf_q3k_gemv_batch_bf16")]] kernel void gguf_q3k_gemv_kernel<GGUF_BATCH_MAX>(device const bfloat *, device const void *, device void *, constant GgufBatchParams &, uint3, uint, uint);

// Tiled GEMM parameters and tile geometry, shared with the MXFP4 GEMM below.

struct GgufMatmulParams {
    uint m_total;
    uint n_total;
    uint k_total;
};

constant uint GGUF_MM_BM = 16;
constant uint GGUF_MM_BN = 16;
constant uint GGUF_MM_TG = GGUF_MM_BM * GGUF_MM_BN;

// Dense BF16 decode GEMV: out[m, n] = sum_k x[m, k] * w[n, k] with x [M, K] and
// w [N, K] in BF16, F32 accumulation, two simdgroups of BF16_ROWS_PER_SG rows
// per threadgroup, each lane eight consecutive elements of k per step. One
// template instantiated for one row and for a batch, so a row gets the same
// bits whatever it is batched with. K must be a multiple of eight.
#define BF16_ROWS_PER_SG 2
template <int MAXM>
kernel void bf16_gemv(
    device const bfloat        *x        [[buffer(0)]],
    device const bfloat        *w        [[buffer(1)]],
    device       void          *out      [[buffer(2)]],
    constant     GgufBatchParams &p      [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N = p.out_features, K = p.in_features;
    const uint M = min(p.m_batch, (uint)MAXM);
    const uint first_row = (tgpig.x * 2 + sgitg) * BF16_ROWS_PER_SG;
    if (first_row >= N) return;
    float acc[BF16_ROWS_PER_SG][MAXM];
    for (uint r = 0; r < BF16_ROWS_PER_SG; ++r)
        for (uint m = 0; m < MAXM; ++m) acc[r][m] = 0.f;
    for (uint k = tiisg * 8; k + 8 <= K; k += 32 * 8) {
        float xv[MAXM][8];
        for (uint m = 0; m < M; ++m) {
            device const bfloat4 *xp = (device const bfloat4 *)(x + m * K + k);
            const float4 a = float4(xp[0]);
            const float4 b = float4(xp[1]);
            xv[m][0] = a.x; xv[m][1] = a.y; xv[m][2] = a.z; xv[m][3] = a.w;
            xv[m][4] = b.x; xv[m][5] = b.y; xv[m][6] = b.z; xv[m][7] = b.w;
        }
        for (uint r = 0; r < BF16_ROWS_PER_SG; ++r) {
            const uint row = first_row + r;
            if (row >= N) break;
            device const bfloat4 *wp = (device const bfloat4 *)(w + row * K + k);
            const float4 c = float4(wp[0]);
            const float4 d = float4(wp[1]);
            const float wv[8] = {c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w};
            for (uint m = 0; m < M; ++m) {
                float s = 0.f;
                for (uint j = 0; j < 8; ++j) s += xv[m][j] * wv[j];
                acc[r][m] += s;
            }
        }
    }
    for (uint r = 0; r < BF16_ROWS_PER_SG; ++r) {
        const uint row = first_row + r;
        for (uint m = 0; m < M; ++m) {
            const float tot = simd_sum(acc[r][m]);
            if (tiisg == 0 && row < N) {
                gguf_store(out, (ulong)m * (ulong)N + row, tot, p.out_f32);
            }
        }
    }
}

template [[host_name("bf16_gemv_bf16")]] kernel void bf16_gemv<1>(device const bfloat *, device const bfloat *, device void *, constant GgufBatchParams &, uint3, uint, uint);
template [[host_name("bf16_gemv_batch_bf16")]] kernel void bf16_gemv<GGUF_BATCH_MAX>(device const bfloat *, device const bfloat *, device void *, constant GgufBatchParams &, uint3, uint, uint);

#define MXFP4_BLOCK 32
constant float MXFP4_LUT16[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// Batched decode GEMV (1 <= M <= GGUF_BATCH_MAX), 2 simdgroups x GGUF_N_DST rows.
kernel void mxfp4_gemv_batch_bf16(
    device const bfloat        *x       [[buffer(0)]],   // [M, K]
    device const uint8_t       *blocks  [[buffer(1)]],   // [N, K/32, 16] flat
    device const uint8_t       *scales  [[buffer(2)]],   // [N, K/32] flat
    device       bfloat        *out     [[buffer(3)]],   // [M, N]
    constant     GgufBatchParams &p     [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint N  = p.out_features;
    const uint K  = p.in_features;
    const uint M  = min(p.m_batch, (uint)GGUF_BATCH_MAX);
    const uint nb = K / MXFP4_BLOCK;

    const uint r0 = tgpig.x;
    const uint first_row = (r0 * GGUF_N_SIMDGROUP + sgitg) * GGUF_N_DST;
    if (first_row >= N) return;

    const uint ix = tiisg / 4;
    const uint il = tiisg % 4;
    const uint y_off = 8 * il;

    float sumf[GGUF_N_DST][GGUF_BATCH_MAX];
    for (uint row = 0; row < GGUF_N_DST; ++row)
        for (uint m = 0; m < GGUF_BATCH_MAX; ++m) sumf[row][m] = 0.f;

    for (uint ib = ix; ib < nb; ib += GGUF_N_SIMDWIDTH / 4) {
        for (uint row = 0; row < GGUF_N_DST; ++row) {
            if (first_row + row >= N) continue;
            const ulong base = (ulong)(first_row + row) * (ulong)nb + ib;
            device const uint8_t *qs = blocks + base * 16 + 4 * il;
            const float scale = exp2((float)scales[base] - 127.0f);
            // Dequantize once into registers (scale folded); m-loop is pure FMA.
            float w[8];
            for (uint i = 0; i < 4; ++i) {
                const uint8_t byte = qs[i];
                w[2*i]   = scale * MXFP4_LUT16[byte & 0xF];
                w[2*i+1] = scale * MXFP4_LUT16[byte >> 4];
            }
            for (uint m = 0; m < M; ++m) {
                device const bfloat *y =
                    x + (ulong)m * (ulong)K + (ulong)ib * MXFP4_BLOCK + y_off;
                float acc = 0.f;
                for (uint j = 0; j < 8; ++j) {
                    acc += (float)y[j] * w[j];
                }
                sumf[row][m] += acc;
            }
        }
    }

    for (uint row = 0; row < GGUF_N_DST; ++row) {
        const uint r = first_row + row;
        for (uint m = 0; m < M; ++m) {
            const float tot = simd_sum(sumf[row][m]);
            if (tiisg == 0 && r < N) out[(ulong)m * (ulong)N + r] = (bfloat)tot;
        }
    }
}

inline void mxfp4_dequant_strip_into(
    threadgroup bfloat (&w_tile)[GGUF_MM_BN][MXFP4_BLOCK],
    device const uint8_t *blocks,
    device const uint8_t *scales,
    uint n_base,
    uint n_total,
    uint nb,
    uint b,
    uint tg_tid)
{
    for (uint kk = tg_tid; kk < GGUF_MM_BN * MXFP4_BLOCK; kk += GGUF_MM_TG) {
        const uint c = kk / MXFP4_BLOCK;
        const uint j = kk % MXFP4_BLOCK;
        const uint n = n_base + c;
        if (n >= n_total) { continue; }
        const ulong base = (ulong)n * (ulong)nb + b;
        const uint8_t byte = blocks[base * 16 + j / 2];
        const uint8_t nib = (j & 1) ? (byte >> 4) : (byte & 0xF);
        w_tile[c][j] = (bfloat)(exp2((float)scales[base] - 127.0f) * MXFP4_LUT16[nib]);
    }
}

kernel void mxfp4_mul_mm_bf16(
    device const bfloat            *x       [[buffer(0)]],
    device const uint8_t           *blocks  [[buffer(1)]],
    device const uint8_t           *scales  [[buffer(2)]],
    device       bfloat            *out     [[buffer(3)]],
    constant     GgufMatmulParams  &p       [[buffer(4)]],
    uint3  tg_tid_xy [[thread_position_in_threadgroup]],
    uint3  tgpig     [[threadgroup_position_in_grid]])
{
    threadgroup bfloat w_tile[GGUF_MM_BN][MXFP4_BLOCK];

    const uint M  = p.m_total;
    const uint N  = p.n_total;
    const uint K  = p.k_total;
    const uint nb = K / MXFP4_BLOCK;

    const uint n_base = tgpig.x * GGUF_MM_BN;
    const uint m_base = tgpig.y * GGUF_MM_BM;
    const uint n_off  = tg_tid_xy.x;
    const uint m_off  = tg_tid_xy.y;
    const uint n      = n_base + n_off;
    const uint m      = m_base + m_off;
    const uint tg_tid = m_off * GGUF_MM_BN + n_off;

    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        mxfp4_dequant_strip_into(w_tile, blocks, scales, n_base, N, nb, b, tg_tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (m < M && n < N) {
            device const bfloat *x_chunk = x + (ulong)m * (ulong)K + (ulong)b * (ulong)MXFP4_BLOCK;
            float partial = 0.0f;
            for (uint j = 0; j < MXFP4_BLOCK; ++j) {
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

// ── Decode SDPA with attention sinks (GPT-OSS) ──────────────────────────────
// One simdgroup per (q-head): online softmax over the kv positions with the
// per-head sink logit folded into the denominator (the sink contributes no
// value vector). Reads K/V per kv-head directly, GQA without repeat_kv.
// q: [H, D] (q_len = 1), k/v: [KVH, L, D], sinks: [H], out: [H, D]. BF16 I/O,
// F32 accumulation. D <= SDPA_SINK_MAX_D.

#define SDPA_SINK_MAX_D 128

struct SdpaSinkParams {
    uint  n_heads;
    uint  n_kv_heads;
    uint  kv_len;
    uint  head_dim;
    float scale;
};

kernel void sdpa_vector_sink_bf16(
    device const bfloat  *q      [[buffer(0)]],
    device const bfloat  *k      [[buffer(1)]],
    device const bfloat  *v      [[buffer(2)]],
    device const bfloat  *sinks  [[buffer(3)]],
    device       bfloat  *out    [[buffer(4)]],
    constant SdpaSinkParams &p   [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    if (h >= p.n_heads) return;
    const uint D = p.head_dim;
    const uint kvh = h / (p.n_heads / p.n_kv_heads);

    threadgroup float q_s[SDPA_SINK_MAX_D];
    for (uint d = tiisg; d < D; d += 32) {
        q_s[d] = (float)q[(ulong)h * D + d] * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const bfloat *k_base = k + (ulong)kvh * (ulong)p.kv_len * D;
    device const bfloat *v_base = v + (ulong)kvh * (ulong)p.kv_len * D;

    // Per-thread online softmax over a strided slice of kv positions.
    float m = -INFINITY;
    float l = 0.0f;
    float acc[SDPA_SINK_MAX_D];
    for (uint d = 0; d < D; ++d) {
        acc[d] = 0.0f;
    }

    for (uint j = tiisg; j < p.kv_len; j += 32) {
        device const bfloat *kj = k_base + (ulong)j * D;
        float s = 0.0f;
        for (uint d = 0; d < D; ++d) {
            s += q_s[d] * (float)kj[d];
        }
        const float m_new = max(m, s);
        const float corr = exp(m - m_new);
        const float w = exp(s - m_new);
        l = l * corr + w;
        device const bfloat *vj = v_base + (ulong)j * D;
        for (uint d = 0; d < D; ++d) {
            acc[d] = acc[d] * corr + w * (float)vj[d];
        }
        m = m_new;
    }

    // Combine the 32 per-thread partials, then fold in the sink logit.
    const float m_g = simd_max(m);
    const float corr_g = (m == -INFINITY) ? 0.0f : exp(m - m_g);
    float l_g = simd_sum(l * corr_g);

    const float sink = (float)sinks[h];
    const float m_f = max(m_g, sink);
    const float denom = l_g * exp(m_g - m_f) + exp(sink - m_f);
    const float acc_scale = exp(m_g - m_f) / denom;

    for (uint d = 0; d < D; ++d) {
        const float num = simd_sum(acc[d] * corr_g);
        if (tiisg == 0) {
            out[(ulong)h * D + d] = (bfloat)(num * acc_scale);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic split-K epilogues for the AWQ/GPTQ GEMV kernels.
//
// The `*_atomic` kernels above accumulate their split-K partials with
// `atomic_fetch_add` on floats: the scheduling-dependent addition order makes
// the result vary run to run, which flips borderline tokens at temperature 0.
// These variants keep the same per-thread main loop but place every k-split
// of an output column in one threadgroup and reduce the partials in
// threadgroup memory in a fixed order with a single writer per output, so the
// result is bitwise reproducible. The threadgroup is (QDET_TGW, S): tid.x is
// the output column (memory coalescing over adjacent columns is preserved),
// tid.y the k-split. Dispatch must use exactly S rows so every partial slot
// is written (inactive threads store zeros).
// ─────────────────────────────────────────────────────────────────────────────

constant uint QDET_TGW = 32u;
constant uint QDET_S = 16u;       // k-splits for the GPTQ M=1 GEMV (TG = 512)
constant uint QDET_S_BATCH = 16u; // k-splits for the GPTQ batched variant
constant uint QDET_A_TGW = 32u;   // AWQ M=1: TG = 1024, epilogue in two
constant uint QDET_A_S = 32u;     // half-pack rounds (scratch 16 KB at 4-bit)
constant uint QDET_AB_TGW = 16u;  // AWQ batch: narrower tile, more k-splits
constant uint QDET_AB_S = 32u;    // (TG = 512, threadgroup scratch 16 KB at 4-bit)

// The deterministic kernels store each output once, in F32 for the logits or
// in the activation dtype for the hidden states, so nothing zeroes or casts
// an accumulator around them; the atomic bench variants keep their F32 one.
template<typename T>
inline void quant_store(device void* out, ulong idx, float v, uint as_f32) {
    if (as_f32) {
        ((device float*)out)[idx] = v;
    } else {
        ((device T*)out)[idx] = T(v);
    }
}

template<typename T, uint BITS, uint TGW, uint S>
inline void awq_gemv_det_impl(
    device const T*       x,
    device const uint*    qweight,
    device const uint*    qzeros,
    device const T*       scales,
    device void*          out,
    constant W4A16Params& p,
    threadgroup float*    partials,   // [S][TGW][PACK_FACTOR / 2]
    uint2 tgid,
    uint2 tid)
{
    constexpr uint PACK_FACTOR = 32u / BITS;
    // The epilogue reduces half of the pack slots per round so the scratch
    // stays within the 32 KB threadgroup limit at S = 32.
    constexpr uint HALF = PACK_FACTOR / 2u;

    uint j = tgid.x * TGW + tid.x;
    uint ks = tid.y;

    float acc[PACK_FACTOR];
    for (uint k = 0; k < PACK_FACTOR; ++k) {
        acc[k] = 0.0f;
    }

    if (j < p.packed_out && ks < p.k_splits) {
        uint i_start = ks * p.chunk;
        uint i_end = min(i_start + p.chunk, p.in_features);

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
    }

    for (uint round = 0; round < 2u; ++round) {
        for (uint h = 0; h < HALF; ++h) {
            partials[(ks * TGW + tid.x) * HALF + h] = acc[round * HALF + h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Row `ty` reduces and stores pack slot `round * HALF + ty`.
        if (tid.y < HALF && j < p.packed_out) {
            uint k = round * HALF + tid.y;
            float sum = 0.0f;
            for (uint s = 0; s < S; ++s) {
                sum += partials[(s * TGW + tid.x) * HALF + (k - round * HALF)];
            }
            quant_store<T>(out, j * PACK_FACTOR + pack_position<BITS>(k), sum, p.out_f32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template<typename T, uint BITS, uint TGW, uint S>
inline void awq_gemv_batch_det_impl(
    device const T*       x,        // [m, in_features]
    device const uint*    qweight,
    device const uint*    qzeros,
    device const T*       scales,
    device void*          out,      // [m, out_features]
    constant W4A16Params& p,
    threadgroup float*    partials, // [S][TGW][PACK_FACTOR]
    uint2 tgid,
    uint2 tid)
{
    constexpr uint PACK_FACTOR = 32u / BITS;

    uint j = tgid.x * TGW + tid.x;
    uint ks = tid.y;
    uint m_rows = min(p.m, AWQ_BATCH_MAX);

    float acc[AWQ_BATCH_MAX][PACK_FACTOR];
    for (uint m = 0; m < m_rows; ++m) {
        for (uint k = 0; k < PACK_FACTOR; ++k) {
            acc[m][k] = 0.0f;
        }
    }

    if (j < p.packed_out && ks < p.k_splits) {
        uint i_start = ks * p.chunk;
        uint i_end = min(i_start + p.chunk, p.in_features);

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
            float w[PACK_FACTOR];
            for (uint k = 0; k < PACK_FACTOR; ++k) {
                w[k] = (float(unpack<BITS>(ww, k)) - zero_slot[k]) * scale_v[k];
            }
            for (uint m = 0; m < m_rows; ++m) {
                float xv = float(x[m * p.in_features + i]);
                for (uint k = 0; k < PACK_FACTOR; ++k) {
                    acc[m][k] += xv * w[k];
                }
            }
        }
    }

    // One reduction round per activation row, reusing the same scratch. All
    // threads see the same m_rows, so the barriers stay uniform. The write is
    // spread over the first PACK_FACTOR split rows: row `ty` reduces and
    // stores pack slot `ty` of its column.
    for (uint m = 0; m < m_rows; ++m) {
        for (uint k = 0; k < PACK_FACTOR; ++k) {
            partials[(ks * TGW + tid.x) * PACK_FACTOR + k] = acc[m][k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid.y < PACK_FACTOR && j < p.packed_out) {
            uint k = tid.y;
            float sum = 0.0f;
            for (uint s = 0; s < S; ++s) {
                sum += partials[(s * TGW + tid.x) * PACK_FACTOR + k];
            }
            quant_store<T>(out, m * p.out_features + j * PACK_FACTOR + pack_position<BITS>(k), sum, p.out_f32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One implementation serves one row and a batch alike, so a row gets the
// same bits whatever it is batched with (the single-row kernel is this at
// m = 1); the split depth is the same for both.
template<typename T, uint BITS>
inline void gptq_gemv_batch_det_impl(
    device const T*       x,        // [m, in_features]
    device const uint*    qweight,
    device const uint*    qzeros,
    device const T*       scales,
    device void*          out,      // [m, out_features]
    constant GptqParams&  p,
    threadgroup float*    partials, // [QDET_S_BATCH][QDET_TGW]
    uint2 tgid,
    uint2 tid)
{
    constexpr uint PACK_FACTOR = 32u / BITS;
    constexpr uint MASK = (1u << BITS) - 1u;

    uint o = tgid.x * QDET_TGW + tid.x;
    uint ks = tid.y;
    uint m_rows = min(p.m, AWQ_BATCH_MAX);

    float acc[AWQ_BATCH_MAX];
    for (uint m = 0; m < m_rows; ++m) {
        acc[m] = 0.0f;
    }

    if (o < p.out_features && ks < p.k_splits) {
        uint i_start = ks * p.chunk;
        uint i_end = min(i_start + p.chunk, p.in_features);
        uint w_start = i_start / PACK_FACTOR;
        uint w_end = (i_end + PACK_FACTOR - 1u) / PACK_FACTOR;

        uint o_word = o / PACK_FACTOR;
        uint o_slot = o % PACK_FACTOR;
        uint o_shift = BITS * o_slot;
        uint qzeros_inner = p.out_features / PACK_FACTOR;

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
            float w[PACK_FACTOR];
            for (uint k = 0; k < PACK_FACTOR; ++k) {
                uint q = (ww >> (BITS * k)) & MASK;
                w[k] = scale_v * (float(q) - zp1);
            }
            for (uint m = 0; m < m_rows; ++m) {
                device const T* xr = x + (m * p.in_features + i_base);
                float a = 0.0f;
                for (uint k = 0; k < PACK_FACTOR; ++k) {
                    a += float(xr[k]) * w[k];
                }
                acc[m] += a;
            }
        }
    }

    for (uint m = 0; m < m_rows; ++m) {
        partials[ks * QDET_TGW + tid.x] = acc[m];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid.y == 0 && o < p.out_features) {
            float sum = 0.0f;
            for (uint s = 0; s < QDET_S_BATCH; ++s) {
                sum += partials[s * QDET_TGW + tid.x];
            }
            quant_store<T>(out, m * p.out_features + o, sum, p.out_f32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <typename T, uint BITS, uint TGW, uint S>
kernel void awq_det_kernel(
    device const T*       x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const T*       scales   [[buffer(3)]],
    device void*          out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]])
{
    threadgroup float partials[S * TGW * (32u / BITS) / 2u];
    awq_gemv_det_impl<T, BITS, TGW, S>(x, qweight, qzeros, scales, out, p, partials, tgid, tid);
}

template <typename T, uint BITS, uint S>
kernel void awq_det_batch_kernel(
    device const T*       x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const T*       scales   [[buffer(3)]],
    device void*          out      [[buffer(4)]],
    constant W4A16Params& p        [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]])
{
    threadgroup float partials[S * QDET_AB_TGW * (32u / BITS)];
    awq_gemv_batch_det_impl<T, BITS, QDET_AB_TGW, S>(x, qweight, qzeros, scales, out, p, partials, tgid, tid);
}

template <typename T, uint BITS>
kernel void gptq_det_kernel(
    device const T*       x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const T*       scales   [[buffer(3)]],
    device void*          out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]])
{
    threadgroup float partials[QDET_S_BATCH * QDET_TGW];
    gptq_gemv_batch_det_impl<T, BITS>(x, qweight, qzeros, scales, out, p, partials, tgid, tid);
}

template <typename T, uint BITS>
kernel void gptq_det_batch_kernel(
    device const T*       x        [[buffer(0)]],
    device const uint*    qweight  [[buffer(1)]],
    device const uint*    qzeros   [[buffer(2)]],
    device const T*       scales   [[buffer(3)]],
    device void*          out      [[buffer(4)]],
    constant GptqParams&  p        [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]])
{
    threadgroup float partials[QDET_S_BATCH * QDET_TGW];
    gptq_gemv_batch_det_impl<T, BITS>(x, qweight, qzeros, scales, out, p, partials, tgid, tid);
}

// Three M=1 geometries, host-selected by `packed_out`: narrow outputs need a
// deep split to keep the GPU busy, mid ones a narrow tile for threadgroup
// balance, wide ones have enough threadgroups at S=16.
template [[host_name("w4a16_gemv_f16")]] kernel void awq_det_kernel<half, 4, 32u, 32u>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w4a16_gemv_bf16")]] kernel void awq_det_kernel<bfloat, 4, 32u, 32u>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w8a16_gemv_f16")]] kernel void awq_det_kernel<half, 8, 32u, 32u>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w8a16_gemv_bf16")]] kernel void awq_det_kernel<bfloat, 8, 32u, 32u>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w4a16_gemv_t16_f16")]] kernel void awq_det_kernel<half, 4, 16u, 32u>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w4a16_gemv_t16_bf16")]] kernel void awq_det_kernel<bfloat, 4, 16u, 32u>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w8a16_gemv_t16_f16")]] kernel void awq_det_kernel<half, 8, 16u, 32u>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w8a16_gemv_t16_bf16")]] kernel void awq_det_kernel<bfloat, 8, 16u, 32u>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w4a16_gemv_s16_f16")]] kernel void awq_det_kernel<half, 4, 32u, 16u>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w4a16_gemv_s16_bf16")]] kernel void awq_det_kernel<bfloat, 4, 32u, 16u>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w8a16_gemv_s16_f16")]] kernel void awq_det_kernel<half, 8, 32u, 16u>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w8a16_gemv_s16_bf16")]] kernel void awq_det_kernel<bfloat, 8, 32u, 16u>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant W4A16Params&, uint2, uint2);
// The batch kernels split K as the single-row kernel of the same output
// width does, since the split decides the order of the sums: a row must get
// the same bits decoded alone and in a batch.
template [[host_name("w4a16_gemv_batch_f16")]] kernel void awq_det_batch_kernel<half, 4, QDET_AB_S>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w4a16_gemv_batch_bf16")]] kernel void awq_det_batch_kernel<bfloat, 4, QDET_AB_S>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w8a16_gemv_batch_f16")]] kernel void awq_det_batch_kernel<half, 8, QDET_AB_S>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w8a16_gemv_batch_bf16")]] kernel void awq_det_batch_kernel<bfloat, 8, QDET_AB_S>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w4a16_gemv_batch_s16_f16")]] kernel void awq_det_batch_kernel<half, 4, QDET_S>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w4a16_gemv_batch_s16_bf16")]] kernel void awq_det_batch_kernel<bfloat, 4, QDET_S>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w8a16_gemv_batch_s16_f16")]] kernel void awq_det_batch_kernel<half, 8, QDET_S>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("w8a16_gemv_batch_s16_bf16")]] kernel void awq_det_batch_kernel<bfloat, 8, QDET_S>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant W4A16Params&, uint2, uint2);
template [[host_name("gptq4_gemv_f16")]] kernel void gptq_det_kernel<half, 4>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant GptqParams&, uint2, uint2);
template [[host_name("gptq4_gemv_bf16")]] kernel void gptq_det_kernel<bfloat, 4>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant GptqParams&, uint2, uint2);
template [[host_name("gptq8_gemv_f16")]] kernel void gptq_det_kernel<half, 8>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant GptqParams&, uint2, uint2);
template [[host_name("gptq8_gemv_bf16")]] kernel void gptq_det_kernel<bfloat, 8>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant GptqParams&, uint2, uint2);
template [[host_name("gptq4_gemv_batch_f16")]] kernel void gptq_det_batch_kernel<half, 4>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant GptqParams&, uint2, uint2);
template [[host_name("gptq4_gemv_batch_bf16")]] kernel void gptq_det_batch_kernel<bfloat, 4>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant GptqParams&, uint2, uint2);
template [[host_name("gptq8_gemv_batch_f16")]] kernel void gptq_det_batch_kernel<half, 8>(device const half*, device const uint*, device const uint*, device const half*, device void*, constant GptqParams&, uint2, uint2);
template [[host_name("gptq8_gemv_batch_bf16")]] kernel void gptq_det_batch_kernel<bfloat, 8>(device const bfloat*, device const uint*, device const uint*, device const bfloat*, device void*, constant GptqParams&, uint2, uint2);

// ─────────────────────────────────────────────────────────────────────────────
// F8E4M3 (fn variant) to F32 elementwise cast. candle 0.11's Metal backend
// supports F8E4M3 as storage only (no cast kernels), but the FP8 runtime
// dequant reads the resident F8 weights and widens them on every forward.
// E4M3fn: 1 sign, 4 exponent (bias 7), 3 mantissa; exp=15/mant=7 is NaN and
// there is no infinity encoding.
// ─────────────────────────────────────────────────────────────────────────────

// Native GEMV over block-scaled F8E4M3 weights (streamed FP8 experts).
//
// Reads the raw F8 bytes and the F32 block-scale grid directly, so the
// BF16 weight tensor is never materialized. Each weight value replays the
// resident dequantization chain bit-for-bit (decode, BF16 round, F32 scale
// fold, BF16 round) before the F32 multiply-accumulate; only the summation
// order differs from the dequantize-then-matmul path. Deterministic by the
// same construction as the AWQ/GPTQ kernels: every k-split of an output
// lives in one threadgroup and reduces in a fixed order with one writer.
// The threadgroup is (QDET_TGW, QDET_S); dispatch must use exactly QDET_S
// rows so every partial slot is written.

struct F8GemvParams {
    uint in_features;
    uint out_features;
    uint k_splits;
    uint chunk;
    uint block_r;
    uint block_c;
    uint s_cols;
    uint m;
};

inline float f8_weight_value(uchar v, float sc) {
    uint sgn = v >> 7;
    uint e = (v >> 3) & 0xFu;
    uint mant = v & 0x7u;
    float val;
    if (e == 0u) {
        val = ldexp(float(mant) * 0.125f, -6);
    } else if (e == 15u && mant == 7u) {
        val = NAN;
    } else {
        val = ldexp(1.0f + float(mant) * 0.125f, int(e) - 7);
    }
    val = sgn != 0u ? -val : val;
    return float(bfloat(float(bfloat(val)) * sc));
}

kernel void f8_gemv_bf16(
    device const bfloat*  x       [[buffer(0)]],
    device const uchar*   w       [[buffer(1)]],
    device const float*   scales  [[buffer(2)]],
    device bfloat*        out     [[buffer(3)]],
    constant F8GemvParams& p      [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]])
{
    threadgroup float partials[QDET_S * QDET_TGW];
    uint o = tgid.x * QDET_TGW + tid.x;
    uint ks = tid.y;

    float acc = 0.0f;
    if (o < p.out_features && ks < p.k_splits) {
        uint i = ks * p.chunk;
        uint i_end = min(i + p.chunk, p.in_features);
        uint srow_off = (o / p.block_r) * p.s_cols;
        device const uchar* wrow = w + ulong(o) * p.in_features;
        while (i < i_end) {
            float sc = scales[srow_off + i / p.block_c];
            uint blk_end = min(i_end, (i / p.block_c + 1u) * p.block_c);
            for (; i < blk_end; ++i) {
                acc += float(x[i]) * f8_weight_value(wrow[i], sc);
            }
        }
    }

    partials[ks * QDET_TGW + tid.x] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid.y == 0 && o < p.out_features) {
        float sum = 0.0f;
        for (uint sslot = 0; sslot < QDET_S; ++sslot) {
            sum += partials[sslot * QDET_TGW + tid.x];
        }
        out[o] = bfloat(sum);
    }
}

// Batch variant for 2 <= m <= 8 rows: each weight byte decodes once and
// multiplies into per-row register accumulators, then one reduction round
// per row reuses the same threadgroup scratch.
kernel void f8_gemv_batch_bf16(
    device const bfloat*  x       [[buffer(0)]],
    device const uchar*   w       [[buffer(1)]],
    device const float*   scales  [[buffer(2)]],
    device bfloat*        out     [[buffer(3)]],
    constant F8GemvParams& p      [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]])
{
    threadgroup float partials[QDET_S * QDET_TGW];
    uint o = tgid.x * QDET_TGW + tid.x;
    uint ks = tid.y;

    float acc[8] = {0.0f};
    if (o < p.out_features && ks < p.k_splits) {
        uint i = ks * p.chunk;
        uint i_end = min(i + p.chunk, p.in_features);
        uint srow_off = (o / p.block_r) * p.s_cols;
        device const uchar* wrow = w + ulong(o) * p.in_features;
        while (i < i_end) {
            float sc = scales[srow_off + i / p.block_c];
            uint blk_end = min(i_end, (i / p.block_c + 1u) * p.block_c);
            for (; i < blk_end; ++i) {
                float wv = f8_weight_value(wrow[i], sc);
                for (uint r = 0; r < p.m; ++r) {
                    acc[r] += float(x[ulong(r) * p.in_features + i]) * wv;
                }
            }
        }
    }

    for (uint r = 0; r < p.m; ++r) {
        partials[ks * QDET_TGW + tid.x] = acc[r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid.y == 0 && o < p.out_features) {
            float sum = 0.0f;
            for (uint sslot = 0; sslot < QDET_S; ++sslot) {
                sum += partials[sslot * QDET_TGW + tid.x];
            }
            out[ulong(r) * p.out_features + o] = bfloat(sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Block-scaled F8E4M3 to BF16 dequantization for streamed FP8 experts.
// Replicates the resident loader's cast chain bit-for-bit: the F8 value is
// first rounded to BF16 (the loader materializes the BF16 tensor before the
// scale fold), then re-widened to F32, multiplied by the block scale in F32,
// and rounded back to BF16.
kernel void f8_block_dequant_bf16(
    device const uchar* input   [[buffer(0)]],
    device const float* scales  [[buffer(1)]],
    device bfloat*      output  [[buffer(2)]],
    constant uint&      cols    [[buffer(3)]],
    constant uint&      n       [[buffer(4)]],
    constant uint&      block_r [[buffer(5)]],
    constant uint&      block_c [[buffer(6)]],
    constant uint&      s_cols  [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    uchar v = input[gid];
    uint s = v >> 7;
    uint e = (v >> 3) & 0xFu;
    uint m = v & 0x7u;
    float val;
    if (e == 0u) {
        val = ldexp(float(m) * 0.125f, -6);
    } else if (e == 15u && m == 7u) {
        val = NAN;
    } else {
        val = ldexp(1.0f + float(m) * 0.125f, int(e) - 7);
    }
    val = s != 0u ? -val : val;
    bfloat w = bfloat(val);
    uint r = gid / cols;
    uint c = gid - r * cols;
    float sc = scales[(r / block_r) * s_cols + (c / block_c)];
    output[gid] = bfloat(float(w) * sc);
}

kernel void cast_f8e4m3_f32(
    device const uchar* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      n      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    uchar v = input[gid];
    uint s = v >> 7;
    uint e = (v >> 3) & 0xFu;
    uint m = v & 0x7u;
    float val;
    if (e == 0u) {
        val = ldexp(float(m) * 0.125f, -6);
    } else if (e == 15u && m == 7u) {
        val = NAN;
    } else {
        val = ldexp(1.0f + float(m) * 0.125f, int(e) - 7);
    }
    output[gid] = s != 0u ? -val : val;
}
