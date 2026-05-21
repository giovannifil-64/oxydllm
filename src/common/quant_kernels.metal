// W4A16 quantized matmul kernels for Apple Metal.
//
// Fused dequantization + matmul for AWQ-style 4-bit weight-only quantization.
// The packed weights stay resident (no bf16 expansion at load): decode reads
// 4-bit data straight from HBM, cutting weight bandwidth ~4x.
//
// On-disk AWQ layout (autoawq, GEMM kernel):
//   qweight  u32 [in_features, out_features / 8]   8 nibbles/word along out
//   qzeros   u32 [in_features / group_size, out_features / 8]
//   scales   T   [in_features / group_size, out_features]
// Nibble k (bits 4k..4k+3) of a word maps to output column 8*j + AWQ_PACK_ORDER[k].
// Dequant:  W[o][i] = (qw_nibble - qz_nibble) * scale[group(i)][o].

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Must stay in sync with `W4A16Params` in metal_ops.rs.
struct W4A16Params {
    uint in_features;
    uint out_features;
    uint packed_out;    // out_features / 8
    uint group_size;    // dequant kernel only
    uint group_shift;   // log2(group_size) — gemv kernel only
    uint k_splits;      // in_features partitions — gemv kernel only
    uint chunk;         // ceil(in_features / k_splits) — gemv kernel only
};

constant uint AWQ_PACK_ORDER[8] = {0u, 2u, 4u, 6u, 1u, 3u, 5u, 7u};

inline uint awq_nibble(uint word, uint k) {
    return (word >> (4u * k)) & 0xFu;
}

// ── Fused W4A16 GEMV (M=1), split-K: out += x @ dequant(W)ᵀ ──────────────────
//
// Grid: (packed_out, k_splits). A naive one-thread-per-output GEMV launches only
// `packed_out` threads (~hundreds) and leaves the GPU idle with HBM load latency
// fully exposed. Here the reduction is split `k_splits` ways, multiplying the
// thread count so enough simdgroups are resident to hide latency.
//
// Each thread owns one packed-out column `j` and a CONTIGUOUS in_features chunk
// `[ks*chunk, (ks+1)*chunk)`. Contiguous (not strided) is essential: a thread
// stays inside a quant group for `group_size` steps, so the per-group scale/zero
// reload is amortised. Reads stay coalesced — consecutive threads (consecutive
// j) read consecutive qweight words. The `k_splits` partial sums are combined
// straight into `out` with relaxed atomic adds: this avoids materialising a
// `[k_splits, out]` partial buffer and a separate (strided, slow) reduction.
// `out` must be zero-initialised by the host.
template<typename T>
inline void w4a16_gemv_impl(
    device const T*       x,
    device const uint*    qweight,
    device const uint*    qzeros,
    device const T*       scales,
    device atomic_float*  out,
    constant W4A16Params& p,
    uint2 gid)
{
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

    float acc[8];
    for (uint k = 0; k < 8u; ++k) {
        acc[k] = 0.0f;
    }

    // Zero-point and scale depend only on the group → cached, refreshed on change.
    float zero_nib[8];
    float scale_v[8];
    uint last_g = 0xFFFFFFFFu;

    for (uint i = i_start; i < i_end; ++i) {
        uint g = i >> p.group_shift;
        if (g != last_g) {
            uint zw = qzeros[g * p.packed_out + j];
            for (uint k = 0; k < 8u; ++k) {
                zero_nib[k] = float(awq_nibble(zw, k));
                scale_v[k] = float(scales[g * p.out_features + j * 8u + AWQ_PACK_ORDER[k]]);
            }
            last_g = g;
        }

        uint ww = qweight[i * p.packed_out + j];
        float xv = float(x[i]);
        for (uint k = 0; k < 8u; ++k) {
            float w = (float(awq_nibble(ww, k)) - zero_nib[k]) * scale_v[k];
            acc[k] += xv * w;
        }
    }

    for (uint k = 0; k < 8u; ++k) {
        uint o = j * 8u + AWQ_PACK_ORDER[k];
        atomic_fetch_add_explicit(&out[o], acc[k], memory_order_relaxed);
    }
}

// ── Dequantize only: weight[in, out] = dequant(packed) ───────────────────────
//
// Grid: (in_features, packed_out). Produces a plain row-major [in, out] weight
// ready for a standard matmul — used for the prefill / batched (M>1) path, where
// a tuned GEMM beats a custom kernel.
template<typename T>
inline void dequantize_w4_impl(
    device const uint*    qweight,
    device const uint*    qzeros,
    device const T*       scales,
    device       T*       weight,
    constant W4A16Params& p,
    uint2 gid)
{
    uint i = gid.x;
    uint j = gid.y;
    if (i >= p.in_features || j >= p.packed_out) {
        return;
    }

    uint g = i / p.group_size;
    uint ww = qweight[i * p.packed_out + j];
    uint zw = qzeros[g * p.packed_out + j];

    for (uint k = 0; k < 8u; ++k) {
        uint o = j * 8u + AWQ_PACK_ORDER[k];
        float val = (float(awq_nibble(ww, k)) - float(awq_nibble(zw, k)))
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
    w4a16_gemv_impl<half>(x, qweight, qzeros, scales, out, p, gid);
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
    w4a16_gemv_impl<bfloat>(x, qweight, qzeros, scales, out, p, gid);
}

kernel void dequantize_w4_f16(
    device const uint*    qweight  [[buffer(0)]],
    device const uint*    qzeros   [[buffer(1)]],
    device const half*    scales   [[buffer(2)]],
    device       half*    weight   [[buffer(3)]],
    constant W4A16Params& p        [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    dequantize_w4_impl<half>(qweight, qzeros, scales, weight, p, gid);
}

kernel void dequantize_w4_bf16(
    device const uint*    qweight  [[buffer(0)]],
    device const uint*    qzeros   [[buffer(1)]],
    device const bfloat*  scales   [[buffer(2)]],
    device       bfloat*  weight   [[buffer(3)]],
    constant W4A16Params& p        [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    dequantize_w4_impl<bfloat>(qweight, qzeros, scales, weight, p, gid);
}
