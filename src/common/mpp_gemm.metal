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

struct MppFaParams {
    int t_q;
    int t_kv;
    int h;
    int h_kv;
    float scale;
    int prefix_len;
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
    device bfloat* kb = k + (size_t(b) * p.h_kv + hkv) * size_t(p.t_kv) * D;
    device bfloat* vb = v + (size_t(b) * p.h_kv + hkv) * size_t(p.t_kv) * D;

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
    if (lane < uint(FA_BR)) {
        tg_m[lane] = -INFINITY;
        tg_l[lane] = 0.0f;
    }

    int kv_max = min(p.t_kv, p.prefix_len + q0 + br);
    for (int kv0 = 0; kv0 < kv_max; kv0 += FA_BC) {
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
                if (kv0 + n > p.prefix_len + q0 + m) {
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

        if (lane < uint(FA_BR)) {
            float m_new = max(tg_m[lane], tg_r[lane]);
            tg_a[lane] = (tg_m[lane] == -INFINITY) ? 0.0f : exp(tg_m[lane] - m_new);
            tg_m[lane] = m_new;
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
                float pv = exp(sT[i] - tg_m[m]);
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

        if (lane < uint(FA_BR)) {
            tg_l[lane] = tg_l[lane] * tg_a[lane] + tg_r[lane];
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
