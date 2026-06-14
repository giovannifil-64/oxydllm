// Flash Attention prefill kernel for Apple Metal
//
// Algorithm based on FlashAttention (Dao et al., 2022-2024)
// Original repository: https://github.com/Dao-AILab/flash-attention
// Original license:
// BSD 3-Clause License
// 
// Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// This file is an independent reimplementation in Metal Shading Language.
// No source code was copied or translated from the original CUDA kernels.
// The online softmax recurrence and IO-aware tiling algorithm are derived
// from the FlashAttention papers.
//
// Phase 1: FA2 baseline (tiled, online softmax, causal+prefix mask, GQA, softcap)
// Phase 2: FA4 portable improvements (conditional rescaling, exp2, simdgroup_matrix)

#include <metal_stdlib>
#include <metal_math>
using namespace metal;

struct FlashAttnParams {
    uint  t_q;
    uint  t_kv;
    uint  h;
    uint  h_kv;        // h % h_kv == 0
    uint  d;
    uint  br;
    uint  bc;
    float scale;
    float softcap;     // 0.0 = disabled
    uint  prefix_len;  // = t_kv - t_q
};

constant float M_LOG2E_F_CONST = 1.4426950408889634f;
constant float FA4_TAU = 8.0f;  // log2(256)

// Grid: batch * heads * ceil(t_q / br). 128 threads (4 SIMD-groups) per group.
// kv_tile rows padded by 4/sizeof(T) words to avoid 32-way bank conflicts on
// Apple GPU threadgroup memory (32 banks × 32-bit).

template<typename T>
inline void flash_attention_prefill_impl(
    device const T*               Q,
    device const T*               K,
    device const T*               V,
    device       T*               O,
    constant     FlashAttnParams& p,
    threadgroup  uchar*           tg_mem,
    uint                          gid,
    uint                          tid,
    uint                          tg_size
) {
    const uint n_q_blocks   = (p.t_q + p.br - 1) / p.br;
    const uint q_block_idx  = gid % n_q_blocks;
    const uint head_idx     = (gid / n_q_blocks) % p.h;
    const uint batch_idx    = gid / (n_q_blocks * p.h);
    const uint q_start      = q_block_idx * p.br;
    const uint br_act       = min(p.br, p.t_q - q_start);

    const uint n_rep   = p.h / p.h_kv;
    const uint kv_head = head_idx / n_rep;

    const uint kv_pad    = (uint)(4 / sizeof(T));
    const uint kv_stride = p.d + kv_pad;

    threadgroup float* o_acc     = (threadgroup float*)(tg_mem);
    threadgroup float* m_row     = o_acc + p.br * p.d;
    threadgroup float* l_row     = m_row + p.br;
    threadgroup float* s_scratch = l_row + p.br;
    threadgroup T*     q_tile    = (threadgroup T*)(s_scratch + p.br * p.bc);
    threadgroup T*     kv_tile   = q_tile + p.br * p.d;

    for (uint i = tid; i < br_act; i += tg_size) {
        m_row[i] = -INFINITY;
        l_row[i] = 0.0f;
    }
    for (uint k = tid; k < br_act * p.d; k += tg_size) {
        o_acc[k] = 0.0f;
    }

    const uint q_base = ((batch_idx * p.h) + head_idx) * p.t_q * p.d + q_start * p.d;
    for (uint k = tid; k < br_act * p.d; k += tg_size) {
        q_tile[k] = Q[q_base + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint first_q_global = p.prefix_len + q_start;
    const uint last_q_global  = p.prefix_len + q_start + br_act - 1;
    const uint mask_boundary  = first_q_global / p.bc;
    const uint n_block_max    = min(
        (p.t_kv + p.bc - 1) / p.bc,
        (last_q_global + p.bc) / p.bc
    );

    for (uint n = 0; n < n_block_max; n++) {
        const uint kv_start  = n * p.bc;
        const uint bc_act    = min(p.bc, p.t_kv - kv_start);
        const bool need_mask = (n >= mask_boundary);

        // Zero-fill rows >= bc_act so the unrolled PV loop sees 0 (no NaN propagation).
        const uint k_base = ((batch_idx * p.h_kv) + kv_head) * p.t_kv * p.d + kv_start * p.d;
        for (uint k = tid; k < p.bc * p.d; k += tg_size) {
            uint row = k / p.d;
            uint col = k % p.d;
            T value = (row < bc_act) ? K[k_base + row * p.d + col] : T(0);
            kv_tile[row * kv_stride + col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q @ K^T * scale. 4-way unroll requires D % 4 == 0 (head_dims 64/128/256).
        for (uint kk = tid; kk < br_act * p.bc; kk += tg_size) {
            uint i = kk / p.bc;
            uint j = kk % p.bc;
            float dot;
            if (j >= bc_act) {
                dot = -INFINITY;
            } else {
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                const uint q_base_dd  = i * p.d;
                const uint kv_base_dd = j * kv_stride;
                for (uint dd = 0; dd < p.d; dd += 4) {
                    d0 = fma(float(q_tile[q_base_dd  + dd + 0]),
                             float(kv_tile[kv_base_dd + dd + 0]), d0);
                    d1 = fma(float(q_tile[q_base_dd  + dd + 1]),
                             float(kv_tile[kv_base_dd + dd + 1]), d1);
                    d2 = fma(float(q_tile[q_base_dd  + dd + 2]),
                             float(kv_tile[kv_base_dd + dd + 2]), d2);
                    d3 = fma(float(q_tile[q_base_dd  + dd + 3]),
                             float(kv_tile[kv_base_dd + dd + 3]), d3);
                }
                dot = (d0 + d1) + (d2 + d3);
                dot *= p.scale;
                if (p.softcap > 0.0f) {
                    dot = p.softcap * precise::tanh(dot / p.softcap);
                }
                if (need_mask) {
                    uint q_pos  = p.prefix_len + q_start + i;
                    uint kv_pos = kv_start + j;
                    if (kv_pos > q_pos) {
                        dot = -INFINITY;
                    }
                }
            }
            s_scratch[i * p.bc + j] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // FA4 conditional rescaling: when (row_max - m_prev) < log2(256), the
        // rescale factor is ~1 in BF16/FP16; skip the multiply and keep m_prev.
        // Iterates full p.bc (not bc_act) to zero the tail for the unrolled PV loop.
        for (uint i = tid; i < br_act; i += tg_size) {
            float row_max = -INFINITY;
            for (uint j = 0; j < bc_act; j++) {
                row_max = max(row_max, s_scratch[i * p.bc + j]);
            }

            if (row_max == -INFINITY) {
                for (uint j = 0; j < p.bc; j++) {
                    s_scratch[i * p.bc + j] = 0.0f;
                }
                continue;
            }

            float shift = row_max - m_row[i];

            if (shift < FA4_TAU) {
                float ref = m_row[i];
                float row_sum = 0.0f;
                for (uint j = 0; j < p.bc; j++) {
                    float s_val = s_scratch[i * p.bc + j];
                    float e = (s_val == -INFINITY)
                                ? 0.0f
                                : exp2((s_val - ref) * M_LOG2E_F_CONST);
                    s_scratch[i * p.bc + j] = e;
                    row_sum += e;
                }
                l_row[i] += row_sum;
            } else {
                float m_new = max(m_row[i], row_max);
                float exp_factor = (m_row[i] == -INFINITY)
                                    ? 0.0f
                                    : exp2((m_row[i] - m_new) * M_LOG2E_F_CONST);
                float row_sum = 0.0f;
                for (uint j = 0; j < p.bc; j++) {
                    float s_val = s_scratch[i * p.bc + j];
                    float e = (s_val == -INFINITY)
                                ? 0.0f
                                : exp2((s_val - m_new) * M_LOG2E_F_CONST);
                    s_scratch[i * p.bc + j] = e;
                    row_sum += e;
                }
                l_row[i] = l_row[i] * exp_factor + row_sum;
                for (uint dd = 0; dd < p.d; dd++) {
                    o_acc[i * p.d + dd] *= exp_factor;
                }
                m_row[i] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V reuses kv_tile (K no longer needed).
        const uint v_base = ((batch_idx * p.h_kv) + kv_head) * p.t_kv * p.d + kv_start * p.d;
        for (uint k = tid; k < p.bc * p.d; k += tg_size) {
            uint row = k / p.d;
            uint col = k % p.d;
            T value = (row < bc_act) ? V[v_base + row * p.d + col] : T(0);
            kv_tile[row * kv_stride + col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4-way unroll requires p.bc % 4 == 0 (supported tile sizes: 16, 32).
        for (uint kk = tid; kk < br_act * p.d; kk += tg_size) {
            uint i  = kk / p.d;
            uint dd = kk % p.d;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            const uint s_base = i * p.bc;
            for (uint j = 0; j < p.bc; j += 4) {
                a0 = fma(s_scratch[s_base + j + 0],
                         float(kv_tile[(j + 0) * kv_stride + dd]), a0);
                a1 = fma(s_scratch[s_base + j + 1],
                         float(kv_tile[(j + 1) * kv_stride + dd]), a1);
                a2 = fma(s_scratch[s_base + j + 2],
                         float(kv_tile[(j + 2) * kv_stride + dd]), a2);
                a3 = fma(s_scratch[s_base + j + 3],
                         float(kv_tile[(j + 3) * kv_stride + dd]), a3);
            }
            o_acc[i * p.d + dd] += (a0 + a1) + (a2 + a3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint o_base = ((batch_idx * p.h) + head_idx) * p.t_q * p.d + q_start * p.d;
    for (uint kk = tid; kk < br_act * p.d; kk += tg_size) {
        uint i  = kk / p.d;
        uint dd = kk % p.d;
        float inv_l = (l_row[i] > 0.0f) ? (1.0f / l_row[i]) : 0.0f;
        O[o_base + i * p.d + dd] = T(o_acc[i * p.d + dd] * inv_l);
    }
}

// MMA variant: hardware simdgroup_matrix on Apple GPU family 8+ (M3+).
// Host MUST gate on `supportsFamily(MTLGPUFamily.apple8)`, on M1/M2 the path
// is emulated and slower than the scalar kernel above.
template<typename T>
inline void flash_attention_prefill_mma_impl(
    device const T*               Q,
    device const T*               K,
    device const T*               V,
    device       T*               O,
    constant     FlashAttnParams& p,
    threadgroup  uchar*           tg_mem,
    uint                          gid,
    uint                          tid,
    uint                          tg_size,
    uint                          sg_idx
) {
    const uint n_q_blocks   = (p.t_q + p.br - 1) / p.br;
    const uint q_block_idx  = gid % n_q_blocks;
    const uint head_idx     = (gid / n_q_blocks) % p.h;
    const uint batch_idx    = gid / (n_q_blocks * p.h);
    const uint q_start      = q_block_idx * p.br;
    const uint br_act       = min(p.br, p.t_q - q_start);

    const uint n_rep   = p.h / p.h_kv;
    const uint kv_head = head_idx / n_rep;

    const uint kv_pad    = (uint)(4 / sizeof(T));
    const uint kv_stride = p.d + kv_pad;

    threadgroup float* o_acc     = (threadgroup float*)(tg_mem);
    threadgroup float* m_row     = o_acc + p.br * p.d;
    threadgroup float* l_row     = m_row + p.br;
    threadgroup float* s_scratch = l_row + p.br;
    threadgroup T*     p_tile    = (threadgroup T*)(s_scratch + p.br * p.bc);
    threadgroup T*     q_tile    = p_tile + p.br * p.bc;
    threadgroup T*     kv_tile   = q_tile + p.br * p.d;

    // Init FULL br rows so MMA dummy rows are zeroed.
    for (uint i = tid; i < p.br; i += tg_size) {
        m_row[i] = -INFINITY;
        l_row[i] = 0.0f;
    }
    for (uint k = tid; k < p.br * p.d; k += tg_size) {
        o_acc[k] = 0.0f;
    }
    for (uint k = tid; k < p.br * p.bc; k += tg_size) {
        p_tile[k] = T(0);
    }

    const uint q_base = ((batch_idx * p.h) + head_idx) * p.t_q * p.d + q_start * p.d;
    for (uint k = tid; k < p.br * p.d; k += tg_size) {
        uint row = k / p.d;
        uint col = k % p.d;
        q_tile[row * p.d + col] = (row < br_act) ? Q[q_base + row * p.d + col] : T(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint first_q_global = p.prefix_len + q_start;
    const uint last_q_global  = p.prefix_len + q_start + br_act - 1;
    const uint mask_boundary  = first_q_global / p.bc;
    const uint n_block_max    = min(
        (p.t_kv + p.bc - 1) / p.bc,
        (last_q_global + p.bc) / p.bc
    );

    const uint num_simdgroups = tg_size / 32;
    const uint s_tiles_m = p.br / 8;
    const uint s_tiles_n = p.bc / 8;
    const uint s_tiles_k = p.d  / 8;
    const uint o_tiles_m = p.br / 8;
    const uint o_tiles_n = p.d  / 8;
    const uint o_tiles_k = p.bc / 8;
    const uint total_s_tiles = s_tiles_m * s_tiles_n;
    const uint total_o_tiles = o_tiles_m * o_tiles_n;

    for (uint n = 0; n < n_block_max; n++) {
        const uint kv_start  = n * p.bc;
        const uint bc_act    = min(p.bc, p.t_kv - kv_start);
        const bool need_mask = (n >= mask_boundary);

        const uint k_base = ((batch_idx * p.h_kv) + kv_head) * p.t_kv * p.d + kv_start * p.d;
        for (uint k = tid; k < p.bc * p.d; k += tg_size) {
            uint row = k / p.d;
            uint col = k % p.d;
            T value = (row < bc_act) ? K[k_base + row * p.d + col] : T(0);
            kv_tile[row * kv_stride + col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = sg_idx; t < total_s_tiles; t += num_simdgroups) {
            uint i_blk = t / s_tiles_n;
            uint j_blk = t % s_tiles_n;
            simdgroup_matrix<float, 8, 8> s_mat = simdgroup_matrix<float, 8, 8>(0);
            for (uint k_blk = 0; k_blk < s_tiles_k; k_blk++) {
                simdgroup_matrix<T, 8, 8> q_mat, k_mat;
                simdgroup_load(q_mat,
                               q_tile + i_blk * 8 * p.d + k_blk * 8,
                               p.d,
                               ulong2(0, 0),
                               false);
                // transpose=true so the MMA receives K^T.
                simdgroup_load(k_mat,
                               kv_tile + j_blk * 8 * kv_stride + k_blk * 8,
                               kv_stride,
                               ulong2(0, 0),
                               true);
                simdgroup_multiply_accumulate(s_mat, q_mat, k_mat, s_mat);
            }
            simdgroup_store(s_mat,
                            s_scratch + i_blk * 8 * p.bc + j_blk * 8,
                            p.bc,
                            ulong2(0, 0),
                            false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = tid; kk < p.br * p.bc; kk += tg_size) {
            uint i = kk / p.bc;
            uint j = kk % p.bc;
            float val = s_scratch[i * p.bc + j];
            if (i >= br_act || j >= bc_act) {
                val = -INFINITY;
            } else {
                val *= p.scale;
                if (p.softcap > 0.0f) {
                    val = p.softcap * precise::tanh(val / p.softcap);
                }
                if (need_mask) {
                    uint q_pos  = p.prefix_len + q_start + i;
                    uint kv_pos = kv_start + j;
                    if (kv_pos > q_pos) {
                        val = -INFINITY;
                    }
                }
            }
            s_scratch[i * p.bc + j] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Fused softmax to P-as-T conversion (writes p_tile directly, saving 2 barriers).
        for (uint i = tid; i < br_act; i += tg_size) {
            float row_max = -INFINITY;
            for (uint j = 0; j < bc_act; j++) {
                row_max = max(row_max, s_scratch[i * p.bc + j]);
            }
            if (row_max == -INFINITY) {
                for (uint j = 0; j < p.bc; j++) {
                    p_tile[i * p.bc + j] = T(0);
                }
                continue;
            }
            float shift = row_max - m_row[i];
            if (shift < FA4_TAU) {
                float ref = m_row[i];
                float row_sum = 0.0f;
                for (uint j = 0; j < p.bc; j++) {
                    float s_val = s_scratch[i * p.bc + j];
                    float e = (s_val == -INFINITY)
                                ? 0.0f
                                : exp2((s_val - ref) * M_LOG2E_F_CONST);
                    p_tile[i * p.bc + j] = T(e);
                    row_sum += e;
                }
                l_row[i] += row_sum;
            } else {
                float m_new = max(m_row[i], row_max);
                float exp_factor = (m_row[i] == -INFINITY)
                                    ? 0.0f
                                    : exp2((m_row[i] - m_new) * M_LOG2E_F_CONST);
                float row_sum = 0.0f;
                for (uint j = 0; j < p.bc; j++) {
                    float s_val = s_scratch[i * p.bc + j];
                    float e = (s_val == -INFINITY)
                                ? 0.0f
                                : exp2((s_val - m_new) * M_LOG2E_F_CONST);
                    p_tile[i * p.bc + j] = T(e);
                    row_sum += e;
                }
                l_row[i] = l_row[i] * exp_factor + row_sum;
                for (uint dd = 0; dd < p.d; dd++) {
                    o_acc[i * p.d + dd] *= exp_factor;
                }
                m_row[i] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint v_base = ((batch_idx * p.h_kv) + kv_head) * p.t_kv * p.d + kv_start * p.d;
        for (uint k = tid; k < p.bc * p.d; k += tg_size) {
            uint row = k / p.d;
            uint col = k % p.d;
            T value = (row < bc_act) ? V[v_base + row * p.d + col] : T(0);
            kv_tile[row * kv_stride + col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = sg_idx; t < total_o_tiles; t += num_simdgroups) {
            uint i_blk  = t / o_tiles_n;
            uint dd_blk = t % o_tiles_n;
            simdgroup_matrix<float, 8, 8> o_mat;
            simdgroup_load(o_mat,
                           o_acc + i_blk * 8 * p.d + dd_blk * 8,
                           p.d,
                           ulong2(0, 0),
                           false);
            for (uint j_blk = 0; j_blk < o_tiles_k; j_blk++) {
                simdgroup_matrix<T, 8, 8> p_mat, v_mat;
                simdgroup_load(p_mat,
                               p_tile + i_blk * 8 * p.bc + j_blk * 8,
                               p.bc,
                               ulong2(0, 0),
                               false);
                simdgroup_load(v_mat,
                               kv_tile + j_blk * 8 * kv_stride + dd_blk * 8,
                               kv_stride,
                               ulong2(0, 0),
                               false);
                simdgroup_multiply_accumulate(o_mat, p_mat, v_mat, o_mat);
            }
            simdgroup_store(o_mat,
                            o_acc + i_blk * 8 * p.d + dd_blk * 8,
                            p.d,
                            ulong2(0, 0),
                            false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint o_base = ((batch_idx * p.h) + head_idx) * p.t_q * p.d + q_start * p.d;
    for (uint kk = tid; kk < br_act * p.d; kk += tg_size) {
        uint i  = kk / p.d;
        uint dd = kk % p.d;
        float inv_l = (l_row[i] > 0.0f) ? (1.0f / l_row[i]) : 0.0f;
        O[o_base + i * p.d + dd] = T(o_acc[i * p.d + dd] * inv_l);
    }
}

kernel void flash_attention_prefill_f32(
    device const float*           Q      [[buffer(0)]],
    device const float*           K      [[buffer(1)]],
    device const float*           V      [[buffer(2)]],
    device       float*           O      [[buffer(3)]],
    constant     FlashAttnParams& p      [[buffer(4)]],
    threadgroup  uchar*           tg_mem [[threadgroup(0)]],
    uint                          gid    [[threadgroup_position_in_grid]],
    uint                          tid    [[thread_position_in_threadgroup]],
    uint                          tg_sz  [[threads_per_threadgroup]]
) {
    flash_attention_prefill_impl<float>(Q, K, V, O, p, tg_mem, gid, tid, tg_sz);
}

kernel void flash_attention_prefill_f16(
    device const half*            Q      [[buffer(0)]],
    device const half*            K      [[buffer(1)]],
    device const half*            V      [[buffer(2)]],
    device       half*            O      [[buffer(3)]],
    constant     FlashAttnParams& p      [[buffer(4)]],
    threadgroup  uchar*           tg_mem [[threadgroup(0)]],
    uint                          gid    [[threadgroup_position_in_grid]],
    uint                          tid    [[thread_position_in_threadgroup]],
    uint                          tg_sz  [[threads_per_threadgroup]]
) {
    flash_attention_prefill_impl<half>(Q, K, V, O, p, tg_mem, gid, tid, tg_sz);
}

kernel void flash_attention_prefill_mma_f32(
    device const float*           Q      [[buffer(0)]],
    device const float*           K      [[buffer(1)]],
    device const float*           V      [[buffer(2)]],
    device       float*           O      [[buffer(3)]],
    constant     FlashAttnParams& p      [[buffer(4)]],
    threadgroup  uchar*           tg_mem [[threadgroup(0)]],
    uint                          gid    [[threadgroup_position_in_grid]],
    uint                          tid    [[thread_position_in_threadgroup]],
    uint                          tg_sz  [[threads_per_threadgroup]],
    uint                          sg_idx [[simdgroup_index_in_threadgroup]]
) {
    flash_attention_prefill_mma_impl<float>(Q, K, V, O, p, tg_mem, gid, tid, tg_sz, sg_idx);
}

kernel void flash_attention_prefill_mma_f16(
    device const half*            Q      [[buffer(0)]],
    device const half*            K      [[buffer(1)]],
    device const half*            V      [[buffer(2)]],
    device       half*            O      [[buffer(3)]],
    constant     FlashAttnParams& p      [[buffer(4)]],
    threadgroup  uchar*           tg_mem [[threadgroup(0)]],
    uint                          gid    [[threadgroup_position_in_grid]],
    uint                          tid    [[thread_position_in_threadgroup]],
    uint                          tg_sz  [[threads_per_threadgroup]],
    uint                          sg_idx [[simdgroup_index_in_threadgroup]]
) {
    flash_attention_prefill_mma_impl<half>(Q, K, V, O, p, tg_mem, gid, tid, tg_sz, sg_idx);
}

#if defined(__HAVE_BFLOAT__)
kernel void flash_attention_prefill_mma_bf16(
    device const bfloat*          Q      [[buffer(0)]],
    device const bfloat*          K      [[buffer(1)]],
    device const bfloat*          V      [[buffer(2)]],
    device       bfloat*          O      [[buffer(3)]],
    constant     FlashAttnParams& p      [[buffer(4)]],
    threadgroup  uchar*           tg_mem [[threadgroup(0)]],
    uint                          gid    [[threadgroup_position_in_grid]],
    uint                          tid    [[thread_position_in_threadgroup]],
    uint                          tg_sz  [[threads_per_threadgroup]],
    uint                          sg_idx [[simdgroup_index_in_threadgroup]]
) {
    flash_attention_prefill_mma_impl<bfloat>(Q, K, V, O, p, tg_mem, gid, tid, tg_sz, sg_idx);
}
#endif

#if defined(__HAVE_BFLOAT__)
kernel void flash_attention_prefill_bf16(
    device const bfloat*          Q      [[buffer(0)]],
    device const bfloat*          K      [[buffer(1)]],
    device const bfloat*          V      [[buffer(2)]],
    device       bfloat*          O      [[buffer(3)]],
    constant     FlashAttnParams& p      [[buffer(4)]],
    threadgroup  uchar*           tg_mem [[threadgroup(0)]],
    uint                          gid    [[threadgroup_position_in_grid]],
    uint                          tid    [[thread_position_in_threadgroup]],
    uint                          tg_sz  [[threads_per_threadgroup]]
) {
    flash_attention_prefill_impl<bfloat>(Q, K, V, O, p, tg_mem, gid, tid, tg_sz);
}
#endif
