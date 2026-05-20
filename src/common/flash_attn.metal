// Flash Attention prefill kernel for Apple Metal
//
// Algorithm based on FlashAttention (Dao et al., 2022-2024)
// Original repository: https://github.com/Dao-AILab/flash-attention
// Original license: BSD 3-Clause ("Revised BSD License")
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

// ── Parameter struct (must match Rust FlashAttnParams) ──────────────────────
struct FlashAttnParams {
    uint  t_q;        // query sequence length (this segment)
    uint  t_kv;       // full KV cache length (prefix + t_q)
    uint  h;          // number of query heads
    uint  h_kv;       // number of KV heads (h % h_kv == 0)
    uint  d;          // head dimension
    uint  br;         // Q tile rows
    uint  bc;         // KV tile cols
    float scale;      // attention scale (1/sqrt(d))
    float softcap;    // Gemma softcap (0.0 = disabled)
    uint  prefix_len; // cached prefix length = t_kv - t_q
};

constant float M_LOG2E_F_CONST = 1.4426950408889634f;
constant float FA4_TAU = 8.0f;  // log2(256) — FA4 conditional rescaling threshold

// ── Core kernel template ─────────────────────────────────────────────────────
//
// Grid layout:
//   grid_x = batch * heads * ceil(t_q / br)   — one threadgroup per Q-block
// Threads per group: 128 (4 SIMD-groups × 32 threads)
//
// kv_tile bank-conflict mitigation:
//   Apple GPU threadgroup memory has 32 banks × 32-bit words.  For D=64/128/256
//   the natural kv_tile row stride is an exact multiple of 32 banks, so the
//   SIMD-group QKᵀ pattern (32 lanes reading kv_tile[j*D + d] for j=0..31, d
//   constant) hits a single bank → 32-way serial access.
//   We pad each kv_tile row by 1 word (4 / sizeof(T) elements) so consecutive
//   rows land in different banks: row n hits bank (n + d/word_size) mod 32.
//
// Threadgroup memory layout (byte-aligned, allocated by host):
//   o_acc[br][d]                              br*d        fp32
//   m_row[br]                                 br          fp32
//   l_row[br]                                 br          fp32
//   s_scratch[br][bc]                         br*bc       fp32
//   q_tile[br][d]                             br*d        T
//   kv_tile[bc][d + KV_PAD]                   bc*(d+pad)  T   (K and V alternate)

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

    // kv_tile row stride (with padding to avoid 32-way bank conflicts).
    // KV_PAD = 4 / sizeof(T): 1 element for fp32, 2 elements for fp16/bf16.
    const uint kv_pad    = (uint)(4 / sizeof(T));
    const uint kv_stride = p.d + kv_pad;

    // ── Slice threadgroup memory ────────────────────────────────────────────
    threadgroup float* o_acc     = (threadgroup float*)(tg_mem);
    threadgroup float* m_row     = o_acc + p.br * p.d;
    threadgroup float* l_row     = m_row + p.br;
    threadgroup float* s_scratch = l_row + p.br;
    threadgroup T*     q_tile    = (threadgroup T*)(s_scratch + p.br * p.bc);
    threadgroup T*     kv_tile   = q_tile + p.br * p.d;

    // ── Initialize per-row accumulators ─────────────────────────────────────
    for (uint i = tid; i < br_act; i += tg_size) {
        m_row[i] = -INFINITY;
        l_row[i] = 0.0f;
    }
    for (uint k = tid; k < br_act * p.d; k += tg_size) {
        o_acc[k] = 0.0f;
    }

    // ── Load Q tile from global memory ──────────────────────────────────────
    // Q layout: [B, H, T_q, D]
    const uint q_base = ((batch_idx * p.h) + head_idx) * p.t_q * p.d + q_start * p.d;
    for (uint k = tid; k < br_act * p.d; k += tg_size) {
        q_tile[k] = Q[q_base + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Compute KV block iteration range (causal skip) ──────────────────────
    const uint first_q_global = p.prefix_len + q_start;
    const uint last_q_global  = p.prefix_len + q_start + br_act - 1;
    const uint mask_boundary  = first_q_global / p.bc;
    const uint n_block_max    = min(
        (p.t_kv + p.bc - 1) / p.bc,
        (last_q_global + p.bc) / p.bc
    );

    // ── Main loop over KV blocks ────────────────────────────────────────────
    for (uint n = 0; n < n_block_max; n++) {
        const uint kv_start  = n * p.bc;
        const uint bc_act    = min(p.bc, p.t_kv - kv_start);
        const bool need_mask = (n >= mask_boundary);

        // Load K tile (padded stride per row to avoid bank conflicts).
        // Zero-fill rows >= bc_act so the PV inner loop can unroll without a
        // tail (avoiding NaN propagation from stale memory × 0).
        const uint k_base = ((batch_idx * p.h_kv) + kv_head) * p.t_kv * p.d + kv_start * p.d;
        for (uint k = tid; k < p.bc * p.d; k += tg_size) {
            uint row = k / p.d;
            uint col = k % p.d;
            T value = (row < bc_act) ? K[k_base + row * p.d + col] : T(0);
            kv_tile[row * kv_stride + col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute S = Q @ K^T * scale, with optional softcap and causal mask.
        // Inner d-loop is 4-way unrolled with 4 independent FMA accumulators
        // to break the data-dependency chain and expose instruction-level
        // parallelism (Apple GPU can issue multiple FMAs per cycle).
        // D is guaranteed divisible by 4 (supported head_dims: 64, 128, 256).
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

        // ── Online softmax update with FA4 conditional rescaling ────────────
        //
        // Standard FA2:
        //   m_new = max(m_prev, row_max);  exp_factor = exp(m_prev - m_new);
        //   l = exp_factor * l + sum(exp(S - m_new));
        //   O *= exp_factor;  O += exp(S - m_new) @ V;  m = m_new;
        //
        // FA4 (Dao 2026): when (row_max - m_prev) < τ (= log2(256) = 8.0), the
        // rescale factor exp(m_prev - m_new) is close to 1 and contributes
        // negligibly in BF16/FP16 precision.  Skip the multiply and keep
        // m_prev as the reference for this block.  This saves ~15-20% of the
        // multiply-add ops in the inner loop without changing correctness.
        //
        // Also uses exp2(x * log2(e)) instead of exp(x) — single hardware
        // instruction on Apple GPUs (and NVIDIA via MUFU.EX2).
        for (uint i = tid; i < br_act; i += tg_size) {
            float row_max = -INFINITY;
            for (uint j = 0; j < bc_act; j++) {
                row_max = max(row_max, s_scratch[i * p.bc + j]);
            }

            if (row_max == -INFINITY) {
                // All elements in this row+block are masked.  Zero ALL bc
                // slots (not just bc_act) so the unrolled PV loop reads 0,
                // and skip the m/l/O update.
                for (uint j = 0; j < p.bc; j++) {
                    s_scratch[i * p.bc + j] = 0.0f;
                }
                continue;
            }

            // shift = row_max - m_prev.  When m_prev = -INF (first valid block
            // for this row), shift = +INF and we always take the rescale path.
            float shift = row_max - m_row[i];

            if (shift < FA4_TAU) {
                // FA4 skip-rescale path.  Reference stays at m_row[i].
                // m_row[i] is guaranteed finite here (otherwise shift = +INF).
                // Iterate full bc to zero out tail (-INFINITY → 0) so the
                // unrolled PV loop doesn't read stale -INF values.
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
                // m_row[i] unchanged; o_acc unchanged.
            } else {
                // Full rescale path (FA2 standard).
                // Iterate full bc (not bc_act) to zero out the tail too.
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

        // Load V tile (reuses kv_tile area — K is no longer needed).
        // Same zero-fill pattern as K so j >= bc_act reads return 0.
        const uint v_base = ((batch_idx * p.h_kv) + kv_head) * p.t_kv * p.d + kv_start * p.d;
        for (uint k = tid; k < p.bc * p.d; k += tg_size) {
            uint row = k / p.d;
            uint col = k % p.d;
            T value = (row < bc_act) ? V[v_base + row * p.d + col] : T(0);
            kv_tile[row * kv_stride + col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate O += P @ V.
        // Inner j-loop is 4-way unrolled with parallel FMA accumulators.
        // p.bc is guaranteed divisible by 4 (supported tile sizes: 16, 32).
        // Out-of-range V rows are zeroed during load, so iterating to p.bc is
        // safe and avoids a tail loop.
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

    // ── Normalize and write output ──────────────────────────────────────────
    const uint o_base = ((batch_idx * p.h) + head_idx) * p.t_q * p.d + q_start * p.d;
    for (uint kk = tid; kk < br_act * p.d; kk += tg_size) {
        uint i  = kk / p.d;
        uint dd = kk % p.d;
        float inv_l = (l_row[i] > 0.0f) ? (1.0f / l_row[i]) : 0.0f;
        O[o_base + i * p.d + dd] = T(o_acc[i * p.d + dd] * inv_l);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ── MMA variant (Apple GPU family 8+ / M3+ with hardware Matrix Multiply Accum)
// ─────────────────────────────────────────────────────────────────────────────
//
// Replaces the scalar 4-way-unrolled inner loops of QKᵀ and PV with
// `simdgroup_matrix<T, 8, 8>` operations.  On M3+ these compile to the
// hardware MMA units (~5-10× theoretical peak vs scalar FMA).  On M1/M2 the
// same code path is emulated and may be SLOWER than the scalar kernel — the
// runtime dispatcher must check `supportsFamily(MTLGPUFamily.apple8)` before
// selecting this variant.
//
// Mixed-precision contract (Metal 3.0+):
//   simdgroup_multiply_accumulate(float D, T A, T B, float C) — supported for
//   T ∈ {half, bfloat}.  Accumulator stays in fp32 for numerical stability.
//
// Threadgroup memory layout (note the extra p_tile vs base kernel):
//   o_acc[br*d]              fp32
//   m_row[br]                fp32
//   l_row[br]                fp32
//   s_scratch[br*bc]         fp32   (QKᵀ result, softmax in-place)
//   p_tile[br*bc]            T      (P converted to T for PV MMA)
//   q_tile[br*d]             T
//   kv_tile[bc*(d+kv_pad)]   T      (kept padded — same layout as base kernel)

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

    // Threadgroup memory slicing.
    threadgroup float* o_acc     = (threadgroup float*)(tg_mem);
    threadgroup float* m_row     = o_acc + p.br * p.d;
    threadgroup float* l_row     = m_row + p.br;
    threadgroup float* s_scratch = l_row + p.br;
    threadgroup T*     p_tile    = (threadgroup T*)(s_scratch + p.br * p.bc);
    threadgroup T*     q_tile    = p_tile + p.br * p.bc;
    threadgroup T*     kv_tile   = q_tile + p.br * p.d;

    // Initialize FULL br rows (not just br_act) so MMA dummy rows are zeroed.
    for (uint i = tid; i < p.br; i += tg_size) {
        m_row[i] = -INFINITY;
        l_row[i] = 0.0f;
    }
    for (uint k = tid; k < p.br * p.d; k += tg_size) {
        o_acc[k] = 0.0f;
    }
    // Zero p_tile once up front.  Rows >= br_act are never written by the
    // softmax, so they stay 0 and contribute 0 to the PV MMA — no per-block
    // "zero dummy rows" pass needed.
    for (uint k = tid; k < p.br * p.bc; k += tg_size) {
        p_tile[k] = T(0);
    }

    // Load Q tile, zero-fill rows >= br_act.
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

        // Load K tile (zero-fill rows >= bc_act).
        const uint k_base = ((batch_idx * p.h_kv) + kv_head) * p.t_kv * p.d + kv_start * p.d;
        for (uint k = tid; k < p.bc * p.d; k += tg_size) {
            uint row = k / p.d;
            uint col = k % p.d;
            T value = (row < bc_act) ? K[k_base + row * p.d + col] : T(0);
            kv_tile[row * kv_stride + col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QKᵀ via simdgroup_matrix: each SIMD-group owns total_s_tiles/num_sg tiles.
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
                // K loaded with transpose=true so the MMA gets K^T.
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

        // Apply scale, softcap, causal mask (scalar).  Out-of-range rows/cols
        // get -INF (will be zeroed after softmax).
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

        // Online softmax with FA4 conditional rescaling.  Writes the
        // exponentiated probabilities DIRECTLY into p_tile as type T — this
        // fuses what used to be three separate passes (softmax → s_scratch,
        // zero dummy rows, convert float→T) into one, removing two
        // threadgroup barriers per KV block.
        for (uint i = tid; i < br_act; i += tg_size) {
            float row_max = -INFINITY;
            for (uint j = 0; j < bc_act; j++) {
                row_max = max(row_max, s_scratch[i * p.bc + j]);
            }
            if (row_max == -INFINITY) {
                // Fully-masked row: zero its P slot, skip m/l/O update.
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

        // Load V tile (reuses kv_tile area, zero-fill rows >= bc_act).
        const uint v_base = ((batch_idx * p.h_kv) + kv_head) * p.t_kv * p.d + kv_start * p.d;
        for (uint k = tid; k < p.bc * p.d; k += tg_size) {
            uint row = k / p.d;
            uint col = k % p.d;
            T value = (row < bc_act) ? V[v_base + row * p.d + col] : T(0);
            kv_tile[row * kv_stride + col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O += P @ V via simdgroup_matrix.
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

    // Final normalization (only valid rows).
    const uint o_base = ((batch_idx * p.h) + head_idx) * p.t_q * p.d + q_start * p.d;
    for (uint kk = tid; kk < br_act * p.d; kk += tg_size) {
        uint i  = kk / p.d;
        uint dd = kk % p.d;
        float inv_l = (l_row[i] > 0.0f) ? (1.0f / l_row[i]) : 0.0f;
        O[o_base + i * p.d + dd] = T(o_acc[i * p.d + dd] * inv_l);
    }
}

// ── Kernel entry points (one per dtype) ──────────────────────────────────────

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

// ── MMA kernel entry points (M3+ / Apple GPU family 8+) ─────────────────────
//
// Host dispatcher must check `supportsFamily(MTLGPUFamily.apple8)` before
// launching these.  On older hardware simdgroup_matrix is emulated and
// substantially slower than the scalar 4-way-unroll kernels above.

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
