#include <metal_stdlib>
#include <metal_math>
using namespace metal;

METAL_FUNC float silu_f32(float x) {
    return x / (1.0f + exp(-x));
}

// ── GatedSiLU: [*, 2*N] → [*, N]  (gate in first half, up in second) ─────────
// For output index gid: row = gid/half_n, col = gid%half_n
//   gate = x[row*2*half_n + col],  up = x[row*2*half_n + half_n + col]

kernel void gated_silu_f32(
    device const float* x      [[buffer(0)]],
    device       float* out    [[buffer(1)]],
    constant uint&      n_out  [[buffer(2)]],
    constant uint&      half_n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_out) return;
    uint row = gid / half_n;
    uint col = gid % half_n;
    float g = x[row * 2u * half_n + col];
    float u = x[row * 2u * half_n + half_n + col];
    out[gid] = silu_f32(g) * u;
}

kernel void gated_silu_f16(
    device const half* x      [[buffer(0)]],
    device       half* out    [[buffer(1)]],
    constant uint&     n_out  [[buffer(2)]],
    constant uint&     half_n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_out) return;
    uint row = gid / half_n;
    uint col = gid % half_n;
    float g = (float)x[row * 2u * half_n + col];
    float u = (float)x[row * 2u * half_n + half_n + col];
    out[gid] = (half)(silu_f32(g) * u);
}

#if defined(__HAVE_BFLOAT__)
kernel void gated_silu_bf16(
    device const bfloat* x      [[buffer(0)]],
    device       bfloat* out    [[buffer(1)]],
    constant uint&       n_out  [[buffer(2)]],
    constant uint&       half_n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_out) return;
    uint row = gid / half_n;
    uint col = gid % half_n;
    float g = (float)x[row * 2u * half_n + col];
    float u = (float)x[row * 2u * half_n + half_n + col];
    out[gid] = (bfloat)(silu_f32(g) * u);
}
#endif

// ── SiLU-Mul: element-wise silu(gate[i]) * up[i] ─────────────────────────────

kernel void silu_mul_f32(
    device const float* gate  [[buffer(0)]],
    device const float* up    [[buffer(1)]],
    device       float* out   [[buffer(2)]],
    constant uint&      n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float g = gate[gid];
    out[gid] = silu_f32(g) * up[gid];
}

kernel void silu_mul_f16(
    device const half* gate  [[buffer(0)]],
    device const half* up    [[buffer(1)]],
    device       half* out   [[buffer(2)]],
    constant uint&     n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float g = (float)gate[gid];
    float u = (float)up[gid];
    out[gid] = (half)(silu_f32(g) * u);
}

#if defined(__HAVE_BFLOAT__)
kernel void silu_mul_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up   [[buffer(1)]],
    device       bfloat* out  [[buffer(2)]],
    constant uint&       n    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float g = (float)gate[gid];
    float u = (float)up[gid];
    out[gid] = (bfloat)(silu_f32(g) * u);
}
#endif

// ── Softcap: out[i] = cap * tanh(x[i] / cap) ─────────────────────────────────
// Used by Gemma2 attention scores (cap=50) and Gemma2/Gemma4 logits (cap=30).
// Replaces the 3-op fallback `(x/cap).tanh()*cap` with one kernel pass.

kernel void softcap_f32(
    device const float* x   [[buffer(0)]],
    device       float* out [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    constant float&     cap [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = cap * tanh(x[gid] / cap);
}

kernel void softcap_f16(
    device const half* x   [[buffer(0)]],
    device       half* out [[buffer(1)]],
    constant uint&     n   [[buffer(2)]],
    constant float&    cap [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float v = (float)x[gid];
    out[gid] = (half)(cap * tanh(v / cap));
}

#if defined(__HAVE_BFLOAT__)
kernel void softcap_bf16(
    device const bfloat* x   [[buffer(0)]],
    device       bfloat* out [[buffer(1)]],
    constant uint&       n   [[buffer(2)]],
    constant float&      cap [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float v = (float)x[gid];
    out[gid] = (bfloat)(cap * tanh(v / cap));
}
#endif
