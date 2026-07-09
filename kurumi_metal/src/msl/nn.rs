//! Fused nn-primitive kernel sources (softmax, ...). One kernel replaces the decomposed
//! reduce+pointwise chain. Launched by `context/dispatch/nn.rs`; checked against `interp/nn`.

// `softmax_k`: stable softmax over an axis (layout outer x axis_len x inner), one thread per
// line. exp/sum accumulate in float, store as `ty` (one cast at store, matching the oracle).
pub(crate) fn softmax_msl(ty: &str) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void softmax_k(device const {ty}* in [[buffer(0)]], device {ty}* out [[buffer(1)]],\n\
                            constant uint& axis_len [[buffer(2)]], constant uint& inner [[buffer(3)]],\n\
                            uint gid [[thread_position_in_grid]]) {{\n\
             uint base = (gid / inner) * axis_len * inner + (gid % inner);\n\
             float m = -INFINITY;\n\
             for (uint k = 0; k < axis_len; k++) m = max(m, (float)in[base + k * inner]);\n\
             float sum = 0.0f;\n\
             for (uint k = 0; k < axis_len; k++) sum += exp((float)in[base + k * inner] - m);\n\
             for (uint k = 0; k < axis_len; k++) out[base + k * inner] = ({ty})(exp((float)in[base + k * inner] - m) / sum);\n}}"
    )
}

/// Max head-dim the flash kernel's thread-local accumulator holds; a larger `dh` falls to the
/// CPU oracle (see `eval_nn`). 256 covers every common transformer head width (64/80/128).
pub(crate) const SDPA_MAX_DH: usize = 256;

// `sdpa_flash_k`: fused scaled-dot-product attention FORWARD via ONLINE softmax -- ONE thread
// per (batch, query-row). All leading dims flatten to a batch index b (grid = batch*S); thread
// gid -> b = gid/S, query i = gid%S. Each thread streams keys j (causal: only j<=i), keeping a
// running max `m`, denominator `l`, and weighted value `acc[dh]` (rescaled by exp(m_old-m_new)
// on every new max) -- it NEVER materializes the SxS scores (O(dh) state, the flash memory win).
// `scale` = 1/sqrt(dh) is host-computed and passed in, matching the oracle's exact scale. f32
// only; `dh <= SDPA_MAX_DH` (the thread-local acc bound).
pub(crate) fn sdpa_flash_msl() -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void sdpa_flash_k(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],\n\
                            device const float* v [[buffer(2)]], device float* out [[buffer(3)]],\n\
                            constant uint& S [[buffer(4)]], constant uint& dh [[buffer(5)]],\n\
                            constant float& scale [[buffer(6)]], constant uint& causal [[buffer(7)]],\n\
                            uint gid [[thread_position_in_grid]]) {{\n\
             uint b = gid / S, i = gid % S;\n\
             uint qoff = (b * S + i) * dh;\n\
             float acc[{max_dh}];\n\
             for (uint d = 0; d < dh; d++) acc[d] = 0.0f;\n\
             float m = -INFINITY, l = 0.0f;\n\
             uint jmax = causal ? (i + 1) : S;\n\
             for (uint j = 0; j < jmax; j++) {{\n\
                 uint koff = (b * S + j) * dh;\n\
                 float s = 0.0f;\n\
                 for (uint d = 0; d < dh; d++) s += q[qoff + d] * k[koff + d];\n\
                 s *= scale;\n\
                 float m_new = max(m, s);\n\
                 float p = exp(s - m_new);\n\
                 float corr = exp(m - m_new);\n\
                 l = l * corr + p;\n\
                 for (uint d = 0; d < dh; d++) acc[d] = acc[d] * corr + p * v[koff + d];\n\
                 m = m_new;\n\
             }}\n\
             for (uint d = 0; d < dh; d++) out[qoff + d] = acc[d] / l;\n}}",
        max_dh = SDPA_MAX_DH
    )
}

// `rmsnorm_k`: x / sqrt(mean(x^2) + eps) over an axis, one thread per line. sum-of-squares in
// float, store as `ty`.
pub(crate) fn rmsnorm_msl(ty: &str) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void rmsnorm_k(device const {ty}* in [[buffer(0)]], device {ty}* out [[buffer(1)]],\n\
                            constant uint& axis_len [[buffer(2)]], constant uint& inner [[buffer(3)]],\n\
                            constant float& eps [[buffer(4)]], uint gid [[thread_position_in_grid]]) {{\n\
             uint base = (gid / inner) * axis_len * inner + (gid % inner);\n\
             float ss = 0.0f;\n\
             for (uint k = 0; k < axis_len; k++) {{ float x = (float)in[base + k * inner]; ss += x * x; }}\n\
             float rms = sqrt(ss / (float)axis_len + eps);\n\
             for (uint k = 0; k < axis_len; k++) out[base + k * inner] = ({ty})((float)in[base + k * inner] / rms);\n}}"
    )
}
