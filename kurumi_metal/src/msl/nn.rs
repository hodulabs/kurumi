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
