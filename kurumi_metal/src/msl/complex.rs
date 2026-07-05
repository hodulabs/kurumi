//! Complex (C64 = float2) kernel sources: build from real+imag parts, extract a part, and
//! the real->C64 cast. Launched by `context/dispatch/complex.rs`.

// `complex_k`: two f32 buffers (re, im) -> one float2 buffer.
pub(crate) const COMPLEX_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
     kernel void complex_k(device const float* re [[buffer(0)]], device const float* im [[buffer(1)]],\n\
                           device float2* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n\
         out[i] = float2(re[i], im[i]); }";

// `r2c_k`: real f32 -> float2 with imag 0 (the f32 -> C64 cast).
pub(crate) const R2C_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
     kernel void r2c_k(device const float* re [[buffer(0)]], device float2* out [[buffer(1)]],\n\
                       uint i [[thread_position_in_grid]]) { out[i] = float2(re[i], 0.0f); }";

// `part_k`: float2 -> a real component (`comp` = "x" real / "y" imag).
pub(crate) fn part_msl(comp: &str) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void part_k(device const float2* z [[buffer(0)]], device float* out [[buffer(1)]],\n\
                            uint i [[thread_position_in_grid]]) {{ out[i] = z[i].{comp}; }}"
    )
}
