//! Host-offload GEMM kernel sources: the naive per-output kernel (any dtype) and the
//! simdgroup-matrix f32/f16 kernels. Launched by `context/dispatch/hostgemm.rs`; back the
//! CPU-offload fallback in `backend/hostgemm.rs`. (Device-resident GEMM uses MPS, see matmul.)

pub(crate) fn naive_mm_msl(ty: &str) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void mm(device const {ty}* A [[buffer(0)]],\n\
                        device const {ty}* B [[buffer(1)]],\n\
                        device {ty}* C       [[buffer(2)]],\n\
                        constant uint& M [[buffer(3)]],\n\
                        constant uint& N [[buffer(4)]],\n\
                        constant uint& K [[buffer(5)]],\n\
                        uint2 gid [[thread_position_in_grid]]) {{\n\
             if (gid.x >= N || gid.y >= M) return;\n\
             {ty} acc = {ty}(0);\n\
             for (uint k = 0; k < K; k++) acc += A[gid.y * K + k] * B[k * N + gid.x];\n\
             C[gid.y * N + gid.x] = acc;\n}}"
    )
}

// minimal simdgroup_matrix GEMM: each threadgroup (one 32-thread simdgroup)
// computes one 8x8 tile of C, accumulating over K in 8-wide steps. fp32 accum.
pub(crate) const SGEMM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void sgemm(device const float* A [[buffer(0)]],
                  device const float* B [[buffer(1)]],
                  device float* C       [[buffer(2)]],
                  constant uint& M [[buffer(3)]],
                  constant uint& N [[buffer(4)]],
                  constant uint& K [[buffer(5)]],
                  uint2 gid [[threadgroup_position_in_grid]]) {
    uint row = gid.y * 8;
    uint col = gid.x * 8;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < K; k += 8) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, A + row * K + k, K);
        simdgroup_load(b, B + k * N + col, N);
        simdgroup_multiply_accumulate(acc, a, b, acc);
    }
    simdgroup_store(acc, C + row * N + col, N);
}
"#;

// f16 variant: simdgroup_half8x8, half accumulate (native on the GPU matrix units).
pub(crate) const SGEMM_F16_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void sgemm_h(device const half* A [[buffer(0)]],
                    device const half* B [[buffer(1)]],
                    device half* C       [[buffer(2)]],
                    constant uint& M [[buffer(3)]],
                    constant uint& N [[buffer(4)]],
                    constant uint& K [[buffer(5)]],
                    uint2 gid [[threadgroup_position_in_grid]]) {
    uint row = gid.y * 8;
    uint col = gid.x * 8;
    simdgroup_half8x8 acc = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    for (uint k = 0; k < K; k += 8) {
        simdgroup_half8x8 a, b;
        simdgroup_load(a, A + row * K + k, K);
        simdgroup_load(b, B + k * N + col, N);
        simdgroup_multiply_accumulate(acc, a, b, acc);
    }
    simdgroup_store(acc, C + row * N + col, N);
}
"#;
