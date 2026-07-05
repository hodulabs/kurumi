//! Device GEMM kernel source. The real f32/f16 path is MPS (API calls, no MSL); this is
//! the naive complex (float2) GEMM MPS has no path for. Launched by
//! `context/dispatch/matmul.rs`.

// `cmatmul`: complex (C64/float2) batched GEMM, one thread per output, accumulating cmul over
// k. trans_l/trans_r pick each operand's physical layout (op(A)[M,K] @ op(B)[K,N] per batch).
pub(crate) const CMATMUL_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
     static inline float2 cmul(float2 a, float2 b){ return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }\n\
     kernel void cmatmul(device const float2* A [[buffer(0)]], device const float2* B [[buffer(1)]],\n\
                        device float2* C [[buffer(2)]], constant uint& M [[buffer(3)]],\n\
                        constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],\n\
                        constant uint& TL [[buffer(6)]], constant uint& TR [[buffer(7)]],\n\
                        uint gid [[thread_position_in_grid]]) {\n\
         uint bi = gid / (M * N); uint rem = gid % (M * N); uint i = rem / N; uint j = rem % N;\n\
         uint ab = bi * M * K; uint bb = bi * K * N; float2 acc = float2(0);\n\
         for (uint t = 0; t < K; t++) {\n\
             float2 av = TL ? A[ab + t * M + i] : A[ab + i * K + t];\n\
             float2 bv = TR ? B[bb + j * K + t] : B[bb + t * N + j];\n\
             acc += cmul(av, bv); }\n\
         C[gid] = acc; }";
