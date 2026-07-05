//! Dense linalg kernel sources (f32; Metal has no double). One thread per batch matrix runs
//! the serial LU/Cholesky in place. eigh/qr/eigvals sources are in the sibling `eigen`.
//! Launched by `context/dispatch/linalg.rs`; mirror `interp/linalg`.

pub(crate) mod eigen;

// `solve_k`: solve A*X = B per batch via LU with partial pivoting (working copy in `aa`).
pub(crate) const SOLVE_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
     kernel void solve_k(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],\n\
                        device float* aa [[buffer(2)]], device float* X [[buffer(3)]],\n\
                        constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]],\n\
                        uint bi [[thread_position_in_grid]]) {\n\
         uint an = bi*N*N, bn = bi*N*K;\n\
         for (uint i=0;i<N*N;i++) aa[an+i]=A[an+i];\n\
         for (uint i=0;i<N*K;i++) X[bn+i]=B[bn+i];\n\
         for (uint col=0; col<N; col++) {\n\
             uint piv=col; float best=fabs(aa[an+col*N+col]);\n\
             for (uint r=col+1;r<N;r++){ float v=fabs(aa[an+r*N+col]); if(v>best){best=v;piv=r;} }\n\
             if (piv!=col){ for(uint c=0;c<N;c++){ float t=aa[an+col*N+c]; aa[an+col*N+c]=aa[an+piv*N+c]; aa[an+piv*N+c]=t; }\n\
                            for(uint c=0;c<K;c++){ float t=X[bn+col*K+c]; X[bn+col*K+c]=X[bn+piv*K+c]; X[bn+piv*K+c]=t; } }\n\
             float diag=aa[an+col*N+col];\n\
             for (uint r=col+1;r<N;r++){ float f=aa[an+r*N+col]/diag;\n\
                 for(uint c=col;c<N;c++) aa[an+r*N+c]-=f*aa[an+col*N+c];\n\
                 for(uint c=0;c<K;c++) X[bn+r*K+c]-=f*X[bn+col*K+c]; } }\n\
         for (int row=(int)N-1; row>=0; row--){ uint rr=(uint)row; for(uint c=0;c<K;c++){ float s=X[bn+rr*K+c];\n\
             for(uint l=rr+1;l<N;l++) s-=aa[an+rr*N+l]*X[bn+l*K+c];\n\
             X[bn+rr*K+c]=s/aa[an+rr*N+rr]; } } }";

// `det_k`: determinant per batch via LU (product of pivots x row-swap sign).
pub(crate) const DET_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
     kernel void det_k(device const float* A [[buffer(0)]], device float* aa [[buffer(1)]],\n\
                        device float* out [[buffer(2)]], constant uint& N [[buffer(3)]],\n\
                        uint bi [[thread_position_in_grid]]) {\n\
         uint an = bi*N*N;\n\
         for (uint i=0;i<N*N;i++) aa[an+i]=A[an+i];\n\
         float det=1.0f;\n\
         for (uint col=0; col<N; col++) {\n\
             uint piv=col; float best=fabs(aa[an+col*N+col]);\n\
             for (uint r=col+1;r<N;r++){ float v=fabs(aa[an+r*N+col]); if(v>best){best=v;piv=r;} }\n\
             if (best==0.0f){ det=0.0f; break; }\n\
             if (piv!=col){ for(uint c=0;c<N;c++){ float t=aa[an+col*N+c]; aa[an+col*N+c]=aa[an+piv*N+c]; aa[an+piv*N+c]=t; } det=-det; }\n\
             float diag=aa[an+col*N+col]; det*=diag;\n\
             for (uint r=col+1;r<N;r++){ float f=aa[an+r*N+col]/diag;\n\
                 for(uint c=col;c<N;c++) aa[an+r*N+c]-=f*aa[an+col*N+c]; } }\n\
         out[bi]=det; }";

// `chol_k`: Cholesky per batch, A = L*L^T with lower-triangular L.
pub(crate) const CHOL_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
     kernel void chol_k(device const float* A [[buffer(0)]], device float* L [[buffer(1)]],\n\
                        constant uint& N [[buffer(2)]], uint bi [[thread_position_in_grid]]) {\n\
         uint an = bi*N*N;\n\
         for (uint i=0;i<N*N;i++) L[an+i]=0.0f;\n\
         for (uint i=0;i<N;i++) { for (uint j=0;j<=i;j++) {\n\
             float s=A[an+i*N+j];\n\
             for (uint c=0;c<j;c++) s-=L[an+i*N+c]*L[an+j*N+c];\n\
             if (i==j) L[an+i*N+j]=sqrt(max(0.0f,s)); else L[an+i*N+j]=s/L[an+j*N+j]; } } }";
