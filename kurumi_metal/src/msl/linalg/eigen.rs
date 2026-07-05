//! Eigendecomposition & QR kernel sources (f32): symmetric eigh (cyclic Jacobi), reduced QR
//! (Householder), general eigvals (QR algorithm). One thread per batch, scratch in device
//! memory (runtime N -> no local arrays). Launched by `context/dispatch/linalg/eigen.rs`;
//! mirror `interp/linalg/eigen`.

// `eigh_k`: symmetric eigendecomposition, cyclic Jacobi. Output packs `[.., N, N+1]` (columns
// 0..N eigenvectors, column N eigenvalues; ascending). Port of `jacobi_eigh_t`.
pub(crate) const EIGH_MSL: &str = r"#include <metal_stdlib>
using namespace metal;
kernel void eigh_k(device const float* A [[buffer(0)]], device float* M [[buffer(1)]],
                   device float* V [[buffer(2)]], device float* out [[buffer(3)]],
                   constant uint& N [[buffer(4)]], uint bi [[thread_position_in_grid]]) {
    uint an = bi*N*N;
    for (uint i=0;i<N*N;i++) M[an+i]=A[an+i];
    for (uint i=0;i<N*N;i++) V[an+i]=0.0f;
    for (uint i=0;i<N;i++) V[an+i*N+i]=1.0f;
    float scale=0.0f; for(uint i=0;i<N*N;i++) scale+=M[an+i]*M[an+i];
    float thresh = 1e-30f + scale*1e-26f;
    for (uint sweep=0; sweep<100u; sweep++) {
        float off=0.0f;
        for(uint p=0;p<N;p++) for(uint q=p+1;q<N;q++) off += M[an+p*N+q]*M[an+p*N+q];
        if (off <= thresh) break;
        for (uint p=0;p<N;p++) for (uint q=p+1;q<N;q++) {
            float apq=M[an+p*N+q];
            if (fabs(apq)<=0.0f) continue;
            float app=M[an+p*N+p], aqq=M[an+q*N+q];
            float theta=(aqq-app)/(2.0f*apq);
            float sg = theta<0.0f ? -1.0f : 1.0f;
            float t = sg/(fabs(theta)+sqrt(theta*theta+1.0f));
            float c = 1.0f/sqrt(t*t+1.0f); float s=t*c;
            for (uint i=0;i<N;i++) if(i!=p&&i!=q){
                float aip=M[an+i*N+p], aiq=M[an+i*N+q];
                float nip=c*aip-s*aiq, niq=s*aip+c*aiq;
                M[an+i*N+p]=nip; M[an+p*N+i]=nip; M[an+i*N+q]=niq; M[an+q*N+i]=niq; }
            M[an+p*N+p]=c*c*app-2.0f*s*c*apq+s*s*aqq;
            M[an+q*N+q]=s*s*app+2.0f*s*c*apq+c*c*aqq;
            M[an+p*N+q]=0.0f; M[an+q*N+p]=0.0f;
            for (uint i=0;i<N;i++){ float vip=V[an+i*N+p], viq=V[an+i*N+q];
                V[an+i*N+p]=c*vip-s*viq; V[an+i*N+q]=s*vip+c*viq; } }
    }
    for (uint i=0;i<N;i++){ uint mi=i; float mv=M[an+i*N+i];
        for(uint j=i+1;j<N;j++) if(M[an+j*N+j]<mv){ mv=M[an+j*N+j]; mi=j; }
        if(mi!=i){ float tt=M[an+i*N+i]; M[an+i*N+i]=M[an+mi*N+mi]; M[an+mi*N+mi]=tt;
            for(uint r=0;r<N;r++){ float tv=V[an+r*N+i]; V[an+r*N+i]=V[an+r*N+mi]; V[an+r*N+mi]=tv; } } }
    uint ob=bi*N*(N+1);
    for(uint i=0;i<N;i++){ for(uint j=0;j<N;j++) out[ob+i*(N+1)+j]=V[an+i*N+j];
        out[ob+i*(N+1)+N]=M[an+i*N+i]; }
}";

// `qr_k`: reduced Householder QR. `WR` picks R `[.., K, N]` else Q `[.., M, K]`, K=min(M,N).
// Port of `qr_t`.
pub(crate) const QR_MSL: &str = r"#include <metal_stdlib>
using namespace metal;
kernel void qr_k(device const float* A [[buffer(0)]], device float* R [[buffer(1)]],
                 device float* Q [[buffer(2)]], device float* vv [[buffer(3)]], device float* out [[buffer(4)]],
                 constant uint& M [[buffer(5)]], constant uint& N [[buffer(6)]], constant uint& WR [[buffer(7)]],
                 uint bi [[thread_position_in_grid]]) {
    uint k = min(M,N);
    uint rn=bi*M*N, qn=bi*M*M, vn=bi*M;
    for(uint i=0;i<M*N;i++) R[rn+i]=A[rn+i];
    for(uint i=0;i<M*M;i++) Q[qn+i]=0.0f;
    for(uint i=0;i<M;i++) Q[qn+i*M+i]=1.0f;
    for(uint j=0;j<k;j++){
        float norm=0.0f; for(uint i=j;i<M;i++) norm+=R[rn+i*N+j]*R[rn+i*N+j]; norm=sqrt(norm);
        if(norm<=0.0f) continue;
        float alpha = R[rn+j*N+j]<0.0f ? norm : -norm;
        for(uint i=j;i<M;i++) vv[vn+i]=R[rn+i*N+j]; vv[vn+j]-=alpha;
        float vn2=0.0f; for(uint i=j;i<M;i++) vn2+=vv[vn+i]*vv[vn+i];
        if(vn2<=0.0f) continue;
        for(uint col=0;col<N;col++){ float dot=0.0f; for(uint i=j;i<M;i++) dot+=vv[vn+i]*R[rn+i*N+col];
            float f=2.0f*dot/vn2; for(uint i=j;i<M;i++) R[rn+i*N+col]-=f*vv[vn+i]; }
        for(uint row=0;row<M;row++){ float dot=0.0f; for(uint l=j;l<M;l++) dot+=Q[qn+row*M+l]*vv[vn+l];
            float f=2.0f*dot/vn2; for(uint i=j;i<M;i++) Q[qn+row*M+i]-=f*vv[vn+i]; } }
    if(WR!=0u){ uint ob=bi*k*N; for(uint i=0;i<k;i++) for(uint c=0;c<N;c++) out[ob+i*N+c]=R[rn+i*N+c]; }
    else { uint ob=bi*M*k; for(uint row=0;row<M;row++) for(uint c=0;c<k;c++) out[ob+row*k+c]=Q[qn+row*M+c]; }
}";

// `eigvals_k`: eigenvalues of a general (nonsymmetric) real matrix via the unshifted QR
// algorithm -> complex `[.., N]` (C64/float2). Port of `eigvals_t` + `qr_full_t`.
pub(crate) const EIGVALS_MSL: &str = r"#include <metal_stdlib>
using namespace metal;
kernel void eigvals_k(device const float* A [[buffer(0)]], device float* H [[buffer(1)]],
                      device float* Q [[buffer(2)]], device float* R [[buffer(3)]], device float* vv [[buffer(4)]],
                      device float* NH [[buffer(5)]], device float2* out [[buffer(6)]],
                      constant uint& N [[buffer(7)]], uint bi [[thread_position_in_grid]]) {
    uint hn=bi*N*N, vn=bi*N;
    for(uint i=0;i<N*N;i++) H[hn+i]=A[hn+i];
    float scale=0.0f; for(uint i=0;i<N*N;i++) scale+=H[hn+i]*H[hn+i]; scale=sqrt(scale);
    float eps=1e-10f*(scale+1.0f);
    uint iters=60u*max(N,1u);
    for(uint it=0; it<iters; it++){
        for(uint i=0;i<N*N;i++) R[hn+i]=H[hn+i];
        for(uint i=0;i<N*N;i++) Q[hn+i]=0.0f;
        for(uint i=0;i<N;i++) Q[hn+i*N+i]=1.0f;
        for(uint j=0;j<N;j++){
            float norm=0.0f; for(uint i=j;i<N;i++) norm+=R[hn+i*N+j]*R[hn+i*N+j]; norm=sqrt(norm);
            if(norm<=0.0f) continue;
            float alpha = R[hn+j*N+j]<0.0f ? norm : -norm;
            for(uint i=j;i<N;i++) vv[vn+i]=R[hn+i*N+j]; vv[vn+j]-=alpha;
            float vn2=0.0f; for(uint i=j;i<N;i++) vn2+=vv[vn+i]*vv[vn+i];
            if(vn2<=0.0f) continue;
            for(uint col=0;col<N;col++){ float dot=0.0f; for(uint i=j;i<N;i++) dot+=vv[vn+i]*R[hn+i*N+col];
                float f=2.0f*dot/vn2; for(uint i=j;i<N;i++) R[hn+i*N+col]-=f*vv[vn+i]; }
            for(uint row=0;row<N;row++){ float dot=0.0f; for(uint l=j;l<N;l++) dot+=Q[hn+row*N+l]*vv[vn+l];
                float f=2.0f*dot/vn2; for(uint i=j;i<N;i++) Q[hn+row*N+i]-=f*vv[vn+i]; } }
        for(uint i=0;i<N;i++) for(uint kk=0;kk<N;kk++){ float s=0.0f;
            for(uint l=0;l<N;l++) s+=R[hn+i*N+l]*Q[hn+l*N+kk]; NH[hn+i*N+kk]=s; }
        for(uint i=0;i<N*N;i++) H[hn+i]=NH[hn+i];
    }
    uint ob=bi*N; uint i=0;
    while(i<N){
        if(i+1>=N || fabs(H[hn+(i+1)*N+i])<=eps){ out[ob+i]=float2(H[hn+i*N+i],0.0f); i+=1u; }
        else {
            float aa=H[hn+i*N+i], bb=H[hn+i*N+i+1], cc=H[hn+(i+1)*N+i], dd=H[hn+(i+1)*N+i+1];
            float tr=aa+dd, det=aa*dd-bb*cc; float disc=tr*tr-4.0f*det;
            if(disc>=0.0f){ float sq=sqrt(disc); out[ob+i]=float2((tr+sq)/2.0f,0.0f); out[ob+i+1]=float2((tr-sq)/2.0f,0.0f); }
            else { float sq=sqrt(-disc); out[ob+i]=float2(tr/2.0f,sq/2.0f); out[ob+i+1]=float2(tr/2.0f,-sq/2.0f); }
            i+=2u;
        }
    }
}";
