//! Host-path GEMM launches: the hand-written simdgroup f32/f16 kernels and the
//! naive integer/bfloat kernel. The device-resident MPS GEMM path lives in
//! `dispatch/matmul.rs`; these back the CPU-offload fallback (`backend/hostgemm`).

use crate::Pipeline;
use crate::context::{MetalContext, read_t, set_u32};
use crate::msl::hostgemm::{SGEMM_F16_MSL, SGEMM_MSL};
use half::f16;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize};

impl MetalContext {
    /// Compile the simdgroup-matrix GEMM kernel once (reuse the PSO across calls).
    pub fn gemm_pipeline(&self) -> Pipeline {
        self.pipeline(SGEMM_MSL, "sgemm")
    }

    /// C[M,N] = A[M,K] @ B[K,N], fp32, via 8x8 `simdgroup_matrix` tiles.
    /// M, N, K must be multiples of 8. Reuses a compiled `pso`.
    pub fn matmul(&self, pso: &Pipeline, a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert!(m.is_multiple_of(8) && n.is_multiple_of(8) && k.is_multiple_of(8), "M,N,K must be multiples of 8");
        let (ba, bb, bc) = (self.buffer_of(a), self.buffer_of(b), self.empty_buffer(m * n));
        let (mu, nu, ku) = (m as u32, n as u32, k as u32);
        let cmd = self.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&ba), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&bb), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&bc), 0, 2);
            set_u32(&enc, mu, 3);
            set_u32(&enc, nu, 4);
            set_u32(&enc, ku, 5);
        }
        // one threadgroup (= one simdgroup, 32 threads) per 8x8 output tile
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: n / 8, height: m / 8, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        read_t::<f32>(&bc, m * n)
    }

    /// f16 GEMM pipeline (compile once).
    pub fn gemm_f16_pipeline(&self) -> Pipeline {
        self.pipeline(SGEMM_F16_MSL, "sgemm_h")
    }

    /// C[M,N] = A[M,K] @ B[K,N], f16 in/out (f16 accumulate), via `simdgroup_half8x8`.
    /// Native on the GPU matrix units; M,N,K multiples of 8.
    pub fn matmul_f16(&self, pso: &Pipeline, a: &[f16], m: usize, k: usize, b: &[f16], n: usize) -> Vec<f16> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert!(m.is_multiple_of(8) && n.is_multiple_of(8) && k.is_multiple_of(8), "M,N,K mult of 8");
        let ba = self.buffer_bytes(a);
        let bb = self.buffer_bytes(b);
        let bc = self.empty_bytes(m * n * std::mem::size_of::<f16>());
        let (mu, nu, ku) = (m as u32, n as u32, k as u32);
        let cmd = self.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&ba), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&bb), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&bc), 0, 2);
            set_u32(&enc, mu, 3);
            set_u32(&enc, nu, 4);
            set_u32(&enc, ku, 5);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: n / 8, height: m / 8, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        read_t::<f16>(&bc, m * n)
    }

    /// Naive one-thread-per-output GEMM for a given MSL element type (`int`, `uint`,
    /// `long`, `uchar`, `bfloat`): integer types have no `simdgroup_matrix`, so this
    /// covers the long-tail dtypes; floats use the fast simdgroup kernels.
    pub fn matmul_naive<T: Copy>(&self, pso: &Pipeline, a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        let ba = self.buffer_bytes(a);
        let bb = self.buffer_bytes(b);
        let bc = self.empty_bytes(m * n * std::mem::size_of::<T>());
        let (mu, nu, ku) = (m as u32, n as u32, k as u32);
        let cmd = self.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&ba), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&bb), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&bc), 0, 2);
            set_u32(&enc, mu, 3);
            set_u32(&enc, nu, 4);
            set_u32(&enc, ku, 5);
        }
        // one thread per output element (grid N x M, bounds-checked in the kernel)
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: n, height: m, depth: 1 },
            MTLSize { width: 16, height: 16, depth: 1 },
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        read_t::<T>(&bc, m * n)
    }
}
