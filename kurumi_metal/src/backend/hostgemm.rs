//! Host-path GEMM: a storage-level `matmul` for the dtypes the device can run
//! (f32/f16 via simdgroup_matrix, bf16/int via the naive kernel), used by the CPU-
//! fallback branch of `eval` (`host_op`). Device-resident GEMM is `mps_matmul_dev`.

use crate::backend::MetalBackend;
use crate::msl::hostgemm::naive_mm_msl;
use crate::{MetalContext, Pipeline};
use half::f16;
use kurumi_core::Storage;

// a GPU GEMM entry point for element type T (simdgroup f32/f16 kernels).
type GpuMatmul<T> = fn(&MetalContext, &Pipeline, &[T], usize, usize, &[T], usize) -> Vec<T>;

impl MetalBackend {
    /// GPU GEMM for the dtypes the device can do (f32/f16/bf16 + integers); f64
    /// (no GPU double) and bool error: `eval` then falls back to the CPU.
    pub fn matmul(
        &self,
        a: &Storage,
        m: usize,
        k: usize,
        b: &Storage,
        n: usize,
    ) -> Result<Storage, kurumi_core::Error> {
        if a.dtype() != b.dtype() {
            return Err(kurumi_core::Error::backend(format!(
                "metal matmul dtype mismatch: {:?} vs {:?}",
                a.dtype(),
                b.dtype()
            )));
        }
        match (a, b) {
            // fast path: f32/f16 on the GPU matrix units (simdgroup_matrix). f16
            // is native here: the CPU can only emulate it (upcast to f32).
            (Storage::F32(av), Storage::F32(bv)) => {
                Ok(Storage::F32(self.run_padded(&self.gemm_f32, av, m, k, bv, n, 0.0f32, MetalContext::matmul)))
            }
            (Storage::F16(av), Storage::F16(bv)) => {
                Ok(Storage::F16(self.run_padded(&self.gemm_f16, av, m, k, bv, n, f16::ZERO, MetalContext::matmul_f16)))
            }
            // long-tail dtypes via the naive kernel (no integer simdgroup_matrix)
            (Storage::BF16(av), Storage::BF16(bv)) => Ok(Storage::BF16(self.naive(av, m, k, bv, n, "bfloat"))),
            (Storage::I32(av), Storage::I32(bv)) => Ok(Storage::I32(self.naive(av, m, k, bv, n, "int"))),
            (Storage::I64(av), Storage::I64(bv)) => Ok(Storage::I64(self.naive(av, m, k, bv, n, "long"))),
            (Storage::U32(av), Storage::U32(bv)) => Ok(Storage::U32(self.naive(av, m, k, bv, n, "uint"))),
            (Storage::U8(av), Storage::U8(bv)) => Ok(Storage::U8(self.naive(av, m, k, bv, n, "uchar"))),
            // F64 (Apple GPUs have no double) and BOOL (matmul is meaningless) error
            _ => Err(kurumi_core::Error::backend(format!(
                "metal: matmul not supported for dtype {:?} (f64 has no GPU support; bool has no matmul)",
                a.dtype()
            ))),
        }
    }

    // compile + run the naive kernel for an MSL element type (cheap dtypes; the
    // shader compile is a few ms, fine for these cold paths)
    fn naive<T: Copy>(&self, a: &[T], m: usize, k: usize, b: &[T], n: usize, ty: &str) -> Vec<T> {
        let pso = self.ctx.pipeline(&naive_mm_msl(ty), "mm");
        self.ctx.matmul_naive(&pso, a, m, k, b, n)
    }

    // the simdgroup kernels need multiples of 8: zero-pad, run, crop (the zeros
    // contribute nothing). generic over the element type & its GPU matmul fn.
    fn run_padded<T: Copy + Default>(
        &self,
        pso: &Pipeline,
        a: &[T],
        m: usize,
        k: usize,
        b: &[T],
        n: usize,
        _zero: T,
        gpu: GpuMatmul<T>,
    ) -> Vec<T> {
        let r8 = |x: usize| x.div_ceil(8) * 8;
        let (mp, kp, np) = (r8(m), r8(k), r8(n));
        if (mp, kp, np) == (m, k, n) {
            return gpu(&self.ctx, pso, a, m, k, b, n);
        }
        let ap = pad2d(a, m, k, mp, kp);
        let bp = pad2d(b, k, n, kp, np);
        let cp = gpu(&self.ctx, pso, &ap, mp, kp, &bp, np);
        crop2d(&cp, np, m, n)
    }
}

fn pad2d<T: Copy + Default>(src: &[T], rows: usize, cols: usize, prows: usize, pcols: usize) -> Vec<T> {
    let mut out = vec![T::default(); prows * pcols];
    for r in 0..rows {
        out[r * pcols..r * pcols + cols].copy_from_slice(&src[r * cols..r * cols + cols]);
    }
    out
}

fn crop2d<T: Copy>(src: &[T], src_cols: usize, rows: usize, cols: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        out.extend_from_slice(&src[r * src_cols..r * src_cols + cols]);
    }
    out
}
