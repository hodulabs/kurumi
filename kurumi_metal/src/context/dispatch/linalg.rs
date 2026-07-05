//! Dense linalg launchers (f32 only: Metal has no double; the dtype-native engine does f32 linalg
//! in f32, so these match the oracle). ONE thread per batch matrix runs the serial LU/Cholesky in
//! place: batch-parallel, good for the many-small case. One large matrix is a single GPU thread
//! (O(N^3) serial); add a blocked panel factorization if big-single solves get hot. eigh/qr/eigvals
//! in sibling `eigen`. Kernel sources are in `msl::linalg`.

mod eigen;

use crate::Buffer;
use crate::context::{MetalContext, set_u32};
use crate::msl::linalg::{CHOL_MSL, DET_MSL, SOLVE_MSL};
use kurumi_core::DType;
use objc2_metal::MTLComputeCommandEncoder;

impl MetalContext {
    /// Solve A*X = B per batch (f32). A: batch*N*N, B: batch*N*K -> X: batch*N*K.
    /// LU with partial pivoting, one thread per batch (working copy in `aa`).
    pub(crate) fn solve_dev(&self, a: &Buffer, b: &Buffer, batch: usize, n: usize, k: usize) -> Buffer {
        let pso = self.cached(SOLVE_MSL, "solve_k");
        let aa = self.empty(batch * n * n, DType::F32);
        let x = self.empty(batch * n * k, DType::F32);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(b), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&aa), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&x), 0, 3);
                set_u32(enc, n as u32, 4);
                set_u32(enc, k as u32, 5);
            },
            batch,
        );
        x
    }

    /// Determinant per batch (f32) via LU: product of pivots x row-swap sign. A: batch*N*N -> batch.
    pub(crate) fn det_dev(&self, a: &Buffer, batch: usize, n: usize) -> Buffer {
        let pso = self.cached(DET_MSL, "det_k");
        let aa = self.empty(batch * n * n, DType::F32);
        let out = self.empty(batch, DType::F32);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&aa), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 2);
                set_u32(enc, n as u32, 3);
            },
            batch,
        );
        out
    }

    /// Cholesky per batch (f32): A = L*L^T, lower-triangular L. A: batch*N*N -> batch*N*N.
    pub(crate) fn cholesky_dev(&self, a: &Buffer, batch: usize, n: usize) -> Buffer {
        let pso = self.cached(CHOL_MSL, "chol_k");
        let out = self.empty(batch * n * n, DType::F32);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
                set_u32(enc, n as u32, 2);
            },
            batch,
        );
        out
    }
}
