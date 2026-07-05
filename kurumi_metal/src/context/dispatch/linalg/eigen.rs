//! Eigendecomposition & QR device launchers (f32): symmetric eigh (cyclic Jacobi),
//! reduced QR (Householder), general eigvals (QR algorithm). ONE thread per batch,
//! scratch in device memory (runtime N -> no local arrays). Kernel sources are in
//! `msl::linalg::eigen`; direct LU/Cholesky launchers are in the parent `linalg`.

use crate::Buffer;
use crate::context::{MetalContext, set_u32};
use crate::msl::linalg::eigen::{EIGH_MSL, EIGVALS_MSL, QR_MSL};
use kurumi_core::DType;
use objc2_metal::MTLComputeCommandEncoder;

impl MetalContext {
    /// Symmetric eigendecomposition per batch (f32), cyclic Jacobi: one thread per
    /// batch, M/V scratch in device memory. Output packs `[.., N, N+1]` (columns 0..N
    /// eigenvectors, column N eigenvalues; ascending).
    pub(crate) fn eigh_dev(&self, a: &Buffer, batch: usize, n: usize) -> Buffer {
        let pso = self.cached(EIGH_MSL, "eigh_k");
        let m = self.empty(batch * n * n, DType::F32);
        let v = self.empty(batch * n * n, DType::F32);
        let out = self.empty(batch * n * (n + 1), DType::F32);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&m), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&v), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 3);
                set_u32(enc, n as u32, 4);
            },
            batch,
        );
        out
    }

    /// Reduced Householder QR per batch (f32). `want_r` picks R `[.., K, N]` else Q
    /// `[.., M, K]`, K=min(M,N). R/Q/v scratch in device memory.
    pub(crate) fn qr_dev(&self, a: &Buffer, batch: usize, m: usize, n: usize, want_r: bool) -> Buffer {
        let pso = self.cached(QR_MSL, "qr_k");
        let k = m.min(n);
        let rbuf = self.empty(batch * m * n, DType::F32);
        let qbuf = self.empty(batch * m * m, DType::F32);
        let vbuf = self.empty(batch * m, DType::F32);
        let out = self.empty(batch * if want_r { k * n } else { m * k }, DType::F32);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&rbuf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&qbuf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&vbuf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 4);
                set_u32(enc, m as u32, 5);
                set_u32(enc, n as u32, 6);
                set_u32(enc, want_r as u32, 7);
            },
            batch,
        );
        out
    }

    /// Eigenvalues of a general (nonsymmetric) real matrix per batch (f32) via the
    /// unshifted QR algorithm -> complex `[.., N]` (C64/float2). H/Q/R/NH/v scratch.
    pub(crate) fn eigvals_dev(&self, a: &Buffer, batch: usize, n: usize) -> Buffer {
        let pso = self.cached(EIGVALS_MSL, "eigvals_k");
        let h = self.empty(batch * n * n, DType::F32);
        let q = self.empty(batch * n * n, DType::F32);
        let rr = self.empty(batch * n * n, DType::F32);
        let vv = self.empty(batch * n, DType::F32);
        let nh = self.empty(batch * n * n, DType::F32);
        let out = self.empty(batch * n, DType::C64);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&h), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&q), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&rr), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&vv), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&nh), 0, 5);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 6);
                set_u32(enc, n as u32, 7);
            },
            batch,
        );
        out
    }
}
