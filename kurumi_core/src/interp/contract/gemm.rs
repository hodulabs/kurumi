//! f32 dot_general fast path: 2D x 2D, one contract axis, no batch (the matmul case) via
//! Accelerate cblas_sgemm on macOS / the `gemm` crate elsewhere. Every other shape/dtype
//! falls back to the generic loop (`dot_num`) in the parent `contract`.

use crate::interp::contract::dot_num;
use crate::{Storage, TensorVal, row_major_strides};

pub(crate) fn dot_general(
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
    lhs_contract: &[usize],
    rhs_contract: &[usize],
    lhs_batch: &[usize],
    rhs_batch: &[usize],
) -> TensorVal {
    // fast path: 2D x 2D, one contract axis, no batch (the matmul case): tight
    // triple loop with incremental addressing instead of per-element operand_flat
    if a_shape.len() == 2
        && b_shape.len() == 2
        && lhs_batch.is_empty()
        && rhs_batch.is_empty()
        && lhs_contract.len() == 1
        && rhs_contract.len() == 1
    {
        let (lc, rc) = (lhs_contract[0], rhs_contract[0]);
        let (lf, rf) = (1 - lc, 1 - rc);
        let (m, k, n) = (a_shape[lf], a_shape[lc], b_shape[rf]);
        let (a_st, b_st) = (row_major_strides(a_shape), row_major_strides(b_shape));
        let (a_fs, a_cs, b_fs, b_cs) = (a_st[lf], a_st[lc], b_st[rf], b_st[rc]);
        let mut out = vec![0f32; m * n];
        if m != 0 && n != 0 && k != 0 {
            matmul_2d(m, n, k, &mut out, a_data, a_fs, a_cs, b_data, b_fs, b_cs);
        }
        return TensorVal { shape: vec![m, n], storage: Storage::F32(out) };
    }

    // general (batched / multi-axis) case: same generic loop as other dtypes
    dot_num(a_data, a_shape, b_data, b_shape, lhs_contract, rhs_contract, lhs_batch, rhs_batch)
}

// f32 2D matmul: C[m,n] = A[m,k] * B[k,n]. macOS -> Accelerate cblas_sgemm (AMX-backed,
// ~10x the gemm crate); else the gemm crate. Inputs are contiguous, so exactly one of each
// matrix's (row, col) strides is 1, which BLAS expresses as a Trans flag + leading dimension.
fn matmul_2d(
    m: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
    a: &[f32],
    a_fs: usize,
    a_cs: usize,
    b: &[f32],
    b_fs: usize,
    b_cs: usize,
) {
    #[cfg(target_os = "macos")]
    {
        debug_assert!(a_cs == 1 || a_fs == 1, "gemm operand A not BLAS-expressible");
        debug_assert!(b_cs == 1 || b_fs == 1, "gemm operand B not BLAS-expressible");
        // A is m x k with row stride a_fs, col stride a_cs; unit col stride = row-major NoTrans
        // (lda = row stride), unit row stride = transpose. Same for B. Leading dim is clamped to
        // the BLAS minimum: a dim of 1 has degenerate stride (=1) but BLAS needs ld >= spanned dim.
        let (ta, lda) = if a_cs == 1 { (accelerate::NO_TRANS, a_fs.max(k)) } else { (accelerate::TRANS, a_cs.max(m)) };
        let (tb, ldb) = if b_fs == 1 { (accelerate::NO_TRANS, b_cs.max(n)) } else { (accelerate::TRANS, b_fs.max(k)) };
        // SAFETY: m x k / k x n / m x n matrices lie fully in a / b / out; lda/ldb/ldc
        // are the positive strides just asserted; beta = 0 so out is written, not read.
        unsafe {
            accelerate::cblas_sgemm(
                accelerate::ROW_MAJOR,
                ta,
                tb,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                0.0,
                out.as_mut_ptr(),
                n as i32,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    // SAFETY: strides describe valid m x k / k x n / m x n matrices fully in bounds.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            a_cs as isize,
            a_fs as isize,
            b.as_ptr(),
            b_fs as isize,
            b_cs as isize,
            0.0_f32,
            1.0_f32,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
}

// Accelerate (system framework, no crate dep) CBLAS single-precision GEMM.
#[cfg(target_os = "macos")]
mod accelerate {
    pub const ROW_MAJOR: u32 = 101; // CblasRowMajor
    pub const NO_TRANS: u32 = 111; // CblasNoTrans
    pub const TRANS: u32 = 112; // CblasTrans
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: u32,
            transa: u32,
            transb: u32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}
