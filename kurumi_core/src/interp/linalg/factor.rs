//! Factorization-based ops: Cholesky (A = L*L^T) and determinant via LU (product of pivots x
//! row-swap sign), in the operand's own float dtype. The `LinFloat` abstraction and the op
//! dispatcher live in the parent `linalg.rs`; the linear solve in `solve`.

use super::LinFloat;
use crate::{DType, Elem, Storage};

/// Cholesky (A = L*L^T), computed natively in the operand dtype (f32/f64).
pub(crate) fn cholesky(a: &Storage, batch: usize, n: usize) -> Storage {
    match a.dtype() {
        DType::F64 => Storage::F64(cholesky_t(<f64 as Elem>::slice(a), batch, n)),
        DType::F32 => Storage::F32(cholesky_t(<f32 as Elem>::slice(a), batch, n)),
        dt => unreachable!("linalg builder gates f32/f64, got {dt:?}"),
    }
}

/// Determinant, computed natively in the operand dtype (f32/f64).
pub(crate) fn det(a: &Storage, batch: usize, n: usize) -> Storage {
    match a.dtype() {
        DType::F64 => Storage::F64(det_t(<f64 as Elem>::slice(a), batch, n)),
        DType::F32 => Storage::F32(det_t(<f32 as Elem>::slice(a), batch, n)),
        dt => unreachable!("linalg builder gates f32/f64, got {dt:?}"),
    }
}

// Cholesky per batch: A = L*L^T, L lower-triangular with positive diagonal (assumes
// A symmetric positive-definite). `a`: batch*N*N (row-major) -> L, batch*N*N.
fn cholesky_t<T: LinFloat>(a: &[T], batch: usize, n: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; batch * n * n];
    for bi in 0..batch {
        let src = &a[bi * n * n..(bi + 1) * n * n];
        let l = &mut out[bi * n * n..(bi + 1) * n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = src[i * n + j];
                for k in 0..j {
                    s = s - l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    l[i * n + j] = s.max0().sqrt(); // clamp guards tiny negative from roundoff
                } else {
                    l[i * n + j] = s / l[j * n + j];
                }
            }
        }
    }
    out
}

// det per batch via LU (product of pivots x row-swap sign). `a`: batch*N*N.
fn det_t<T: LinFloat>(a: &[T], batch: usize, n: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; batch];
    for bi in 0..batch {
        let mut aa: Vec<T> = a[bi * n * n..(bi + 1) * n * n].to_vec();
        let mut det = T::ONE;
        for col in 0..n {
            let mut piv = col;
            let mut best = aa[col * n + col].abs();
            for row in (col + 1)..n {
                let v = aa[row * n + col].abs();
                if v > best {
                    best = v;
                    piv = row;
                }
            }
            if best == T::ZERO {
                det = T::ZERO;
                break;
            }
            if piv != col {
                for c in 0..n {
                    aa.swap(col * n + c, piv * n + c);
                }
                det = -det;
            }
            let diag = aa[col * n + col];
            det = det * diag;
            for row in (col + 1)..n {
                let factor = aa[row * n + col] / diag;
                for c in col..n {
                    aa[row * n + c] = aa[row * n + c] - factor * aa[col * n + c];
                }
            }
        }
        out[bi] = det;
    }
    out
}
