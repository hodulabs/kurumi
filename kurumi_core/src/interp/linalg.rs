//! Dense linear algebra: LU with partial pivoting + Cholesky, computed in the operand's own
//! float dtype (f32->f32, f64->f64): no silent promotion, matching the strict-dtype contract and
//! NumPy/PyTorch/JAX. Builders gate f32/f64 only -- there's no canonical low-precision LU, so
//! f16/bf16/fp8 are rejected at record time (picking a compute precision = promotion = frontend's
//! job). Batched over leading dims; trailing `[N, N]` is the matrix per batch.

mod eigen;

pub(crate) use eigen::{eigh, eigvals, qr};

use crate::{DType, Elem, Storage};

// the float types linalg runs in (f32/f64 native). Only the ops LU/Cholesky/det need.
pub(crate) trait LinFloat:
    Copy
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn max0(self) -> Self {
        if self > Self::ZERO { self } else { Self::ZERO }
    }
}
impl LinFloat for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    fn abs(self) -> f32 {
        f32::abs(self)
    }
    fn sqrt(self) -> f32 {
        f32::sqrt(self)
    }
    fn from_f64(x: f64) -> f32 {
        x as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl LinFloat for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    fn from_f64(x: f64) -> f64 {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// Solve A*X = B, computed natively in the operand dtype (f32/f64; builder gates others).
pub(crate) fn solve(a: &Storage, b: &Storage, batch: usize, n: usize, k: usize) -> Storage {
    match a.dtype() {
        DType::F64 => Storage::F64(solve_t(<f64 as Elem>::slice(a), <f64 as Elem>::slice(b), batch, n, k)),
        DType::F32 => Storage::F32(solve_t(<f32 as Elem>::slice(a), <f32 as Elem>::slice(b), batch, n, k)),
        dt => unreachable!("linalg builder gates f32/f64, got {dt:?}"),
    }
}

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

// solve A*X = B per batch. `a`: batch*N*N, `b`: batch*N*K (row-major) -> batch*N*K.
fn solve_t<T: LinFloat>(a: &[T], b: &[T], batch: usize, n: usize, k: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; batch * n * k];
    for bi in 0..batch {
        let mut aa: Vec<T> = a[bi * n * n..(bi + 1) * n * n].to_vec();
        let mut bb: Vec<T> = b[bi * n * k..(bi + 1) * n * k].to_vec();
        for col in 0..n {
            // partial pivot: largest magnitude in the column at/below the diagonal
            let mut piv = col;
            let mut best = aa[col * n + col].abs();
            for row in (col + 1)..n {
                let v = aa[row * n + col].abs();
                if v > best {
                    best = v;
                    piv = row;
                }
            }
            if piv != col {
                for c in 0..n {
                    aa.swap(col * n + c, piv * n + c);
                }
                for c in 0..k {
                    bb.swap(col * k + c, piv * k + c);
                }
            }
            let diag = aa[col * n + col];
            for row in (col + 1)..n {
                let factor = aa[row * n + col] / diag;
                for c in col..n {
                    aa[row * n + c] = aa[row * n + c] - factor * aa[col * n + c];
                }
                for c in 0..k {
                    bb[row * k + c] = bb[row * k + c] - factor * bb[col * k + c];
                }
            }
        }
        // back-substitution
        let xb = &mut out[bi * n * k..(bi + 1) * n * k];
        for row in (0..n).rev() {
            for c in 0..k {
                let mut s = bb[row * k + c];
                for l in (row + 1)..n {
                    s = s - aa[row * n + l] * xb[l * k + c];
                }
                xb[row * k + c] = s / aa[row * n + row];
            }
        }
    }
    out
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
