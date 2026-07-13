//! Linear solve A*X = B via LU with partial pivoting + back-substitution, in the operand's own
//! float dtype. The `LinFloat` abstraction and the op dispatcher live in the parent `linalg.rs`;
//! det/Cholesky in `factor`.

use super::LinFloat;
use crate::{DType, Elem, Storage};

/// Solve A*X = B, computed natively in the operand dtype (f32/f64; builder gates others).
pub(crate) fn solve(a: &Storage, b: &Storage, batch: usize, n: usize, k: usize) -> Storage {
    match a.dtype() {
        DType::F64 => Storage::F64(solve_t(<f64 as Elem>::slice(a), <f64 as Elem>::slice(b), batch, n, k)),
        DType::F32 => Storage::F32(solve_t(<f32 as Elem>::slice(a), <f32 as Elem>::slice(b), batch, n, k)),
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
