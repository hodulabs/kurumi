//! Householder QR (interp kernels): the reduced factorization plus the full nxn QR helper
//! the eigenvalue iteration (`eigvals`) reuses.

use crate::interp::linalg::LinFloat;
use crate::{DType, Elem, Storage};

/// Householder QR (reduced), operand dtype. `want_r` picks R `[.., K, N]` else Q
/// `[.., M, K]`, `K = min(M,N)`.
pub(crate) fn qr(a: &Storage, batch: usize, m: usize, n: usize, want_r: bool) -> Storage {
    match a.dtype() {
        DType::F64 => Storage::F64(qr_t(<f64 as Elem>::slice(a), batch, m, n, want_r)),
        DType::F32 => Storage::F32(qr_t(<f32 as Elem>::slice(a), batch, m, n, want_r)),
        dt => unreachable!("linalg builder gates f32/f64, got {dt:?}"),
    }
}

// full nxn QR (Q, R both [n,n]): helper for the eigenvalue iteration.
#[allow(clippy::needless_range_loop)] // index math is clearer than iterator gymnastics
pub(super) fn qr_full_t<T: LinFloat>(a: &[T], n: usize) -> (Vec<T>, Vec<T>) {
    let mut r = a.to_vec();
    let mut q = vec![T::ZERO; n * n];
    for i in 0..n {
        q[i * n + i] = T::ONE;
    }
    let two = T::from_f64(2.0);
    for j in 0..n {
        let mut norm = T::ZERO;
        for i in j..n {
            norm = norm + r[i * n + j] * r[i * n + j];
        }
        norm = norm.sqrt();
        if norm <= T::ZERO {
            continue;
        }
        let alpha = if r[j * n + j] < T::ZERO { norm } else { T::ZERO - norm };
        let mut v = vec![T::ZERO; n];
        for i in j..n {
            v[i] = r[i * n + j];
        }
        v[j] = v[j] - alpha;
        let mut vn2 = T::ZERO;
        for i in j..n {
            vn2 = vn2 + v[i] * v[i];
        }
        if vn2 <= T::ZERO {
            continue;
        }
        for col in 0..n {
            let mut dot = T::ZERO;
            for i in j..n {
                dot = dot + v[i] * r[i * n + col];
            }
            let f = two * dot / vn2;
            for i in j..n {
                r[i * n + col] = r[i * n + col] - f * v[i];
            }
        }
        for row in 0..n {
            let mut dot = T::ZERO;
            for l in j..n {
                dot = dot + q[row * n + l] * v[l];
            }
            let f = two * dot / vn2;
            for i in j..n {
                q[row * n + i] = q[row * n + i] - f * v[i];
            }
        }
    }
    (q, r)
}

// Householder QR per batch. Returns the reduced R [k,n] or Q [m,k], k=min(m,n).
#[allow(clippy::needless_range_loop)] // index math (i*n+j) is clearer than iterator gymnastics
fn qr_t<T: LinFloat>(a: &[T], batch: usize, m: usize, n: usize, want_r: bool) -> Vec<T> {
    let k = m.min(n);
    let mut out = vec![T::ZERO; if want_r { batch * k * n } else { batch * m * k }];
    let two = T::from_f64(2.0);
    for bi in 0..batch {
        let mut r: Vec<T> = a[bi * m * n..(bi + 1) * m * n].to_vec(); // [m,n]
        let mut q = vec![T::ZERO; m * m]; // [m,m], starts as I
        for i in 0..m {
            q[i * m + i] = T::ONE;
        }
        for j in 0..k {
            let mut norm = T::ZERO;
            for i in j..m {
                norm = norm + r[i * n + j] * r[i * n + j];
            }
            norm = norm.sqrt();
            if norm <= T::ZERO {
                continue;
            }
            let alpha = if r[j * n + j] < T::ZERO { norm } else { -norm }; // -sign(x0)*||x||
            let mut vv = vec![T::ZERO; m];
            for i in j..m {
                vv[i] = r[i * n + j];
            }
            vv[j] = vv[j] - alpha;
            let mut vn2 = T::ZERO;
            for i in j..m {
                vn2 = vn2 + vv[i] * vv[i];
            }
            if vn2 <= T::ZERO {
                continue;
            }
            // R <- H R  (H = I - 2 v v^T / vn2), columns 0..n, rows j..m
            for col in 0..n {
                let mut dot = T::ZERO;
                for i in j..m {
                    dot = dot + vv[i] * r[i * n + col];
                }
                let f = two * dot / vn2;
                for i in j..m {
                    r[i * n + col] = r[i * n + col] - f * vv[i];
                }
            }
            // Q <- Q H, columns j..m per row
            for row in 0..m {
                let mut dot = T::ZERO;
                for l in j..m {
                    dot = dot + q[row * m + l] * vv[l];
                }
                let f = two * dot / vn2;
                for i in j..m {
                    q[row * m + i] = q[row * m + i] - f * vv[i];
                }
            }
        }
        if want_r {
            let ob = &mut out[bi * k * n..(bi + 1) * k * n];
            for i in 0..k {
                ob[i * n..(i + 1) * n].copy_from_slice(&r[i * n..(i + 1) * n]);
            }
        } else {
            let ob = &mut out[bi * m * k..(bi + 1) * m * k];
            for row in 0..m {
                for col in 0..k {
                    ob[row * k + col] = q[row * m + col];
                }
            }
        }
    }
    out
}
