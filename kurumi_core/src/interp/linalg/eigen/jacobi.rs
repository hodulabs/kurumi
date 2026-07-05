//! Symmetric eigendecomposition (interp kernel): cyclic Jacobi.

use crate::interp::linalg::LinFloat;
use crate::{DType, Elem, Storage};

/// Symmetric eigendecomposition (cyclic Jacobi), operand dtype (f32/f64). Output packs `[.., N, N+1]`:
/// columns `0..N` the eigenvectors, column `N` the eigenvalues (both sorted ascending by eigenvalue).
pub(crate) fn eigh(a: &Storage, batch: usize, n: usize) -> Storage {
    match a.dtype() {
        DType::F64 => Storage::F64(jacobi_eigh_t(<f64 as Elem>::slice(a), batch, n)),
        DType::F32 => Storage::F32(jacobi_eigh_t(<f32 as Elem>::slice(a), batch, n)),
        dt => unreachable!("linalg builder gates f32/f64, got {dt:?}"),
    }
}

// cyclic Jacobi symmetric eigensolver per batch -> packed [n, n+1] (vectors | values).
fn jacobi_eigh_t<T: LinFloat>(a: &[T], batch: usize, n: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; batch * n * (n + 1)];
    let two = T::from_f64(2.0);
    for bi in 0..batch {
        let mut m: Vec<T> = a[bi * n * n..(bi + 1) * n * n].to_vec(); // diagonalized in place
        let mut v = vec![T::ZERO; n * n]; // eigenvectors (columns)
        for i in 0..n {
            v[i * n + i] = T::ONE;
        }
        let scale: T = m.iter().fold(T::ZERO, |acc, &x| acc + x * x);
        let thresh = T::from_f64(1e-30) + scale * T::from_f64(1e-26);
        for _ in 0..100 {
            let mut off = T::ZERO;
            for p in 0..n {
                for q in (p + 1)..n {
                    off = off + m[p * n + q] * m[p * n + q];
                }
            }
            if off <= thresh {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = m[p * n + q];
                    if apq.abs() <= T::ZERO {
                        continue;
                    }
                    let (app, aqq) = (m[p * n + p], m[q * n + q]);
                    let theta = (aqq - app) / (two * apq);
                    let sign = if theta < T::ZERO { -T::ONE } else { T::ONE };
                    let t = sign / (theta.abs() + (theta * theta + T::ONE).sqrt());
                    let c = T::ONE / (t * t + T::ONE).sqrt();
                    let s = t * c;
                    for i in 0..n {
                        if i == p || i == q {
                            continue;
                        }
                        let (aip, aiq) = (m[i * n + p], m[i * n + q]);
                        let (nip, niq) = (c * aip - s * aiq, s * aip + c * aiq);
                        m[i * n + p] = nip;
                        m[p * n + i] = nip;
                        m[i * n + q] = niq;
                        m[q * n + i] = niq;
                    }
                    m[p * n + p] = c * c * app - two * s * c * apq + s * s * aqq;
                    m[q * n + q] = s * s * app + two * s * c * apq + c * c * aqq;
                    m[p * n + q] = T::ZERO;
                    m[q * n + p] = T::ZERO;
                    for i in 0..n {
                        let (vip, viq) = (v[i * n + p], v[i * n + q]);
                        v[i * n + p] = c * vip - s * viq;
                        v[i * n + q] = s * vip + c * viq;
                    }
                }
            }
        }
        // sort ascending by eigenvalue, then pack columns (vectors) + last col (values)
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| {
            m[i * n + i].to_f64().partial_cmp(&m[j * n + j].to_f64()).unwrap_or(std::cmp::Ordering::Equal)
        });
        let base = bi * n * (n + 1);
        for i in 0..n {
            for j in 0..n {
                out[base + i * (n + 1) + j] = v[i * n + idx[j]];
            }
            out[base + i * (n + 1) + n] = m[idx[i] * n + idx[i]];
        }
    }
    out
}
