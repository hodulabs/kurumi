//! General (nonsymmetric) eigenvalues (interp kernel): unshifted QR algorithm, reusing the
//! full QR helper from the sibling `qr`.

use crate::interp::linalg::LinFloat;
use crate::interp::linalg::eigen::qr::qr_full_t;
use crate::{DType, Elem, Storage};
use num_complex::Complex;

/// Eigenvalues of a general (nonsymmetric) real square matrix `[.., N, N]` via the unshifted QR
/// algorithm -> complex `[.., N]` (C64/C128). No Hessenberg/shifts: converges for distinct-magnitude
/// spectra (incl. clean complex pairs); defective / equal-magnitude cases want shifted Francis QR.
pub(crate) fn eigvals(a: &Storage, batch: usize, n: usize) -> Storage {
    match a.dtype() {
        DType::F64 => {
            let (re, im) = eigvals_t(<f64 as Elem>::slice(a), batch, n);
            Storage::C128(re.iter().zip(&im).map(|(&r, &i)| Complex::new(r, i)).collect())
        }
        DType::F32 => {
            let (re, im) = eigvals_t(<f32 as Elem>::slice(a), batch, n);
            Storage::C64(re.iter().zip(&im).map(|(&r, &i)| Complex::new(r, i)).collect())
        }
        dt => unreachable!("linalg builder gates f32/f64, got {dt:?}"),
    }
}

// unshifted QR iteration (H <- RQ) to (quasi-)upper-triangular, then read eigenvalues
// off 1x1 (real) and 2x2 (complex-pair) diagonal blocks.
fn eigvals_t<T: LinFloat>(a: &[T], batch: usize, n: usize) -> (Vec<T>, Vec<T>) {
    let mut re = vec![T::ZERO; batch * n];
    let mut im = vec![T::ZERO; batch * n];
    let two = T::from_f64(2.0);
    let four = T::from_f64(4.0);
    for bi in 0..batch {
        let mut h = a[bi * n * n..(bi + 1) * n * n].to_vec();
        let scale = h.iter().fold(T::ZERO, |s, &x| s + x * x).sqrt();
        let eps = T::from_f64(1e-10) * (scale + T::ONE);
        for _ in 0..(60 * n.max(1)) {
            let (q, r) = qr_full_t(&h, n);
            let mut nh = vec![T::ZERO; n * n];
            for i in 0..n {
                for k in 0..n {
                    let mut s = T::ZERO;
                    for l in 0..n {
                        s = s + r[i * n + l] * q[l * n + k];
                    }
                    nh[i * n + k] = s;
                }
            }
            h = nh;
        }
        let (rb, ib) = (&mut re[bi * n..(bi + 1) * n], &mut im[bi * n..(bi + 1) * n]);
        let mut i = 0;
        while i < n {
            if i + 1 >= n || h[(i + 1) * n + i].abs() <= eps {
                rb[i] = h[i * n + i]; // 1x1 real block
                i += 1;
            } else {
                let (aa, bb) = (h[i * n + i], h[i * n + i + 1]);
                let (cc, dd) = (h[(i + 1) * n + i], h[(i + 1) * n + i + 1]);
                let tr = aa + dd;
                let det = aa * dd - bb * cc;
                let disc = tr * tr - four * det;
                if disc >= T::ZERO {
                    let sq = disc.sqrt();
                    rb[i] = (tr + sq) / two;
                    rb[i + 1] = (tr - sq) / two;
                } else {
                    let sq = (T::ZERO - disc).sqrt();
                    rb[i] = tr / two;
                    ib[i] = sq / two;
                    rb[i + 1] = tr / two;
                    ib[i + 1] = T::ZERO - sq / two;
                }
                i += 2;
            }
        }
    }
    (re, im)
}
