//! Dense linear algebra: LU with partial pivoting + Cholesky, computed in the operand's own
//! float dtype (f32->f32, f64->f64): no silent promotion, matching the strict-dtype contract and
//! NumPy/PyTorch/JAX. Builders gate f32/f64 only -- there's no canonical low-precision LU, so
//! f16/bf16/fp8 are rejected at record time (picking a compute precision = promotion = frontend's
//! job). Batched over leading dims; trailing `[N, N]` is the matrix per batch.

mod eigen;
mod factor;
mod solve;

pub(crate) use eigen::{eigh, eigvals, qr};
pub(crate) use factor::{cholesky, det};
pub(crate) use solve::solve;

use crate::{Op, TensorVal};

pub(super) fn eval(op: &Op, inputs: &[&TensorVal]) -> TensorVal {
    match op {
        Op::Solve => {
            let (a, b) = (inputs[0], inputs[1]);
            let (ar, br) = (a.shape.len(), b.shape.len());
            let n = a.shape[ar - 1];
            let batch: usize = a.shape[..ar - 2].iter().product();
            let k = b.shape[br - 1];
            TensorVal { shape: b.shape.clone(), storage: solve(&a.storage, &b.storage, batch, n, k) }
        }
        Op::Det => {
            let a = inputs[0];
            let ar = a.shape.len();
            let n = a.shape[ar - 1];
            let batch: usize = a.shape[..ar - 2].iter().product();
            TensorVal { shape: a.shape[..ar - 2].to_vec(), storage: det(&a.storage, batch, n) }
        }
        Op::Cholesky => {
            let a = inputs[0];
            let ar = a.shape.len();
            let n = a.shape[ar - 1];
            let batch: usize = a.shape[..ar - 2].iter().product();
            TensorVal { shape: a.shape.clone(), storage: cholesky(&a.storage, batch, n) }
        }
        Op::Eigh => {
            let a = inputs[0];
            let ar = a.shape.len();
            let n = a.shape[ar - 1];
            let batch: usize = a.shape[..ar - 2].iter().product();
            let mut shape = a.shape.clone();
            *shape.last_mut().unwrap() += 1; // [.., N, N+1]
            TensorVal { shape, storage: eigh(&a.storage, batch, n) }
        }
        Op::Qr { r_factor } => {
            let a = inputs[0];
            let ar = a.shape.len();
            let (m, n) = (a.shape[ar - 2], a.shape[ar - 1]);
            let batch: usize = a.shape[..ar - 2].iter().product();
            let k = m.min(n);
            let mut shape = a.shape.clone();
            if *r_factor {
                shape[ar - 2] = k
            } else {
                shape[ar - 1] = k
            }
            TensorVal { shape, storage: qr(&a.storage, batch, m, n, *r_factor) }
        }
        Op::Eigvals => {
            let a = inputs[0];
            let ar = a.shape.len();
            let n = a.shape[ar - 1];
            let batch: usize = a.shape[..ar - 2].iter().product();
            TensorVal { shape: a.shape[..ar - 1].to_vec(), storage: eigvals(&a.storage, batch, n) }
        }
        _ => unreachable!("linalg::eval: non-linalg op"),
    }
}

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
