//! Dense linear-algebra solvers: solve / det / cholesky (LU / Cholesky in the operand
//! dtype), and inv (decomposes to solve). Contraction (dot_general/einsum) is in
//! `contract.rs`.

mod eigen;

use crate::{DType, Error, Graph, NodeId, Op};

impl Graph {
    /// Solve the batched linear system `A*X = B` (LU with partial pivoting, in the operand
    /// dtype). `a` is `[.., N, N]`, `b` is `[.., N, K]` -> `[.., N, K]`. f32/f64 only (no
    /// canonical low-precision LU, so f16/bf16/fp8 are rejected; frontend promotes if it
    /// wants). Differentiable.
    pub fn solve(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        let (ash, bsh) = (self.shape(a), self.shape(b));
        if ash.len() < 2 || ash[ash.len() - 1] != ash[ash.len() - 2] {
            return Err(Error::shape("solve", format!("A must be [.., N, N], got {ash:?}")));
        }
        let n = ash[ash.len() - 1];
        if bsh.len() != ash.len() || bsh[bsh.len() - 2] != n || bsh[..bsh.len() - 2] != ash[..ash.len() - 2] {
            return Err(Error::shape("solve", format!("B {bsh:?} incompatible with A {ash:?}")));
        }
        self.require("solve", a, matches!(self.dtype(a), DType::F32 | DType::F64), "f32 or f64")?;
        self.same_dtype("solve", a, b)?;
        Ok(self.push(Op::Solve, vec![a, b]))
    }

    /// Determinant of each batched matrix `[.., N, N]` -> `[..]` (LU, in the operand
    /// dtype; f32/f64 only, like `solve`).
    pub fn det(&mut self, a: NodeId) -> Result<NodeId, Error> {
        let ash = self.shape(a);
        if ash.len() < 2 || ash[ash.len() - 1] != ash[ash.len() - 2] {
            return Err(Error::shape("det", format!("A must be [.., N, N], got {ash:?}")));
        }
        self.require("det", a, matches!(self.dtype(a), DType::F32 | DType::F64), "f32 or f64")?;
        Ok(self.push(Op::Det, vec![a]))
    }

    /// Cholesky factor of each batched symmetric positive-definite `[.., N, N]`:
    /// lower-triangular `L` with `A = L*L^T` (computed in the operand dtype; f32/f64
    /// only, like `solve`). Differentiable.
    pub fn cholesky(&mut self, a: NodeId) -> Result<NodeId, Error> {
        let ash = self.shape(a);
        if ash.len() < 2 || ash[ash.len() - 1] != ash[ash.len() - 2] {
            return Err(Error::shape("cholesky", format!("A must be [.., N, N], got {ash:?}")));
        }
        self.require("cholesky", a, matches!(self.dtype(a), DType::F32 | DType::F64), "f32 or f64")?;
        Ok(self.push(Op::Cholesky, vec![a]))
    }

    // batched matmul over the trailing two dims: [.., m, k] @ [.., k, n] -> [.., m, n].
    fn bmm(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        let r = self.shape(a).len();
        let batch: Vec<usize> = (0..r - 2).collect();
        self.dot_general(a, b, vec![r - 1], vec![r - 2], batch.clone(), batch)
    }
    // transpose the trailing two dims.
    fn transpose_last(&mut self, a: NodeId) -> Result<NodeId, Error> {
        let r = self.shape(a).len();
        self.transpose(a, r - 2, r - 1)
    }
    // batched identity `[.., n, n]` matching `a`'s batch dims + dtype.
    fn eye_like(&mut self, a: NodeId, n: usize) -> Result<NodeId, Error> {
        let sh = self.shape(a);
        let r = sh.len();
        let mut data = vec![0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let eye = self.constant(data, vec![n, n]);
        let dt = self.dtype(a);
        let eye = if dt == DType::F32 { eye } else { self.cast(eye, dt) };
        let mut full = vec![1usize; r];
        full[r - 2] = n;
        full[r - 1] = n;
        let er = self.reshape(eye, full)?;
        self.broadcast_to(er, sh)
    }

    /// Sign and log-absolute-determinant `(sign, ln|det|)` of each batched `[.., N, N]`.
    /// Via `det`, so it inherits det's overflow for large/ill-scaled matrices (a fused
    /// LU-logdet primitive would avoid it).
    pub fn slogdet(&mut self, a: NodeId) -> Result<(NodeId, NodeId), Error> {
        let d = self.det(a)?;
        let s = self.sign(d);
        let ad = self.abs(d);
        Ok((s, self.ln(ad)))
    }

    /// Least-squares solution of `A*x ~= b` via the normal equations `A^T A x = A^T b`
    /// (full-rank `A = [.., m, n]`, `m >= n`; `b = [.., m, k]` -> `[.., n, k]`).
    pub fn lstsq(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        let at = self.transpose_last(a)?;
        let ata = self.bmm(at, a)?;
        let atb = self.bmm(at, b)?;
        self.solve(ata, atb)
    }

    /// Moore-Penrose pseudo-inverse of `A = [.., m, n]` (full rank): `(A^T A)^-1 A^T` for
    /// tall, `A^T(A A^T)^-1` for wide. Normal-equations (needs full column/row rank); an
    /// SVD-based pinv would handle rank deficiency.
    pub fn pinv(&mut self, a: NodeId) -> Result<NodeId, Error> {
        let sh = self.shape(a);
        let r = sh.len();
        let (m, n) = (sh[r - 2], sh[r - 1]);
        let at = self.transpose_last(a)?;
        if m >= n {
            let ata = self.bmm(at, a)?; // [.., n, n]
            self.solve(ata, at) // (A^T A)^-1 A^T
        } else {
            let aat = self.bmm(a, at)?; // [.., m, m]
            let iaat = self.inv(aat)?;
            self.bmm(at, iaat) // A^T(A A^T)^-1
        }
    }

    /// Matrix exponential `exp(A)` of each batched square `[.., N, N]` via a truncated
    /// Taylor series `sum A^k/k!`. No scaling-and-squaring (needs a runtime norm we don't
    /// compute statically), so accuracy holds for moderate ||A||.
    pub fn matrix_exp(&mut self, a: NodeId) -> Result<NodeId, Error> {
        let sh = self.shape(a);
        let r = sh.len();
        let n = sh[r - 1];
        if r < 2 || sh[r - 2] != n {
            return Err(Error::shape("matrix_exp", "expects [.., N, N]"));
        }
        let eye = self.eye_like(a, n)?;
        let mut acc = eye;
        let mut term = eye; // A^0/0! = I
        for k in 1..=18u32 {
            let at = self.bmm(term, a)?;
            let inv = self.scalar(at, 1.0 / k as f32);
            term = self.mul(at, inv)?; // term*A/k
            acc = self.add(acc, term)?;
        }
        Ok(acc)
    }

    /// Matrix inverse of each batched `[.., N, N]`: `solve(A, I)`.
    pub fn inv(&mut self, a: NodeId) -> Result<NodeId, Error> {
        let ash = self.shape(a);
        let r = ash.len();
        let n = ash[r - 1];
        let mut data = vec![0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let eye = self.constant(data, vec![n, n]);
        let mut full = vec![1usize; r];
        full[r - 2] = n;
        full[r - 1] = n;
        let eye_r = self.reshape(eye, full)?;
        let eye_b = self.expand(eye_r, ash)?;
        let dt = self.dtype(a);
        let eye_b = if dt == DType::F32 { eye_b } else { self.cast(eye_b, dt) };
        self.solve(a, eye_b)
    }
}
