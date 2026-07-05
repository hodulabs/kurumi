//! Eigendecomposition & orthogonal-factorization builders (eigh/qr/svd/eigvals),
//! decomposing onto the eigen primitives + the shared solver helpers in `linalg`.

use crate::{DType, Error, Graph, NodeId, Op};

impl Graph {
    /// Symmetric eigendecomposition of each batched `[.., N, N]` (assumed symmetric):
    /// returns `(eigenvalues [.., N], eigenvectors [.., N, N])`, ascending, columns are
    /// the eigenvectors. Cyclic Jacobi; f32/f64. Forward-only (no VJP yet).
    pub fn eigh(&mut self, a: NodeId) -> Result<(NodeId, NodeId), Error> {
        let sh = self.shape(a);
        let r = sh.len();
        if r < 2 || sh[r - 1] != sh[r - 2] {
            return Err(Error::shape("eigh", "expects [.., N, N]"));
        }
        self.require("eigh", a, matches!(self.dtype(a), DType::F32 | DType::F64), "f32 or f64")?;
        let n = sh[r - 1];
        let packed = self.push(Op::Eigh, vec![a]); // [.., N, N+1]
        let psh = self.shape(packed);
        let mut vr: Vec<(usize, usize)> = psh.iter().map(|&d| (0, d)).collect();
        vr[r - 1] = (0, n);
        let vecs = self.slice(packed, vr)?; // [.., N, N] eigenvectors
        let mut sr: Vec<(usize, usize)> = psh.iter().map(|&d| (0, d)).collect();
        sr[r - 1] = (n, n + 1);
        let col = self.slice(packed, sr)?;
        let vals = self.squeeze(col, r - 1)?; // [.., N] eigenvalues
        Ok((vals, vecs))
    }

    /// Eigenvalues of a general (nonsymmetric) real square `[.., N, N]` -> complex
    /// `[.., N]` (C64/C128). Unshifted QR algorithm; forward-only. For symmetric inputs
    /// prefer `eigh` (real, with eigenvectors, differentiable).
    pub fn eigvals(&mut self, a: NodeId) -> Result<NodeId, Error> {
        let sh = self.shape(a);
        let r = sh.len();
        if r < 2 || sh[r - 1] != sh[r - 2] {
            return Err(Error::shape("eigvals", "expects [.., N, N]"));
        }
        self.require("eigvals", a, matches!(self.dtype(a), DType::F32 | DType::F64), "f32 or f64")?;
        Ok(self.push(Op::Eigvals, vec![a]))
    }

    /// Reduced QR of each batched `[.., M, N]`: `(Q [.., M, K], R [.., K, N])` with
    /// `K = min(M,N)`, `A = Q*R`, `Q` orthonormal columns, `R` upper-triangular.
    /// Householder; f32/f64. Forward-only.
    pub fn qr(&mut self, a: NodeId) -> Result<(NodeId, NodeId), Error> {
        if self.shape(a).len() < 2 {
            return Err(Error::shape("qr", "expects [.., M, N]"));
        }
        self.require("qr", a, matches!(self.dtype(a), DType::F32 | DType::F64), "f32 or f64")?;
        let q = self.push(Op::Qr { r_factor: false }, vec![a]);
        let r = self.push(Op::Qr { r_factor: true }, vec![a]);
        Ok((q, r))
    }

    /// Singular value decomposition of `A = [.., M, N]` -> `(U, S, V)` with
    /// `A = U*diag(S)*V^T`, singular values descending. Full-rank; decomposed via the
    /// eigendecomposition of `A^T A` (tall) / `A A^T` (wide).
    pub fn svd(&mut self, a: NodeId) -> Result<(NodeId, NodeId, NodeId), Error> {
        let sh = self.shape(a);
        let r = sh.len();
        if r < 2 {
            return Err(Error::shape("svd", "expects [.., M, N]"));
        }
        let (m, n) = (sh[r - 2], sh[r - 1]);
        if m < n {
            let at = self.transpose_last(a)?;
            let (u, s, v) = self.svd(at)?; // svd(A^T) = (V, S, U)
            return Ok((v, s, u));
        }
        // tall/square: A^T A = V diag(sigma^2) V^T ; sigma = sqrt ; U = A V / sigma
        let at = self.transpose_last(a)?;
        let ata = self.bmm(at, a)?; // [.., N, N]
        let (evals, evecs) = self.eigh(ata)?; // ascending
        let ev = self.clamp_min(evals, 0.0)?;
        let sig = self.sqrt(ev); // [.., N] ascending
        let sr = self.shape(sig).len();
        let s = self.flip(sig, vec![sr - 1])?; // descending
        let vr = self.shape(evecs).len();
        let v = self.flip(evecs, vec![vr - 1])?; // columns descending
        let av = self.bmm(a, v)?; // [.., M, N]
        let ssh = self.shape(s);
        let mut s_re = ssh.clone();
        s_re.insert(ssh.len() - 1, 1); // [.., 1, N]
        let s_row = self.reshape(s, s_re)?;
        let avsh = self.shape(av);
        let s_b = self.broadcast_to(s_row, avsh)?;
        let u = self.div(av, s_b)?;
        Ok((u, s, v))
    }
}
