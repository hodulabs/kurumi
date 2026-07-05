//! Triangular & diagonal matrix ops: diagonal/trace, tril/triu (host-built mask).

use crate::{DType, Error, Graph, NodeId};

impl Graph {
    /// Main diagonal of a 2-D matrix `[M, N]` -> `[min(M,N)]` (stride `N+1`).
    pub fn diagonal(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        let (m, n) = (sh[0], sh[1]);
        let k = m.min(n);
        let flat = self.reshape(x, vec![m * n])?;
        self.slice_step(flat, vec![(0, (k - 1) * (n + 1) + 1, n + 1)])
    }

    /// Trace (sum of the main diagonal) of a 2-D matrix.
    pub fn trace(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let d = self.diagonal(x)?;
        self.sum(d, 0)
    }

    /// Build a diagonal matrix from a vector: `[.., N] -> [.., N, N]` with the
    /// values on the main diagonal (inverse of `diagonal`): `v.unsqueeze(-1) * I`.
    pub fn diag_embed(&mut self, v: NodeId) -> Result<NodeId, Error> {
        let sh = self.shape(v);
        let n = *sh.last().expect("diag_embed needs a trailing axis");
        let vu = self.unsqueeze(v, sh.len())?; // [.., N, 1]
        let mut full = sh.clone();
        full.push(n); // [.., N, N]
        let vb = self.broadcast_to(vu, full.clone())?;
        // identity over the trailing [N, N], broadcast to `full`
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let eye = self.constant(data, vec![n, n]);
        let dt = self.dtype(v);
        let eye = if dt == DType::F32 { eye } else { self.cast(eye, dt) };
        let mut er = vec![1usize; full.len()];
        let l = full.len();
        er[l - 2] = n;
        er[l - 1] = n;
        let eye_r = self.reshape(eye, er)?;
        let eye_b = self.expand(eye_r, full)?;
        self.mul(vb, eye_b)
    }

    /// Lower-triangular part (zeros above the `diagonal`-th diagonal).
    pub fn tril(&mut self, x: NodeId, diagonal: i64) -> Result<NodeId, Error> {
        self.triangular(x, diagonal, true)
    }
    /// Upper-triangular part (zeros below the `diagonal`-th diagonal).
    pub fn triu(&mut self, x: NodeId, diagonal: i64) -> Result<NodeId, Error> {
        self.triangular(x, diagonal, false)
    }
    // multiply by a host-built 0/1 triangular mask over the trailing [M, N].
    fn triangular(&mut self, x: NodeId, diagonal: i64, lower: bool) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        let r = sh.len();
        let (m, n) = (sh[r - 2], sh[r - 1]);
        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let (i, j) = (i as i64, j as i64);
                let keep = if lower { j <= i + diagonal } else { j >= i + diagonal };
                if keep {
                    data[(i * n as i64 + j) as usize] = 1.0;
                }
            }
        }
        let mask = self.constant(data, vec![m, n]);
        let mut full = vec![1usize; r];
        full[r - 2] = m;
        full[r - 1] = n;
        let mr = self.reshape(mask, full)?;
        let mb = self.expand(mr, sh)?;
        let dt = self.dtype(x);
        let mb = if dt == DType::F32 { mb } else { self.cast(mb, dt) };
        self.mul(x, mb)
    }
}
