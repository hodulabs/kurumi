//! VJP rules for the dense-linalg primitives (Solve, Det, Cholesky), plus the batched
//! matrix helpers they share. Eigen VJPs are in `linalg/eigen`.

mod eigen;

pub(super) use eigen::{eigh_vjp, qr_vjp};

use crate::grad::acc;
use crate::{Error, Graph, NodeId};
use std::collections::HashMap;

// batched matmul over the trailing two axes: [.., m, k] @ [.., k, n] -> [.., m, n].
fn matmul_l2(g: &mut Graph, x: NodeId, y: NodeId) -> Result<NodeId, Error> {
    let r = g.shape(x).len();
    let batch: Vec<usize> = (0..r - 2).collect();
    g.dot_general(x, y, vec![r - 1], vec![r - 2], batch.clone(), batch)
}

// swap the trailing two axes (matrix transpose of a batched matrix).
fn transpose_last2(g: &mut Graph, x: NodeId) -> Result<NodeId, Error> {
    let r = g.shape(x).len();
    let mut perm: Vec<usize> = (0..r).collect();
    perm.swap(r - 2, r - 1);
    g.permute(x, perm)
}

// batched identity matrix shaped like `ash` (trailing [N, N], leading dims = 1 then
// broadcast). Used by the determinant VJP.
fn identity_like(g: &mut Graph, ash: &[usize]) -> Result<NodeId, Error> {
    let r = ash.len();
    let n = ash[r - 1];
    let mut data = vec![0f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    let eye = g.constant(data, vec![n, n]);
    let mut full = vec![1usize; r];
    full[r - 2] = n;
    full[r - 1] = n;
    let eye_r = g.reshape(eye, full)?;
    g.expand(eye_r, ash.to_vec())
}

/// X = A^-1 B. dL/dB = solve(A^T, ct); dL/dA = -(dL/dB) X^T (contract over K).
pub(super) fn solve_vjp(
    g: &mut Graph,
    id: NodeId,
    s: &[NodeId],
    ct: NodeId,
    cot: &mut HashMap<NodeId, NodeId>,
) -> Result<(), Error> {
    let at = transpose_last2(g, s[0])?;
    let gb = g.solve(at, ct)?;
    acc(g, cot, s[1], gb)?;
    // ga = -(gb @ X^T): contract the K axis of gb and X (= this node `id`)
    let rk = g.shape(gb).len();
    let batch: Vec<usize> = (0..rk - 2).collect();
    let ga0 = g.dot_general(gb, id, vec![rk - 1], vec![rk - 1], batch.clone(), batch)?;
    let ga = g.neg(ga0);
    acc(g, cot, s[0], ga)
}

/// dL/dA = ct*det*(A^-1)^T. (A^-1)^T = solve(A^T, I); ct*det broadcast to [..,N,N].
pub(super) fn det_vjp(
    g: &mut Graph,
    id: NodeId,
    s: &[NodeId],
    ct: NodeId,
    cot: &mut HashMap<NodeId, NodeId>,
) -> Result<(), Error> {
    let ash = g.shape(s[0]);
    let r = ash.len();
    let n_ = ash[r - 1];
    let at = transpose_last2(g, s[0])?;
    let eye = identity_like(g, &ash)?;
    let inv_t = g.solve(at, eye)?; // (A^-1)^T
    let coef = g.mul(ct, id)?; // ct * det, shape [..]
    // broadcast coef [..] to [.., N, N]
    let mut full = ash.clone();
    full[r - 1] = n_;
    full[r - 2] = n_;
    let mut keep = g.shape(coef);
    keep.push(1);
    keep.push(1);
    let coef_r = g.reshape(coef, keep)?;
    let coef_b = g.expand(coef_r, full)?;
    let ga = g.mul(coef_b, inv_t)?;
    acc(g, cot, s[0], ga)
}

/// L = chol(A). Reverse-mode (Murray 2016): with L_bar = ct,
///   A_bar = L^-T * Phi(L^T L_bar) * L^-1,  Phi = lower-tri with the diagonal halved.
/// `id` is L (this node's output). The result is folded onto the lower triangle A reads.
pub(super) fn cholesky_vjp(
    g: &mut Graph,
    id: NodeId,
    s: &[NodeId],
    ct: NodeId,
    cot: &mut HashMap<NodeId, NodeId>,
) -> Result<(), Error> {
    let ash = g.shape(s[0]);
    let r = ash.len();
    let n = ash[r - 1];
    let lt = transpose_last2(g, id)?;
    let m = matmul_l2(g, lt, ct)?; // L^T L_bar
    // Phi mask: 1 below the diagonal, 1/2 on it, 0 above.
    let mut data = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = if i > j {
                1.0
            } else if i == j {
                0.5
            } else {
                0.0
            };
        }
    }
    let mask = g.constant(data, vec![n, n]);
    let mut full = vec![1usize; r];
    full[r - 2] = n;
    full[r - 1] = n;
    let mask_r = g.reshape(mask, full)?;
    let mask_b = g.expand(mask_r, ash.clone())?;
    let p = g.mul(m, mask_b)?; // Phi(L^T L_bar)
    let linv = g.inv(id)?; // L^-1
    let linvt = transpose_last2(g, linv)?; // L^-T
    let lp = matmul_l2(g, linvt, p)?;
    let sfull = matmul_l2(g, lp, linv)?; // S = L^-T Phi L^-1
    // A reads only its lower triangle, so fold S's symmetric contribution into it:
    // ga[i,j] = S[i,j]+S[j,i] (i>j), S[i,i] (diag), 0 (i<j).
    let st = transpose_last2(g, sfull)?;
    let lower = g.tril(sfull, 0)?;
    let strict = g.tril(st, -1)?;
    let ga = g.add(lower, strict)?;
    acc(g, cot, s[0], ga)
}
