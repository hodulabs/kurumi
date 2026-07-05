//! Eigendecomposition & QR VJPs (differentiable eigh/qr). Both use the batched
//! matrix helpers in the parent `linalg`.

use crate::grad::acc;
use crate::grad::linalg::{identity_like, matmul_l2, transpose_last2};
use crate::{Error, Graph, NodeId};
use std::collections::HashMap;

// copyltu(M) = tril(M,0) + tril(M,-1)^T : the symmetric matrix built from M's lower
// triangle (used by the QR backward).
fn copyltu(g: &mut Graph, m: NodeId) -> Result<NodeId, Error> {
    let lower = g.tril(m, 0)?;
    let strict = g.tril(m, -1)?;
    let strict_t = transpose_last2(g, strict)?;
    g.add(lower, strict_t)
}

/// QR VJP for the tall/square case (`A = [.., M, N]`, `M >= N`, reduced `A = Q*R`).
/// The backward is linear in `(Q_bar, R_bar)`, so the Q-node and R-node primitives each
/// contribute their half (`acc` sums them): `A_bar = (Q_bar + Q*copyltu(M)) R^-T`, with
/// `M = R_bar R^T` (R-node) or `M = -Q^T Q_bar` (Q-node). Wide `M<N` is forward-only (skipped).
pub(crate) fn qr_vjp(
    g: &mut Graph,
    s: &[NodeId],
    ct: NodeId,
    r_factor: bool,
    cot: &mut HashMap<NodeId, NodeId>,
) -> Result<(), Error> {
    let a = s[0];
    let ash = g.shape(a);
    let r = ash.len();
    if ash[r - 2] < ash[r - 1] {
        // wide QR (M < N) backward is unimplemented; error loudly rather than return a
        // silently-zero gradient.
        return Err(Error::shape("qr backward", "wide QR (M < N) is not differentiable"));
    }
    let (q, rr) = g.qr(a)?; // Q [.., M, N], R [.., N, N] (recomputed; forward is cached)
    // M = R*R_bar^T - Q_bar^T*Q ; split per output (the other cotangent is 0).
    let (m_mat, qbar) = if r_factor {
        let rbart = transpose_last2(g, ct)?; // R_bar^T
        (matmul_l2(g, rr, rbart)?, None) // M = R*R_bar^T, Q_bar = 0
    } else {
        let qbart = transpose_last2(g, ct)?; // Q_bar^T
        let qtq = matmul_l2(g, qbart, q)?; // Q_bar^T*Q
        (g.neg(qtq), Some(ct)) // M = -Q_bar^T*Q
    };
    let cl = copyltu(g, m_mat)?;
    let qcl = matmul_l2(g, q, cl)?; // Q copyltu(M)
    let x = match qbar {
        Some(qb) => g.add(qb, qcl)?,
        None => qcl,
    };
    let rinv = g.inv(rr)?;
    let rinvt = transpose_last2(g, rinv)?;
    let ga = matmul_l2(g, x, rinvt)?; // X R^-T
    acc(g, cot, a, ga)
}

/// Symmetric-eigendecomposition VJP. `id`/`ct` are the packed `[.., N, N+1]` (columns
/// `0..N` = eigenvectors V, column N = eigenvalues w). With V_bar/w_bar read from `ct`:
///   A_bar = sym( V (diag(w_bar) + F .* (V^T V_bar)) V^T ),  F_ij = 1/(w_j - w_i) (i!=j), 0 diag.
/// Degenerate eigenvalues make `F` blow up (the known eigh-backward limitation).
pub(crate) fn eigh_vjp(
    g: &mut Graph,
    id: NodeId,
    s: &[NodeId],
    ct: NodeId,
    cot: &mut HashMap<NodeId, NodeId>,
) -> Result<(), Error> {
    let psh = g.shape(id);
    let r = psh.len();
    let n = psh[r - 1] - 1;
    // slice the packed tensor's last axis to [a, b)
    let sl = |g: &mut Graph, x: NodeId, a: usize, b: usize| -> Result<NodeId, Error> {
        let mut rg: Vec<(usize, usize)> = g.shape(x).iter().map(|&d| (0, d)).collect();
        rg[r - 1] = (a, b);
        g.slice(x, rg)
    };
    let v = sl(g, id, 0, n)?; // [.., N, N]
    let w = {
        let c = sl(g, id, n, n + 1)?;
        g.squeeze(c, r - 1)?
    }; // [.., N]
    let gv = sl(g, ct, 0, n)?;
    let gw = {
        let c = sl(g, ct, n, n + 1)?;
        g.squeeze(c, r - 1)?
    };
    let vt = transpose_last2(g, v)?;
    let m = matmul_l2(g, vt, gv)?; // V^T V_bar
    // F_ij = (1 - delta_ij) / (w_j - w_i)
    let nn: Vec<usize> = {
        let mut s = psh.clone();
        s[r - 1] = n;
        s
    }; // [.., N, N]
    let wsh = g.shape(w);
    let wrow = {
        let mut rs = wsh.clone();
        rs.push(1);
        let rw = g.reshape(w, rs)?;
        g.broadcast_to(rw, nn.clone())?
    };
    let wcol = {
        let mut cs = wsh.clone();
        cs.insert(r - 2, 1);
        let cw = g.reshape(w, cs)?;
        g.broadcast_to(cw, nn.clone())?
    };
    let diff = g.sub(wcol, wrow)?; // w_j - w_i
    let eye = identity_like(g, &nn)?;
    let dsafe = g.add(diff, eye)?; // + I so the diagonal divide is 1, not 0
    let rdiff = g.recip(dsafe);
    let ones = g.ones_like(eye);
    let offdiag = g.sub(ones, eye)?;
    let f = g.mul(offdiag, rdiff)?;
    let fm = g.mul(f, m)?;
    let dg = g.diag_embed(gw)?; // diag(w_bar)
    let inner = g.add(dg, fm)?;
    let vi = matmul_l2(g, v, inner)?;
    let ga0 = matmul_l2(g, vi, vt)?; // V inner V^T
    let ga0t = transpose_last2(g, ga0)?;
    let sum = g.add(ga0, ga0t)?;
    let half = g.scalar(sum, 0.5);
    let ga = g.mul(sum, half)?; // symmetrize (A is symmetric)
    acc(g, cot, s[0], ga)
}
