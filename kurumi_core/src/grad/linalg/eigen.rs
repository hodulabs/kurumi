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

// Square/tall QR backward (`A1 = Q*R1`, square R1): A1_bar = (Q_bar + Q*copyltu(M)) R1^-T
// with M = R1*R1_bar^T - Q_bar^T*Q. Either cotangent may be absent (treated as 0). Shared
// by the tall/square path (one cotangent) and the wide path's A1 sub-problem (both present).
fn square_qr_bwd(
    g: &mut Graph,
    q: NodeId,
    r1: NodeId,
    q_bar: Option<NodeId>,
    r1_bar: Option<NodeId>,
) -> Result<NodeId, Error> {
    let mut m_mat: Option<NodeId> = None;
    if let Some(rb) = r1_bar {
        let rbart = transpose_last2(g, rb)?;
        m_mat = Some(matmul_l2(g, r1, rbart)?); // R1*R1_bar^T
    }
    if let Some(qb) = q_bar {
        let qbart = transpose_last2(g, qb)?;
        let qtq = matmul_l2(g, qbart, q)?; // Q_bar^T*Q
        let neg = g.neg(qtq);
        m_mat = Some(match m_mat {
            Some(a) => g.add(a, neg)?,
            None => neg,
        });
    }
    let m_mat = m_mat.ok_or_else(|| Error::shape("qr backward", "no cotangent"))?;
    let cl = copyltu(g, m_mat)?;
    let qcl = matmul_l2(g, q, cl)?; // Q copyltu(M)
    let x = match q_bar {
        Some(qb) => g.add(qb, qcl)?,
        None => qcl,
    };
    let rinv = g.inv(r1)?;
    let rinvt = transpose_last2(g, rinv)?;
    matmul_l2(g, x, rinvt) // X R1^-T
}

/// QR VJP. Tall/square (`M >= N`, reduced `A = Q*R`): one cotangent flows to the shared
/// `square_qr_bwd`. Wide (`M < N`): split A = [A1|A2], R = [R1|R2] at column M (R2 = Q^T A2).
/// R-path R_bar = [R1_bar|R2_bar]: A2_bar = Q*R2_bar, effective Q_bar for A1 = A2*R2_bar^T,
/// A1_bar = square_qr_bwd(Q, R1, A2*R2_bar^T, R1_bar). Q-path (R_bar=0): A2_bar = 0,
/// A1_bar = square_qr_bwd(Q, R1, Q_bar, 0). A_bar = concat([A1_bar, A2_bar]).
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
    let (m, n) = (ash[r - 2], ash[r - 1]);
    let (q, rr) = g.qr(a)?; // Q [.., M, K], R [.., K, N], K = min(M,N) (forward is cached)
    if m >= n {
        // tall/square: reduced A = Q*R, only one output has a nonzero cotangent.
        let (qbar, rbar) = if r_factor { (None, Some(ct)) } else { (Some(ct), None) };
        let ga = square_qr_bwd(g, q, rr, qbar, rbar)?;
        return acc(g, cot, a, ga);
    }
    // wide (M < N): slice trailing axis [lo, hi).
    let sl = |g: &mut Graph, x: NodeId, lo: usize, hi: usize| -> Result<NodeId, Error> {
        let mut rg: Vec<(usize, usize)> = g.shape(x).iter().map(|&d| (0, d)).collect();
        let rk = rg.len();
        rg[rk - 1] = (lo, hi);
        g.slice(x, rg)
    };
    let a2 = sl(g, a, m, n)?; // [.., M, N-M]
    let r1 = sl(g, rr, 0, m)?; // [.., M, M] upper-tri
    let (a1_bar, a2_bar) = if r_factor {
        let r1_bar = sl(g, ct, 0, m)?;
        let r2_bar = sl(g, ct, m, n)?; // [.., M, N-M]
        let a2_bar = matmul_l2(g, q, r2_bar)?; // A2_bar = Q*R2_bar
        let r2bart = transpose_last2(g, r2_bar)?;
        let qbar = matmul_l2(g, a2, r2bart)?; // effective Q_bar = A2*R2_bar^T
        let a1_bar = square_qr_bwd(g, q, r1, Some(qbar), Some(r1_bar))?;
        (a1_bar, a2_bar)
    } else {
        // R_bar = 0 => R2_bar = 0 => A2_bar = 0; A1 sub-problem sees only Q_bar = ct.
        let a1_bar = square_qr_bwd(g, q, r1, Some(ct), None)?;
        (a1_bar, g.zeros_like(a2))
    };
    let ga = g.concat(&[a1_bar, a2_bar], r - 1)?;
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
