//! VJPs for fused nn primitives (softmax, ...). The forward node id is the output y.

use crate::grad::acc;
use crate::{Error, Graph, Node, NodeId, Op};
use std::collections::HashMap;

pub(super) fn vjp(
    g: &mut Graph,
    id: NodeId,
    n: &Node,
    ct: NodeId,
    cot: &mut HashMap<NodeId, NodeId>,
) -> Result<(), Error> {
    let s = &n.src;
    match &n.op {
        // softmax: dx = y * (ct - sum(y*ct, axis)), y = the forward output (id).
        Op::Softmax { axis } => {
            let full = g.shape(s[0]);
            let yc = g.mul(id, ct)?;
            let dot = g.sum(yc, *axis)?;
            let dotb = g.broadcast_back(dot, &full, *axis)?;
            let diff = g.sub(ct, dotb)?;
            let gx = g.mul(id, diff)?;
            acc(g, cot, s[0], gx)?;
        }
        // rmsnorm: dx = (ct - y * S/N) / r, y = id, r = rms, S = sum(y*ct, axis), N = axis_len.
        Op::RmsNorm { axis, eps } => {
            let x = s[0];
            let full = g.shape(x);
            let inv_n = 1.0 / full[*axis] as f32;
            // recompute rinv = 1 / sqrt(mean(x^2) + eps) (not stored on the node)
            let sq = g.mul(x, x)?;
            let ssq = g.sum(sq, *axis)?;
            let sc = g.scalar(ssq, inv_n);
            let ms = g.mul(ssq, sc)?;
            let eps_c = g.scalar(ms, *eps);
            let var = g.add(ms, eps_c)?;
            let rms = g.sqrt(var);
            let rinv = g.recip(rms);
            let rinv_b = g.broadcast_back(rinv, &full, *axis)?;
            // num = ct - y * S/N
            let ys = g.mul(id, ct)?;
            let ssum = g.sum(ys, *axis)?;
            let ssc = g.scalar(ssum, inv_n);
            let ssum_n = g.mul(ssum, ssc)?;
            let ssum_nb = g.broadcast_back(ssum_n, &full, *axis)?;
            let y_s = g.mul(id, ssum_nb)?;
            let num = g.sub(ct, y_s)?;
            let dx = g.mul(num, rinv_b)?;
            acc(g, cot, s[0], dx)?;
        }
        // SDPA: out = softmax(scale*q@k^T [+causal]) @ v. Recompute P (the attention weights),
        // then the standard attention backward, built from the SAME ops the decomposition's
        // autograd uses -> matches it bit-for-bit. Causal is handled implicitly: P is exactly 0
        // in the masked upper triangle, so dScores (= P * ...) vanishes there (no explicit mask).
        Op::Sdpa { causal } => {
            let (q, k, v) = (s[0], s[1], s[2]);
            let qs = g.shape(q);
            let r = qs.len();
            let scale = 1.0 / (qs[r - 1] as f32).sqrt(); // 1/sqrt(dh)
            let batch: Vec<usize> = (0..r - 2).collect();
            // recompute P = softmax(scale*q@k^T [+causal], keys), exactly as the forward
            let raw = g.dot_general(q, k, vec![r - 1], vec![r - 1], batch.clone(), batch.clone())?;
            let inv = g.scalar(raw, scale);
            let mut scores = g.mul(raw, inv)?;
            if *causal {
                let bias = causal_bias(g, qs[r - 2], &g.shape(scores))?;
                scores = g.add(scores, bias)?;
            }
            let full = g.shape(scores);
            let p = g.softmax(scores, r - 1)?;
            // dV = P^T @ ct, contracting the query axis (r-2) -> [.., S, dh]
            let dv = g.dot_general(p, ct, vec![r - 2], vec![r - 2], batch.clone(), batch.clone())?;
            acc(g, cot, v, dv)?;
            // dP = ct @ v^T, contracting dh (r-1) -> [.., S, S]
            let dp = g.dot_general(ct, v, vec![r - 1], vec![r - 1], batch.clone(), batch.clone())?;
            // dScores = P * (dP - rowsum(dP*P over keys))  (softmax jacobian along the key axis)
            let pdp = g.mul(dp, p)?;
            let rs = g.sum(pdp, r - 1)?;
            let rs_b = g.broadcast_back(rs, &full, r - 1)?;
            let diff = g.sub(dp, rs_b)?;
            let dscores = g.mul(p, diff)?;
            // d(raw) = scale * dScores (the scale is a constant multiply), then the q@k^T backward
            let sc = g.scalar(dscores, scale);
            let draw = g.mul(dscores, sc)?;
            // dQ = d(raw) @ k, contracting the key axis (r-1) of d(raw) with k's S (r-2) -> [.., S, dh]
            let dq = g.dot_general(draw, k, vec![r - 1], vec![r - 2], batch.clone(), batch.clone())?;
            acc(g, cot, q, dq)?;
            // dK = d(raw)^T @ q, contracting the query axis (r-2) of both -> [.., S, dh]
            let dk = g.dot_general(draw, q, vec![r - 2], vec![r - 2], batch.clone(), batch)?;
            acc(g, cot, k, dk)?;
        }
        _ => unreachable!("nn::vjp: {:?}", n.op),
    }
    Ok(())
}

// additive causal bias broadcast to `out_shape` (trailing [S,S]): 0 on/below the diagonal,
// -inf above. Mirrors the decomposition's mask so the recomputed P is exactly 0 in masked
// positions (and dScores vanishes there). A constant: no gradient flows through it.
fn causal_bias(g: &mut Graph, s: usize, out_shape: &[usize]) -> Result<NodeId, Error> {
    let mut data = vec![0.0f32; s * s];
    for i in 0..s {
        for j in (i + 1)..s {
            data[i * s + j] = f32::NEG_INFINITY;
        }
    }
    let bias = g.constant(data, vec![s, s]);
    let rank = out_shape.len();
    let mut full = vec![1usize; rank];
    full[rank - 2] = s;
    full[rank - 1] = s;
    let biased = g.reshape(bias, full)?;
    g.expand(biased, out_shape.to_vec())
}
