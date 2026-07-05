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
        _ => unreachable!("nn::vjp: {:?}", n.op),
    }
    Ok(())
}
