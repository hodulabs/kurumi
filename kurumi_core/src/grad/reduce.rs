//! VJPs for reductions: sum broadcasts the cotangent back; max routes it to the
//! argmax positions (ties share); prod uses the product/element ratio.

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
        Op::Sum { axis } => {
            let full = g.shape(s[0]);
            let gx = g.broadcast_back(ct, &full, *axis)?;
            acc(g, cot, s[0], gx)?;
        }
        Op::ReduceMax { axis } => {
            // grad flows to argmax positions (ties share)
            let full = g.shape(s[0]);
            let yb = g.broadcast_back(id, &full, *axis)?;
            let mask = g.cmp_eq(s[0], yb)?;
            let ctb = g.broadcast_back(ct, &full, *axis)?;
            let zero = g.zeros_like(ctb);
            let gx = g.select(mask, ctb, zero)?;
            acc(g, cot, s[0], gx)?;
        }
        Op::Prod { axis } => {
            // dy/da_i = (prod / a_i) via recip: NaN/inf if any a_i == 0 (a stable fix
            // needs log-domain accumulation of the product; not done).
            let full = g.shape(s[0]);
            let yb = g.broadcast_back(id, &full, *axis)?;
            let ctb = g.broadcast_back(ct, &full, *axis)?;
            let inv = g.recip(s[0]);
            let t = g.mul(yb, inv)?;
            let gx = g.mul(ctb, t)?;
            acc(g, cot, s[0], gx)?;
        }
        _ => unreachable!("reduce::vjp: {:?}", n.op),
    }
    Ok(())
}
