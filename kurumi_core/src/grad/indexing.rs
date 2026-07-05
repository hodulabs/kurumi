//! VJPs for gather/scatter: gather backward scatter-adds the cotangent at the
//! gathered indices; scatter backward gathers it for the updates and passes/masks it
//! for the operand (Add passes through, Set/Max/Min zero the written positions).

use crate::grad::acc;
use crate::{Error, Graph, Node, NodeId, Op, ScatterOp};
use std::collections::HashMap;

pub(super) fn vjp(g: &mut Graph, n: &Node, ct: NodeId, cot: &mut HashMap<NodeId, NodeId>) -> Result<(), Error> {
    let s = &n.src;
    match &n.op {
        Op::Gather { axis } => {
            // operand grad: scatter-add ct at the gathered indices; indices: none
            let zeros = g.zeros_like(s[0]);
            let gx = g.scatter(zeros, s[1], ct, *axis, ScatterOp::Add)?;
            acc(g, cot, s[0], gx)?;
        }
        Op::Scatter { axis, combine } => {
            // updates grad = gather(ct) at the indices
            let gu = g.gather(ct, s[1], *axis)?;
            acc(g, cot, s[2], gu)?;
            // operand grad: Add passes ct through; Set zeros the written positions
            match combine {
                ScatterOp::Add => acc(g, cot, s[0], ct)?,
                ScatterOp::Set | ScatterOp::Max | ScatterOp::Min => {
                    let zu = g.zeros_like(s[2]);
                    let masked = g.scatter(ct, s[1], zu, *axis, ScatterOp::Set)?;
                    acc(g, cot, s[0], masked)?;
                }
            }
        }
        Op::GatherAlong { axis } => {
            // operand grad: scatter-add ct back at the per-position indices
            let zeros = g.zeros_like(s[0]);
            let gx = g.scatter_along(zeros, s[1], ct, *axis, ScatterOp::Add)?;
            acc(g, cot, s[0], gx)?;
        }
        Op::ScatterAlong { axis, combine } => {
            let gu = g.gather_along(ct, s[1], *axis)?;
            acc(g, cot, s[2], gu)?;
            match combine {
                ScatterOp::Add => acc(g, cot, s[0], ct)?,
                ScatterOp::Set | ScatterOp::Max | ScatterOp::Min => {
                    let zu = g.zeros_like(s[2]);
                    let masked = g.scatter_along(ct, s[1], zu, *axis, ScatterOp::Set)?;
                    acc(g, cot, s[0], masked)?;
                }
            }
        }
        _ => unreachable!("indexing::vjp: {:?}", n.op),
    }
    Ok(())
}
