//! VJPs for pointwise arithmetic (add/mul/max/neg + the unary transcendentals) and
//! `where` select. Holomorphic complex rules conjugate the derivative factor via the
//! shared `cfactor` (identity for real dtypes).

use crate::grad::{acc, cfactor};
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
        Op::Cast { .. } => {
            // grad flows only into a float source (integer src => Zero)
            let src_dt = g.dtype(s[0]);
            if src_dt.is_float() {
                let gx = g.cast(ct, src_dt);
                acc(g, cot, s[0], gx)?;
            }
        }
        Op::Add => {
            acc(g, cot, s[0], ct)?;
            acc(g, cot, s[1], ct)?;
        }
        Op::Mul => {
            let cb = cfactor(g, s[1])?;
            let ga = g.mul(ct, cb)?;
            acc(g, cot, s[0], ga)?;
            let ca = cfactor(g, s[0])?;
            let gb = g.mul(ct, ca)?;
            acc(g, cot, s[1], gb)?;
        }
        Op::Max => {
            // ties go to the first operand (a<b false => a wins)
            let lt = g.cmp_lt(s[0], s[1])?;
            let zero = g.zeros_like(ct);
            let ga = g.select(lt, zero, ct)?; // a<b ? 0 : ct
            acc(g, cot, s[0], ga)?;
            let gb = g.select(lt, ct, zero)?; // a<b ? ct : 0
            acc(g, cot, s[1], gb)?;
        }
        Op::Neg => {
            let gx = g.neg(ct);
            acc(g, cot, s[0], gx)?;
        }
        Op::Recip => {
            // y = 1/a ; dy/da = -y^2  (complex: conj the factor)
            let y2 = g.mul(id, id)?;
            let cf = cfactor(g, y2)?;
            let t = g.mul(ct, cf)?;
            let gx = g.neg(t);
            acc(g, cot, s[0], gx)?;
        }
        Op::Sqrt => {
            // y = sqrt(a) ; dy/da = 0.5/y
            let inv = g.recip(id);
            let half = g.scalar(id, 0.5);
            let scaled = g.mul(half, inv)?;
            let cf = cfactor(g, scaled)?;
            let gx = g.mul(ct, cf)?;
            acc(g, cot, s[0], gx)?;
        }
        Op::Exp2 => {
            // y = 2^a ; dy/da = y * ln2
            let ln2 = g.scalar(id, std::f32::consts::LN_2);
            let yl = g.mul(id, ln2)?;
            let cf = cfactor(g, yl)?;
            let gx = g.mul(ct, cf)?;
            acc(g, cot, s[0], gx)?;
        }
        Op::Log2 => {
            // dy/da = 1/(a * ln2)
            let inv = g.recip(s[0]);
            let inv_ln2 = g.scalar(s[0], 1.0 / std::f32::consts::LN_2);
            let d = g.mul(inv, inv_ln2)?;
            let cf = cfactor(g, d)?;
            let gx = g.mul(ct, cf)?;
            acc(g, cot, s[0], gx)?;
        }
        Op::Sin => {
            // dy/da = cos(a) = sin(pi/2 - a)
            let hp = g.scalar(s[0], std::f32::consts::FRAC_PI_2);
            let arg = g.sub(hp, s[0])?;
            let cos = g.sin(arg);
            let cf = cfactor(g, cos)?;
            let gx = g.mul(ct, cf)?;
            acc(g, cot, s[0], gx)?;
        }
        Op::Where => {
            // cond(s[0]) is non-differentiable; route ct to the picked branch
            let cond = s[0];
            let zero = g.zeros_like(ct);
            let ga = g.select(cond, ct, zero)?;
            acc(g, cot, s[1], ga)?;
            let gb = g.select(cond, zero, ct)?;
            acc(g, cot, s[2], gb)?;
        }
        _ => unreachable!("elementwise::vjp: {:?}", n.op),
    }
    Ok(())
}
