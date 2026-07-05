//! VJPs for movement (reshape/permute/expand/slice/flip/pad): each backward is the
//! inverse movement: reshape<->reshape, permute<->inverse-permute, expand->sum,
//! slice<->pad (with dilation for strided slices), flip<->flip.

use crate::grad::acc;
use crate::{Error, Graph, Node, NodeId, Op};
use std::collections::HashMap;

pub(super) fn vjp(g: &mut Graph, n: &Node, ct: NodeId, cot: &mut HashMap<NodeId, NodeId>) -> Result<(), Error> {
    let s = &n.src;
    match &n.op {
        Op::Reshape { .. } => {
            let in_shape = g.shape(s[0]);
            let gx = g.reshape(ct, in_shape)?;
            acc(g, cot, s[0], gx)?;
        }
        Op::Permute { perm } => {
            let mut inv = vec![0usize; perm.len()];
            for (i, &p) in perm.iter().enumerate() {
                inv[p] = i;
            }
            let gx = g.permute(ct, inv)?;
            acc(g, cot, s[0], gx)?;
        }
        Op::Expand { shape } => {
            // sum out the broadcast axes, then reshape back to the input shape
            let in_shape = g.shape(s[0]);
            let mut axes: Vec<usize> = (0..shape.len()).filter(|&d| in_shape[d] == 1 && shape[d] != 1).collect();
            axes.sort_unstable_by(|a, b| b.cmp(a)); // descending: removal keeps lower indices valid
            let mut cur = ct;
            for d in axes {
                cur = g.sum(cur, d)?;
            }
            let gx = g.reshape(cur, in_shape)?;
            acc(g, cot, s[0], gx)?;
        }
        Op::Slice { ranges } => {
            // place ct back at the sampled positions of a zero input: dilate each axis
            // by its step (insert step-1 zeros between elements), then pad to in_shape.
            let in_shape = g.shape(s[0]);
            let mut cur = ct;
            for (d, &(_, _, step)) in ranges.iter().enumerate() {
                if step > 1 {
                    cur = dilate(g, cur, d, step)?;
                }
            }
            let cur_shape = g.shape(cur);
            let pads: Vec<(usize, usize)> = ranges
                .iter()
                .zip(&in_shape)
                .zip(&cur_shape)
                .map(|((&(lo, _, _), &dim), &cl)| (lo, dim - lo - cl))
                .collect();
            let gx = g.pad(cur, pads)?;
            acc(g, cot, s[0], gx)?;
        }
        Op::Flip { axes } => {
            let gx = g.flip(ct, axes.clone())?;
            acc(g, cot, s[0], gx)?;
        }
        Op::Pad { pads } => {
            // extract the original (unpadded) region
            let in_shape = g.shape(s[0]);
            let ranges: Vec<(usize, usize)> = pads.iter().zip(&in_shape).map(|(&(lo, _), &d)| (lo, lo + d)).collect();
            let gx = g.slice(ct, ranges)?;
            acc(g, cot, s[0], gx)?;
        }
        _ => unreachable!("movement::vjp: {:?}", n.op),
    }
    Ok(())
}

// insert `step-1` zeros after each element along `axis` (length k -> (k-1)*step+1).
// Used by the strided-slice VJP to spread the cotangent over the sampled positions.
fn dilate(g: &mut Graph, x: NodeId, axis: usize, step: usize) -> Result<NodeId, Error> {
    let sh = g.shape(x);
    let k = sh[axis];
    let mut s1 = sh.clone();
    s1.insert(axis + 1, 1); // [.., k, 1, ..]
    let r1 = g.reshape(x, s1)?;
    let mut pads = vec![(0, 0); sh.len() + 1];
    pads[axis + 1] = (0, step - 1); // [.., k, step, ..]
    let p = g.pad(r1, pads)?;
    let mut s2 = sh.clone();
    s2[axis] = k * step; // [.., k*step, ..]
    let m = g.reshape(p, s2)?;
    let want = (k - 1) * step + 1; // drop trailing step-1 zeros
    let mut ranges: Vec<(usize, usize)> = g.shape(m).iter().map(|&d| (0, d)).collect();
    ranges[axis] = (0, want);
    g.slice(m, ranges)
}
