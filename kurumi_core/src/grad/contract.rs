//! VJP for the contraction (dot_general): grad of a contraction is two contractions
//! (contract the cotangent against the other operand over its free axes, then permute
//! into the operand's axis order). Complex conjugates the other operand.

use crate::grad::{acc, cfactor};
use crate::{Error, Graph, Node, NodeId, Op, free_axes};
use std::collections::HashMap;

pub(super) fn vjp(g: &mut Graph, n: &Node, ct: NodeId, cot: &mut HashMap<NodeId, NodeId>) -> Result<(), Error> {
    let Op::DotGeneral { lhs_contract, rhs_contract, lhs_batch, rhs_batch } = &n.op else {
        unreachable!("contract::vjp: {:?}", n.op)
    };
    dot_general_vjp(g, ct, n.src[0], n.src[1], lhs_contract, rhs_contract, lhs_batch, rhs_batch, cot)
}

fn dot_general_vjp(
    g: &mut Graph,
    ct: NodeId,
    a: NodeId,
    b: NodeId,
    lc: &[usize],
    rc: &[usize],
    lb: &[usize],
    rb: &[usize],
    cot: &mut HashMap<NodeId, NodeId>,
) -> Result<(), Error> {
    let (ra, rb_rank) = (g.shape(a).len(), g.shape(b).len());
    let a_free = free_axes(ra, lb, lc);
    let b_free = free_axes(rb_rank, rb, rc);
    let (nb, naf, nbf, nc) = (lb.len(), a_free.len(), b_free.len(), lc.len());

    // ct layout: [batch(nb)] [a_free(naf)] [b_free(nbf)]
    let ct_batch: Vec<usize> = (0..nb).collect();
    let ct_a_free: Vec<usize> = (nb..nb + naf).collect();
    let ct_b_free: Vec<usize> = (nb + naf..nb + naf + nbf).collect();

    // grad_a = dot(ct, b) contracting ct's b_free with b's free; result =
    // [batch][a_free][b_contract], permute to a's [batch/free/contract] layout.
    // complex: conjugate the other operand (grad_a = ct @ conj(b)).
    let cb = cfactor(g, b)?;
    let ga_raw = g.dot_general(ct, cb, ct_b_free, b_free.clone(), ct_batch.clone(), rb.to_vec())?;
    let ga = g.permute(ga_raw, perm_blocks(ra, lb, &a_free, lc))?;
    acc(g, cot, a, ga)?;

    // grad_b = dot(a, ct) contracting a's free with ct's a_free; result =
    // [batch][a_contract][b_free], permute to b's layout (contract block first).
    let ca = cfactor(g, a)?;
    let gb_raw = g.dot_general(ca, ct, a_free, ct_a_free, lb.to_vec(), ct_batch)?;
    let gb = g.permute(gb_raw, perm_blocks(rb_rank, rb, rc, &b_free))?;
    acc(g, cot, b, gb)?;

    let _ = nc;
    Ok(())
}

// raw layout is [batch][block1][block2]; build the permutation that sends each
// raw axis to its target position in the operand's axis order.
fn perm_blocks(rank: usize, batch: &[usize], block1: &[usize], block2: &[usize]) -> Vec<usize> {
    let (nb, n1) = (batch.len(), block1.len());
    let mut perm = vec![0usize; rank];
    for (j, &p) in batch.iter().enumerate() {
        perm[p] = j;
    }
    for (k, &p) in block1.iter().enumerate() {
        perm[p] = nb + k;
    }
    for (k, &p) in block2.iter().enumerate() {
        perm[p] = nb + n1 + k;
    }
    perm
}
