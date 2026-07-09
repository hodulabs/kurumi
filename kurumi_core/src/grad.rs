//! Reverse-mode autograd as an IR->IR transform. Each primitive has a VJP rule
//! that emits backward primitives into the same graph; composite/surface ops get
//! their gradient for free via their decomposition (a transformation contract,
//! not JAX-style JVP+transpose).

mod complex;
mod contract;
mod elementwise;
mod indexing;
mod linalg;
mod movement;
mod nn;
mod reduce;

use crate::{DType, Error, Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};

/// Gradients of `sum(output)` w.r.t. each node in `wrt` (reverse-mode VJP).
/// Requires an f32 graph (the training dtype); non-differentiable ops (integers,
/// bool, comparisons, floor, iota, bitcast) contribute zero.
pub fn grad(g: &mut Graph, output: NodeId, wrt: &[NodeId]) -> Result<Vec<NodeId>, Error> {
    if g.dtype(output) != DType::F32 {
        return Err(Error::shape("grad", format!("output must be f32, got {:?}", g.dtype(output))));
    }
    let reachable = reachable_from(g, output);
    let mut cot: HashMap<NodeId, NodeId> = HashMap::new();
    let seed = g.ones_like(output);
    cot.insert(output, seed);

    // arena order is topological (input id < consumer id), so decreasing-id order is
    // reverse-topological: every consumer is processed (cotangent fully accumulated)
    // before we reach the node itself.
    for i in (0..=output.0).rev() {
        let id = NodeId(i);
        if !reachable.contains(&id) {
            continue;
        }
        if let Some(&ct) = cot.get(&id) {
            vjp(g, id, ct, &mut cot)?;
        }
    }

    let mut out = Vec::with_capacity(wrt.len());
    for &w in wrt {
        out.push(match cot.get(&w) {
            Some(&c) => c,
            None => g.zeros_like(w),
        });
    }
    Ok(out)
}

fn reachable_from(g: &Graph, root: NodeId) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if seen.insert(id) {
            stack.extend_from_slice(&g.node(id).src);
        }
    }
    seen
}

// accumulate a cotangent contribution into `node` (sum across fan-out consumers)
fn acc(g: &mut Graph, cot: &mut HashMap<NodeId, NodeId>, node: NodeId, contrib: NodeId) -> Result<(), Error> {
    let merged = match cot.get(&node) {
        Some(&existing) => g.add(existing, contrib)?,
        None => contrib,
    };
    cot.insert(node, merged);
    Ok(())
}

// holomorphic complex VJPs conjugate the derivative factor (real-pair CR:
// grad_z = ct*conj(f'(z)); complex mul grad_z = ct*conj(w); complex matmul
// conjugates the other operand). Real dtypes: identity, so real autodiff is unchanged.
fn cfactor(g: &mut Graph, x: NodeId) -> Result<NodeId, Error> {
    if g.dtype(x).is_complex() { g.conj(x) } else { Ok(x) }
}

// push the cotangent `ct` (grad w.r.t. this node's output) back to its inputs.
fn vjp(g: &mut Graph, id: NodeId, ct: NodeId, cot: &mut HashMap<NodeId, NodeId>) -> Result<(), Error> {
    let n = g.node(id).clone();
    let s = &n.src;
    match &n.op {
        // leaves / non-differentiable: no cotangent flows
        // (Input is a leaf: cotangent accumulates at it -- the param grad.)
        // (Detach deliberately blocks the gradient.)
        Op::Const { .. }
        | Op::Input { .. }
        | Op::Iota { .. }
        | Op::Bitcast { .. }
        | Op::Detach
        | Op::Floor
        | Op::IDiv
        | Op::And
        | Op::Or
        | Op::Xor
        | Op::Shl
        | Op::Shr
        | Op::CmpLt
        | Op::CmpEq
        | Op::ArgReduce { .. }
        | Op::Argsort { .. }
        | Op::RandUniform { .. }
        | Op::QuantMatmul { .. } => {} // frozen weights, inference-only

        // general (non-symmetric) eigenvalues: the VJP is unimplemented, so error loudly
        // rather than return a silent zero gradient (use eigh for symmetric matrices).
        Op::Eigvals => {
            return Err(Error::shape(
                "eigvals backward",
                "eigvals is not differentiable (use eigh for symmetric matrices)",
            ));
        }

        // dense linalg VJPs (batched matrix helpers live with them in grad/linalg.rs)
        Op::Solve => linalg::solve_vjp(g, id, s, ct, cot)?,
        Op::Det => linalg::det_vjp(g, id, s, ct, cot)?,
        Op::Cholesky => linalg::cholesky_vjp(g, id, s, ct, cot)?,
        Op::Eigh => linalg::eigh_vjp(g, id, s, ct, cot)?,
        Op::Qr { r_factor } => linalg::qr_vjp(g, s, ct, *r_factor, cot)?,
        // real<->complex seam (no conjugation; the holomorphic conj is in `cfactor`)
        Op::Complex => complex::complex_vjp(g, s, ct, cot)?,
        Op::Real => complex::real_vjp(g, s, ct, cot)?,
        Op::Imag => complex::imag_vjp(g, s, ct, cot)?,

        // pointwise arithmetic + transcendentals + where
        Op::Cast { .. }
        | Op::Add
        | Op::Mul
        | Op::Max
        | Op::Neg
        | Op::Recip
        | Op::Sqrt
        | Op::Exp2
        | Op::Log2
        | Op::Sin
        | Op::Where => elementwise::vjp(g, id, &n, ct, cot)?,
        // reductions
        Op::Sum { .. } | Op::ReduceMax { .. } | Op::Prod { .. } => reduce::vjp(g, id, &n, ct, cot)?,
        // fused nn primitives
        Op::Softmax { .. } | Op::RmsNorm { .. } | Op::Sdpa { .. } => nn::vjp(g, id, &n, ct, cot)?,
        // movement (reshape/permute/expand/slice/flip/pad)
        Op::Reshape { .. }
        | Op::Permute { .. }
        | Op::Expand { .. }
        | Op::Slice { .. }
        | Op::Flip { .. }
        | Op::Pad { .. } => movement::vjp(g, &n, ct, cot)?,
        // contraction
        Op::DotGeneral { .. } => contract::vjp(g, &n, ct, cot)?,
        // gather / scatter
        Op::Gather { .. } | Op::Scatter { .. } | Op::GatherAlong { .. } | Op::ScatterAlong { .. } => {
            indexing::vjp(g, &n, ct, cot)?
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
