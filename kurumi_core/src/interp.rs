//! Reference interpreter (the oracle): walks the IR memoized and routes each primitive to its
//! category's `eval` in the submodules -- each owns BOTH the per-dtype dispatch and the kernels
//! (mirroring `grad`'s per-category `vjp`), so a new op touches one file. Every backend is
//! checked against this; device backends reuse `eval_op` for the ops they skip.

mod complex;
mod contract;
mod elementwise;
mod indexing;
mod linalg;
mod movement;
mod nn;
mod random;
mod reduce;
mod sort;

pub(crate) use contract::{dot_dispatch, dot_general};
pub(crate) use reduce::reduce_v;

use crate::{Feeds, Graph, NodeId, Op, TensorVal};
use std::collections::HashMap;

/// Reference interpreter: the oracle every backend is checked against. Memoized so a
/// shared subgraph computes once (decompositions are diamond-heavy; naive recursion is exponential).
pub fn interpret(g: &Graph, id: NodeId) -> TensorVal {
    interpret_with(g, id, &Feeds::new())
}

/// Interpret with `Input` nodes supplied by `feeds` (build the graph once, feed
/// params/data per step). Panics if an `Input` reached has no feed.
pub fn interpret_with(g: &Graph, id: NodeId, feeds: &Feeds) -> TensorVal {
    interpret_memo(g, id, feeds, &mut HashMap::new())
}

/// Interpret several outputs sharing one memo pass: a subgraph common to the
/// requested nodes (the forward trunk under many grads) computes once.
pub fn interpret_many(g: &Graph, ids: &[NodeId], feeds: &Feeds) -> Vec<TensorVal> {
    let mut memo = std::collections::HashMap::new();
    ids.iter().map(|&id| interpret_memo(g, id, feeds, &mut memo)).collect()
}

fn interpret_memo(g: &Graph, id: NodeId, feeds: &Feeds, memo: &mut HashMap<NodeId, TensorVal>) -> TensorVal {
    if let Some(v) = memo.get(&id) {
        return v.clone();
    }
    let v = if matches!(g.node(id).op, Op::Input { .. }) {
        feeds.get(&id).expect("interpret: missing feed for an Input node").clone()
    } else {
        let src = g.node(id).src.clone();
        let inputs: Vec<TensorVal> = src.iter().map(|&s| interpret_memo(g, s, feeds, memo)).collect();
        let refs: Vec<&TensorVal> = inputs.iter().collect();
        eval_op(&g.node(id).op, &refs)
    };
    memo.insert(id, v.clone());
    v
}

/// Run one primitive given its materialized inputs. The CPU reference for every op + dtype;
/// device backends reuse it for ops they don't accelerate (so every op runs on every backend).
/// A thin router: each arm delegates to its category module's `eval`, which owns the per-dtype
/// dispatch AND the kernels. The match is exhaustive -- a new `Op` fails to compile until routed.
pub fn eval_op(op: &Op, inputs: &[&TensorVal]) -> TensorVal {
    match op {
        Op::Input { .. } => unreachable!("Input must be resolved from feeds before eval_op"),
        Op::Detach => inputs[0].clone(), // identity in the forward
        Op::Const { data, shape } => TensorVal { shape: shape.clone(), storage: data.clone() },

        Op::Add
        | Op::Mul
        | Op::Max
        | Op::IDiv
        | Op::Shl
        | Op::Shr
        | Op::And
        | Op::Or
        | Op::Xor
        | Op::CmpLt
        | Op::CmpEq
        | Op::Where
        | Op::Neg
        | Op::Recip
        | Op::Sqrt
        | Op::Exp2
        | Op::Log2
        | Op::Sin
        | Op::Floor
        | Op::Cast { .. }
        | Op::Bitcast { .. } => elementwise::eval(op, inputs),

        Op::Sum { .. } | Op::Prod { .. } | Op::ReduceMax { .. } | Op::ArgReduce { .. } => reduce::eval(op, inputs),
        Op::Argsort { .. } => sort::eval(op, inputs),

        Op::Softmax { .. } | Op::RmsNorm { .. } | Op::Sdpa { .. } => nn::eval(op, inputs),

        Op::Reshape { .. }
        | Op::Permute { .. }
        | Op::Expand { .. }
        | Op::Slice { .. }
        | Op::Flip { .. }
        | Op::Pad { .. } => movement::eval(op, inputs),

        Op::Gather { .. } | Op::Scatter { .. } | Op::GatherAlong { .. } | Op::ScatterAlong { .. } => {
            indexing::eval(op, inputs)
        }

        Op::Iota { .. } | Op::RandUniform { .. } => random::eval(op, inputs),

        Op::Solve | Op::Det | Op::Cholesky | Op::Eigh | Op::Qr { .. } | Op::Eigvals => linalg::eval(op, inputs),
        Op::Complex | Op::Real | Op::Imag => complex::eval(op, inputs),
        Op::DotGeneral { .. } | Op::QuantMatmul { .. } => contract::eval(op, inputs),
    }
}
