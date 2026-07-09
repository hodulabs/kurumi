//! Reference interpreter (the oracle): walks the IR memoized and dispatches each
//! primitive (per dtype) to the generic kernels in the submodules. Every backend
//! is checked against this; device backends reuse `eval_op` for ops they skip.

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

pub(crate) use contract::{dot_dispatch, dot_general, quant_matmul};
pub(crate) use reduce::reduce_v;

use crate::{
    Bitwise, Float, Graph, Int, NodeId, Num, Op, ScatterOp, Signed, Storage, TensorVal, bitcast, cast, iota_storage,
};
use elementwise::{cmp_map, map1, select_k, zip_map};
use indexing::{gather_along_k, gather_k, indices_i64, scatter_along_k, scatter_k, set_combine};
use movement::{expand_k, flip_k, pad_k, permute_k, slice_k};
use random::rand_uniform_gen;
use reduce::{arg_reduce, reduce_prod, reduce_sum};
use sort::argsort;
use std::collections::HashMap;

/// Values supplied for `Op::Input` nodes at eval time (params/data), keyed by node.
pub type Feeds = HashMap<NodeId, TensorVal>;

// scatter Add/Max/Min combiner dispatch (numeric only; Set uses dispatch_pair! +
// set_combine). `$k` is the scatter kernel (`scatter_k` or `scatter_along_k`).
macro_rules! scatter_num {
    ($k:path, $op:expr, $up:expr, $f:path, $sh:expr, $ax:expr, $idx:expr, $is:expr) => {
        match ($op, $up) {
            (Storage::U8(o), Storage::U8(u)) => Storage::U8($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::U32(o), Storage::U32(u)) => Storage::U32($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::I32(o), Storage::I32(u)) => Storage::I32($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::I64(o), Storage::I64(u)) => Storage::I64($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::F16(o), Storage::F16(u)) => Storage::F16($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::BF16(o), Storage::BF16(u)) => Storage::BF16($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::F32(o), Storage::F32(u)) => Storage::F32($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::F64(o), Storage::F64(u)) => Storage::F64($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::F8E4M3(o), Storage::F8E4M3(u)) => Storage::F8E4M3($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::F8E5M2(o), Storage::F8E5M2(u)) => Storage::F8E5M2($k(o, $sh, $ax, $idx, $is, u, $f)),
            _ => unreachable!("scatter Add/Max/Min on non-numeric dtype"),
        }
    };
}

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
pub fn eval_op(op: &Op, inputs: &[&TensorVal]) -> TensorVal {
    match op {
        Op::Input { .. } => unreachable!("Input must be resolved from feeds before eval_op"),
        Op::Detach => inputs[0].clone(), // identity in the forward
        Op::Const { data, shape } => TensorVal { shape: shape.clone(), storage: data.clone() },
        Op::Iota { shape, axis, dtype } => {
            TensorVal { shape: shape.clone(), storage: iota_storage(shape, *axis, *dtype) }
        }
        Op::Cast { to } => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: cast(&a.storage, *to) }
        }
        Op::Bitcast { to } => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: bitcast(&a.storage, *to) }
        }
        Op::Add => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: num_binary!(&a.storage, &b.storage, Num::add) }
        }
        Op::Mul => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: num_binary!(&a.storage, &b.storage, Num::mul) }
        }
        Op::Max => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: num_binary!(&a.storage, &b.storage, Num::max) }
        }
        Op::IDiv => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: int_binary!(&a.storage, &b.storage, Int::idiv) }
        }
        Op::Shl => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: int_binary!(&a.storage, &b.storage, Int::shl) }
        }
        Op::Shr => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: int_binary!(&a.storage, &b.storage, Int::shr) }
        }
        Op::And => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: bitwise_binary!(&a.storage, &b.storage, Bitwise::and) }
        }
        Op::Or => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: bitwise_binary!(&a.storage, &b.storage, Bitwise::or) }
        }
        Op::Xor => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: bitwise_binary!(&a.storage, &b.storage, Bitwise::xor) }
        }
        Op::CmpLt => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: cmp_binary!(&a.storage, &b.storage, PartialOrd::lt) }
        }
        Op::CmpEq => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: cmp_binary!(&a.storage, &b.storage, PartialEq::eq) }
        }
        Op::Where => {
            let (c, a, b) = (inputs[0], inputs[1], inputs[2]);
            let cond = match &c.storage {
                Storage::BOOL(v) => v,
                _ => unreachable!("where cond must be BOOL"),
            };
            let storage = dispatch_pair!(&a.storage, &b.storage, |x, y| select_k(cond, x, y));
            TensorVal { shape: a.shape.clone(), storage }
        }
        Op::Neg => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: signed_unary!(&a.storage, Signed::neg) }
        }
        Op::Recip => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: float_unary!(&a.storage, Float::recip) }
        }
        Op::Sqrt => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: float_unary!(&a.storage, Float::sqrt) }
        }
        Op::Exp2 => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: float_unary!(&a.storage, Float::exp2) }
        }
        Op::Log2 => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: float_unary!(&a.storage, Float::log2) }
        }
        Op::Sin => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: float_unary!(&a.storage, Float::sin) }
        }
        Op::Floor => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: float_unary!(&a.storage, Float::floor) }
        }
        Op::Sum { axis } => reduce_sum(&inputs[0].storage, &inputs[0].shape, *axis),
        Op::Prod { axis } => reduce_prod(&inputs[0].storage, &inputs[0].shape, *axis),
        Op::ReduceMax { axis } => num_reduce!(&inputs[0].storage, &inputs[0].shape, *axis, Num::lowest, Num::max),
        Op::ArgReduce { axis, kind } => arg_reduce(inputs[0], *axis, *kind),
        Op::Softmax { axis } => nn::softmax_v(&inputs[0].storage, &inputs[0].shape, *axis),
        Op::RmsNorm { axis, eps } => nn::rmsnorm_v(&inputs[0].storage, &inputs[0].shape, *axis, *eps),
        Op::Sdpa { causal } => nn::sdpa_v(inputs[0], inputs[1], inputs[2], *causal),
        Op::Reshape { shape } => TensorVal { shape: shape.clone(), storage: inputs[0].storage.clone() },
        Op::Permute { perm } => {
            let a = inputs[0];
            let out: Vec<usize> = perm.iter().map(|&p| a.shape[p]).collect();
            TensorVal { shape: out, storage: dispatch!(&a.storage, |v| permute_k(v, &a.shape, perm)) }
        }
        Op::Expand { shape } => {
            let a = inputs[0];
            TensorVal { shape: shape.clone(), storage: dispatch!(&a.storage, |v| expand_k(v, &a.shape, shape)) }
        }
        Op::Slice { ranges } => {
            let a = inputs[0];
            let out: Vec<usize> = ranges.iter().map(|&(s, e, st)| (e - s).div_ceil(st)).collect();
            TensorVal { shape: out, storage: dispatch!(&a.storage, |v| slice_k(v, &a.shape, ranges)) }
        }
        Op::Flip { axes } => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: dispatch!(&a.storage, |v| flip_k(v, &a.shape, axes)) }
        }
        Op::Pad { pads } => {
            let a = inputs[0];
            let out: Vec<usize> = pads.iter().zip(&a.shape).map(|(&(lo, hi), &s)| lo + s + hi).collect();
            TensorVal { shape: out, storage: dispatch!(&a.storage, |v| pad_k(v, &a.shape, pads)) }
        }
        Op::Gather { axis } => {
            let (op, ix) = (inputs[0], inputs[1]);
            let idx = indices_i64(&ix.storage);
            let out: Vec<usize> =
                op.shape[..*axis].iter().chain(&ix.shape).chain(&op.shape[*axis + 1..]).copied().collect();
            let storage = dispatch!(&op.storage, |v| gather_k(v, &op.shape, *axis, &idx, &ix.shape));
            TensorVal { shape: out, storage }
        }
        Op::Scatter { axis, combine } => {
            let (op, ix, up) = (inputs[0], inputs[1], inputs[2]);
            let idx = indices_i64(&ix.storage);
            let storage = match combine {
                ScatterOp::Set => {
                    dispatch_pair!(&op.storage, &up.storage, |o, u| scatter_k(
                        o,
                        &op.shape,
                        *axis,
                        &idx,
                        &ix.shape,
                        u,
                        set_combine
                    ))
                }
                ScatterOp::Add => {
                    scatter_num!(scatter_k, &op.storage, &up.storage, Num::add, &op.shape, *axis, &idx, &ix.shape)
                }
                ScatterOp::Max => {
                    scatter_num!(scatter_k, &op.storage, &up.storage, Num::max, &op.shape, *axis, &idx, &ix.shape)
                }
                ScatterOp::Min => {
                    scatter_num!(scatter_k, &op.storage, &up.storage, Num::min, &op.shape, *axis, &idx, &ix.shape)
                }
            };
            TensorVal { shape: op.shape.clone(), storage }
        }
        Op::GatherAlong { axis } => {
            let (op, ix) = (inputs[0], inputs[1]);
            let idx = indices_i64(&ix.storage);
            let storage = dispatch!(&op.storage, |v| gather_along_k(v, &op.shape, *axis, &idx, &ix.shape));
            TensorVal { shape: ix.shape.clone(), storage }
        }
        Op::ScatterAlong { axis, combine } => {
            let (op, ix, up) = (inputs[0], inputs[1], inputs[2]);
            let idx = indices_i64(&ix.storage);
            let storage = match combine {
                ScatterOp::Set => {
                    dispatch_pair!(&op.storage, &up.storage, |o, u| scatter_along_k(
                        o,
                        &op.shape,
                        *axis,
                        &idx,
                        &ix.shape,
                        u,
                        set_combine
                    ))
                }
                ScatterOp::Add => {
                    scatter_num!(scatter_along_k, &op.storage, &up.storage, Num::add, &op.shape, *axis, &idx, &ix.shape)
                }
                ScatterOp::Max => {
                    scatter_num!(scatter_along_k, &op.storage, &up.storage, Num::max, &op.shape, *axis, &idx, &ix.shape)
                }
                ScatterOp::Min => {
                    scatter_num!(scatter_along_k, &op.storage, &up.storage, Num::min, &op.shape, *axis, &idx, &ix.shape)
                }
            };
            TensorVal { shape: op.shape.clone(), storage }
        }
        Op::Argsort { axis, descending } => argsort(inputs[0], *axis, *descending),
        Op::RandUniform { shape } => {
            let seed = indices_i64(&inputs[0].storage)[0] as u64;
            rand_uniform_gen(seed, shape)
        }
        Op::Solve => {
            let (a, b) = (inputs[0], inputs[1]);
            let (ar, br) = (a.shape.len(), b.shape.len());
            let n = a.shape[ar - 1];
            let batch: usize = a.shape[..ar - 2].iter().product();
            let k = b.shape[br - 1];
            TensorVal { shape: b.shape.clone(), storage: linalg::solve(&a.storage, &b.storage, batch, n, k) }
        }
        Op::Det => {
            let a = inputs[0];
            let ar = a.shape.len();
            let n = a.shape[ar - 1];
            let batch: usize = a.shape[..ar - 2].iter().product();
            TensorVal { shape: a.shape[..ar - 2].to_vec(), storage: linalg::det(&a.storage, batch, n) }
        }
        Op::Cholesky => {
            let a = inputs[0];
            let ar = a.shape.len();
            let n = a.shape[ar - 1];
            let batch: usize = a.shape[..ar - 2].iter().product();
            TensorVal { shape: a.shape.clone(), storage: linalg::cholesky(&a.storage, batch, n) }
        }
        Op::Eigh => {
            let a = inputs[0];
            let ar = a.shape.len();
            let n = a.shape[ar - 1];
            let batch: usize = a.shape[..ar - 2].iter().product();
            let mut shape = a.shape.clone();
            *shape.last_mut().unwrap() += 1; // [.., N, N+1]
            TensorVal { shape, storage: linalg::eigh(&a.storage, batch, n) }
        }
        Op::Qr { r_factor } => {
            let a = inputs[0];
            let ar = a.shape.len();
            let (m, n) = (a.shape[ar - 2], a.shape[ar - 1]);
            let batch: usize = a.shape[..ar - 2].iter().product();
            let k = m.min(n);
            let mut shape = a.shape.clone();
            if *r_factor {
                shape[ar - 2] = k
            } else {
                shape[ar - 1] = k
            }
            TensorVal { shape, storage: linalg::qr(&a.storage, batch, m, n, *r_factor) }
        }
        Op::Eigvals => {
            let a = inputs[0];
            let ar = a.shape.len();
            let n = a.shape[ar - 1];
            let batch: usize = a.shape[..ar - 2].iter().product();
            TensorVal { shape: a.shape[..ar - 1].to_vec(), storage: linalg::eigvals(&a.storage, batch, n) }
        }
        Op::Complex => {
            let storage = complex::complex_k(&inputs[0].storage, &inputs[1].storage);
            TensorVal { shape: inputs[0].shape.clone(), storage }
        }
        Op::Real => TensorVal { shape: inputs[0].shape.clone(), storage: complex::real_k(&inputs[0].storage) },
        Op::Imag => TensorVal { shape: inputs[0].shape.clone(), storage: complex::imag_k(&inputs[0].storage) },
        Op::DotGeneral { lhs_contract, rhs_contract, lhs_batch, rhs_batch } => {
            dot_dispatch(inputs[0], inputs[1], lhs_contract, rhs_contract, lhs_batch, rhs_batch)
        }
        Op::QuantMatmul { bits, group_size, symmetric } => {
            let mins = (!*symmetric).then(|| inputs[3]);
            quant_matmul(inputs[0], inputs[1], inputs[2], mins, *bits, *group_size)
        }
    }
}
