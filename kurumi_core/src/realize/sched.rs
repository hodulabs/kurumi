//! The scheduler walk: graph -> `Realized` nodes. Movement rewrites the read view over a shared
//! source buffer (0 copies); elementwise ops fuse into one lazy expression; a boundary (reduce,
//! contraction, movement on a fused result, multi-consumer node) materializes once. `go` is the
//! memoized recursion; the fused_bin/unary/movement/leaf_of helpers build each node. The
//! one-shot entries + fused-path gate live in `eval`.

use super::bump_kernel;
use super::expr::{Expr, Realized, Repr, UnOp};
use crate::lower::index::{self, View};
use crate::{Feeds, Graph, NodeId, Op, TensorVal, dot_general, reduce_v};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

pub fn realize(g: &Graph, id: NodeId) -> Realized {
    let counts = consumer_counts(g, id);
    let feeds = Feeds::new();
    let s = Sched { g, counts: &counts, feeds: &feeds, consts: None };
    let mut memo = HashMap::new();
    go(&s, id, &mut memo)
}

// Scheduling env threaded through `go`: graph + consumer counts, plus run-time leaf
// bindings -- `feeds` for Input nodes, and (on a compiled Plan) const buffers
// materialized once so replay never re-copies weights.
pub(crate) struct Sched<'a> {
    pub(crate) g: &'a Graph,
    pub(crate) counts: &'a HashMap<NodeId, usize>,
    pub(crate) feeds: &'a Feeds,
    pub(crate) consts: Option<&'a HashMap<NodeId, Rc<[f32]>>>,
}

pub(crate) fn go(s: &Sched, id: NodeId, memo: &mut HashMap<NodeId, Realized>) -> Realized {
    if let Some(r) = memo.get(&id) {
        return r.clone();
    }
    let n = s.g.node(id);
    let r = match &n.op {
        Op::Const { data, shape } => {
            // one-shot realize copies; a Plan hands back the buffer it materialized
            // once at compile (shared Rc: weights never re-copied on replay).
            let buf = match s.consts {
                Some(cache) => cache[&id].clone(),
                None => Rc::from(data.as_f32()),
            };
            Realized(Repr::Leaf { buf, view: View::source(shape.clone()) })
        }
        // per-step data (only reached via a Plan; `op_fused` excludes Input so the
        // no-feeds `force`/`realize` never routes an Input graph here).
        Op::Input { .. } => {
            let tv = s.feeds.get(&id).expect("realize: missing feed for an Input node");
            Realized(Repr::Leaf { buf: Rc::from(tv.storage.as_f32()), view: View::source(tv.shape.clone()) })
        }
        // reached only on all-f32 graphs (force() falls back otherwise), so a cast
        // is f32 -> f32, i.e. identity.
        Op::Cast { .. } => go(s, n.src[0], memo),

        Op::Permute { perm } => movement(go(s, n.src[0], memo), |v| Some(v.permute(perm))),
        Op::Expand { shape } => movement(go(s, n.src[0], memo), |v| Some(v.expand(shape))),
        Op::Slice { ranges } => movement(go(s, n.src[0], memo), |v| Some(v.slice(ranges))),
        Op::Flip { axes } => movement(go(s, n.src[0], memo), |v| Some(v.flip(axes))),
        Op::Pad { pads } => movement(go(s, n.src[0], memo), |v| Some(v.pad(pads))),
        Op::Reshape { shape } => movement(go(s, n.src[0], memo), |v| v.reshape(shape.clone())),

        Op::Neg => fused_unary(go(s, n.src[0], memo), UnOp::Neg),
        Op::Recip => fused_unary(go(s, n.src[0], memo), UnOp::Recip),
        Op::Sqrt => fused_unary(go(s, n.src[0], memo), UnOp::Sqrt),
        Op::Exp2 => fused_unary(go(s, n.src[0], memo), UnOp::Exp2),
        Op::Log2 => fused_unary(go(s, n.src[0], memo), UnOp::Log2),
        Op::Sin => fused_unary(go(s, n.src[0], memo), UnOp::Sin),
        Op::Add => fused_bin(go(s, n.src[0], memo), go(s, n.src[1], memo), Expr::Add),
        Op::Mul => fused_bin(go(s, n.src[0], memo), go(s, n.src[1], memo), Expr::Mul),
        Op::Max => fused_bin(go(s, n.src[0], memo), go(s, n.src[1], memo), Expr::Max),

        Op::Sum { axis } => {
            let r = go(s, n.src[0], memo);
            let (d, sh) = r.force_contig();
            Realized(leaf_of(reduce_v(&d, &sh, *axis, 0.0, |a, x| a + x)))
        }
        Op::ReduceMax { axis } => {
            let r = go(s, n.src[0], memo);
            let (d, sh) = r.force_contig();
            Realized(leaf_of(reduce_v(&d, &sh, *axis, f32::NEG_INFINITY, f32::max)))
        }
        Op::DotGeneral { lhs_contract, rhs_contract, lhs_batch, rhs_batch } => {
            let (ra, rb) = (go(s, n.src[0], memo), go(s, n.src[1], memo));
            let ((ad, ash), (bd, bsh)) = (ra.force_contig(), rb.force_contig());
            Realized(leaf_of(dot_general(&ad, &ash, &bd, &bsh, lhs_contract, rhs_contract, lhs_batch, rhs_batch)))
        }
        // ops the fused executor doesn't lower; force() falls back to interpret
        // for any graph containing them (op_fused gate), so this is unreachable.
        _ => unreachable!("op not lowered by the fused executor: {:?}", n.op),
    };

    // multi-consumer fused node: materialize once so consumers share the buffer,
    // not re-evaluate a shared subtree per consumer
    let r = if r.is_fused() && s.counts.get(&id).copied().unwrap_or(0) > 1 {
        let (buf, view) = r.into_leaf();
        Realized(Repr::Leaf { buf, view })
    } else {
        r
    };
    memo.insert(id, r.clone());
    r
}

fn fused_bin(a: Realized, b: Realized, mk: fn(Rc<Expr>, Rc<Expr>) -> Expr) -> Realized {
    // graph builder guarantees a.shape == b.shape (no broadcast at binary)
    let shape = a.shape().to_vec();
    Realized(Repr::Fused { shape, expr: mk(Rc::new(a.as_expr()), Rc::new(b.as_expr())) })
}

fn fused_unary(a: Realized, op: UnOp) -> Realized {
    let shape = a.shape().to_vec();
    Realized(Repr::Fused { shape, expr: Expr::Unary(op, Rc::new(a.as_expr())) })
}

fn movement(src: Realized, f: impl Fn(&View) -> Option<View>) -> Realized {
    let (buf, view) = src.into_leaf();
    // a guarded (padded) source must materialize before further movement
    let (buf, view) = if view.guards.is_empty() {
        (buf, view)
    } else {
        bump_kernel(); // gather the padded source: one pass
        (Rc::from(index::read(&buf, &view)), View::source(view.shape))
    };
    match f(&view) {
        Some(view) => Realized(Repr::Leaf { buf, view }),
        None => {
            // non-contiguous reshape: materialize, then it lowers freely
            bump_kernel();
            let buf: Rc<[f32]> = Rc::from(index::read(&buf, &view));
            let base = View::source(view.shape);
            let view = f(&base).expect("contiguous reshape always lowers");
            Realized(Repr::Leaf { buf, view })
        }
    }
}

fn leaf_of(tv: TensorVal) -> Repr {
    bump_kernel(); // reduce / contraction output = one pass
    Repr::Leaf { buf: Rc::from(tv.storage.into_f32()), view: View::source(tv.shape) }
}

// in-degree of each reachable node = how many times it is read
pub(crate) fn consumer_counts(g: &Graph, root: NodeId) -> HashMap<NodeId, usize> {
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        for &s in &g.node(id).src {
            *counts.entry(s).or_insert(0) += 1;
            stack.push(s);
        }
    }
    counts
}
