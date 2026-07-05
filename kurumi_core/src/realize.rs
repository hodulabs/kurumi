//! View-fused evaluator. Movement only rewrites the read view over a shared source
//! buffer (Rc, 0 copies); elementwise ops fuse into one lazy expression read at the
//! output coordinate. Materializes only at a boundary (reduce, contraction, output,
//! movement on a fused result, multi-consumer node), so a movement+elementwise subtree
//! runs in ONE pass and a shared node computes once. Checked against `interpret`.
//! Submodules: repr (types), tape (executor), plan (compile-once replay).
//! This file is the scheduler: graph -> `Realized` nodes plus the one-shot eval entries.

mod plan;
mod repr;
mod tape;

pub use plan::Plan;
pub use repr::Realized;

use crate::lower::index::{self, View};
use crate::{Feeds, Graph, NodeId, Op, TensorVal, dot_general, reduce_v};
use repr::{Expr, Repr, UnOp};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

thread_local! {
    // fusion metric: compute passes over data this realize takes. Bumped per fused
    // elementwise group (eval_fused), reduce/contraction (leaf_of), and materializing
    // gather (movement); a view-only movement chain rides the source buffer, adds 0.
    static KERNELS: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn bump_kernel() {
    KERNELS.with(|k| k.set(k.get() + 1));
}

pub fn force(g: &Graph, id: NodeId) -> TensorVal {
    // fused executor is f32-only (ML hot path). Other dtypes fall back to the interpreter
    // oracle: correct, and f16/bf16 have no CPU perf win anyway (perf is on Metal).
    // Genericize over float dtype when Metal needs it.
    if realize_supported(g, id) { realize(g, id).force() } else { crate::interpret(g, id) }
}

/// Realize `id` into a reused output buffer (eval-loop / replay path: no per-call output
/// alloc, so a memory-bound op runs at streaming bandwidth, no page faults on a fresh
/// buffer). Returns the row-major shape.
pub fn force_into(g: &Graph, id: NodeId, out: &mut Vec<f32>) -> Vec<usize> {
    if !realize_supported(g, id) {
        let tv = crate::interpret(g, id);
        out.clear();
        out.extend_from_slice(&tv.storage.into_f32());
        return tv.shape;
    }
    let r = realize(g, id);
    let shape = r.shape().to_vec();
    r.force_into(out);
    shape
}

/// Realize `id` and report compute kernels (fused passes over data) taken: the fusion
/// metric. Elementwise groups, reductions, contractions, and materializing gathers each
/// count 1; a view-only movement chain adds 0. `None` means the graph left the f32 fused
/// path (an op the executor doesn't lower, e.g. Detach/Where/Cast-to-int) -> interpret.
pub fn force_counted(g: &Graph, id: NodeId) -> (TensorVal, Option<usize>) {
    if !realize_supported(g, id) {
        return (crate::interpret(g, id), None);
    }
    KERNELS.with(|k| k.set(0));
    let out = realize(g, id).force();
    (out, Some(KERNELS.with(|k| k.get())))
}

// fused path runs only when every reachable node is f32 AND uses an op the executor
// lowers; anything else (other dtypes, ops like where/cmp/iota) falls back to interpret.
fn realize_supported(g: &Graph, id: NodeId) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![id];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if g.dtype(n) != crate::DType::F32 || !op_fused(&g.node(n).op) {
            return false;
        }
        stack.extend_from_slice(&g.node(n).src);
    }
    true
}

pub(super) fn op_fused(op: &Op) -> bool {
    matches!(
        op,
        Op::Const { .. }
            | Op::Cast { .. }
            | Op::Add
            | Op::Mul
            | Op::Max
            | Op::Neg
            | Op::Recip
            | Op::Sqrt
            | Op::Exp2
            | Op::Log2
            | Op::Sin
            | Op::Sum { .. }
            | Op::ReduceMax { .. }
            | Op::Reshape { .. }
            | Op::Permute { .. }
            | Op::Expand { .. }
            | Op::Slice { .. }
            | Op::Flip { .. }
            | Op::Pad { .. }
            | Op::DotGeneral { .. }
    )
}

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
pub(super) struct Sched<'a> {
    pub(super) g: &'a Graph,
    pub(super) counts: &'a HashMap<NodeId, usize>,
    pub(super) feeds: &'a Feeds,
    pub(super) consts: Option<&'a HashMap<NodeId, Rc<[f32]>>>,
}

pub(super) fn go(s: &Sched, id: NodeId, memo: &mut HashMap<NodeId, Realized>) -> Realized {
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
pub(super) fn consumer_counts(g: &Graph, root: NodeId) -> HashMap<NodeId, usize> {
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

#[cfg(test)]
mod fuzz;
#[cfg(test)]
mod tests;
