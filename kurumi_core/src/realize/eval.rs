//! One-shot eval entries + the fused-path gate + the fusion metric. `force`/`force_into`/
//! `force_counted` drive a single realize (or fall back to the interpreter oracle);
//! `fused_supported`/`op_fused` decide whether the f32 fused executor can run the graph at all.
//! The scheduler walk they invoke lives in `sched`.

use super::realize;
use crate::{Graph, NodeId, Op, TensorVal};
use std::cell::Cell;
use std::collections::HashSet;

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
    if fused_supported(g, id, false) { realize(g, id).force() } else { crate::interpret(g, id) }
}

/// Realize `id` into a reused output buffer (eval-loop / replay path: no per-call output
/// alloc, so a memory-bound op runs at streaming bandwidth, no page faults on a fresh
/// buffer). Returns the row-major shape.
pub fn force_into(g: &Graph, id: NodeId, out: &mut Vec<f32>) -> Vec<usize> {
    if !fused_supported(g, id, false) {
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
    if !fused_supported(g, id, false) {
        return (crate::interpret(g, id), None);
    }
    KERNELS.with(|k| k.set(0));
    let out = realize(g, id).force();
    (out, Some(KERNELS.with(|k| k.get())))
}

// fused path runs only when every reachable node is f32 AND uses an op the executor
// lowers; anything else (other dtypes, ops like where/cmp/iota) falls back to interpret.
// `allow_input` lets a compiled `Plan` treat an f32 Input as a valid leaf (fed at run time);
// one-shot `force` does not (no feeds).
pub(crate) fn fused_supported(g: &Graph, id: NodeId, allow_input: bool) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![id];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let node = g.node(n);
        let op_ok = op_fused(&node.op) || (allow_input && matches!(node.op, Op::Input { .. }));
        if g.dtype(n) != crate::DType::F32 || !op_ok {
            return false;
        }
        stack.extend_from_slice(&node.src);
    }
    true
}

fn op_fused(op: &Op) -> bool {
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
