//! Automatic mixed precision (AMP): an IR->IR pass running the compute-heavy,
//! numerically-forgiving ops in f16 while keeping the graph f32 elsewhere (sensitive
//! reductions / norms / loss stay full precision). Policy is matmul-in-f16: each f32
//! `DotGeneral` gets its inputs cast to f16, runs the f16 GEMM (2x on Metal via MPS f16),
//! and casts the result back to f32. Every other node keeps f32, so the transform is
//! local and the result matches the f32 graph within f16 tolerance (the test checks that
//! against the oracle). Bottom-up rebuild on the append-only arena, same shape as `simplify`.

use crate::DType;
use crate::graph::inspect::reachable;
use crate::graph::{Graph, NodeId, Op};
use std::collections::HashMap;

/// Rewrite `root` so f32 matmuls run in f16 (inputs cast down, output cast back up).
/// Returns the new root; appends the casts/f16 ops to the arena.
pub fn amp(g: &mut Graph, root: NodeId) -> NodeId {
    let mut remap: HashMap<NodeId, NodeId> = HashMap::new();
    for id in reachable(g, root) {
        let src_old = g.node(id).src.clone();
        if src_old.is_empty() {
            remap.insert(id, id); // leaf (const/input/iota): unchanged
            continue;
        }
        let src: Vec<NodeId> = src_old.iter().map(|s| remap[s]).collect();
        let op = g.node(id).op.clone();
        let new = if matches!(op, Op::DotGeneral { .. }) && g.dtype(id) == DType::F32 {
            // run this matmul in f16: cast operands down, GEMM, cast result up.
            let a = g.cast(src[0], DType::F16);
            let b = g.cast(src[1], DType::F16);
            let mm = g.push(op, vec![a, b]);
            g.cast(mm, DType::F32)
        } else if src == src_old {
            id // unchanged (srcs didn't move)
        } else {
            g.push(op, src)
        };
        remap.insert(id, new);
    }
    remap[&root]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, interpret};

    #[test]
    fn amp_runs_matmul_in_f16_and_matches_within_tol() {
        let mut g = Graph::new();
        let a = g.constant((0..32).map(|i| (i as f32 * 0.1).sin() * 0.3).collect(), vec![4, 8]);
        let b = g.constant((0..32).map(|i| (i as f32 * 0.2).cos() * 0.3).collect(), vec![8, 4]);
        let c = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
        let y = g.relu(c); // a following f32 op stays f32
        let want = interpret(&g, y).storage.into_f32();

        let amped = amp(&mut g, y);
        let got = interpret(&g, amped).storage.into_f32();
        assert_eq!(g.dtype(y), DType::F32); // output still f32
        for (w, g_) in want.iter().zip(&got) {
            assert!((w - g_).abs() < 2e-2, "amp {g_} vs f32 {w}");
        }
        // the matmul inside the amp graph is now f16
        let has_f16_matmul = reachable(&g, amped)
            .iter()
            .any(|&n| matches!(g.node(n).op, Op::DotGeneral { .. }) && g.dtype(n) == DType::F16);
        assert!(has_f16_matmul, "amp did not lower the matmul to f16");
    }
}
