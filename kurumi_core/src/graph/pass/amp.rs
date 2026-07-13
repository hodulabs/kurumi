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
    use crate::{Graph, grad, interpret};

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

    #[test]
    fn amp_then_grad_is_finite_and_matches_f32() {
        // amp runs before grad, so lowering the forward matmuls to f16 also makes the
        // backward matmuls f16. build a training-shaped graph (two matmul+gelu layers into
        // a scalar loss = sum of the last matmul) and check the param grads of the amped
        // graph are finite and close to the f32 grads within f16 tolerance.
        let mut g = Graph::new();
        let x = g.constant((0..24).map(|i| (i as f32 * 0.1).sin() * 0.5).collect(), vec![4, 6]);
        let w1 = g.constant((0..30).map(|i| (i as f32 * 0.2).cos() * 0.3).collect(), vec![6, 5]);
        let w2 = g.constant((0..5).map(|i| (i as f32 * 0.3).sin() * 0.4).collect(), vec![5, 1]);
        let params = [w1, w2];

        let h = g.dot_general(x, w1, vec![1], vec![0], vec![], vec![]).unwrap();
        let a = g.gelu(h);
        let loss = g.dot_general(a, w2, vec![1], vec![0], vec![], vec![]).unwrap(); // [4,1]; grad sums it

        // baseline grads with the graph left in f32.
        let base: Vec<Vec<f32>> =
            grad(&mut g, loss, &params).unwrap().iter().map(|&gn| interpret(&g, gn).f32().to_vec()).collect();

        // amp the forward loss, then differentiate the amped (f16-matmul) graph.
        let amped = amp(&mut g, loss);
        let got: Vec<Vec<f32>> =
            grad(&mut g, amped, &params).unwrap().iter().map(|&gn| interpret(&g, gn).f32().to_vec()).collect();

        for (p, (b, a)) in base.iter().zip(&got).enumerate() {
            for (j, (bv, av)) in b.iter().zip(a).enumerate() {
                assert!(av.is_finite(), "param{p}[{j}] grad not finite: {av}");
                let tol = 3e-3 + 2e-2 * bv.abs();
                assert!((av - bv).abs() <= tol, "param{p}[{j}] amp grad {av} vs f32 {bv}");
            }
        }
    }
}
