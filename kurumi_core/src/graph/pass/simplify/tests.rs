use crate::graph::pass::simplify::*;
use crate::{Graph, grad, interpret, node_count};

// every rule is checked the same way: the rewritten root evaluates to the same
// values as the original (the oracle), and the graph is no larger.
fn f32s(g: &Graph, id: NodeId) -> Vec<f32> {
    interpret(g, id).storage.into_f32()
}

#[test]
fn double_neg_cancels() {
    let mut g = Graph::new();
    let x = g.constant(vec![1.0, -2.0, 3.0], vec![3]);
    let z = g.neg(x);
    let z = g.neg(z); // neg(neg(x)) = x
    let want = f32s(&g, z);
    let s = simplify(&mut g, z);
    assert_eq!(s, x); // collapsed straight to x
    assert_eq!(f32s(&g, s), want);
}

#[test]
fn mul_by_ones_and_add_zero_drop() {
    let mut g = Graph::new();
    let x = g.constant(vec![1.0, 2.0, 3.0], vec![3]);
    let ones = g.constant(vec![1.0; 3], vec![3]);
    let zeros = g.constant(vec![0.0; 3], vec![3]);
    let y = g.mul(x, ones).unwrap();
    let y = g.add(zeros, y).unwrap();
    let want = f32s(&g, y);
    let s = simplify(&mut g, y);
    assert_eq!(s, x);
    assert_eq!(f32s(&g, s), want);
}

#[test]
fn movement_chains_collapse() {
    let mut g = Graph::new();
    let x = g.constant((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);
    let r = g.reshape(x, vec![6, 4]).unwrap();
    let r = g.reshape(r, vec![24]).unwrap(); // reshape after reshape -> reshape(x,[24])
    let want = f32s(&g, r);
    let s = simplify(&mut g, r);
    assert_eq!(f32s(&g, s), want);
    assert!(node_count(&g, s) < node_count(&g, r));
}

#[test]
fn double_transpose_cancels() {
    let mut g = Graph::new();
    let x = g.constant((0..6).map(|i| i as f32).collect(), vec![2, 3]);
    let t = g.transpose(x, 0, 1).unwrap();
    let t = g.transpose(t, 0, 1).unwrap(); // permute after permute = identity
    let want = f32s(&g, t);
    let s = simplify(&mut g, t);
    assert_eq!(f32s(&g, s), want);
    assert_eq!(node_count(&g, s), 1); // just x
}

#[test]
fn cse_merges_identical_subgraphs() {
    let mut g = Graph::new();
    let x = g.constant(vec![1.0, 2.0], vec![2]);
    let a = g.neg(x);
    let b = g.neg(x); // structurally identical to a
    let y = g.add(a, b).unwrap();
    let before = node_count(&g, y); // x, a, b, y = 4
    let s = simplify(&mut g, y);
    assert!(node_count(&g, s) < before); // a and b merged
    assert_eq!(f32s(&g, s), f32s(&g, y));
}

// simplify shrinks a real backward graph and preserves the gradient.
#[test]
fn simplify_shrinks_backward_matches_oracle() {
    let mut g = Graph::new();
    let x = g.constant((0..12).map(|i| (i as f32 * 0.1).sin()).collect(), vec![3, 4]);
    let w = g.constant((0..8).map(|i| (i as f32 * 0.2).cos()).collect(), vec![4, 2]);
    let h = g.dot_general(x, w, vec![1], vec![0], vec![], vec![]).unwrap();
    let a = g.relu(h);
    let loss = g.sum(a, 0).unwrap();
    let loss = g.sum(loss, 0).unwrap(); // scalar
    let gw = grad(&mut g, loss, &[w]).unwrap()[0];

    let before = node_count(&g, gw);
    let want = f32s(&g, gw);
    let s = simplify(&mut g, gw);
    let after = node_count(&g, s);
    assert_eq!(f32s(&g, s), want, "simplify changed the gradient");
    assert!(after <= before);
    eprintln!("backward nodes: {before} -> {after} ({} removed)", before - after);
}
