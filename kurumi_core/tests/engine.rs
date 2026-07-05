//! Core engine integration tests (public API): IR ops, reductions, movement, matmul,
//! and realize-vs-interpret parity. dtype/cast -> dtype.rs; primitives & decomposed
//! ops -> ops.rs / decomposed.rs.

use kurumi_core::*;

#[test]
fn add_const() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3.], vec![3]);
    let b = g.constant(vec![4., 5., 6.], vec![3]);
    let y = g.add(a, b).unwrap();
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![3], storage: Storage::F32(vec![5., 7., 9.]) });
}

#[test]
fn mul_neg_sum_chain() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let b = g.constant(vec![1., 1., 1., 2., 2., 2.], vec![2, 3]);
    let p = g.mul(a, b).unwrap();
    let np = g.neg(p);
    let y = g.sum(np, 1).unwrap();
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![2], storage: Storage::F32(vec![-6., -30.]) });
}

#[test]
fn sum_to_scalar() {
    let mut g = Graph::new();
    let a = g.constant(vec![2., 3., 5.], vec![3]);
    let y = g.sum(a, 0).unwrap();
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![], storage: Storage::F32(vec![10.]) });
}

#[test]
fn permute_transpose() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let y = g.permute(a, vec![1, 0]).unwrap();
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![3, 2], storage: Storage::F32(vec![1., 4., 2., 5., 3., 6.]) });
}

#[test]
fn expand_broadcast_row() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2.], vec![1, 2]);
    let y = g.expand(a, vec![3, 2]).unwrap();
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![3, 2], storage: Storage::F32(vec![1., 2., 1., 2., 1., 2.]) });
}

// 2D matmul as dot_general(contract = (1,0), batch = ())
#[test]
fn dot_general_2d() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let b = g.constant(vec![1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.], vec![3, 4]);
    let y = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
    assert_eq!(
        interpret(&g, y),
        TensorVal { shape: vec![2, 4], storage: Storage::F32(vec![1., 2., 3., 0., 4., 5., 6., 0.]) }
    );
}

// batched contraction: [2,1,2] x [2,2,1] over batch axis 0 -> [2,1,1]
#[test]
fn dot_general_batched() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3., 4.], vec![2, 1, 2]);
    let b = g.constant(vec![5., 6., 7., 8.], vec![2, 2, 1]);
    let y = g.dot_general(a, b, vec![2], vec![1], vec![0], vec![0]).unwrap();
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![2, 1, 1], storage: Storage::F32(vec![17., 53.]) });
}

#[test]
fn mlp_forward_relu() {
    // y = relu(x @ w + b),  x:[2,3] w:[3,4] b:[4]
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let w = g.constant(vec![1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.], vec![3, 4]);
    let b = g.constant(vec![0., 0., 0., -10.], vec![4]);

    let xw = g.dot_general(x, w, vec![1], vec![0], vec![], vec![]).unwrap(); // [2,4]
    let b = g.reshape(b, vec![1, 4]).unwrap();
    let b = g.expand(b, vec![2, 4]).unwrap();
    let z = g.add(xw, b).unwrap();
    let zero = g.constant(vec![0.; 8], vec![2, 4]);
    let y = g.max(z, zero).unwrap();

    let out = interpret(&g, y);
    assert_eq!(out.shape, vec![2, 4]);
    assert_eq!(out.f32().to_vec(), vec![1., 2., 3., 0., 4., 5., 6., 0.]);
}

#[test]
fn errors_point_at_record_time() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2.], vec![2]);
    let b = g.constant(vec![1., 2., 3.], vec![3]);
    assert!(matches!(g.add(a, b), Err(Error::Shape { op: "add", .. })));
    assert!(matches!(g.sum(a, 5), Err(Error::Shape { op: "sum", .. })));
    assert!(matches!(g.reshape(a, vec![5]), Err(Error::Shape { op: "reshape", .. })));
    assert!(matches!(g.permute(a, vec![0, 0]), Err(Error::Shape { op: "permute", .. })));
    // contract dim 2 vs 3 mismatch
    assert!(matches!(
        g.dot_general(a, b, vec![0], vec![0], vec![], vec![]),
        Err(Error::Shape { op: "dot_general", .. })
    ));
    assert!(matches!(g.slice(a, vec![(0, 5)]), Err(Error::Shape { op: "slice", .. })));
    assert!(matches!(g.flip(a, vec![3]), Err(Error::Shape { op: "flip", .. })));
}

#[test]
fn slice_basic() {
    let mut g = Graph::new();
    let a = g.constant((0..12).map(|x| x as f32).collect(), vec![3, 4]);
    let y = g.slice(a, vec![(1, 3), (1, 3)]).unwrap();
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![2, 2], storage: Storage::F32(vec![5., 6., 9., 10.]) });
}

#[test]
fn flip_basic() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let y = g.flip(a, vec![1]).unwrap();
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![2, 3], storage: Storage::F32(vec![3., 2., 1., 6., 5., 4.]) });
}

#[test]
fn pad_basic() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3.], vec![3]);
    let y = g.pad(a, vec![(1, 2)]).unwrap();
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![6], storage: Storage::F32(vec![0., 1., 2., 3., 0., 0.]) });
}

#[test]
fn unary_primitives() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 4., 9.], vec![3]);
    let s = g.sqrt(a);
    assert_eq!(interpret(&g, s).f32().to_vec(), vec![1., 2., 3.]);
    let b = g.constant(vec![0., 1., 3.], vec![3]);
    let e = g.exp2(b);
    assert_eq!(interpret(&g, e).f32().to_vec(), vec![1., 2., 8.]);
}

#[test]
fn reduce_max_basic() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 5., 2., 8., 3., 0.], vec![2, 3]);
    let m = g.reduce_max(a, 1).unwrap();
    assert_eq!(interpret(&g, m), TensorVal { shape: vec![2], storage: Storage::F32(vec![5., 8.]) });
}

#[test]
fn exp_matches_reference() {
    let mut g = Graph::new();
    let x = g.constant(vec![0., 1., 2.], vec![3]);
    let y = g.exp(x);
    let got = interpret(&g, y).f32().to_vec();
    let expect = [1f32, 1f32.exp(), 2f32.exp()];
    for (a, b) in got.iter().zip(&expect) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

fn ref_softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

#[test]
fn softmax_matches_reference() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3., 0., 0., 0.], vec![2, 3]);
    let y = g.softmax(x, 1).unwrap();
    let got = interpret(&g, y).f32().to_vec();
    // fused path agrees with the oracle
    assert_eq!(kurumi_core::realize::force(&g, y).f32().to_vec(), got);
    // and matches a direct per-row softmax within 1e-5
    let mut expect = ref_softmax(&[1., 2., 3.]);
    expect.extend(ref_softmax(&[0., 0., 0.]));
    for (a, b) in got.iter().zip(&expect) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}

#[test]
fn gelu_matches_reference() {
    let mut g = Graph::new();
    let xs = vec![-2., -0.5, 0., 0.5, 2.];
    let x = g.constant(xs.clone(), vec![5]);
    let y = g.gelu(x);
    let got = interpret(&g, y).f32().to_vec();
    for (&xv, &gv) in xs.iter().zip(&got) {
        let k = (2.0_f32 / std::f32::consts::PI).sqrt();
        let r = 0.5 * xv * (1.0 + (k * (xv + 0.044715 * xv.powi(3))).tanh());
        assert!((gv - r).abs() < 1e-5, "{gv} vs {r}");
    }
}

#[test]
fn layernorm_matches_reference() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3., 4., 10., 0., -10., 0.], vec![2, 4]);
    let y = g.layernorm(x, 1, 1e-5).unwrap();
    let got = interpret(&g, y).f32().to_vec();
    let ref_ln = |row: &[f32]| -> Vec<f32> {
        let n = row.len() as f32;
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std = (var + 1e-5).sqrt();
        row.iter().map(|v| (v - mean) / std).collect()
    };
    let mut expect = ref_ln(&[1., 2., 3., 4.]);
    expect.extend(ref_ln(&[10., 0., -10., 0.]));
    for (a, b) in got.iter().zip(&expect) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

// full GPT-2-style block (single head): realize (fused) matches the oracle
#[test]
fn transformer_block_matches_oracle() {
    fn w(g: &mut Graph, rows: usize, cols: usize, seed: f32) -> NodeId {
        let data = (0..rows * cols).map(|i| ((i as f32 + 1.0) * seed).sin() * 0.1).collect();
        g.constant(data, vec![rows, cols])
    }
    let (s, d, h) = (2usize, 4usize, 8usize);
    let mut g = Graph::new();
    let x = g.constant((0..s * d).map(|i| i as f32 * 0.1).collect(), vec![s, d]);
    let (wq, wk, wv, wo) = (w(&mut g, d, d, 0.3), w(&mut g, d, d, 0.5), w(&mut g, d, d, 0.7), w(&mut g, d, d, 0.9));
    let (w1, w2) = (w(&mut g, d, h, 0.2), w(&mut g, h, d, 0.4));

    let q = g.dot_general(x, wq, vec![1], vec![0], vec![], vec![]).unwrap();
    let k = g.dot_general(x, wk, vec![1], vec![0], vec![], vec![]).unwrap();
    let v = g.dot_general(x, wv, vec![1], vec![0], vec![], vec![]).unwrap();
    let scores = g.dot_general(q, k, vec![1], vec![1], vec![], vec![]).unwrap();
    let scale = g.scalar(scores, 1.0 / (d as f32).sqrt());
    let scaled = g.mul(scores, scale).unwrap();
    let attn = g.softmax(scaled, 1).unwrap();
    let ctx = g.dot_general(attn, v, vec![1], vec![0], vec![], vec![]).unwrap();
    let proj = g.dot_general(ctx, wo, vec![1], vec![0], vec![], vec![]).unwrap();
    let ln1 = g.add(x, proj).unwrap();
    let ln1 = g.layernorm(ln1, 1, 1e-5).unwrap();
    let hpre = g.dot_general(ln1, w1, vec![1], vec![0], vec![], vec![]).unwrap();
    let hact = g.gelu(hpre);
    let mlp = g.dot_general(hact, w2, vec![1], vec![0], vec![], vec![]).unwrap();
    let res2 = g.add(ln1, mlp).unwrap();
    let out = g.layernorm(res2, 1, 1e-5).unwrap();

    let oracle = interpret(&g, out);
    let fused = kurumi_core::realize::force(&g, out);
    assert_eq!(fused.shape, vec![s, d]);
    for (a, b) in fused.f32().to_vec().iter().zip(&oracle.f32().to_vec()) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}
