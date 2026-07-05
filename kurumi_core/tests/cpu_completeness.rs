//! Coverage probes from the engine audit: matmul shapes (1D dot, 1xN, rank-N@2D),
//! integer ops, mixed-dtype autograd, min/clamp.
use kurumi_core::*;

#[test]
fn probe_vector_dot() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3.], vec![3]);
    let b = g.constant(vec![4., 5., 6.], vec![3]);
    let d = g.dot_general(a, b, vec![0], vec![0], vec![], vec![]).unwrap();
    let o = interpret(&g, d);
    assert_eq!(o.shape.len(), 0, "vector dot -> scalar, got {:?}", o.shape);
    assert!((o.f32()[0] - 32.0).abs() < 1e-5);
}

#[test]
fn probe_matmul_1x2_2x3() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2.], vec![1, 2]);
    let b = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let m = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
    let o = interpret(&g, m);
    assert_eq!(o.shape, vec![1, 3]);
    assert_eq!(o.f32(), &[9., 12., 15.]);
}

#[test]
fn probe_rank3_at_2d() {
    let mut g = Graph::new();
    let a = g.constant((0..2 * 4 * 3).map(|i| i as f32).collect(), vec![2, 4, 3]);
    let w = g.constant((0..3 * 5).map(|i| i as f32).collect(), vec![3, 5]);
    let m = g.dot_general(a, w, vec![2], vec![0], vec![], vec![]).unwrap();
    let o = interpret(&g, m);
    assert_eq!(o.shape, vec![2, 4, 5]);
}

#[test]
fn probe_int_matmul_and_add() {
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![1, 2, 3, 4]), vec![2, 2]);
    let b = g.const_storage(Storage::I32(vec![1, 0, 0, 1]), vec![2, 2]);
    let m = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
    assert_eq!(interpret(&g, m).storage, Storage::I32(vec![1, 2, 3, 4]));
    let s = g.add(a, b).unwrap();
    assert_eq!(interpret(&g, s).storage, Storage::I32(vec![2, 2, 3, 5]));
}

#[test]
fn probe_min_clamp() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 5., 3.], vec![3]);
    let b = g.constant(vec![4., 2., 3.], vec![3]);
    let mn = g.min(a, b).unwrap();
    assert_eq!(interpret(&g, mn).f32(), &[1., 2., 3.]);
    let cl = g.clamp(a, 2.0, 4.0).unwrap();
    assert_eq!(interpret(&g, cl).f32(), &[2., 4., 3.]); // clamp([1,5,3], 2, 4)
}

#[test]
fn probe_fp8_cast_arith_matmul() {
    use float8::F8E4M3;
    let f8 = |xs: Vec<f32>| Storage::F8E4M3(xs.into_iter().map(F8E4M3::from_f32).collect());
    let mut g = Graph::new();
    // f32 -> fp8 -> f32 roundtrip on exactly-representable values
    let x = g.constant(vec![0.5, 1.0, 2.0, -1.5], vec![4]);
    let x8 = g.cast(x, DType::F8E4M3);
    assert_eq!(g.dtype(x8), DType::F8E4M3);
    let back = g.cast(x8, DType::F32);
    assert_eq!(interpret(&g, back).f32(), &[0.5, 1.0, 2.0, -1.5]);
    // fp8 arithmetic (upcast-compute-round) + fp8 matmul
    let a = g.const_storage(f8(vec![1.0, 2.0]), vec![2]);
    let b = g.const_storage(f8(vec![0.5, 1.0]), vec![2]);
    let ab = g.add(a, b).unwrap();
    let sum = g.cast(ab, DType::F32);
    assert_eq!(interpret(&g, sum).f32(), &[1.5, 3.0]);
    let dot = g.dot_general(a, b, vec![0], vec![0], vec![], vec![]).unwrap(); // 1*0.5 + 2*1 = 2.5
    let dot32 = g.cast(dot, DType::F32);
    assert!((interpret(&g, dot32).f32()[0] - 2.5).abs() < 0.1);
    // fp8 float op (sqrt upcasts to f32, rounds): sqrt(4) = 2
    let four = g.const_storage(f8(vec![4.0]), vec![1]);
    let r = g.sqrt(four);
    let r32 = g.cast(r, DType::F32);
    assert!((interpret(&g, r32).f32()[0] - 2.0).abs() < 0.1);
}

#[test]
fn probe_fp8_grad() {
    use float8::F8E4M3;
    let mut g = Graph::new();
    let w8 = g.const_storage(Storage::F8E4M3(vec![F8E4M3::from_f32(0.5); 4]), vec![2, 2]);
    let w = g.cast(w8, DType::F32);
    let x = g.constant(vec![1., 2., 3., 4.], vec![2, 2]);
    let y = g.mul(w, x).unwrap();
    let s1 = g.sum(y, 1).unwrap();
    let loss = g.sum(s1, 0).unwrap();
    let gw = grad(&mut g, loss, &[w8]).unwrap()[0];
    assert_eq!(g.dtype(gw), DType::F8E4M3); // grad of an fp8 param is fp8
    interpret(&g, gw); // must not panic
}

#[test]
fn probe_mixed_dtype_grad() {
    use half::f16;
    let mut g = Graph::new();
    let w16 = g.const_storage(Storage::F16(vec![f16::from_f32(0.5); 6]), vec![2, 3]);
    let w = g.cast(w16, DType::F32);
    let x = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
    let m = g.dot_general(w, x, vec![1], vec![0], vec![], vec![]).unwrap();
    let s1 = g.sum(m, 1).unwrap();
    let loss = g.sum(s1, 0).unwrap();
    let gw = grad(&mut g, loss, &[w16]).unwrap()[0];
    assert_eq!(g.dtype(gw), DType::F16, "grad of f16 param should be f16");
    assert_eq!(g.shape(gw), vec![2, 3]);
    interpret(&g, gw); // must not panic
}
