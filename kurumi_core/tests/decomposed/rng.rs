use kurumi_core::*;

#[test]
fn rng_uniform_normal_dropout() {
    let mut g = Graph::new();
    let n = 10000usize;
    // uniform: range [0,1), mean ~0.5
    let u = g.rand_uniform(vec![n], 42);
    let uv = interpret(&g, u).f32().to_vec();
    assert!(uv.iter().all(|&v| (0.0..1.0).contains(&v)), "uniform out of [0,1)");
    let umean: f32 = uv.iter().sum::<f32>() / n as f32;
    assert!((umean - 0.5).abs() < 0.02, "uniform mean {umean}");
    // reproducible: same seed -> identical; different seed -> different
    let u2 = g.rand_uniform(vec![n], 42);
    assert_eq!(interpret(&g, u2).f32().to_vec(), uv);
    let u3 = g.rand_uniform(vec![n], 43);
    assert_ne!(interpret(&g, u3).f32().to_vec(), uv);
    // normal: mean ~0, std ~1
    let z = g.randn(vec![n], 7);
    let zv = interpret(&g, z).f32().to_vec();
    let zmean: f32 = zv.iter().sum::<f32>() / n as f32;
    let zvar: f32 = zv.iter().map(|v| (v - zmean).powi(2)).sum::<f32>() / n as f32;
    assert!(zmean.abs() < 0.05, "normal mean {zmean}");
    assert!((zvar.sqrt() - 1.0).abs() < 0.05, "normal std {}", zvar.sqrt());
    // bernoulli p=0.3 -> mean ~0.3
    let b = g.bernoulli(vec![n], 9, 0.3).unwrap();
    let bmean: f32 = interpret(&g, b).f32().iter().sum::<f32>() / n as f32;
    assert!((bmean - 0.3).abs() < 0.02, "bernoulli mean {bmean}");
    // dropout p=0.5: ~half zeros, survivors scaled by 2; mean preserved
    let x = g.constant(vec![1.0; n], vec![n]);
    let d = g.dropout(x, 0.5, 11).unwrap();
    let dv = interpret(&g, d).f32().to_vec();
    let zeros = dv.iter().filter(|&&v| v == 0.0).count();
    assert!((zeros as f32 / n as f32 - 0.5).abs() < 0.03, "dropout zero frac {}", zeros as f32 / n as f32);
    let dmean: f32 = dv.iter().sum::<f32>() / n as f32;
    assert!((dmean - 1.0).abs() < 0.05, "dropout mean (should preserve) {dmean}");
    // randint in [3,10)
    let ri = g.randint(vec![n], 5, 3, 10);
    match interpret(&g, ri).storage {
        Storage::I64(v) => assert!(v.iter().all(|&x| (3..10).contains(&x)), "randint out of range"),
        s => panic!("want i64, got {:?}", s.dtype()),
    }
}
