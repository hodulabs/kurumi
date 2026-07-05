//! Grab-bag surface-coverage tests: broad op sweeps + end-to-end training/quantum.

use kurumi_core::*;

#[test]
fn low_value_ops_batch() {
    let mut g = Graph::new();
    // cumprod handles negatives (sign parity) and zeros (log0->exp0)
    let a = g.constant(vec![1., 2., 3., 4.], vec![4]);
    let cp = g.cumprod(a, 0).unwrap();
    for (got, want) in interpret(&g, cp).f32().iter().zip([1., 2., 6., 24.]) {
        assert!((got - want).abs() < 1e-3, "cumprod pos {:?}", interpret(&g, cp).f32());
    }
    let b = g.constant(vec![2., -3., 0., 5.], vec![4]);
    let cpb = g.cumprod(b, 0).unwrap();
    let cv = interpret(&g, cpb).f32().to_vec();
    for (got, want) in cv.iter().zip([2., -6., 0., 0.]) {
        assert!((got - want).abs() < 1e-4, "cumprod {cv:?}");
    }
    // pool3d + pool1d variants (via the general pool_nd helper)
    let vol = g.constant((0..8).map(|i| i as f32).collect(), vec![1, 1, 2, 2, 2]);
    let mp3 = g.max_pool3d(vol, (2, 2, 2), (1, 1, 1)).unwrap();
    assert_eq!(interpret(&g, mp3).f32(), &[7.]);
    let seq = g.constant(vec![1., 2., 3., 4.], vec![1, 1, 4]);
    let ap1 = g.avg_pool1d(seq, 2, 2).unwrap();
    assert_eq!(interpret(&g, ap1).f32(), &[1.5, 3.5]);
    // lrn with alpha=0 -> out = x / k^beta
    let li = g.constant(vec![2., 4.], vec![1, 2, 1, 1]);
    let ln = g.lrn(li, 1, 0.0, 1.0, 2.0).unwrap();
    assert_eq!(interpret(&g, ln).f32(), &[1., 2.]);
    // bicubic of a constant field is constant (cubic weights partition unity)
    let cst = g.constant(vec![5.0; 16], vec![1, 1, 4, 4]);
    let bc = g.resize_bicubic(cst, 8, 8).unwrap();
    assert_eq!(interpret(&g, bc).shape, vec![1, 1, 8, 8]);
    assert!(interpret(&g, bc).f32().iter().all(|&v| (v - 5.0).abs() < 1e-4), "bicubic const");
}

#[test]
fn surface_completeness_batch3() {
    let mut g = Graph::new();
    // bitwise_not on int: ~5 = -6 (i32)
    let xi = g.const_storage(Storage::I32(vec![5, 0, -1]), vec![3]);
    let bn = g.bitwise_not(xi).unwrap();
    let bnv = match interpret(&g, bn).storage {
        Storage::I32(v) => v,
        s => panic!("{s:?}"),
    };
    assert_eq!(bnv, vec![-6, -1, 0]);
    // min/sum pool2d on a 1x1x2x2
    let img = g.constant(vec![1., 4., 3., 2.], vec![1, 1, 2, 2]);
    let mp = g.min_pool2d(img, (2, 2), (1, 1)).unwrap();
    assert_eq!(interpret(&g, mp).f32(), &[1.]);
    let sp = g.sum_pool2d(img, (2, 2), (1, 1)).unwrap();
    assert_eq!(interpret(&g, sp).f32(), &[10.]);
    // upsample_nearest2d 2x: [[1,2],[3,4]] -> each pixel 2x2
    let s = g.constant(vec![1., 2., 3., 4.], vec![1, 1, 2, 2]);
    let up = g.upsample_nearest2d(s, 2).unwrap();
    assert_eq!(interpret(&g, up).shape, vec![1, 1, 4, 4]);
    assert_eq!(interpret(&g, up).f32(), &[1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.]);
}

#[test]
fn surface_completeness_batch2() {
    let mut g = Graph::new();
    // atan2 quadrants: (1,1)->pi/4, (1,-1)->3pi/4, (-1,-1)->-3pi/4
    let y = g.constant(vec![1., 1., -1.], vec![3]);
    let x = g.constant(vec![1., -1., -1.], vec![3]);
    let a2 = g.atan2(y, x).unwrap();
    let av = interpret(&g, a2).f32().to_vec();
    let pi = std::f32::consts::PI;
    for (got, want) in av.iter().zip([pi / 4., 3. * pi / 4., -3. * pi / 4.]) {
        assert!((got - want).abs() < 1e-4, "atan2 {av:?}");
    }
    // angle of a complex number: angle(1+i) = pi/4
    let z = {
        let r = g.constant(vec![1.], vec![1]);
        let i = g.constant(vec![1.], vec![1]);
        g.complex(r, i).unwrap()
    };
    let ang = g.angle(z).unwrap();
    assert!((interpret(&g, ang).f32()[0] - pi / 4.).abs() < 1e-4, "angle");
    // depth_to_space round-trips with space_to_depth
    let img = g.constant((0..2 * 4 * 2 * 2).map(|i| i as f32).collect(), vec![2, 4, 2, 2]); // [N=2, C*r^2=4, H=2, W=2], r=2, C=1
    let d2s = g.depth_to_space(img, 2).unwrap();
    assert_eq!(interpret(&g, d2s).shape, vec![2, 1, 4, 4]);
    let back = g.space_to_depth(d2s, 2).unwrap();
    assert_eq!(interpret(&g, back).f32(), interpret(&g, img).f32(), "d2s/s2d round-trip");
}

#[test]
fn surface_completeness_batch() {
    let mut g = Graph::new();
    // softsign: x/(1+|x|)
    let x = g.constant(vec![-3., 0., 1.], vec![3]);
    let ss = g.softsign(x);
    let sv = interpret(&g, ss).f32().to_vec();
    for (got, want) in sv.iter().zip([-3. / 4., 0., 1. / 2.]) {
        assert!((got - want).abs() < 1e-6, "softsign {sv:?}");
    }
    // selu: x>0 -> lambda*x ; here x=2 -> 1.0507*2
    let p = g.constant(vec![2.], vec![1]);
    let se = g.selu(p);
    assert!((interpret(&g, se).f32()[0] - 1.050_701 * 2.0).abs() < 1e-4);
    // celu(x>0)=x ; celu(x<0)=alpha(e^{x/alpha}-1)
    let cn = g.constant(vec![-1.], vec![1]);
    let ce = g.celu(cn, 1.0);
    assert!((interpret(&g, ce).f32()[0] - ((-1.0f32).exp() - 1.0)).abs() < 1e-5, "celu");
    // norm_p: 3-norm of [1,2,2] = (1+8+8)^(1/3) = 17^(1/3)
    let v = g.constant(vec![1., 2., 2.], vec![3]);
    let np = g.norm_p(v, 3.0, 0).unwrap();
    assert!((interpret(&g, np).f32()[0] - 17f32.powf(1.0 / 3.0)).abs() < 1e-3);
    // diag_embed: [1,2,3] -> diag
    let d = g.constant(vec![1., 2., 3.], vec![3]);
    let de = g.diag_embed(d).unwrap();
    assert_eq!(interpret(&g, de).shape, vec![3, 3]);
    assert_eq!(interpret(&g, de).f32(), &[1., 0., 0., 0., 2., 0., 0., 0., 3.]);
}

#[test]
fn training_converges() {
    // end-to-end learning: fit y = 2x+1 by gradient descent through the
    // Input/feeds param loop. Loss must drop sharply and (W,b) -> (2,1).
    let mut g = Graph::new();
    let n = 4usize;
    let x = g.constant(vec![0., 1., 2., 3.], vec![n, 1]);
    let target = g.constant(vec![1., 3., 5., 7.], vec![n, 1]); // 2x+1
    let w = g.input(vec![1, 1], DType::F32);
    let b = g.input(vec![1], DType::F32);
    let pred = {
        let xw = g.dot_general(x, w, vec![1], vec![0], vec![], vec![]).unwrap();
        let bb = g.broadcast_to(b, vec![n, 1]).unwrap();
        g.add(xw, bb).unwrap()
    };
    let loss = {
        let d = g.sub(pred, target).unwrap();
        let sq = g.square(d);
        g.mean_all(sq).unwrap()
    };
    let grads = grad(&mut g, loss, &[w, b]).unwrap();
    let (gw, gb) = (grads[0], grads[1]);
    let feeds_of = |wv: &[f32], bv: &[f32]| {
        Feeds::from([
            (w, TensorVal { shape: vec![1, 1], storage: Storage::F32(wv.to_vec()) }),
            (b, TensorVal { shape: vec![1], storage: Storage::F32(bv.to_vec()) }),
        ])
    };
    let (mut wv, mut bv, lr) = (vec![0.0f32], vec![0.0f32], 0.05f32);
    let l0 = interpret_with(&g, loss, &feeds_of(&wv, &bv)).f32()[0];
    for _ in 0..500 {
        let f = feeds_of(&wv, &bv);
        let dgw = interpret_with(&g, gw, &f).f32().to_vec();
        let dgb = interpret_with(&g, gb, &f).f32().to_vec();
        wv[0] -= lr * dgw[0];
        bv[0] -= lr * dgb[0];
    }
    let lf = interpret_with(&g, loss, &feeds_of(&wv, &bv)).f32()[0];
    assert!(lf < l0 * 0.001, "loss {l0} -> {lf}");
    assert!((wv[0] - 2.0).abs() < 0.05 && (bv[0] - 1.0).abs() < 0.05, "W={} b={}", wv[0], bv[0]);
}
