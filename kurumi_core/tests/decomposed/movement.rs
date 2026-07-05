use crate::approx;
use kurumi_core::*;

#[test]
fn shape_concat() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    // transpose / t
    let xt = g.t(x).unwrap();
    let o = interpret(&g, xt);
    assert_eq!(o.shape, vec![3, 2]);
    assert_eq!(o.f32(), &[1., 4., 2., 5., 3., 6.]);
    // unsqueeze / squeeze round-trip
    let u = g.unsqueeze(x, 1).unwrap();
    assert_eq!(interpret(&g, u).shape, vec![2, 1, 3]);
    let sq = g.squeeze(u, 1).unwrap();
    assert_eq!(interpret(&g, sq).shape, vec![2, 3]);
    // flatten
    let f = g.flatten(x).unwrap();
    assert_eq!(interpret(&g, f).shape, vec![6]);
    // concat along axis 0
    let a = g.constant(vec![1., 2.], vec![1, 2]);
    let b = g.constant(vec![3., 4., 5., 6.], vec![2, 2]);
    let c = g.concat(&[a, b], 0).unwrap();
    let oc = interpret(&g, c);
    assert_eq!(oc.shape, vec![3, 2]);
    assert_eq!(oc.f32(), &[1., 2., 3., 4., 5., 6.]);
    // stack
    let p = g.constant(vec![1., 2.], vec![2]);
    let q = g.constant(vec![3., 4.], vec![2]);
    let st = g.stack(&[p, q], 0).unwrap();
    let os = interpret(&g, st);
    assert_eq!(os.shape, vec![2, 2]);
    assert_eq!(os.f32(), &[1., 2., 3., 4.]);
    // split
    let parts = g.split(x, &[1, 2], 1).unwrap();
    assert_eq!(interpret(&g, parts[0]).shape, vec![2, 1]);
    assert_eq!(interpret(&g, parts[1]).f32(), &[2., 3., 5., 6.]);
}

#[test]
fn mask_tri_onehot() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3., 4., 5., 6., 7., 8., 9.], vec![3, 3]);
    // tril keeps lower triangle (incl diagonal)
    let lo = g.tril(x, 0).unwrap();
    assert_eq!(interpret(&g, lo).f32(), &[1., 0., 0., 4., 5., 0., 7., 8., 9.]);
    let up = g.triu(x, 0).unwrap();
    assert_eq!(interpret(&g, up).f32(), &[1., 2., 3., 0., 5., 6., 0., 0., 9.]);
    // clamp_min / clamp_max
    let v = g.constant(vec![-1., 0.5, 2.0], vec![3]);
    let cmin = g.clamp_min(v, 0.0).unwrap();
    let cmax = g.clamp_max(v, 1.0).unwrap();
    approx(&g, cmin, &[0., 0.5, 2.0]);
    approx(&g, cmax, &[-1., 0.5, 1.0]);
    // onehot
    let idx = g.const_storage(Storage::I64(vec![0, 2, 1]), vec![3]);
    let oh = g.onehot(idx, 3).unwrap();
    let ooh = interpret(&g, oh);
    assert_eq!(ooh.shape, vec![3, 3]);
    assert_eq!(ooh.f32(), &[1., 0., 0., 0., 0., 1., 0., 1., 0.]);
    // log_softmax sums (in exp) to 1
    let lg = g.constant(vec![1., 2., 3.], vec![3]);
    let ls = g.log_softmax(lg, 0).unwrap();
    let sm: f32 = interpret(&g, ls).f32().iter().map(|v| v.exp()).sum();
    assert!((sm - 1.0).abs() < 1e-5, "sum exp(log_softmax) = {sm}");
}

#[test]
fn strided_slice_fwd_bwd() {
    // forward: a[0:6:2] -> [0,2,4]; a[1:6:2] -> [1,3,5]
    let mut g = Graph::new();
    let a = g.constant((0..6).map(|i| i as f32).collect(), vec![6]);
    let s0 = g.slice_step(a, vec![(0, 6, 2)]).unwrap();
    assert_eq!(interpret(&g, s0).shape, vec![3]);
    assert_eq!(interpret(&g, s0).f32(), &[0., 2., 4.]);
    // realize (view) path agrees with the oracle
    assert_eq!(kurumi_core::realize::force(&g, s0).f32().to_vec(), vec![0., 2., 4.]);
    let s1 = g.slice_step(a, vec![(1, 6, 2)]).unwrap();
    assert_eq!(interpret(&g, s1).f32(), &[1., 3., 5.]);
    // 2D strided slice [::2, ::2] of a 4x4
    let m = g.constant((0..16).map(|i| i as f32).collect(), vec![4, 4]);
    let sm = g.slice_step(m, vec![(0, 4, 2), (0, 4, 2)]).unwrap();
    let osm = interpret(&g, sm);
    assert_eq!(osm.shape, vec![2, 2]);
    assert_eq!(osm.f32(), &[0., 2., 8., 10.]);

    // backward: d/da sum(a[0:6:2]) = 1 at even positions, 0 at odd
    let mut g2 = Graph::new();
    let x = g2.constant((0..6).map(|i| i as f32).collect(), vec![6]);
    let sl = g2.slice_step(x, vec![(0, 6, 2)]).unwrap();
    let loss = g2.sum(sl, 0).unwrap();
    let gx = grad(&mut g2, loss, &[x]).unwrap()[0];
    assert_eq!(interpret(&g2, gx).f32(), &[1., 0., 1., 0., 1., 0.]);
}

#[test]
fn manip() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3.], vec![3]);
    // tile vs repeat_interleave
    let t = g.tile(x, 0, 2).unwrap();
    assert_eq!(interpret(&g, t).f32(), &[1., 2., 3., 1., 2., 3.]);
    let ri = g.repeat_interleave(x, 0, 2).unwrap();
    assert_eq!(interpret(&g, ri).f32(), &[1., 1., 2., 2., 3., 3.]);
    // roll
    let r = g.roll(x, 1, 0).unwrap();
    assert_eq!(interpret(&g, r).f32(), &[3., 1., 2.]);
    // diagonal + trace of a 3x3
    let m = g.constant((1..=9).map(|i| i as f32).collect(), vec![3, 3]);
    let d = g.diagonal(m).unwrap();
    assert_eq!(interpret(&g, d).f32(), &[1., 5., 9.]);
    let tr = g.trace(m).unwrap();
    assert_eq!(interpret(&g, tr).f32(), &[15.]);
    // sum_all / mean_all / prod_all
    let sa = g.sum_all(m).unwrap();
    assert_eq!(interpret(&g, sa).f32(), &[45.]);
    let ma = g.mean_all(m).unwrap();
    assert_eq!(interpret(&g, ma).f32(), &[5.]);
    let v = g.constant(vec![1., 2., 3., 4.], vec![4]);
    let pa = g.prod_all(v).unwrap();
    assert_eq!(interpret(&g, pa).f32(), &[24.]);
    // broadcast_to
    let bt = g.broadcast_to(x, vec![2, 3]).unwrap();
    assert_eq!(interpret(&g, bt).f32(), &[1., 2., 3., 1., 2., 3.]);
}

#[test]
fn pad_mode_cases() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3., 4.], vec![4]);
    // reflect (2,2): mirror about the edges, excluding them
    let re = g.pad_mode(x, vec![(2, 2)], "reflect").unwrap();
    assert_eq!(interpret(&g, re).f32(), &[3., 2., 1., 2., 3., 4., 3., 2.]);
    // replicate (2,1): repeat the edge element
    let rp = g.pad_mode(x, vec![(2, 1)], "replicate").unwrap();
    assert_eq!(interpret(&g, rp).f32(), &[1., 1., 1., 2., 3., 4., 4.]);
    // circular (2,2): wrap around
    let ci = g.pad_mode(x, vec![(2, 2)], "circular").unwrap();
    assert_eq!(interpret(&g, ci).f32(), &[3., 4., 1., 2., 3., 4., 1., 2.]);
    // 2-d, pad last axis only, reflect
    let m = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let m2 = g.pad_mode(m, vec![(0, 0), (1, 1)], "reflect").unwrap();
    assert_eq!(interpret(&g, m2).shape, vec![2, 5]);
    assert_eq!(interpret(&g, m2).f32(), &[2., 1., 2., 3., 2., 5., 4., 5., 6., 5.]);
    // grad counts how often each input feeds the padded output
    let loss = g.sum(re, 0).unwrap();
    let gx = grad(&mut g, loss, &[x]).unwrap()[0];
    // out=[x2,x1,x0,x1,x2,x3,x2,x1] -> counts x0:1 x1:3 x2:3 x3:1
    assert_eq!(interpret(&g, gx).f32(), &[1., 3., 3., 1.]);
}
