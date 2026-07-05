use crate::approx;
use kurumi_core::*;

#[test]
fn unary() {
    let mut g = Graph::new();
    let x = g.constant(vec![-2.0, -0.5, 0.0, 1.5, 3.0], vec![5]);
    let a = g.abs(x);
    approx(&g, a, &[2.0, 0.5, 0.0, 1.5, 3.0]);
    let r = g.relu(x);
    approx(&g, r, &[0.0, 0.0, 0.0, 1.5, 3.0]);
    let s = g.sign(x);
    approx(&g, s, &[-1.0, -1.0, 0.0, 1.0, 1.0]);
    let sq = g.square(x);
    approx(&g, sq, &[4.0, 0.25, 0.0, 2.25, 9.0]);
    let c = g.ceil(x);
    approx(&g, c, &[-2.0, -0.0, 0.0, 2.0, 3.0]);
    let rd = g.round(x);
    approx(&g, rd, &[-2.0, -0.0, 0.0, 2.0, 3.0]); // round(-0.5)=floor(0)=0
}

#[test]
fn trig_hyper() {
    let mut g = Graph::new();
    let x = g.constant(vec![0.0, std::f32::consts::PI], vec![2]);
    let c = g.cos(x);
    approx(&g, c, &[1.0, -1.0]);
    let x2 = g.constant(vec![0.0, 1.0], vec![2]);
    let sh = g.sinh(x2);
    approx(&g, sh, &[0.0, 1.0f32.sinh()]);
    let ch = g.cosh(x2);
    approx(&g, ch, &[1.0, 1.0f32.cosh()]);
    // asinh(sinh(t)) = t round-trip
    let t = g.constant(vec![0.3, -1.2], vec![2]);
    let st = g.sinh(t);
    let back = g.asinh(st);
    approx(&g, back, &[0.3, -1.2]);
}

#[test]
fn activations() {
    let mut g = Graph::new();
    let x = g.constant(vec![-1.0, 0.0, 2.0], vec![3]);
    let sp = g.softplus(x);
    approx(&g, sp, &[(1.0 + (-1.0f32).exp()).ln(), 2.0f32.ln(), (1.0 + 2.0f32.exp()).ln()]);
    let lr = g.leaky_relu(x, 0.1);
    approx(&g, lr, &[-0.1, 0.0, 2.0]);
    let e = g.elu(x, 1.0);
    approx(&g, e, &[(-1.0f32).exp() - 1.0, 0.0, 2.0]);
    let hs = g.hardsigmoid(x);
    approx(&g, hs, &[2.0 / 6.0, 0.5, 5.0 / 6.0]);
}

#[test]
fn reductions() {
    let mut g = Graph::new();
    let x = g.constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let m = g.mean(x, 1).unwrap();
    approx(&g, m, &[1.5, 3.5]);
    let mn = g.reduce_min(x, 1).unwrap();
    approx(&g, mn, &[1.0, 3.0]);
    let v = g.var(x, 1).unwrap();
    approx(&g, v, &[0.25, 0.25]);
    let sd = g.std(x, 1).unwrap();
    approx(&g, sd, &[0.5, 0.5]);
    let l2 = g.l2_norm(x, 1).unwrap();
    approx(&g, l2, &[5.0f32.sqrt(), 25.0f32.sqrt()]);
    let l1 = g.l1_norm(x, 1).unwrap();
    approx(&g, l1, &[3.0, 7.0]);
    // logsumexp matches a manual computation
    let lse = g.logsumexp(x, 1).unwrap();
    let want0 = (1.0f32.exp() + 2.0f32.exp()).ln();
    let want1 = (3.0f32.exp() + 4.0f32.exp()).ln();
    approx(&g, lse, &[want0, want1]);
}

#[test]
fn compare_bool() {
    let mut g = Graph::new();
    let a = g.constant(vec![1.0, 2.0, 3.0], vec![3]);
    let b = g.constant(vec![2.0, 2.0, 1.0], vec![3]);
    let asbool = |g: &Graph, y: NodeId| -> Vec<bool> {
        match interpret(g, y).storage {
            Storage::BOOL(v) => v,
            s => panic!("want bool, got {:?}", s.dtype()),
        }
    };
    let gt = g.gt(a, b).unwrap();
    assert_eq!(asbool(&g, gt), vec![false, false, true]);
    let ge = g.ge(a, b).unwrap();
    assert_eq!(asbool(&g, ge), vec![false, true, true]);
    let le = g.le(a, b).unwrap();
    assert_eq!(asbool(&g, le), vec![true, true, false]);
    let ne = g.ne(a, b).unwrap();
    assert_eq!(asbool(&g, ne), vec![true, false, true]);
    // isnan / isinf / isfinite
    let x = g.constant(vec![0.0, f32::NAN, f32::INFINITY], vec![3]);
    let n = g.isnan(x).unwrap();
    let i = g.isinf(x).unwrap();
    let f = g.isfinite(x).unwrap();
    assert_eq!(asbool(&g, n), vec![false, true, false]);
    assert_eq!(asbool(&g, i), vec![false, false, true]);
    assert_eq!(asbool(&g, f), vec![true, false, false]);
}

#[test]
fn rem_pow() {
    let mut g = Graph::new();
    let a = g.constant(vec![5.0, 7.5, -1.0], vec![3]);
    let b = g.constant(vec![3.0, 2.0, 3.0], vec![3]);
    let r = g.rem(a, b).unwrap();
    approx(&g, r, &[2.0, 1.5, 2.0]); // floor-based: -1 - floor(-1/3)*3 = -1-(-1)*3 = 2
    let base = g.constant(vec![2.0, 4.0, 9.0], vec![3]);
    let ex = g.constant(vec![3.0, 0.5, 0.5], vec![3]);
    let p = g.pow(base, ex).unwrap();
    approx(&g, p, &[8.0, 2.0, 3.0]);
}

#[test]
fn any_all() {
    let mut g = Graph::new();
    let asbool = |g: &Graph, y: NodeId| match interpret(g, y).storage {
        Storage::BOOL(v) => v,
        s => panic!("want bool, got {:?}", s.dtype()),
    };
    let x = g.constant(vec![0., 1., 0., 0., 0., 0.], vec![2, 3]);
    let an = g.any(x, 1).unwrap();
    assert_eq!(asbool(&g, an), vec![true, false]);
    let y = g.constant(vec![1., 2., 0., 3., 4., 5.], vec![2, 3]);
    let al = g.all(y, 1).unwrap();
    assert_eq!(asbool(&g, al), vec![false, true]);
}

#[test]
fn cumsum_cases() {
    let mut g = Graph::new();
    // last-axis
    let a = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let c = g.cumsum(a, 1).unwrap();
    assert_eq!(interpret(&g, c).f32(), &[1., 3., 6., 4., 9., 15.]);
    // non-last axis (axis 0)
    let c0 = g.cumsum(a, 0).unwrap();
    assert_eq!(interpret(&g, c0).f32(), &[1., 2., 3., 5., 7., 9.]);
    // rank-1
    let v = g.constant(vec![3., 1., 4., 1.], vec![4]);
    let cv = g.cumsum(v, 0).unwrap();
    assert_eq!(interpret(&g, cv).f32(), &[3., 4., 8., 9.]);
    // grad: d(sum(cumsum(x)))/dx[i] = #{j>=i} = n-i
    let x = g.constant(vec![1., 1., 1., 1.], vec![4]);
    let cs = g.cumsum(x, 0).unwrap();
    let loss = g.sum(cs, 0).unwrap();
    let gx = grad(&mut g, loss, &[x]).unwrap()[0];
    assert_eq!(interpret(&g, gx).f32(), &[4., 3., 2., 1.]);
}
