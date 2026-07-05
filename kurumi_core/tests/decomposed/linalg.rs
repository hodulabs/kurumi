use crate::approx;
use kurumi_core::*;

#[test]
fn linalg_solve_det_inv() {
    let mut g = Graph::new();
    // solve A x = b ;  A=[[2,1],[1,3]] b=[[1],[2]] -> x=[[0.2],[0.6]]
    let a = g.constant(vec![2., 1., 1., 3.], vec![2, 2]);
    let b = g.constant(vec![1., 2.], vec![2, 1]);
    let x = g.solve(a, b).unwrap();
    approx(&g, x, &[0.2, 0.6]);
    // verify residual A@x == b
    let ax = g.dot_general(a, x, vec![1], vec![0], vec![], vec![]).unwrap();
    approx(&g, ax, &[1., 2.]);
    // det
    let m = g.constant(vec![1., 2., 3., 4.], vec![2, 2]);
    let d = g.det(m).unwrap();
    approx(&g, d, &[-2.0]);
    let diag = g.constant(vec![2., 0., 0., 0., 3., 0., 0., 0., 4.], vec![3, 3]);
    let dd = g.det(diag).unwrap();
    approx(&g, dd, &[24.0]);
    // inv: A @ inv(A) == I
    let ia = g.inv(a).unwrap();
    let prod = g.dot_general(a, ia, vec![1], vec![0], vec![], vec![]).unwrap();
    approx(&g, prod, &[1., 0., 0., 1.]);
    // batched solve
    let ab = g.constant(vec![2., 0., 0., 2., 1., 0., 0., 1.], vec![2, 2, 2]);
    let bb = g.constant(vec![4., 6., 3., 5.], vec![2, 2, 1]);
    let xb = g.solve(ab, bb).unwrap();
    approx(&g, xb, &[2., 3., 3., 5.]);
}

#[test]
fn linalg_cholesky_fwd_bwd() {
    // 2x2 SPD: A = [[4,2],[2,3]] -> L = [[2,0],[1,sqrt(2)]], L@L^T = A.
    let abase = vec![4.0f32, 2.0, 2.0, 3.0];
    let mut g = Graph::new();
    let a = g.constant(abase.clone(), vec![2, 2]);
    let l = g.cholesky(a).unwrap();
    let lv = interpret(&g, l).f32().to_vec();
    let s2 = 2.0f32.sqrt();
    for (got, want) in lv.iter().zip([2.0, 0.0, 1.0, s2]) {
        assert!((got - want).abs() < 1e-5, "L {lv:?}");
    }
    // backward: grad of sum(L) wrt A, vs finite differences. cholesky reads only
    // the lower triangle, so the upper entry A[0,1] gets ~0 gradient.
    let loss = {
        let r = g.sum(l, 1).unwrap();
        g.sum(r, 0).unwrap()
    };
    let ga = grad(&mut g, loss, &[a]).unwrap()[0];
    let gv = interpret(&g, ga).f32().to_vec();
    // reference cholesky in f64
    let chol = |m: &[f64]| -> [f64; 4] {
        let l00 = m[0].sqrt();
        let l10 = m[2] / l00;
        let l11 = (m[3] - l10 * l10).sqrt();
        [l00, 0.0, l10, l11]
    };
    let loss_of = |m: &[f64]| chol(m).iter().sum::<f64>();
    let base: Vec<f64> = abase.iter().map(|&x| x as f64).collect();
    let eps = 1e-4;
    for i in 0..4 {
        let (mut mp, mut mm) = (base.clone(), base.clone());
        mp[i] += eps;
        mm[i] -= eps;
        let fd = (loss_of(&mp) - loss_of(&mm)) / (2.0 * eps);
        assert!((gv[i] as f64 - fd).abs() < 1e-2, "dA[{i}] analytic {} vs fd {fd}", gv[i]);
    }
}

#[test]
fn linalg_det_backward() {
    // d(det)/dA = det(A) * inv(A)^T ; check against finite differences
    let base = vec![2.0f32, 1.0, 0.5, 3.0];
    let mut g = Graph::new();
    let a = g.constant(base.clone(), vec![2, 2]);
    let d = g.det(a).unwrap();
    let gx = grad(&mut g, d, &[a]).unwrap()[0];
    let analytic = interpret(&g, gx).f32().to_vec();
    // finite differences
    let det_of = |m: &[f32]| m[0] * m[3] - m[1] * m[2];
    let eps = 1e-3;
    for i in 0..4 {
        let mut mp = base.clone();
        mp[i] += eps;
        let mut mm = base.clone();
        mm[i] -= eps;
        let fd = (det_of(&mp) - det_of(&mm)) / (2.0 * eps);
        assert!((analytic[i] - fd).abs() < 1e-2, "[{i}] analytic {} vs fd {fd}", analytic[i]);
    }
}

#[test]
fn linalg_solve_backward() {
    // grad of sum(solve(A,b)) wrt A and b, vs finite differences
    let abase = vec![2.0f32, 1.0, 1.0, 3.0];
    let bbase = vec![1.0f32, 2.0];
    let mut g = Graph::new();
    let a = g.constant(abase.clone(), vec![2, 2]);
    let b = g.constant(bbase.clone(), vec![2, 1]);
    let x = g.solve(a, b).unwrap();
    let loss = {
        let s = g.sum(x, 1).unwrap();
        g.sum(s, 0).unwrap()
    };
    let grads = grad(&mut g, loss, &[a, b]).unwrap();
    let ga = interpret(&g, grads[0]).f32().to_vec();
    let gb = interpret(&g, grads[1]).f32().to_vec();
    // reference solver
    let solve2 = |m: &[f32], v: &[f32]| {
        let det = m[0] * m[3] - m[1] * m[2];
        [(m[3] * v[0] - m[1] * v[1]) / det, (-m[2] * v[0] + m[0] * v[1]) / det]
    };
    let loss_of = |m: &[f32], v: &[f32]| {
        let x = solve2(m, v);
        x[0] + x[1]
    };
    let eps = 1e-3;
    for i in 0..4 {
        let (mut mp, mut mm) = (abase.clone(), abase.clone());
        mp[i] += eps;
        mm[i] -= eps;
        let fd = (loss_of(&mp, &bbase) - loss_of(&mm, &bbase)) / (2.0 * eps);
        assert!((ga[i] - fd).abs() < 1e-2, "dA[{i}] {} vs {fd}", ga[i]);
    }
    for i in 0..2 {
        let (mut vp, mut vm) = (bbase.clone(), bbase.clone());
        vp[i] += eps;
        vm[i] -= eps;
        let fd = (loss_of(&abase, &vp) - loss_of(&abase, &vm)) / (2.0 * eps);
        assert!((gb[i] - fd).abs() < 1e-2, "dB[{i}] {} vs {fd}", gb[i]);
    }
}
