use crate::{Graph, NodeId, grad, interpret};

// central finite-difference gradient check of sum(build(inputs)) w.r.t. inputs
fn grad_check(inputs: &[(Vec<f32>, Vec<usize>)], build: impl Fn(&mut Graph, &[NodeId]) -> NodeId) {
    let eps = 1e-2f32;
    // analytic gradients
    let mut g = Graph::new();
    let ids: Vec<NodeId> = inputs.iter().map(|(d, s)| g.constant(d.clone(), s.clone())).collect();
    let out = build(&mut g, &ids);
    let gnodes = grad(&mut g, out, &ids).unwrap();
    let analytic: Vec<Vec<f32>> = gnodes.iter().map(|&gn| interpret(&g, gn).f32().to_vec()).collect();

    let loss = |data: &[Vec<f32>]| -> f32 {
        let mut g = Graph::new();
        let ids: Vec<NodeId> = data.iter().zip(inputs).map(|(d, (_, s))| g.constant(d.clone(), s.clone())).collect();
        let out = build(&mut g, &ids);
        interpret(&g, out).f32().iter().sum()
    };

    let base: Vec<Vec<f32>> = inputs.iter().map(|(d, _)| d.clone()).collect();
    for (i, (d, _)) in inputs.iter().enumerate() {
        for j in 0..d.len() {
            let (mut up, mut dn) = (base.clone(), base.clone());
            up[i][j] += eps;
            dn[i][j] -= eps;
            let num = (loss(&up) - loss(&dn)) / (2.0 * eps);
            let ana = analytic[i][j];
            let tol = 3e-2 + 3e-2 * num.abs().max(ana.abs());
            assert!((num - ana).abs() <= tol, "d/dinput{i}[{j}]: numeric {num} vs analytic {ana}");
        }
    }
}

#[test]
fn second_order_grad() {
    // grad-of-grad smoke test: f = sum(x^2) -> df/dx = 2x -> d/dx sum(2x) = 2 everywhere.
    // grad differentiates sum(output), so the second pass exercises backward-of-backward.
    let mut g = Graph::new();
    let x = g.constant(vec![1.0, -2.0, 3.0], vec![3]);
    let sq = g.mul(x, x).unwrap();
    let g1 = grad(&mut g, sq, &[x]).unwrap()[0];
    assert_eq!(interpret(&g, g1).f32(), &[2.0, -4.0, 6.0], "df/dx = 2x");
    let g2 = grad(&mut g, g1, &[x]).unwrap()[0];
    assert_eq!(interpret(&g, g2).f32(), &[2.0, 2.0, 2.0], "d/dx sum(2x) = 2");
}

#[test]
fn grad_mul_add() {
    grad_check(&[(vec![1., -2., 3., 0.5], vec![2, 2]), (vec![4., 5., -1., 2.], vec![2, 2])], |g, x| {
        let m = g.mul(x[0], x[1]).unwrap();
        g.add(m, x[0]).unwrap() // a*b + a
    });
}

#[test]
fn grad_unary_chain() {
    // positive inputs (sqrt/log2/recip domains)
    grad_check(&[(vec![0.5, 1.0, 2.0, 3.0], vec![4])], |g, x| {
        let a = g.sqrt(x[0]);
        let b = g.exp2(a);
        let c = g.recip(b);
        g.log2(c)
    });
}

#[test]
fn grad_relu_via_max() {
    grad_check(&[(vec![-2., -0.5, 0.5, 2.], vec![4])], |g, x| {
        let z = g.zeros_like(x[0]);
        g.max(x[0], z).unwrap()
    });
}

#[test]
fn grad_matmul() {
    let a = (vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let b = ((0..12).map(|i| (i as f32) * 0.1 - 0.5).collect(), vec![3, 4]);
    grad_check(&[a, b], |g, x| g.dot_general(x[0], x[1], vec![1], vec![0], vec![], vec![]).unwrap());
}

#[test]
fn grad_batched_dot() {
    // [2,2,3] x [2,3,2] over batch axis 0 -> [2,2,2]: exercises the general VJP
    let a = ((0..12).map(|i| (i as f32) * 0.2 - 1.0).collect(), vec![2, 2, 3]);
    let b = ((0..12).map(|i| (i as f32) * 0.1).collect(), vec![2, 3, 2]);
    grad_check(&[a, b], |g, x| g.dot_general(x[0], x[1], vec![2], vec![1], vec![0], vec![0]).unwrap());
}

#[test]
fn grad_softmax_decomposed() {
    grad_check(&[(vec![1., 2., 3., 0., -1., 0.5], vec![2, 3])], |g, x| {
        let sm = g.softmax(x[0], 1).unwrap();
        // weight by a fixed vector so the gradient is non-trivial
        let w = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        g.mul(sm, w).unwrap()
    });
}

#[test]
fn grad_rmsnorm() {
    // weighted so the summed loss has a non-trivial gradient through the normalization.
    grad_check(&[(vec![1., 2., 3., -1., 0.5, 2.], vec![2, 3])], |g, x| {
        let n = g.rmsnorm(x[0], 1, 1e-5).unwrap();
        let w = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        g.mul(n, w).unwrap()
    });
}

#[test]
fn grad_prod() {
    // non-zero inputs: d/dx_i prod = prod_{j!=i} x_j (the exactly-zero path is separate).
    grad_check(&[(vec![1.5, -2., 0.5, 2., 1., -1.], vec![2, 3])], |g, x| g.prod(x[0], 1).unwrap());
}

#[test]
fn grad_sdpa_fused() {
    // small non-causal attention over q/k/v: exercises the fused Sdpa VJP end to end.
    let q = (vec![0.1, 0.2, -0.1, 0.0, 0.3, -0.2], vec![1, 2, 3]);
    let k = (vec![0.2, -0.1, 0.1, 0.1, 0.0, 0.2], vec![1, 2, 3]);
    let v = (vec![0.5, -0.3, 0.2, -0.1, 0.4, 0.1], vec![1, 2, 3]);
    grad_check(&[q, k, v], |g, x| g.sdpa(x[0], x[1], x[2], false).unwrap());
}
