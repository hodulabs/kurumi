use kurumi_core::*;

#[test]
#[allow(clippy::excessive_precision)]
fn special_functions() {
    let mut g = Graph::new();
    // erf at known points
    let x = g.constant(vec![0.0, 1.0, -1.0, 0.5, 2.0], vec![5]);
    let e = g.erf(x);
    approx_tol(&g, e, &[0.0, 0.842700, -0.842700, 0.520500, 0.995322], 1e-3);
    // erfc = 1 - erf
    let ec = g.erfc(x);
    approx_tol(&g, ec, &[1.0, 0.157299, 1.842700, 0.479500, 0.004677], 1e-3);
    // gelu_erf matches the exact formula
    let ge = g.gelu_erf(x);
    let want: Vec<f32> =
        [0.0f32, 1.0, -1.0, 0.5, 2.0].iter().map(|&v| 0.5 * v * (1.0 + erf_ref(v / 2f32.sqrt()))).collect();
    approx_tol(&g, ge, &want, 1e-3);
    // atan / asin / acos
    let t = g.constant(vec![0.0, 1.0, -1.0, 3.0, 0.5], vec![5]);
    let at = g.atan(t);
    approx_tol(
        &g,
        at,
        &[0.0, std::f32::consts::FRAC_PI_4, -std::f32::consts::FRAC_PI_4, 3f32.atan(), 0.5f32.atan()],
        1e-3,
    );
    let u = g.constant(vec![0.0, 0.5, -0.5, 0.8], vec![4]);
    let asn = g.asin(u);
    approx_tol(&g, asn, &[0.0, 0.5f32.asin(), (-0.5f32).asin(), 0.8f32.asin()], 2e-3);
    let acs = g.acos(u);
    approx_tol(&g, acs, &[std::f32::consts::FRAC_PI_2, 0.5f32.acos(), (-0.5f32).acos(), 0.8f32.acos()], 2e-3);
}

#[allow(clippy::excessive_precision)]
fn erf_ref(x: f32) -> f32 {
    // high-accuracy reference (A&S, same approx) just for the test target
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    x.signum() * y
}

fn approx_tol(g: &Graph, y: NodeId, want: &[f32], tol: f32) {
    let got = interpret(g, y);
    for (i, (&a, &b)) in got.f32().iter().zip(want).enumerate() {
        assert!((a - b).abs() < tol, "[{i}] {a} vs {b}");
    }
}
