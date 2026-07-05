use kurumi_core::*;

#[test]
fn complex_basics() {
    let mut g = Graph::new();
    let re = g.constant(vec![1., 2.], vec![2]);
    let im = g.constant(vec![3., 4.], vec![2]);
    let z = g.complex(re, im).unwrap();
    assert_eq!(interpret(&g, z).dtype(), DType::C64);
    // real / imag round-trip
    let (zr, zi) = (g.real(z).unwrap(), g.imag(z).unwrap());
    assert_eq!(interpret(&g, zr).f32(), &[1., 2.]);
    assert_eq!(interpret(&g, zi).f32(), &[3., 4.]);
    // conj negates the imaginary part
    let c = g.conj(z).unwrap();
    let ci = g.imag(c).unwrap();
    assert_eq!(interpret(&g, ci).f32(), &[-3., -4.]);
    // magnitude |z| = sqrt(re^2+im^2)
    let m = g.cabs(z).unwrap();
    let mv = interpret(&g, m).f32().to_vec();
    assert!((mv[0] - 10f32.sqrt()).abs() < 1e-5 && (mv[1] - 20f32.sqrt()).abs() < 1e-5, "cabs {mv:?}");
    // complex add: z+z
    let s = g.add(z, z).unwrap();
    let (sr, si) = (g.real(s).unwrap(), g.imag(s).unwrap());
    assert_eq!(interpret(&g, sr).f32(), &[2., 4.]);
    assert_eq!(interpret(&g, si).f32(), &[6., 8.]);
    // complex mul: z*z = (a+bi)^2 = a^2-b^2 + 2ab*i
    let p = g.mul(z, z).unwrap();
    let (pr, pi) = (g.real(p).unwrap(), g.imag(p).unwrap());
    assert_eq!(interpret(&g, pr).f32(), &[-8., -12.]);
    assert_eq!(interpret(&g, pi).f32(), &[6., 16.]);
    // cast real -> complex sets imag 0
    let up = g.cast(re, DType::C64);
    let ui = g.imag(up).unwrap();
    assert_eq!(interpret(&g, ui).f32(), &[0., 0.]);
    // complex sum/prod are valid (add/mul over complex): builder must accept them.
    let cs = g.sum(z, 0).unwrap(); // (1+3i)+(2+4i) = 3+7i
    let (csr, csi) = (g.real(cs).unwrap(), g.imag(cs).unwrap());
    assert_eq!(interpret(&g, csr).f32(), &[3.]);
    assert_eq!(interpret(&g, csi).f32(), &[7.]);
    let cp = g.prod(z, 0).unwrap(); // (1+3i)(2+4i) = (2-12) + (4+6)i = -10+10i
    let (cpr, cpi) = (g.real(cp).unwrap(), g.imag(cp).unwrap());
    assert_eq!(interpret(&g, cpr).f32(), &[-10.]);
    assert_eq!(interpret(&g, cpi).f32(), &[10.]);
}

#[test]
fn complex_autodiff() {
    // real param theta -> complex -> real loss, verified against finite differences.
    // Wirtinger: complex mul/matmul backward conjugates the other operand; a
    // wrong (no-conj) rule flips the sign of the imaginary contribution.
    let d = 1e-3;
    // (1) mul: L = real(e^{itheta}*e^{itheta}) = cos 2theta ; dL/dtheta = -2 sin 2theta.
    let mut g = Graph::new();
    let th = g.constant(vec![0.5], vec![1]);
    let z = {
        let c = g.cos(th);
        let s = g.sin(th);
        g.complex(c, s).unwrap()
    };
    let w = g.mul(z, z).unwrap();
    let l1 = {
        let r = g.real(w).unwrap();
        g.sum(r, 0).unwrap()
    };
    let g1 = grad(&mut g, l1, &[th]).unwrap()[0];
    let a1 = interpret(&g, g1).f32()[0] as f64;
    let f1 = |t: f64| (2.0 * t).cos();
    let n1 = (f1(0.5 + d) - f1(0.5 - d)) / (2.0 * d);
    assert!((a1 - n1).abs() < 1e-2, "mul-vjp {a1} vs fd {n1}");
    // (2) matmul: L = real(e^{itheta}*(2+i)) = 2costheta - sintheta ; dL/dtheta = -2sintheta - costheta.
    let mut g = Graph::new();
    let th = g.constant(vec![0.5], vec![1]);
    let z = {
        let c = g.cos(th);
        let s = g.sin(th);
        g.complex(c, s).unwrap()
    };
    let m = g.reshape(z, vec![1, 1]).unwrap();
    let v = {
        let r = g.constant(vec![2.], vec![1]);
        let i = g.constant(vec![1.], vec![1]);
        g.complex(r, i).unwrap()
    };
    let y = g.dot_general(m, v, vec![1], vec![0], vec![], vec![]).unwrap();
    let l2 = {
        let r = g.real(y).unwrap();
        g.sum(r, 0).unwrap()
    };
    let g2 = grad(&mut g, l2, &[th]).unwrap()[0];
    let a2 = interpret(&g, g2).f32()[0] as f64;
    let f2 = |t: f64| 2.0 * t.cos() - t.sin();
    let n2 = (f2(0.5 + d) - f2(0.5 - d)) / (2.0 * d);
    assert!((a2 - n2).abs() < 1e-2, "dot-vjp {a2} vs fd {n2}");
}

#[test]
fn complex_exp_euler() {
    // e^{ipi} = -1: complex exp works through the EXISTING exp decomposition (exp2
    // after scalar-mul).
    let mut g = Graph::new();
    let pi = std::f32::consts::PI;
    let z = {
        let r = g.constant(vec![0.], vec![1]);
        let i = g.constant(vec![pi], vec![1]);
        g.complex(r, i).unwrap()
    };
    let e = g.exp(z);
    let (er, ei) = (g.real(e).unwrap(), g.imag(e).unwrap());
    assert!((interpret(&g, er).f32()[0] - (-1.0)).abs() < 1e-5, "e^ipi re = {}", interpret(&g, er).f32()[0]);
    assert!(interpret(&g, ei).f32()[0].abs() < 1e-5, "e^ipi im = {}", interpret(&g, ei).f32()[0]);
    // e^{ipi/2} = i
    let z2 = {
        let r = g.constant(vec![0.], vec![1]);
        let i = g.constant(vec![pi / 2.0], vec![1]);
        g.complex(r, i).unwrap()
    };
    let e2 = g.exp(z2);
    let (e2r, e2i) = (g.real(e2).unwrap(), g.imag(e2).unwrap());
    assert!(
        interpret(&g, e2r).f32()[0].abs() < 1e-5 && (interpret(&g, e2i).f32()[0] - 1.0).abs() < 1e-5,
        "e^ipi/2 = i"
    );
}

#[test]
fn vqe_differentiable_quantum() {
    // End-to-end differentiable quantum: apply a parameterized Ry(theta) gate (a
    // complex unitary) to |0> via complex matmul, measure <Z>, and take the
    // gradient wrt theta. Analytic: <Z> = cos theta, d<Z>/dtheta = -sin theta.
    let mut g = Graph::new();
    let theta = g.constant(vec![1.0], vec![1]);
    let half = {
        let hf = g.scalar(theta, 0.5);
        g.mul(theta, hf).unwrap()
    };
    let c = g.cos(half);
    let s = g.sin(half);
    let ns = g.neg(s);
    // Ry(theta) = [[c, -s], [s, c]] as a complex matrix (imag 0)
    let ry_re = {
        let r = g.concat(&[c, ns, s, c], 0).unwrap();
        g.reshape(r, vec![2, 2]).unwrap()
    };
    let ry_im = g.constant(vec![0.; 4], vec![2, 2]);
    let ry = g.complex(ry_re, ry_im).unwrap();
    // |0> = [1, 0]
    let ket = {
        let kr = g.constant(vec![1., 0.], vec![2]);
        let ki = g.constant(vec![0., 0.], vec![2]);
        g.complex(kr, ki).unwrap()
    };
    // |psi> = Ry*|0>  (complex gate contraction)
    let psi = g.dot_general(ry, ket, vec![1], vec![0], vec![], vec![]).unwrap();
    // <Z> = sum diag(Z)*|psi|^2 = |psi_0|^2 - |psi_1|^2
    let probs = {
        let re = g.real(psi).unwrap();
        let im = g.imag(psi).unwrap();
        let a = g.square(re);
        let b = g.square(im);
        g.add(a, b).unwrap()
    };
    let z = g.constant(vec![1., -1.], vec![2]);
    let expz = {
        let m = g.mul(probs, z).unwrap();
        g.sum(m, 0).unwrap()
    };
    // <Z> = cos(theta)
    assert!((interpret(&g, expz).f32()[0] - 1.0f32.cos()).abs() < 1e-5, "expZ={}", interpret(&g, expz).f32()[0]);
    // d<Z>/dtheta = -sin(theta), through the complex gate + matmul + measurement
    let gth = grad(&mut g, expz, &[theta]).unwrap()[0];
    assert!((interpret(&g, gth).f32()[0] - (-1.0f32.sin())).abs() < 1e-4, "grad={}", interpret(&g, gth).f32()[0]);
}

#[test]
fn complex_matmul_quantum() {
    // quantum circuit sim = complex gate matrices contracted with a state vector.
    let mut g = Graph::new();
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let z4 = g.constant(vec![0.; 4], vec![4]);
    let z16 = g.constant(vec![0.; 16], vec![4, 4]);
    // |00> and the two real gates (imag = 0), all in C64.
    let state = {
        let r = g.constant(vec![1., 0., 0., 0.], vec![4]);
        g.complex(r, z4).unwrap()
    };
    // H(x)I  and  CNOT  (control q0, target q1)
    let hxi = {
        let r = g.constant(vec![s, 0., s, 0., 0., s, 0., s, s, 0., -s, 0., 0., s, 0., -s], vec![4, 4]);
        g.complex(r, z16).unwrap()
    };
    let cnot = {
        let r = g.constant(vec![1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.], vec![4, 4]);
        g.complex(r, z16).unwrap()
    };
    // apply gates by contraction: psi = CNOT * (H(x)I * |00>)
    let psi1 = g.dot_general(hxi, state, vec![1], vec![0], vec![], vec![]).unwrap();
    let psi2 = g.dot_general(cnot, psi1, vec![1], vec![0], vec![], vec![]).unwrap();
    // Bell state (|00> + |11>)/sqrt(2) = [s, 0, 0, s], purely real
    let (pr, pi) = (g.real(psi2).unwrap(), g.imag(psi2).unwrap());
    let rv = interpret(&g, pr).f32().to_vec();
    for (got, want) in rv.iter().zip([s, 0., 0., s]) {
        assert!((got - want).abs() < 1e-6, "bell real {rv:?}");
    }
    assert_eq!(interpret(&g, pi).f32(), &[0., 0., 0., 0.]);

    // S gate = diag(1, i) applied to |1> -> i*|1> (imaginary amplitude appears).
    let one = {
        let r = g.constant(vec![0., 1.], vec![2]);
        let z2 = g.constant(vec![0., 0.], vec![2]);
        g.complex(r, z2).unwrap()
    };
    let sg = {
        let sr = g.constant(vec![1., 0., 0., 0.], vec![2, 2]);
        let si = g.constant(vec![0., 0., 0., 1.], vec![2, 2]);
        g.complex(sr, si).unwrap()
    };
    let out = g.dot_general(sg, one, vec![1], vec![0], vec![], vec![]).unwrap();
    let (or, oi) = (g.real(out).unwrap(), g.imag(out).unwrap());
    assert_eq!(interpret(&g, or).f32(), &[0., 0.]);
    assert_eq!(interpret(&g, oi).f32(), &[0., 1.]); // i*|1>
}
