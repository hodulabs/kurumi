//! Tests for the extended engine surface: f32-accumulated reductions (A), stable
//! elementwise (B), signal (E), stats (F), distances (G), losses (I), and
//! higher-order gradients (H).

use kurumi_core::{DType, Graph, grad, interpret};

fn approx(g: &Graph, n: kurumi_core::NodeId, want: &[f32], tol: f32) {
    let got = interpret(g, n);
    let got = got.f32();
    assert_eq!(got.len(), want.len(), "len {} vs {}", got.len(), want.len());
    for (a, b) in got.iter().zip(want) {
        assert!((a - b).abs() < tol, "got {got:?} want {want:?}");
    }
}

// A1: f16 sum accumulates in f32: summing 4000 ones must reach 4000, not stall at
// ~2048 (where a naive f16 accumulator saturates: 2048 + 1 rounds back to 2048).
#[test]
fn reduce_f16_accumulates_in_f32() {
    let mut g = Graph::new();
    let x = g.constant(vec![1.0; 4000], vec![4000]);
    let xf16 = g.cast(x, DType::F16);
    let s = g.sum(xf16, 0).unwrap();
    let sf = g.cast(s, DType::F32); // sum is F16; read it back as f32
    approx(&g, sf, &[4000.0], 1.0);
}

// A2: variance with Bessel correction (ddof=1) vs population (ddof=0).
#[test]
fn var_correction_bessel() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3., 4.], vec![4]); // mean 2.5, SS = 5
    let v0 = g.var(x, 0).unwrap(); // 5/4 = 1.25
    let v1 = g.var_correction(x, 0, 1).unwrap(); // 5/3 = 1.6667
    approx(&g, v0, &[1.25], 1e-5);
    approx(&g, v1, &[5.0 / 3.0], 1e-5);
}

// B: numerically-stable elementwise.
#[test]
fn stable_elementwise() {
    let mut g = Graph::new();
    // log1p / expm1 near 0
    let x = g.constant(vec![0.0, 1.0, -0.5], vec![3]);
    let lp = g.log1p(x).unwrap();
    approx(&g, lp, &[0.0, 2f32.ln(), 0.5f32.ln()], 1e-5);
    let em = g.expm1(x).unwrap();
    approx(&g, em, &[0.0, std::f32::consts::E - 1.0, (-0.5f32).exp() - 1.0], 1e-5);
    // xlogy(0, y) = 0 even where ln(y) is -inf
    let z = g.constant(vec![0.0, 2.0], vec![2]);
    let y = g.constant(vec![0.0, 3.0], vec![2]);
    let xl = g.xlogy(z, y).unwrap();
    approx(&g, xl, &[0.0, 2.0 * 3f32.ln()], 1e-5);
    // logaddexp(a,b) = ln(e^a + e^b)
    let a = g.constant(vec![1.0, 2.0], vec![2]);
    let b = g.constant(vec![2.0, 0.0], vec![2]);
    let la = g.logaddexp(a, b).unwrap();
    approx(&g, la, &[(1f32.exp() + 2f32.exp()).ln(), (2f32.exp() + 1.0).ln()], 1e-5);
    // sinc(0) = 1
    let s0 = g.constant(vec![0.0, 0.5], vec![2]);
    let sc = g.sinc(s0).unwrap();
    approx(&g, sc, &[1.0, (std::f32::consts::FRAC_PI_2).sin() / std::f32::consts::FRAC_PI_2], 1e-5);
}

// E: signal: fft2 roundtrip, rfft shape, hann window, fft_conv (circular).
#[test]
fn signal_ops() {
    let mut g = Graph::new();
    // ifft2(fft2(x)) == x
    let x = g.constant((0..12).map(|i| i as f32).collect(), vec![3, 4]);
    let f = g.fft2(x).unwrap();
    let back = g.ifft2(f).unwrap();
    let re = g.real(back).unwrap();
    approx(&g, re, &(0..12).map(|i| i as f32).collect::<Vec<_>>(), 1e-3);
    // rfft length = n/2 + 1
    let sig = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![6]);
    let rf = g.rfft(sig, 0).unwrap();
    assert_eq!(interpret(&g, rf).shape, vec![4]); // 6/2 + 1
    // hann window sums to ~ n/2 (its DC gain)
    let hann = g.hann_window(8);
    let hs = g.sum(hann, 0).unwrap();
    approx(&g, hs, &[3.5], 1e-4); // sum hann(8) = 3.5
    // circular convolution: conv([1,2,3],[1,0,0]) = [1,2,3]
    let a = g.constant(vec![1., 2., 3.], vec![3]);
    let d = g.constant(vec![1., 0., 0.], vec![3]);
    let c = g.fft_conv(a, d, 0).unwrap();
    approx(&g, c, &[1., 2., 3.], 1e-3);
}

// F: order stats + cumulative + covariance.
#[test]
fn stats_ops() {
    let mut g = Graph::new();
    let x = g.constant(vec![3., 1., 4., 1., 5.], vec![5]);
    let med = g.median(x, 0).unwrap();
    approx(&g, med, &[3.0], 1e-6); // sorted [1,1,3,4,5] -> 3
    let q = g.quantile(x, 0, 0.25).unwrap();
    approx(&g, q, &[1.0], 1e-6); // pos 1.0 -> sorted[1] = 1
    let cm = g.cummax(x, 0).unwrap();
    approx(&g, cm, &[3., 3., 4., 4., 5.], 1e-6);
    let cn = g.cummin(x, 0).unwrap();
    approx(&g, cn, &[3., 1., 1., 1., 1.], 1e-6);
    // cov of [[0,2],[0,4]] (2 features, 2 obs): row0 var 1, row1 var 4, cov 2 (ddof1)
    let d = g.constant(vec![0., 2., 0., 4.], vec![2, 2]);
    let c = g.cov(d).unwrap();
    approx(&g, c, &[2., 4., 4., 8.], 1e-5);
    // mode: most frequent value (smallest on tie)
    let mv = g.constant(vec![1., 2., 2., 3., 3., 3.], vec![6]);
    let md = g.mode(mv, 0).unwrap();
    approx(&g, md, &[3.0], 1e-6);
}

// istft round-trips stft (rectangular window, OLA/overlap-count normalization).
#[test]
fn istft_roundtrip() {
    let mut g = Graph::new();
    let x = g.constant((1..=8).map(|i| i as f32).collect(), vec![8]);
    let frames = g.stft(x, 4, 2, None).unwrap(); // [n_frames, n_fft] complex
    let rec = g.istft(frames, 2, None).unwrap(); // real [8]
    approx(&g, rec, &(1..=8).map(|i| i as f32).collect::<Vec<_>>(), 1e-3);
}

// G: distances.
#[test]
fn distance_ops() {
    let mut g = Graph::new();
    // cdist rows of [[0,0],[3,4]] to itself -> [[0,5],[5,0]]
    let a = g.constant(vec![0., 0., 3., 4.], vec![2, 2]);
    let d = g.cdist(a, a, 2.0).unwrap();
    approx(&g, d, &[0., 5., 5., 0.], 1e-4);
    // cosine similarity of parallel vectors = 1
    let u = g.constant(vec![1., 2., 3.], vec![1, 3]);
    let v = g.constant(vec![2., 4., 6.], vec![1, 3]);
    let cs = g.cosine_similarity(u, v, 1).unwrap();
    approx(&g, cs, &[1.0], 1e-5);
}

// I: losses.
#[test]
fn loss_ops() {
    let mut g = Graph::new();
    let p = g.constant(vec![1., 2., 3.], vec![3]);
    let t = g.constant(vec![1.5, 2.0, 5.0], vec![3]);
    let mse = g.mse_loss(p, t).unwrap();
    approx(&g, mse, &[0.25, 0.0, 4.0], 1e-5);
    let hub = g.huber_loss(p, t, 1.0).unwrap(); // |d|=[.5,0,2]: quad .125,0 ; lin 1*(2-.5)=1.5
    approx(&g, hub, &[0.125, 0.0, 1.5], 1e-5);
    // bce(p=0.5, t in {0,1}) = ln 2
    let pr = g.constant(vec![0.5, 0.5], vec![2]);
    let tg = g.constant(vec![1.0, 0.0], vec![2]);
    let bce = g.bce_loss(pr, tg).unwrap();
    approx(&g, bce, &[2f32.ln(), 2f32.ln()], 1e-5);
}

// C: special functions against known values.
#[test]
fn special_functions() {
    let mut g = Graph::new();
    // lgamma(5) = ln(4!) = ln 24 ; gamma(5) = 24
    let x = g.constant(vec![5.0, 0.5], vec![2]);
    let lg = g.lgamma(x);
    approx(&g, lg, &[24f32.ln(), std::f32::consts::PI.sqrt().ln()], 1e-3);
    let gm = g.gamma(x);
    approx(&g, gm, &[24.0, std::f32::consts::PI.sqrt()], 2e-2);
    // digamma(1) = -gamma ~= -0.5772
    let one = g.constant(vec![1.0, 2.0], vec![2]);
    let dg = g.digamma(one);
    approx(&g, dg, &[-0.577_215_7, 1.0 - 0.577_215_7], 1e-3);
    // erfinv(erf(t)) = t
    let t = g.constant(vec![0.3, -0.6], vec![2]);
    let e = g.erf(t);
    let inv = g.erfinv(e);
    approx(&g, inv, &[0.3, -0.6], 1e-3);
    // i0(0) = 1, i0(1) ~= 1.2660658
    let z = g.constant(vec![0.0, 1.0, 4.0], vec![3]);
    let b = g.i0(z);
    approx(&g, b, &[1.0, 1.266_065_8, 11.301_922], 2e-2);
    // beta(2,3) = 1!*2!/4! = 1/12
    let a2 = g.constant(vec![2.0], vec![1]);
    let b3 = g.constant(vec![3.0], vec![1]);
    let bt = g.beta(a2, b3).unwrap();
    approx(&g, bt, &[1.0 / 12.0], 1e-3);
}

// D: advanced linalg: eigh / qr / svd (reconstruction, sign-robust) + decompositions.
#[test]
fn advanced_linalg() {
    let mm = |g: &mut Graph, a, b| g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
    // eigh: A = V diag(lambda) V^T ; A = [[2,1],[1,2]] -> lambda = [1,3]
    let mut g = Graph::new();
    let a = g.constant(vec![2., 1., 1., 2.], vec![2, 2]);
    let (vals, vecs) = g.eigh(a).unwrap();
    approx(&g, vals, &[1.0, 3.0], 1e-4);
    let d = g.diag_embed(vals).unwrap();
    let vd = mm(&mut g, vecs, d);
    let vt = g.transpose(vecs, 0, 1).unwrap();
    let recon = mm(&mut g, vd, vt);
    approx(&g, recon, &[2., 1., 1., 2.], 1e-4);

    // qr: Q*R = A, R upper-triangular
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
    let (q, r) = g.qr(a).unwrap();
    let qr = mm(&mut g, q, r);
    approx(&g, qr, &[1., 2., 3., 4., 5., 6.], 1e-4);

    // svd: U*diag(S)*V^T = A, S descending
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3., 4.], vec![2, 2]);
    let (u, s, v) = g.svd(a).unwrap();
    let ds = g.diag_embed(s).unwrap();
    let us = mm(&mut g, u, ds);
    let vt = g.transpose(v, 0, 1).unwrap();
    let recon = mm(&mut g, us, vt);
    approx(&g, recon, &[1., 2., 3., 4.], 1e-3);
    // singular values of [[1,2],[3,4]] ~= [5.465, 0.366], descending
    approx(&g, s, &[5.4650, 0.3660], 1e-2);

    // matrix_exp(0) = I ; slogdet, lstsq, pinv
    let mut g = Graph::new();
    let z = g.constant(vec![0.; 4], vec![2, 2]);
    let e = g.matrix_exp(z).unwrap();
    approx(&g, e, &[1., 0., 0., 1.], 1e-6);
    // lstsq exact-determined: A=[[1,0],[0,1]], b=[3,4] -> x=[3,4]
    let a = g.constant(vec![1., 0., 0., 1.], vec![2, 2]);
    let b = g.constant(vec![3., 4.], vec![2, 1]);
    let x = g.lstsq(a, b).unwrap();
    approx(&g, x, &[3., 4.], 1e-4);
    // slogdet of 2*I (2x2) -> det 4 -> sign 1, ln 4
    let a2 = g.constant(vec![2., 0., 0., 2.], vec![2, 2]);
    let (sign, logabs) = g.slogdet(a2).unwrap();
    approx(&g, sign, &[1.0], 1e-6);
    approx(&g, logabs, &[4f32.ln()], 1e-5);
}

// J: einsum with `...` ellipsis (batch dims).
#[test]
fn einsum_ellipsis() {
    let mut g = Graph::new();
    // batched matmul: [2,2,3] @ [2,3,2] -> [2,2,2] via "...ij,...jk->...ik"
    let a = g.constant((0..12).map(|i| i as f32).collect(), vec![2, 2, 3]);
    let b = g.constant((0..12).map(|i| i as f32).collect(), vec![2, 3, 2]);
    let c = g.einsum("...ij,...jk->...ik", &[a, b]).unwrap();
    assert_eq!(interpret(&g, c).shape, vec![2, 2, 2]);
    // check against an explicit batched dot_general
    let ref_ = g.dot_general(a, b, vec![2], vec![1], vec![0], vec![0]).unwrap();
    approx(&g, c, interpret(&g, ref_).f32(), 1e-4);
    // implicit output batch-transpose: "...ij->...ji"
    let t = g.einsum("...ij->...ji", &[a]).unwrap();
    assert_eq!(interpret(&g, t).shape, vec![2, 3, 2]);
}

// eigh VJP: analytic identities (grad sum lambda = I, grad sum lambda^2 = 2A) + symmetric FD on the
// eigenvector path.
#[test]
fn eigh_vjp() {
    // A symmetric 2x2; grad of sum(eigenvalues) = I (trace)
    let mut g = Graph::new();
    let a = g.constant(vec![2., 1., 1., 3.], vec![2, 2]);
    let (vals, _) = g.eigh(a).unwrap();
    let sv = g.sum(vals, 0).unwrap();
    let ga = grad(&mut g, sv, &[a]).unwrap()[0];
    approx(&g, ga, &[1., 0., 0., 1.], 1e-3);
    // grad of sum(eigenvalues^2) = 2A
    let mut g = Graph::new();
    let a = g.constant(vec![2., 1., 1., 3.], vec![2, 2]);
    let (vals, _) = g.eigh(a).unwrap();
    let sq = g.mul(vals, vals).unwrap();
    let ssq = g.sum(sq, 0).unwrap();
    let ga = grad(&mut g, ssq, &[a]).unwrap()[0];
    approx(&g, ga, &[4., 2., 2., 6.], 1e-2);

    // eigenvector path via symmetric finite differences: loss = sum eigenvectors.
    let base = vec![2.0f32, 0.5, 0.5, 3.0];
    let vec_loss = |g: &mut Graph, a: kurumi_core::NodeId| {
        let (_, vecs) = g.eigh(a).unwrap();
        let s0 = g.sum(vecs, 0).unwrap();
        g.sum(s0, 0).unwrap()
    };
    let mut g = Graph::new();
    let a = g.constant(base.clone(), vec![2, 2]);
    let l = vec_loss(&mut g, a);
    let gn = grad(&mut g, l, &[a]).unwrap()[0];
    let ana = interpret(&g, gn).f32().to_vec();
    let eval = |data: &[f32]| -> f32 {
        let mut g = Graph::new();
        let a = g.constant(data.to_vec(), vec![2, 2]);
        let l = vec_loss(&mut g, a);
        interpret(&g, l).f32().iter().sum()
    };
    let eps = 1e-3;
    // symmetric params: (0,0),(1,1) diagonal; (0,1)&(1,0) coupled
    for &(i, j) in &[(0usize, 0usize), (1, 1), (0, 1)] {
        let mut lo = base.clone();
        let mut hi = base.clone();
        lo[i * 2 + j] -= eps;
        hi[i * 2 + j] += eps;
        if i != j {
            lo[j * 2 + i] -= eps;
            hi[j * 2 + i] += eps;
        }
        let fd = (eval(&hi) - eval(&lo)) / (2.0 * eps);
        let a_grad = if i == j { ana[i * 2 + j] } else { ana[i * 2 + j] + ana[j * 2 + i] };
        assert!((a_grad - fd).abs() < 2e-2, "eigh vec grad ({i},{j}): {a_grad} vs fd {fd}");
    }
}

// qr VJP via finite differences (Q-path and R-path), tall A = [3,2].
#[test]
fn qr_vjp() {
    let base = vec![1.0f32, 2., 3., 1., 1., 4.];
    let check = |loss: &dyn Fn(&mut Graph, kurumi_core::NodeId) -> kurumi_core::NodeId| {
        let mut g = Graph::new();
        let a = g.constant(base.clone(), vec![3, 2]);
        let l = loss(&mut g, a);
        let gn = grad(&mut g, l, &[a]).unwrap()[0];
        let ana = interpret(&g, gn).f32().to_vec();
        let eval = |data: &[f32]| -> f32 {
            let mut g = Graph::new();
            let a = g.constant(data.to_vec(), vec![3, 2]);
            let l = loss(&mut g, a);
            interpret(&g, l).f32().iter().sum()
        };
        let eps = 1e-3;
        for i in 0..base.len() {
            let mut lo = base.clone();
            let mut hi = base.clone();
            lo[i] -= eps;
            hi[i] += eps;
            let fd = (eval(&hi) - eval(&lo)) / (2.0 * eps);
            assert!((ana[i] - fd).abs() < 2e-2, "qr grad {i}: {} vs fd {fd}", ana[i]);
        }
    };
    // R-path: loss = sum R
    check(&|g, a| {
        let (_, r) = g.qr(a).unwrap();
        let s = g.sum(r, 0).unwrap();
        g.sum(s, 0).unwrap()
    });
    // Q-path: loss = sum Q
    check(&|g, a| {
        let (q, _) = g.qr(a).unwrap();
        let s = g.sum(q, 0).unwrap();
        g.sum(s, 0).unwrap()
    });
}

// general (nonsymmetric) eigenvalues via eigvals -> complex.
#[test]
fn eigvals_general() {
    // diagonal -> real eigenvalues {2, 3}
    let mut g = Graph::new();
    let a = g.constant(vec![2., 0., 0., 3.], vec![2, 2]);
    let ev = g.eigvals(a).unwrap();
    let re = g.real(ev).unwrap();
    let sre = g.sort(re, 0, false).unwrap();
    approx(&g, sre, &[2., 3.], 1e-3);
    let im = g.imag(ev).unwrap();
    let i2 = g.mul(im, im).unwrap();
    let si2 = g.sum(i2, 0).unwrap();
    approx(&g, si2, &[0.0], 1e-4);

    // rotation [[0,-1],[1,0]] -> eigenvalues +/-i (real part 0, imag +/-1)
    let mut g = Graph::new();
    let a = g.constant(vec![0., -1., 1., 0.], vec![2, 2]);
    let ev = g.eigvals(a).unwrap();
    let re = g.real(ev).unwrap();
    let sre = g.sum(re, 0).unwrap(); // trace = 0
    approx(&g, sre, &[0.0], 1e-3);
    let im = g.imag(ev).unwrap();
    let sim = g.sort(im, 0, false).unwrap();
    approx(&g, sim, &[-1., 1.], 1e-3);

    // upper-triangular -> eigenvalues are the diagonal {1,2,3}
    let mut g = Graph::new();
    let a = g.constant(vec![1., 5., 7., 0., 2., 9., 0., 0., 3.], vec![3, 3]);
    let ev = g.eigvals(a).unwrap();
    let re = g.real(ev).unwrap();
    let sre = g.sort(re, 0, false).unwrap();
    approx(&g, sre, &[1., 2., 3.], 1e-3);
}

// H: higher-order gradients. f = sum(x^3) -> f' = 3x^2 -> f'' = 6x.
#[test]
fn higher_order_grad() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3.], vec![3]);
    let x2 = g.mul(x, x).unwrap();
    let x3 = g.mul(x2, x).unwrap();
    let g1 = grad(&mut g, x3, &[x]).unwrap()[0]; // d sum(x^3)/dx = 3x^2
    approx(&g, g1, &[3., 12., 27.], 1e-4);
    let g2 = grad(&mut g, g1, &[x]).unwrap()[0]; // d sum(3x^2)/dx = 6x
    approx(&g, g2, &[6., 12., 18.], 1e-4);
}
