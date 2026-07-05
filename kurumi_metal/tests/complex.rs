#![cfg(target_os = "macos")]
//! Complex C64 device tests: exact arithmetic, transcendentals, matmul, reduce, pad.

use kurumi_core::{Backend, Graph, Storage, interpret};
use kurumi_metal::MetalBackend;

// Complex reduce (sum = float2 add, prod = cmul) + zero-pad run device-resident (float2),
// exact match to the oracle (same IEEE ops).
#[test]
fn complex_reduce_pad_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let chk = |g: &Graph, n| assert_eq!(metal.eval(g, n).storage, interpret(g, n).storage, "device != oracle");
    let mk = |g: &mut Graph| {
        let re = g.constant(vec![1.0, -2.0, 3.0, 0.5, 2.0, -1.0], vec![2, 3]);
        let im = g.constant(vec![0.5, 1.0, -1.0, 2.0, -0.5, 3.0], vec![2, 3]);
        g.complex(re, im).unwrap()
    };
    // complex sum over the last axis (float2 component add)
    let mut g = Graph::new();
    let z = mk(&mut g);
    let n = g.sum(z, 1).unwrap();
    chk(&g, n);
    // complex prod over the last axis (cmul fold)
    let mut g = Graph::new();
    let z = mk(&mut g);
    let n = g.prod(z, 1).unwrap();
    chk(&g, n);
    // complex zero-pad
    let mut g = Graph::new();
    let z = mk(&mut g);
    let n = g.pad(z, vec![(1, 0), (0, 2)]).unwrap();
    chk(&g, n);
    // fused complex chain feeding a sum (z*z then reduce): exercises the materialized path
    let mut g = Graph::new();
    let z = mk(&mut g);
    let sq = g.mul(z, z).unwrap();
    let n = g.sum(sq, 0).unwrap();
    chk(&g, n);
}

// Complex seam device (construction/extraction/cast/where + conj/cabs/angle/fft/ifft).
// Real/Imag/Complex/conj/complex-where/real<->C64-cast are exact; cabs/angle/fft use
// transcendentals -> tolerance. All must match the CPU oracle on device.
#[test]
fn complex_seam_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let chk = |g: &Graph, n| assert_eq!(metal.eval(g, n).storage, interpret(g, n).storage, "device != oracle");
    let close = |g: &Graph, n| {
        let (gp, cp) = (metal.eval(g, n), interpret(g, n));
        assert_eq!(gp.shape, cp.shape);
        for (p, q) in gp.f32().iter().zip(cp.f32()) {
            assert!((p - q).abs() < 1e-4, "device {p} vs oracle {q}");
        }
    };
    // C64-aware tolerance compare (re/im components)
    let close_c = |g: &Graph, n| {
        let (gp, cp) = (metal.eval(g, n), interpret(g, n));
        assert_eq!(gp.shape, cp.shape);
        match (&gp.storage, &cp.storage) {
            (Storage::C64(a), Storage::C64(b)) => {
                for (x, y) in a.iter().zip(b) {
                    assert!((x.re - y.re).abs() < 1e-3 && (x.im - y.im).abs() < 1e-3, "{x:?} vs {y:?}");
                }
            }
            _ => panic!("want C64 outputs"),
        }
    };
    let re = vec![1.0f32, -2.0, 3.0, 0.5];
    let im = vec![0.5f32, 1.0, -1.0, 2.0];
    let mk = |g: &mut Graph| {
        let r = g.constant(re.clone(), vec![4]);
        let i = g.constant(im.clone(), vec![4]);
        g.complex(r, i).unwrap()
    };
    // Real / Imag extraction (exact data movement)
    let mut g = Graph::new();
    let z = mk(&mut g);
    let r = g.real(z).unwrap();
    chk(&g, r);
    let mut g = Graph::new();
    let z = mk(&mut g);
    let im_n = g.imag(z).unwrap();
    chk(&g, im_n);
    // conj (real - i*imag): exact (neg on the imag part)
    let mut g = Graph::new();
    let z = mk(&mut g);
    let c = g.conj(z).unwrap();
    chk(&g, c);
    // f32 -> C64 cast (imag 0) and C64 -> f32 cast (real part): exact
    let mut g = Graph::new();
    let x = g.constant(re.clone(), vec![4]);
    let up = g.cast(x, kurumi_core::DType::C64);
    chk(&g, up);
    let mut g = Graph::new();
    let z = mk(&mut g);
    let down = g.cast(z, kurumi_core::DType::F32);
    chk(&g, down);
    // complex where/select (float2): cond ? a : b
    let mut g = Graph::new();
    let a = mk(&mut g);
    let b = g.conj(a).unwrap();
    let cond = g.constant(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
    let cb = g.cast(cond, kurumi_core::DType::BOOL);
    let w = g.select(cb, a, b).unwrap();
    chk(&g, w);
    // cabs / angle (sqrt / atan2) -> tolerance
    let mut g = Graph::new();
    let z = mk(&mut g);
    let m = g.cabs(z).unwrap();
    close(&g, m);
    let mut g = Graph::new();
    let z = mk(&mut g);
    let ph = g.angle(z).unwrap();
    close(&g, ph);
    // fft round-trips through the complex construction + C64 GEMM, all device now
    let mut g = Graph::new();
    let z = mk(&mut g);
    let f = g.fft(z, 0).unwrap();
    close_c(&g, f);
    let mut g = Graph::new();
    let z = mk(&mut g);
    let f = g.fft(z, 0).unwrap();
    let back = g.ifft(f, 0).unwrap();
    close_c(&g, back);
}

// Complex C64 device: exact arithmetic add/mul(cmul)/neg/recip(crecip) + strided
// movement run device-resident (float2). Matches the CPU oracle exactly (same IEEE
// formulas). Construction/extraction/reduce also run device (see other tests).
#[test]
fn complex_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let chk = |g: &Graph, n| assert_eq!(metal.eval(g, n).storage, interpret(g, n).storage, "device != oracle");

    // z = re + i*im (Complex op is host); z*z (cmul) + z (add), then neg: device
    let mut g = Graph::new();
    let re = g.constant(vec![1.0, 2.0, 3.0, -1.0], vec![4]);
    let im = g.constant(vec![0.5, -1.0, 2.0, 1.5], vec![4]);
    let z = g.complex(re, im).unwrap();
    let m = g.mul(z, z).unwrap(); // cmul
    let a = g.add(m, z).unwrap();
    let n = g.neg(a);
    chk(&g, n);

    // complex recip (crecip), then z*(1/z): device cmul of the reciprocal
    let mut g = Graph::new();
    let re = g.constant(vec![1.0, 2.0, 0.5], vec![3]);
    let im = g.constant(vec![1.0, -1.0, 2.0], vec![3]);
    let z = g.complex(re, im).unwrap();
    let r = g.recip(z);
    let n = g.mul(r, z).unwrap();
    chk(&g, n);

    // complex strided movement (permute) fused with a device add
    let mut g = Graph::new();
    let re = g.constant((0..6).map(|i| i as f32).collect(), vec![2, 3]);
    let im = g.constant((0..6).map(|i| -(i as f32)).collect(), vec![2, 3]);
    let z = g.complex(re, im).unwrap();
    let p = g.permute(z, vec![1, 0]).unwrap();
    let s = g.add(p, p).unwrap();
    chk(&g, s);
}

// Complex C64 transcendentals: exp/sin/sqrt/log2 run device (float2 helpers), matching
// the CPU oracle within tolerance (num_complex vs MSL math, like real transcendentals).
#[test]
fn complex_transcendental_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let close = |g: &Graph, n| {
        let (gpu, cpu) = (metal.eval(g, n), interpret(g, n));
        let (Storage::C64(a), Storage::C64(b)) = (gpu.storage, cpu.storage) else { panic!("want C64") };
        for (x, y) in a.iter().zip(&b) {
            assert!((x.re - y.re).abs() < 1e-4 && (x.im - y.im).abs() < 1e-4, "device {x} vs oracle {y}");
        }
    };
    let build = |g: &mut Graph| {
        let re = g.constant(vec![0.5, -1.0, 2.0, 0.3], vec![4]);
        let im = g.constant(vec![1.0, 0.7, -0.5, 1.2], vec![4]);
        g.complex(re, im).unwrap()
    };
    // exp (quantum time-evolution phase), sqrt, log2: all device via c* helpers.
    // (complex `sin` is gated out at the builder in kurumi, so it's not exercised here.)
    let mut g = Graph::new();
    let z = build(&mut g);
    let e = g.exp(z);
    close(&g, e);
    let mut g = Graph::new();
    let z = build(&mut g);
    let r = g.sqrt(z);
    close(&g, r);
    let mut g = Graph::new();
    let z = build(&mut g);
    let l = g.log2(z);
    close(&g, l);
    // Euler round-trip: exp(z) then a fused chain: stays device
    let mut g = Graph::new();
    let z = build(&mut g);
    let e = g.exp(z);
    let m = g.mul(e, e).unwrap(); // (e^z)^2 = e^2z: cmul on transcendental output
    close(&g, m);
}

// Complex matmul device: 2D canonical complex GEMM (quantum gate application) runs
// device (naive cmul-accumulate), matching the CPU oracle within tolerance.
#[test]
fn complex_matmul_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let close = |g: &Graph, n| {
        let (gpu, cpu) = (metal.eval(g, n), interpret(g, n));
        let (Storage::C64(a), Storage::C64(b)) = (gpu.storage, cpu.storage) else { panic!("want C64") };
        for (x, y) in a.iter().zip(&b) {
            assert!((x.re - y.re).abs() < 1e-4 && (x.im - y.im).abs() < 1e-4, "device {x} vs oracle {y}");
        }
    };
    // 2x2 complex @ 2x2 complex (canonical dot_general -> device cmatmul)
    let mut g = Graph::new();
    let a = {
        let re = g.constant(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let im = g.constant(vec![0.0, 1.0, -1.0, 0.0], vec![2, 2]);
        g.complex(re, im).unwrap()
    };
    let b = {
        let re = g.constant(vec![0.5, 1.0, 1.0, 0.5], vec![2, 2]);
        let im = g.constant(vec![1.0, 0.0, 0.0, -1.0], vec![2, 2]);
        g.complex(re, im).unwrap()
    };
    let m = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
    close(&g, m);

    // batched [2,2,3]@[2,3,2] (multi-qubit gate) + transposed A^T@B (autograd-backward shape)
    let cpx = |g: &mut Graph, n: usize, sh: Vec<usize>, s: usize| {
        let re = g.constant((0..n).map(|i| (i as f32) * 0.1 + s as f32).collect(), sh.clone());
        let im = g.constant((0..n).map(|i| -(i as f32) * 0.07).collect(), sh);
        g.complex(re, im).unwrap()
    };
    let mut g = Graph::new();
    let a = cpx(&mut g, 12, vec![2, 2, 3], 0);
    let b = cpx(&mut g, 12, vec![2, 3, 2], 1);
    let m = g.dot_general(a, b, vec![2], vec![1], vec![0], vec![0]).unwrap(); // batched
    close(&g, m);
    let mut g = Graph::new();
    let a = cpx(&mut g, 6, vec![3, 2], 0);
    let b = cpx(&mut g, 12, vec![3, 4], 1);
    let m = g.dot_general(a, b, vec![0], vec![0], vec![], vec![]).unwrap(); // A^T@B (trans_l)
    close(&g, m);

    // 3x4 @ 4x2 complex (non-square, rules out shape/index bugs)
    let mut g = Graph::new();
    let a = {
        let re = g.constant((0..12).map(|i| i as f32 * 0.1).collect(), vec![3, 4]);
        let im = g.constant((0..12).map(|i| -(i as f32) * 0.05).collect(), vec![3, 4]);
        g.complex(re, im).unwrap()
    };
    let b = {
        let re = g.constant((0..8).map(|i| (i as f32) * 0.2 - 0.3).collect(), vec![4, 2]);
        let im = g.constant((0..8).map(|i| (i as f32) * 0.15).collect(), vec![4, 2]);
        g.complex(re, im).unwrap()
    };
    let m = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
    close(&g, m);
}
