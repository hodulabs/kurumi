#![cfg(target_os = "macos")]
//! Device-vs-oracle correctness: dtypes, int/bitwise, cast, iota/rand/bitcast,
//! scatter, argsort, matmul routing.

use kurumi_core::{Backend, DType, Graph, ScatterOp, Storage, interpret};
use kurumi_metal::MetalBackend;

// Every dtype pair casts on both CPU (interpret) and Metal (eval), and agrees.
#[test]
fn cast_all_dtype_pairs_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    use DType::*;
    let all = [BOOL, U8, U16, U32, U64, I8, I16, I32, I64, F8E4M3, F8E5M2, F16, BF16, F32, F64, C64, C128];
    // chained f32 -> from -> to so no exotic-storage construction is needed; both casts
    // run on each backend, and metal must match interpret for every pair.
    for &from in &all {
        for &to in &all {
            let mut g = Graph::new();
            let c = g.constant(vec![0.0, 1.0, 2.0, 3.0], vec![4]);
            let a = g.cast(c, from);
            let n = g.cast(a, to);
            assert_eq!(metal.eval(&g, n).storage, interpret(&g, n).storage, "cast {from:?} -> {to:?}");
        }
    }
}

// Cast semantics that a plain C-cast gets wrong: -> bool (!=0), float->int truncation
// toward zero, and saturation to the int range. Device must match the CPU oracle.
#[test]
fn cast_semantics_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let chk = |g: &Graph, n| assert_eq!(metal.eval(g, n).storage, interpret(g, n).storage, "device != oracle");

    // int -> bool: any nonzero -> true (not a value-preserving cast)
    let mut g = Graph::new();
    let c = g.constant(vec![0.0, 5.0, 2.0, 0.0], vec![4]);
    let i = g.cast(c, DType::I32);
    let n = g.cast(i, DType::BOOL);
    chk(&g, n);

    // float -> int: truncate toward zero
    let mut g = Graph::new();
    let c = g.constant(vec![2.7, -1.3, 0.9, -0.9], vec![4]);
    let n = g.cast(c, DType::I32);
    chk(&g, n);

    // float -> int: saturate out-of-range (300 -> u8 255, -5 -> 0; 200 -> i8 127, -200 -> -128)
    let mut g = Graph::new();
    let c = g.constant(vec![300.0, -5.0, 100.0], vec![3]);
    let n = g.cast(c, DType::U8);
    chk(&g, n);
    let mut g = Graph::new();
    let c = g.constant(vec![200.0, -200.0, 50.0], vec![3]);
    let n = g.cast(c, DType::I8);
    chk(&g, n);
}

// Argsort device: stable, asc/desc, ties, inner axis; sort/topk build on it.
#[test]
fn argsort_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let chk = |g: &Graph, n| assert_eq!(metal.eval(g, n).storage, interpret(g, n).storage, "device != oracle");

    // f32 with ties (3.0 twice) -> stable order (ascending index among equals)
    let mut g = Graph::new();
    let a = g.constant(vec![3.0, 1.0, 3.0, 2.0, 0.0], vec![5]);
    let asc = g.argsort(a, 0, false).unwrap();
    chk(&g, asc);
    let mut g = Graph::new();
    let a = g.constant(vec![3.0, 1.0, 3.0, 2.0, 0.0], vec![5]);
    let desc = g.argsort(a, 0, true).unwrap();
    chk(&g, desc);

    // i32, sort along axis 0 of a [3,2] (inner=2, exercises the strided line layout)
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![5, 2, 1, 8, 3, 3]), vec![3, 2]);
    let s = g.argsort(a, 0, false).unwrap();
    chk(&g, s);

    // full sort() (gathers values by the argsort permutation), f32 last axis
    let mut g = Graph::new();
    let a = g.constant(vec![4.0, 2.0, 9.0, 1.0, 7.0, 5.0], vec![2, 3]);
    let s = g.sort(a, 1, true).unwrap();
    chk(&g, s);
}

// general Scatter device: add (with duplicate + OOB indices), set, max, min.
#[test]
fn scatter_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let chk = |g: &Graph, n| assert_eq!(metal.eval(g, n).storage, interpret(g, n).storage, "device != oracle");

    // operand [pre=2, da=3, post=2], axis=1; distinct indices + an OOB index (5 ->
    // dropped). Set/Add/Max/Min all deterministic here (no duplicate-write race).
    let make = |g: &mut Graph, idx: Vec<i64>, combine| {
        let operand = g.constant((0..12).map(|i| i as f32).collect(), vec![2, 3, 2]);
        let k = idx.len();
        let idx = g.const_storage(Storage::I64(idx), vec![k]);
        let updates = g.constant((0..4 * k).map(|i| (i as f32) * 0.5 + 1.0).collect(), vec![2, k, 2]);
        g.scatter(operand, idx, updates, 1, combine).unwrap()
    };
    for combine in [ScatterOp::Set, ScatterOp::Add, ScatterOp::Max, ScatterOp::Min] {
        let mut g = Graph::new();
        let n = make(&mut g, vec![0, 2, 1, 5], combine);
        chk(&g, n);
    }
    // Add with a duplicate index (2 appears twice): commutative, so the atomic
    // accumulation is deterministic and must match the oracle.
    let mut g = Graph::new();
    let n = make(&mut g, vec![0, 2, 2, 5], ScatterOp::Add);
    chk(&g, n);

    // Set for a non-f32 dtype (i32): device direct-write path, general + along-axis
    let mut g = Graph::new();
    let operand = g.const_storage(Storage::I32((0..12).collect()), vec![2, 3, 2]);
    let idx = g.const_storage(Storage::I64(vec![0, 2, 1]), vec![3]);
    let upd = g.const_storage(Storage::I32((100..112).collect()), vec![2, 3, 2]);
    let n = g.scatter(operand, idx, upd, 1, ScatterOp::Set).unwrap();
    chk(&g, n);
    let mut g = Graph::new();
    let operand = g.const_storage(Storage::I32((0..6).collect()), vec![2, 3]);
    let idx = g.const_storage(Storage::I64(vec![2, 0, 1, 1, 2, 0]), vec![2, 3]);
    let upd = g.const_storage(Storage::I32((10..16).collect()), vec![2, 3]);
    let n = g.scatter_along(operand, idx, upd, 1, ScatterOp::Set).unwrap();
    chk(&g, n);

    // f32 scatter_along Add (index_add): device CAS atomic path
    let mut g = Graph::new();
    let operand = g.constant(vec![0.0; 6], vec![2, 3]);
    let idx = g.const_storage(Storage::I64(vec![0, 0, 1, 2, 2, 2]), vec![2, 3]);
    let upd = g.constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let n = g.scatter_along(operand, idx, upd, 1, ScatterOp::Add).unwrap();
    chk(&g, n);

    // i32 scatter Add with duplicate indices (histogram/bincount): native int atomic
    let mut g = Graph::new();
    let operand = g.const_storage(Storage::I32(vec![0; 4]), vec![4]);
    let idx = g.const_storage(Storage::I64(vec![0, 2, 2, 2, 3, 5]), vec![6]);
    let upd = g.const_storage(Storage::I32(vec![1, 1, 1, 1, 1, 1]), vec![6]);
    let n = g.scatter(operand, idx, upd, 0, ScatterOp::Add).unwrap();
    chk(&g, n);
    // u32 scatter_along Max: native int atomic
    let mut g = Graph::new();
    let operand = g.const_storage(Storage::U32(vec![5, 5, 5]), vec![3]);
    let idx = g.const_storage(Storage::I64(vec![0, 0, 1]), vec![3]);
    let upd = g.const_storage(Storage::U32(vec![3, 9, 7]), vec![3]);
    let n = g.scatter_along(operand, idx, upd, 0, ScatterOp::Max).unwrap();
    chk(&g, n);
}

// device-vs-oracle exact match for the full integer/bool dtype set.
#[test]
fn int_bool_dtypes_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let chk = |g: &Graph, n| {
        assert_eq!(metal.eval(g, n).storage, interpret(g, n).storage, "device != oracle");
    };

    // i32 fused elementwise (add, mul, max, neg): device fused kernel
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![1, -2, 3, 4]), vec![2, 2]);
    let b = g.const_storage(Storage::I32(vec![10, 20, 30, 40]), vec![2, 2]);
    let s = g.add(a, b).unwrap();
    let m = g.mul(s, a).unwrap();
    let mx = g.max(m, b).unwrap();
    let n = g.neg(mx);
    chk(&g, n);

    // i64 sum reduce with magnitudes > 2^24 (catches a float accumulator)
    let mut g = Graph::new();
    let big = g.const_storage(Storage::I64(vec![1 << 30, 1 << 30, 1 << 30, 5]), vec![2, 2]);
    let n = g.sum(big, 1).unwrap();
    chk(&g, n);

    // u8 max reduce + i16/u16/i8/u64 add
    let mut g = Graph::new();
    let u = g.const_storage(Storage::U8(vec![3, 250, 7, 9]), vec![2, 2]);
    let n = g.reduce_max(u, 0).unwrap();
    chk(&g, n);
    for pair in [
        Storage::I16(vec![100, -50, 30]),
        Storage::U16(vec![7, 65530, 3]),
        Storage::I8(vec![-5, 100, 20]),
        Storage::U64(vec![5_000_000_000, 1, 9]),
    ] {
        let mut g = Graph::new();
        let x = g.const_storage(pair.clone(), vec![3]);
        let y = g.const_storage(pair, vec![3]);
        let n = g.add(x, y).unwrap();
        chk(&g, n);
    }

    // i32 movement: permute -> slice -> add (strided device views)
    let mut g = Graph::new();
    let x = g.const_storage(Storage::I32((0..12).collect()), vec![3, 4]);
    let p = g.permute(x, vec![1, 0]).unwrap();
    let sl = g.slice(p, vec![(0, 4), (0, 2)]).unwrap();
    let n = g.add(sl, sl).unwrap();
    chk(&g, n);

    // i32 gather (embedding-style)
    let mut g = Graph::new();
    let t = g.const_storage(Storage::I32(vec![10, 11, 20, 21, 30, 31]), vec![3, 2]);
    let idx = g.const_storage(Storage::I64(vec![2, 0, 1]), vec![3]);
    let n = g.gather(t, idx, 0).unwrap();
    chk(&g, n);

    // i32 cmp_lt -> where select, and argmax
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![5, 2, 8, 1]), vec![4]);
    let b = g.const_storage(Storage::I32(vec![3, 3, 3, 3]), vec![4]);
    let c = g.cmp_lt(a, b).unwrap();
    let n = g.select(c, a, b).unwrap();
    chk(&g, n);
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![5, 2, 8, 1]), vec![4]);
    let n = g.argmax(a, 0).unwrap();
    chk(&g, n);

    // cast i32 -> f32 -> i64 (device casts, ints now a device dtype)
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![7, -3, 100]), vec![3]);
    let f = g.cast(a, DType::F32);
    let n = g.cast(f, DType::I64);
    chk(&g, n);
}

// int/bitwise ops fused on-device: idiv (incl. /0), and/or/xor, shl/shr
// (incl. shift-wrap), across widths: exact match to the CPU oracle.
#[test]
fn int_bitwise_ops_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let chk = |g: &Graph, n| assert_eq!(metal.eval(g, n).storage, interpret(g, n).storage, "device != oracle");

    // idiv with a zero divisor (x/0 = 0) and normal division, i32
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![10, -7, 42, 5]), vec![4]);
    let b = g.const_storage(Storage::I32(vec![3, 2, 0, 5]), vec![4]);
    let n = g.idiv(a, b).unwrap();
    chk(&g, n);

    // and/or/xor over i32, then fused with an add (single-dtype chain)
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![0b1100, 0b1010, 0xFF, 5]), vec![4]);
    let b = g.const_storage(Storage::I32(vec![0b1010, 0b0110, 0x0F, 3]), vec![4]);
    let x = g.and(a, b).unwrap();
    let y = g.or(a, b).unwrap();
    let z = g.xor(x, y).unwrap();
    let n = g.add(z, a).unwrap();
    chk(&g, n);

    // shl/shr with in- and out-of-range shift amounts (wrapping), i32 and u8
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![1, 64, -128, 7]), vec![4]);
    let s = g.const_storage(Storage::I32(vec![1, 1, 33, 2]), vec![4]); // 33 wraps to 1
    let l = g.shl(a, s).unwrap();
    let n = g.shr(l, s).unwrap();
    chk(&g, n);
    let mut g = Graph::new();
    let a = g.const_storage(Storage::U8(vec![1, 200, 255, 8]), vec![4]);
    let s = g.const_storage(Storage::U8(vec![1, 1, 9, 3]), vec![4]); // 9 wraps to 1
    let n = g.shl(a, s).unwrap();
    chk(&g, n);

    // bool and/or/xor (uchar-backed) -> device fused
    let mut g = Graph::new();
    let a = g.const_storage(Storage::BOOL(vec![true, true, false, false]), vec![4]);
    let b = g.const_storage(Storage::BOOL(vec![true, false, true, false]), vec![4]);
    let x = g.and(a, b).unwrap();
    let y = g.xor(a, b).unwrap();
    let n = g.or(x, y).unwrap();
    chk(&g, n);
}

// Iota / RandUniform / Bitcast device kernels: exact match to the oracle.
#[test]
fn iota_rand_bitcast_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let chk = |g: &Graph, n| assert_eq!(metal.eval(g, n).storage, interpret(g, n).storage, "device != oracle");

    // iota along each axis, i32 and f32
    let mut g = Graph::new();
    let n = g.iota(vec![3, 4], 1, DType::I32).unwrap();
    chk(&g, n);
    let mut g = Graph::new();
    let n = g.iota(vec![3, 4], 0, DType::F32).unwrap();
    chk(&g, n);

    // rand_uniform: bit-exact splitmix64 match (device vs CPU)
    let mut g = Graph::new();
    let n = g.rand_uniform(vec![5, 7], 0xABCD_1234);
    chk(&g, n);

    // bitcast f32 <-> i32 (same width, bit reinterpret)
    let mut g = Graph::new();
    let a = g.constant(vec![1.0, -2.5, 3.75, 0.0], vec![4]);
    let n = g.bitcast(a, DType::I32).unwrap();
    chk(&g, n);
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![1065353216, -1082130432, 0, 42]), vec![4]);
    let n = g.bitcast(a, DType::F32).unwrap();
    chk(&g, n);
}

#[test]
fn probe_metal_nonf32_and_rank3_route() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    // i32 2D matmul through eval (host_op -> device naive int kernel)
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![1, 2, 3, 4, 5, 6]), vec![2, 3]);
    let b = g.const_storage(Storage::I32(vec![1, 0, 0, 1, 1, 0]), vec![3, 2]);
    let m = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
    assert_eq!(metal.eval(&g, m).storage, interpret(&g, m).storage, "i32 matmul on metal");
    // rank-3 @ 2D f32 (linear layer, not reshaped) -> host fallback, must still match
    let mut g2 = Graph::new();
    let x = g2.constant((0..2 * 4 * 3).map(|i| i as f32 * 0.1).collect(), vec![2, 4, 3]);
    let w = g2.constant((0..3 * 5).map(|i| i as f32 * 0.2).collect(), vec![3, 5]);
    let y = g2.dot_general(x, w, vec![2], vec![0], vec![], vec![]).unwrap();
    let (gp, cp) = (metal.eval(&g2, y), interpret(&g2, y));
    assert_eq!(gp.shape, vec![2, 4, 5]);
    for (p, q) in gp.f32().iter().zip(cp.f32()) {
        assert!((p - q).abs() < 1e-3, "{p} vs {q}");
    }
}

// f32 dense linalg on device (solve/det/cholesky, one thread per batch matrix) matches
// the CPU oracle within tolerance: enabled by the dtype-native linalg (f32 computes f32).
#[test]
fn linalg_f32_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let close = |g: &Graph, n| {
        let (gp, cp) = (metal.eval(g, n), interpret(g, n));
        for (p, q) in gp.f32().iter().zip(cp.f32()) {
            assert!((p - q).abs() < 1e-3, "device {p} vs oracle {q}");
        }
        assert_eq!(gp.shape, cp.shape);
    };
    // solve A*x = b, A = [[4,1],[1,3]], b = [[1],[2]]
    let mut g = Graph::new();
    let a = g.constant(vec![4.0, 1.0, 1.0, 3.0], vec![2, 2]);
    let b = g.constant(vec![1.0, 2.0], vec![2, 1]);
    let x = g.solve(a, b).unwrap();
    close(&g, x);
    // det of a 3x3
    let mut g = Graph::new();
    let a = g.constant(vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0], vec![3, 3]);
    let d = g.det(a).unwrap();
    close(&g, d);
    // cholesky of SPD [[4,2],[2,3]] -> [[2,0],[1,sqrt2]]
    let mut g = Graph::new();
    let a = g.constant(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2]);
    let l = g.cholesky(a).unwrap();
    close(&g, l);
    // batched solve [2,2,2] @ [2,2,1] (batch-parallel path)
    let mut g = Graph::new();
    let a = g.constant(vec![4.0, 1.0, 1.0, 3.0, 2.0, 0.0, 0.0, 5.0], vec![2, 2, 2]);
    let b = g.constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2, 1]);
    let x = g.solve(a, b).unwrap();
    close(&g, x);
    // inv (decomposes to solve) round-trip: A @ inv(A) ~ I
    let mut g = Graph::new();
    let a = g.constant(vec![4.0, 1.0, 1.0, 3.0], vec![2, 2]);
    let ai = g.inv(a).unwrap();
    let prod = g.dot_general(a, ai, vec![1], vec![0], vec![], vec![]).unwrap();
    close(&g, prod);
}

// General resize (1D/3D, coord modes, all interps) + dilated reduce_window: device
// (gather/slice_step/mul/add/max decompositions) must match the CPU oracle.
#[test]
fn resize_reduce_window_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let close = |g: &Graph, n| {
        let (gp, cp) = (metal.eval(g, n), interpret(g, n));
        assert_eq!(gp.shape, cp.shape);
        for (p, q) in gp.f32().iter().zip(cp.f32()) {
            assert!((p - q).abs() < 1e-4, "device {p} vs oracle {q}");
        }
    };
    // dilated max/avg reduce_window over a 2-D field
    let mut g = Graph::new();
    let x = g.constant((0..24).map(|i| i as f32).collect(), vec![1, 1, 4, 6]);
    let rw = g.reduce_window(x, &[2, 2], &[1, 1], &[2, 2], "max").unwrap();
    close(&g, rw);
    let aw = g.reduce_window(x, &[2, 3], &[2, 1], &[1, 2], "avg").unwrap();
    close(&g, aw);
    // resize: 1-D linear/nearest/cubic across every coord mode
    let a = g.constant((0..5).map(|i| (i * i) as f32).collect(), vec![1, 1, 5]);
    for interp in ["nearest", "linear", "cubic"] {
        for coord in ["half_pixel", "align_corners", "asymmetric", "pytorch_half_pixel"] {
            let up = g.resize(a, &[2], &[9], interp, coord).unwrap();
            close(&g, up);
            let down = g.resize(a, &[2], &[3], interp, coord).unwrap();
            close(&g, down);
        }
    }
    // 3-D trilinear resize
    let c = g.constant((0..8).map(|i| i as f32).collect(), vec![1, 1, 2, 2, 2]);
    let tri = g.resize(c, &[2, 3, 4], &[3, 4, 3], "linear", "half_pixel").unwrap();
    close(&g, tri);
}

// Eigen family on device (eigh/qr/svd/eigvals) vs the CPU oracle. Eigenvectors are
// sign-non-unique, so validate via sign-robust invariants: reconstruction and the
// (sorted) eigenvalue/singular-value spectra.
#[test]
fn eigen_device_match_oracle() {
    let Some(metal) = MetalBackend::new() else { return };
    let close = |g: &Graph, n| {
        let (gp, cp) = (metal.eval(g, n), interpret(g, n));
        assert_eq!(gp.shape, cp.shape);
        for (p, q) in gp.f32().iter().zip(cp.f32()) {
            assert!((p - q).abs() < 1e-2, "device {p} vs oracle {q}");
        }
    };
    let mm = |g: &mut Graph, a, b| g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();

    // eigh: eigenvalues (ascending, unique) + reconstruction V*diag(lambda)*V^T
    let mut g = Graph::new();
    let a = g.constant(vec![2., 1., 1., 3.], vec![2, 2]);
    let (vals, vecs) = g.eigh(a).unwrap();
    close(&g, vals);
    let d = g.diag_embed(vals).unwrap();
    let vd = mm(&mut g, vecs, d);
    let vt = g.transpose(vecs, 0, 1).unwrap();
    let recon = mm(&mut g, vd, vt);
    close(&g, recon);

    // qr: Q*R = A
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
    let (q, r) = g.qr(a).unwrap();
    let qr = mm(&mut g, q, r);
    close(&g, qr);

    // svd (rides eigh on device): U*diag(S)*V^T = A + singular values
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3., 4.], vec![2, 2]);
    let (u, s, v) = g.svd(a).unwrap();
    close(&g, s);
    let ds = g.diag_embed(s).unwrap();
    let us = mm(&mut g, u, ds);
    let vt = g.transpose(v, 0, 1).unwrap();
    let recon = mm(&mut g, us, vt);
    close(&g, recon);

    // eigvals: complex spectrum (rotation -> +/-i); compare sorted re / im
    let mut g = Graph::new();
    let a = g.constant(vec![0., -1., 1., 0.], vec![2, 2]);
    let ev = g.eigvals(a).unwrap();
    let re = g.real(ev).unwrap();
    let sre = g.sort(re, 0, false).unwrap();
    close(&g, sre);
    let im = g.imag(ev).unwrap();
    let sim = g.sort(im, 0, false).unwrap();
    close(&g, sim);
}

// The lazy Tensor handle is device-agnostic: the SAME handle code realizes on
// Metal (via Ctx::with_backend) and matches the CPU handle within tolerance.
#[test]
fn tensor_handle_device_agnostic() {
    let Some(metal) = MetalBackend::new() else { return };
    fn compute(ctx: &kurumi_core::Ctx) -> Vec<f32> {
        let a = ctx.constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = ctx.constant(vec![0.5, -1.0, 2.0, 0.25], vec![2, 2]);
        a.matmul(&b).unwrap().add(&a).unwrap().relu().to_vec()
    }
    let cpu = compute(&kurumi_core::Ctx::cpu());
    let dev = compute(&kurumi_core::Ctx::with_backend(Box::new(metal)));
    for (c, d) in cpu.iter().zip(&dev) {
        assert!((c - d).abs() < 1e-4, "cpu {c} vs metal {d}");
    }
}
