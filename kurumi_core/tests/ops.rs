//! Integration tests: primitive & decomposed ops (indexing, RNG, norm, attention).

use kurumi_core::*;

// new ops

fn i32(g: &mut Graph, data: Vec<i32>, shape: Vec<usize>) -> NodeId {
    g.const_storage(Storage::I32(data), shape)
}

#[test]
fn floor_op() {
    let mut g = Graph::new();
    let a = g.constant(vec![1.7, -0.3, 2.0, -2.5], vec![4]);
    let y = g.floor(a);
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![4], storage: Storage::F32(vec![1., -1., 2., -3.]) });
}

#[test]
fn prod_reduce() {
    let mut g = Graph::new();
    let a = i32(&mut g, vec![1, 2, 3, 4], vec![2, 2]);
    let y = g.prod(a, 1).unwrap(); // [1*2, 3*4]
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![2], storage: Storage::I32(vec![2, 12]) });
}

#[test]
fn idiv_and_div_by_zero() {
    let mut g = Graph::new();
    let a = i32(&mut g, vec![7, 8, 9], vec![3]);
    let b = i32(&mut g, vec![2, 2, 0], vec![3]);
    let y = g.idiv(a, b).unwrap(); // [3, 4, 0(=x/0)]
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![3], storage: Storage::I32(vec![3, 4, 0]) });
}

#[test]
fn shifts() {
    let mut g = Graph::new();
    let a = i32(&mut g, vec![1, 2, 16], vec![3]);
    let s = i32(&mut g, vec![1, 2, 1], vec![3]);
    let l = g.shl(a, s).unwrap();
    assert_eq!(interpret(&g, l).storage, Storage::I32(vec![2, 8, 32]));
    let r = g.shr(a, s).unwrap();
    assert_eq!(interpret(&g, r).storage, Storage::I32(vec![0, 0, 8]));
}

#[test]
fn cmp_produces_bool() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3.], vec![3]);
    let b = g.constant(vec![2., 2., 2.], vec![3]);
    let lt = g.cmp_lt(a, b).unwrap();
    assert_eq!(g.dtype(lt), DType::BOOL);
    assert_eq!(interpret(&g, lt).storage, Storage::BOOL(vec![true, false, false]));
    let eq = g.cmp_eq(a, b).unwrap();
    assert_eq!(interpret(&g, eq).storage, Storage::BOOL(vec![false, true, false]));
}

#[test]
fn logical_ops_on_bool() {
    let mut g = Graph::new();
    let a = g.const_storage(Storage::BOOL(vec![true, true, false, false]), vec![4]);
    let b = g.const_storage(Storage::BOOL(vec![true, false, true, false]), vec![4]);
    let (and, or, xor) = (g.and(a, b).unwrap(), g.or(a, b).unwrap(), g.xor(a, b).unwrap());
    assert_eq!(interpret(&g, and).storage, Storage::BOOL(vec![true, false, false, false]));
    assert_eq!(interpret(&g, or).storage, Storage::BOOL(vec![true, true, true, false]));
    assert_eq!(interpret(&g, xor).storage, Storage::BOOL(vec![false, true, true, false]));
}

#[test]
fn select_picks_by_cond() {
    let mut g = Graph::new();
    let cond = g.const_storage(Storage::BOOL(vec![true, false, true]), vec![3]);
    let a = g.constant(vec![1., 2., 3.], vec![3]);
    let b = g.constant(vec![10., 20., 30.], vec![3]);
    let y = g.select(cond, a, b).unwrap();
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![3], storage: Storage::F32(vec![1., 20., 3.]) });
}

#[test]
fn select_requires_bool_cond() {
    let mut g = Graph::new();
    let c = g.constant(vec![1., 0.], vec![2]); // f32, not BOOL
    let a = g.constant(vec![1., 2.], vec![2]);
    let b = g.constant(vec![3., 4.], vec![2]);
    assert!(g.select(c, a, b).is_err());
}

#[test]
fn iota_along_axes() {
    let mut g = Graph::new();
    let row = g.iota(vec![2, 3], 1, DType::I32).unwrap(); // index along cols
    assert_eq!(interpret(&g, row).storage, Storage::I32(vec![0, 1, 2, 0, 1, 2]));
    let col = g.iota(vec![2, 3], 0, DType::I32).unwrap(); // index along rows
    assert_eq!(interpret(&g, col).storage, Storage::I32(vec![0, 0, 0, 1, 1, 1]));
}

// a non-f32 graph using new ops: realize must fall back and match the oracle
#[test]
fn realize_falls_back_for_new_ops() {
    let mut g = Graph::new();
    let a = g.iota(vec![6], 0, DType::I32).unwrap();
    let two = i32(&mut g, vec![2; 6], vec![6]);
    let y = g.idiv(a, two).unwrap(); // [0,0,1,1,2,2]
    assert_eq!(kurumi_core::realize::force(&g, y), interpret(&g, y));
}

#[test]
fn bitcast_reinterprets_bits() {
    let mut g = Graph::new();
    let a = g.constant(vec![1.0f32, 2.0], vec![2]);
    let y = g.bitcast(a, DType::U32).unwrap(); // IEEE-754 bit patterns
    assert_eq!(interpret(&g, y).storage, Storage::U32(vec![0x3F80_0000, 0x4000_0000]));
    // round-trip back to f32
    let z = g.bitcast(y, DType::F32).unwrap();
    assert_eq!(interpret(&g, z).storage, Storage::F32(vec![1.0, 2.0]));
}

#[test]
fn bitcast_width_mismatch_errors() {
    let mut g = Graph::new();
    let a = g.constant(vec![1.0f32], vec![1]); // 4 bytes
    assert!(g.bitcast(a, DType::U8).is_err()); // 1 byte
}

#[test]
fn gather_embedding_lookup() {
    // weight[V=3, E=2], ids[B=2, S=2] -> out[2, 2, 2] (gather rows along axis 0)
    let mut g = Graph::new();
    let w = g.constant(vec![10., 11., 20., 21., 30., 31.], vec![3, 2]); // rows 0,1,2
    let ids = i32(&mut g, vec![2, 0, 1, 2], vec![2, 2]);
    let y = g.gather(w, ids, 0).unwrap();
    assert_eq!(g.shape(y), vec![2, 2, 2]);
    // rows: 2,0 / 1,2
    assert_eq!(interpret(&g, y).storage, Storage::F32(vec![30., 31., 10., 11., 20., 21., 30., 31.]));
}

#[test]
fn gather_clamps_oob() {
    let mut g = Graph::new();
    let a = g.constant(vec![5., 6., 7.], vec![3]);
    let ix = i32(&mut g, vec![-1, 1, 9], vec![3]); // clamp to 0,1,2
    let y = g.gather(a, ix, 0).unwrap();
    assert_eq!(interpret(&g, y).storage, Storage::F32(vec![5., 6., 7.]));
}

#[test]
fn scatter_set_and_add() {
    let mut g = Graph::new();
    let base = g.constant(vec![0., 0., 0., 0.], vec![4]);
    let ix = i32(&mut g, vec![1, 3], vec![2]);
    let up = g.constant(vec![9., 7.], vec![2]);
    let set = g.scatter(base, ix, up, 0, ScatterOp::Set).unwrap();
    assert_eq!(interpret(&g, set).storage, Storage::F32(vec![0., 9., 0., 7.]));
    // add into [1,1,1,1]
    let ones = g.constant(vec![1., 1., 1., 1.], vec![4]);
    let add = g.scatter(ones, ix, up, 0, ScatterOp::Add).unwrap();
    assert_eq!(interpret(&g, add).storage, Storage::F32(vec![1., 10., 1., 8.]));
}

#[test]
fn scatter_drops_oob() {
    let mut g = Graph::new();
    let base = g.constant(vec![0., 0., 0.], vec![3]);
    let ix = i32(&mut g, vec![5, 1], vec![2]); // 5 dropped, 1 written
    let up = g.constant(vec![9., 4.], vec![2]);
    let y = g.scatter(base, ix, up, 0, ScatterOp::Set).unwrap();
    assert_eq!(interpret(&g, y).storage, Storage::F32(vec![0., 4., 0.]));
}

#[test]
fn gather_scatter_round_trip_via_realize() {
    // realize must fall back (int indices -> non-f32 graph) and match the oracle
    let mut g = Graph::new();
    let w = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
    let ids = i32(&mut g, vec![0, 2], vec![2]);
    let y = g.gather(w, ids, 0).unwrap();
    assert_eq!(kurumi_core::realize::force(&g, y), interpret(&g, y));
}

// Input nodes fed at eval time produce the same result as baking the values as
// constants, and grad flows into an Input (a param) exactly as into anything.
#[test]
fn input_feeds_match_baked_consts() {
    let (av, bv) = (vec![1., 2., 3., 4.], vec![5., 6., 7., 8.]);
    let mut g = Graph::new();
    let a = g.input(vec![2, 2], DType::F32);
    let b = g.input(vec![2, 2], DType::F32);
    let m = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
    let r = g.add(m, a).unwrap();
    let dr = grad(&mut g, r, &[a]).unwrap()[0]; // dr/da exists (a is a leaf Input)
    let feeds = Feeds::from([
        (a, TensorVal { shape: vec![2, 2], storage: Storage::F32(av.clone()) }),
        (b, TensorVal { shape: vec![2, 2], storage: Storage::F32(bv.clone()) }),
    ]);
    let fed = interpret_with(&g, r, &feeds);

    let mut g2 = Graph::new();
    let a2 = g2.constant(av, vec![2, 2]);
    let b2 = g2.constant(bv, vec![2, 2]);
    let m2 = g2.dot_general(a2, b2, vec![1], vec![0], vec![], vec![]).unwrap();
    let r2 = g2.add(m2, a2).unwrap();
    assert_eq!(fed, interpret(&g2, r2));
    assert_eq!(interpret_with(&g, dr, &feeds).shape, vec![2, 2]); // grad shape = param shape
}

// RMSNorm gives each row unit mean-square; SiLU(0)=0 and matches x*sigmoid(x).
#[test]
fn rmsnorm_unit_rms_and_silu() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3., 4., -2., 0.5], vec![2, 3]);
    let y = g.rmsnorm(x, 1, 1e-6).unwrap();
    let o = interpret(&g, y);
    for r in 0..2 {
        let ms: f32 = (0..3).map(|c| o.f32()[r * 3 + c].powi(2)).sum::<f32>() / 3.0;
        assert!((ms - 1.0).abs() < 1e-3, "row {r} mean-square = {ms}");
    }
    let mut g2 = Graph::new();
    let a = g2.constant(vec![0., 1., -1.], vec![3]);
    let sl = g2.silu(a);
    let s = interpret(&g2, sl);
    assert!(s.f32()[0].abs() < 1e-6); // silu(0) = 0
    assert!((s.f32()[1] - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-5); // silu(1) = sigmoid(1)
}

// cross-entropy of one-hot targets == -ln(softmax at the target class).
#[test]
fn cross_entropy_matches_neg_log_softmax() {
    let mut g = Graph::new();
    let logits = g.constant(vec![1., 2., 3., 0., -1., 0.5], vec![2, 3]);
    let targets = g.constant(vec![0., 0., 1., 1., 0., 0.], vec![2, 3]); // class 2, class 0
    let ce = g.cross_entropy(logits, targets, 1).unwrap();
    let sm = g.softmax(logits, 1).unwrap();
    let ceo = interpret(&g, ce).f32().to_vec();
    let smo = interpret(&g, sm).f32().to_vec();
    assert!((ceo[0] - (-smo[2].ln())).abs() < 1e-4, "ce0 {} vs {}", ceo[0], -smo[2].ln());
    assert!((ceo[1] - (-smo[3].ln())).abs() < 1e-4, "ce1 {} vs {}", ceo[1], -smo[3].ln());
}

// RoPE is a per-position rotation, so it preserves each position's vector norm.
#[test]
fn rope_preserves_norm() {
    let (s, d) = (4usize, 6usize);
    let mut g = Graph::new();
    let x = g.constant((0..s * d).map(|i| ((i * 7 % 11) as f32) * 0.3 - 1.0).collect(), vec![s, d]);
    let y = g.rope(x).unwrap();
    let (xo, yo) = (interpret(&g, x), interpret(&g, y));
    for pos in 0..s {
        let nx: f32 = (0..d).map(|c| xo.f32()[pos * d + c].powi(2)).sum();
        let ny: f32 = (0..d).map(|c| yo.f32()[pos * d + c].powi(2)).sum();
        assert!((nx - ny).abs() < 1e-3, "pos {pos}: |x|^2={nx} vs |rope|^2={ny}");
    }
}

// causal SDPA: query 0 can only attend to key 0, so out[0,:] == v[0,:].
#[test]
fn sdpa_causal_first_query_attends_to_first_key() {
    let (s, dh) = (4usize, 3usize);
    let mut g = Graph::new();
    let q = g.constant((0..s * dh).map(|i| i as f32 * 0.1).collect(), vec![s, dh]);
    let k = g.constant((0..s * dh).map(|i| (i as f32 * 0.2).sin()).collect(), vec![s, dh]);
    let v = g.constant((0..s * dh).map(|i| i as f32 + 1.0).collect(), vec![s, dh]);
    let out = g.sdpa(q, k, v, true).unwrap();
    let o = interpret(&g, out);
    assert_eq!(o.shape, vec![s, dh]);
    for d in 0..dh {
        assert!((o.f32()[d] - (1.0 + d as f32)).abs() < 1e-5, "out[0,{d}] = {}", o.f32()[d]);
    }
}

// The fused SDPA primitive (Op::Sdpa) is the flash-kernel oracle: its forward AND backward
// must match the dot+softmax decomposition (g.sdpa) bit-close, for both causal and full attn.
#[test]
fn sdpa_fused_matches_decomposition() {
    let (b, h, s, dh) = (2usize, 3usize, 5usize, 4usize);
    let n = b * h * s * dh;
    let mk = |seed: f32| -> Vec<f32> { (0..n).map(|i| (i as f32 * seed).sin() * 0.5).collect() };
    for causal in [false, true] {
        let mut g = Graph::new();
        let q = g.constant(mk(0.7), vec![b, h, s, dh]);
        let k = g.constant(mk(1.3), vec![b, h, s, dh]);
        let v = g.constant(mk(2.1), vec![b, h, s, dh]);
        let fused = g.sdpa_fused(q, k, v, causal).unwrap();
        let decomp = g.sdpa_decomposed(q, k, v, causal).unwrap(); // explicit dot+softmax reference
        // forward: fused == decomposition
        let of = interpret(&g, fused);
        assert_eq!(of.shape, vec![b, h, s, dh]);
        let od = interpret(&g, decomp).f32().to_vec();
        for (i, (a, e)) in of.f32().iter().zip(&od).enumerate() {
            assert!((a - e).abs() < 1e-5, "causal={causal} fwd[{i}] {a} vs {e}");
        }
        // backward: grad(sum(fused)) == grad(sum(decomposition)) w.r.t q, k, v
        let gf = grad(&mut g, fused, &[q, k, v]).unwrap();
        let gd = grad(&mut g, decomp, &[q, k, v]).unwrap();
        for (w, (&nf, &nd)) in ["dq", "dk", "dv"].iter().zip(gf.iter().zip(&gd)) {
            let (a, e) = (interpret(&g, nf).f32().to_vec(), interpret(&g, nd).f32().to_vec());
            for (i, (x, y)) in a.iter().zip(&e).enumerate() {
                assert!((x - y).abs() < 1e-4, "causal={causal} {w}[{i}] {x} vs {y}");
            }
        }
    }

    // finite-difference sanity on a tiny non-causal case: analytic dq vs central difference.
    let (s2, dh2) = (3usize, 2usize);
    let m = s2 * dh2;
    let base: Vec<f32> = (0..3 * m).map(|i| (i as f32 * 0.37).cos() * 0.4).collect();
    let (qb, kb, vb) = (base[..m].to_vec(), base[m..2 * m].to_vec(), base[2 * m..].to_vec());
    let loss = |qd: &[f32]| -> f32 {
        let mut g = Graph::new();
        let q = g.constant(qd.to_vec(), vec![s2, dh2]);
        let k = g.constant(kb.clone(), vec![s2, dh2]);
        let v = g.constant(vb.clone(), vec![s2, dh2]);
        let o = g.sdpa_fused(q, k, v, false).unwrap();
        let r0 = g.sum(o, 1).unwrap();
        let r1 = g.sum(r0, 0).unwrap();
        interpret(&g, r1).f32()[0]
    };
    let mut g = Graph::new();
    let q = g.constant(qb.clone(), vec![s2, dh2]);
    let k = g.constant(kb.clone(), vec![s2, dh2]);
    let v = g.constant(vb.clone(), vec![s2, dh2]);
    let o = g.sdpa_fused(q, k, v, false).unwrap();
    let dq = grad(&mut g, o, &[q]).unwrap()[0];
    let dqa = interpret(&g, dq).f32().to_vec();
    let eps = 1e-3;
    for idx in 0..m {
        let (mut qp, mut qm) = (qb.clone(), qb.clone());
        qp[idx] += eps;
        qm[idx] -= eps;
        let fd = (loss(&qp) - loss(&qm)) / (2.0 * eps);
        assert!((fd - dqa[idx]).abs() < 1e-2, "FD dq[{idx}] {fd} vs {}", dqa[idx]);
    }
}

// diag_embed needs a trailing axis: a rank-0 (scalar) input is a clean error, not a panic.
#[test]
fn diag_embed_rank0_is_err() {
    let mut g = Graph::new();
    let s = g.constant(vec![5.0], vec![]);
    assert!(g.diag_embed(s).is_err());
}

// quant_matmul validates bits/group_size/qweight/scales at record time; a bad value would
// otherwise defer to an eval-time unreachable or a silent-wrong dequant.
#[test]
fn quant_matmul_guard() {
    let mut g = Graph::new();
    let act = g.constant(vec![0.0; 4], vec![1, 4]); // [M=1, K=4]
    let qw = g.const_storage(Storage::U8(vec![0; 4]), vec![2, 2]); // [N=2, K*bits/8=2] for int4
    let sc = g.constant(vec![0.0; 4], vec![2, 2]); // [N=2, K/group_size=2] -> 4 entries
    // valid: int4, group_size 2, consistent qweight cols + scales count
    assert!(g.quant_matmul(act, qw, sc, None, 4, 2).is_ok());
    // bits not in {2,4,8}
    assert!(g.quant_matmul(act, qw, sc, None, 3, 2).is_err());
    // group_size 0, and a non-divisor of K=4
    assert!(g.quant_matmul(act, qw, sc, None, 4, 0).is_err());
    assert!(g.quant_matmul(act, qw, sc, None, 4, 3).is_err());
    // qweight cols inconsistent with K*bits/8: int8 needs cols=4, qw has 2
    assert!(g.quant_matmul(act, qw, sc, None, 8, 2).is_err());
    // scales entry count mismatch (want N*K/group_size = 4)
    let bad_sc = g.constant(vec![0.0; 2], vec![2, 1]);
    assert!(g.quant_matmul(act, qw, bad_sc, None, 4, 2).is_err());
}
