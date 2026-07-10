//! GPT-shape forward throughput benchmarks. The Llama-shape prefill lives in `llama.rs`,
//! the flash-vs-decomp attention microbench in `attention.rs`.

use crate::tests::*;

// Multi-layer GPT (token embed -> N x [RMSNorm + causal MHA + SwiGLU] -> final RMSNorm
// -> lm_head) forward, fully device-resident, f16 vs f32 throughput. f16 = half the
// memory and MPS f16 GEMM uses the native half matrix units, so f16 should win.
#[test]
#[ignore = "benchmark; run with --release"]
fn metal_gpt_throughput_bench() {
    use half::f16;
    use kurumi_core::{Backend, DType, NodeId, Storage, grad};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (vocab, b, s, h, dh, layers) = (1024usize, 8, 128, 4, 32, 4);
    let (dm, dff) = (h * dh, 4 * h * dh);
    let con = |g: &mut Graph, dt: DType, seed: usize, shape: Vec<usize>| -> NodeId {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (((i * 7 + seed) % 23) as f32) * 0.01 - 0.1).collect();
        let st = match dt {
            DType::F16 => Storage::F16(data.iter().map(|&x| f16::from_f32(x)).collect()),
            _ => Storage::F32(data),
        };
        g.const_storage(st, shape)
    };
    let build = |dt: DType| -> (Graph, NodeId, NodeId, NodeId) {
        let mut g = Graph::new();
        let ids: Vec<i32> = (0..b * s).map(|i| (i * 13 % vocab) as i32).collect();
        let tok = g.const_storage(Storage::I32(ids), vec![b, s]);
        let embed = con(&mut g, dt, 0, vec![vocab, dm]);
        let x0 = g.gather(embed, tok, 0).unwrap(); // [b,s,dm]
        let mut x = x0;
        for l in 0..layers {
            let sd = l * 100;
            let hn = g.rmsnorm(x, 2, 1e-5).unwrap();
            let h2 = g.reshape(hn, vec![b * s, dm]).unwrap();
            let head = |g: &mut Graph, seed| {
                let w = con(g, dt, seed, vec![dm, dm]);
                let p = g.dot_general(h2, w, vec![1], vec![0], vec![], vec![]).unwrap();
                let r = g.reshape(p, vec![b, s, h, dh]).unwrap();
                g.permute(r, vec![0, 2, 1, 3]).unwrap()
            };
            let (q, k, v) = (head(&mut g, sd + 1), head(&mut g, sd + 2), head(&mut g, sd + 3));
            let attn = g.sdpa(q, k, v, true).unwrap();
            let attn = g.permute(attn, vec![0, 2, 1, 3]).unwrap();
            let attn = g.reshape(attn, vec![b * s, dm]).unwrap();
            let wo = con(&mut g, dt, sd + 4, vec![dm, dm]);
            let o = g.dot_general(attn, wo, vec![1], vec![0], vec![], vec![]).unwrap();
            let o = g.reshape(o, vec![b, s, dm]).unwrap();
            x = g.add(x, o).unwrap();
            let n2 = g.rmsnorm(x, 2, 1e-5).unwrap();
            let m2 = g.reshape(n2, vec![b * s, dm]).unwrap();
            let wg = con(&mut g, dt, sd + 5, vec![dm, dff]);
            let wu = con(&mut g, dt, sd + 6, vec![dm, dff]);
            let wd = con(&mut g, dt, sd + 7, vec![dff, dm]);
            let gate = {
                let gp = g.dot_general(m2, wg, vec![1], vec![0], vec![], vec![]).unwrap();
                g.silu(gp)
            };
            let up = g.dot_general(m2, wu, vec![1], vec![0], vec![], vec![]).unwrap();
            let gu = g.mul(gate, up).unwrap();
            let mlp = g.dot_general(gu, wd, vec![1], vec![0], vec![], vec![]).unwrap();
            let mlp = g.reshape(mlp, vec![b, s, dm]).unwrap();
            x = g.add(x, mlp).unwrap();
        }
        let fin = g.rmsnorm(x, 2, 1e-5).unwrap();
        let f2 = g.reshape(fin, vec![b * s, dm]).unwrap();
        let lm = con(&mut g, dt, 999, vec![dm, vocab]);
        let logits = g.dot_general(f2, lm, vec![1], vec![0], vec![], vec![]).unwrap();
        (g, logits, embed, x0)
    };
    let bench = |run: &dyn Fn()| {
        run();
        let t = Instant::now();
        for _ in 0..10 {
            run();
        }
        t.elapsed().as_secs_f64() / 10.0
    };
    let toks = (b * s) as f64;
    for dt in [DType::F32, DType::F16] {
        let (mut g, logits, embed, x0) = build(dt);
        // forward-only
        let fwd = bench(&|| {
            metal.eval(&g, logits);
        });
        // forward + full backward: grad of a scalar loss wrt the embedding table
        // (the deepest leaf -> backprop flows through every layer). grad needs an
        // f32 loss, so f16 logits upcast first (mixed precision: f16 fwd, f32 loss).
        let lf = if dt == DType::F32 { logits } else { g.cast(logits, DType::F32) };
        let s1 = g.sum(lf, 1).unwrap();
        let loss = g.sum(s1, 0).unwrap();
        // grad wrt the embedding TABLE includes the gather-vjp scatter (host op);
        // grad wrt the gather OUTPUT x0 stops one step short -> same backprop depth,
        // no scatter. the delta isolates the host scatter's cost in the backward.
        let ge = grad(&mut g, loss, &[embed]).unwrap()[0];
        let gx = grad(&mut g, loss, &[x0]).unwrap()[0];
        let train = bench(&|| {
            metal.eval(&g, ge);
        });
        let noscat = bench(&|| {
            metal.eval(&g, gx);
        });
        eprintln!(
            "{layers}-layer GPT (dm={dm}, {b}x{s} tok) {dt:?}:  fwd {:.2} ms ({:.0} tok/s)  |  fwd+bwd {:.2} ms ({:.0} tok/s)  |  bwd-no-scatter {:.2} ms (scatter ~{:.2} ms)",
            fwd * 1e3,
            toks / fwd,
            train * 1e3,
            toks / train,
            noscat * 1e3,
            (train - noscat) * 1e3
        );
    }
}

// GPT forward across model scales: is the engine compute-bound (GPU dominates -> competitive
// with any Metal lib) at real sizes, or overhead-bound (dispatch/encode) only at toy sizes?
// Run with KURUMI_PHASE=1 KURUMI_GPUTIME=1 to see encode/flush/GPU per scale.
#[test]
#[ignore = "benchmark; run with --release"]
fn metal_gpt_scale_sweep() {
    use half::f16;
    use kurumi_core::{Backend, NodeId, Storage};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let con = |g: &mut Graph, seed: usize, shape: Vec<usize>| -> NodeId {
        let n: usize = shape.iter().product();
        let data: Vec<f16> = (0..n).map(|i| f16::from_f32(((i * 7 + seed) % 23) as f32 * 0.01 - 0.1)).collect();
        g.const_storage(Storage::F16(data), shape)
    };
    let bench = |run: &dyn Fn()| {
        run();
        let t = Instant::now();
        for _ in 0..5 {
            run();
        }
        t.elapsed().as_secs_f64() / 5.0
    };
    let vocab = 2048usize;
    // (dm, layers, batch, seq); heads fixed at 8
    for (dm, layers, b, s) in
        [(128usize, 4usize, 8usize, 128usize), (512, 6, 8, 256), (1024, 8, 4, 512), (2048, 8, 2, 512)]
    {
        let (h, dh, dff) = (8usize, dm / 8, 4 * dm);
        let mut g = Graph::new();
        let ids: Vec<i32> = (0..b * s).map(|i| (i * 13 % vocab) as i32).collect();
        let tok = g.const_storage(Storage::I32(ids), vec![b, s]);
        let embed = con(&mut g, 0, vec![vocab, dm]);
        let mut x = g.gather(embed, tok, 0).unwrap();
        for l in 0..layers {
            let sd = l * 100;
            let hn = g.rmsnorm(x, 2, 1e-5).unwrap();
            let h2 = g.reshape(hn, vec![b * s, dm]).unwrap();
            let head = |g: &mut Graph, seed| {
                let w = con(g, seed, vec![dm, dm]);
                let p = g.dot_general(h2, w, vec![1], vec![0], vec![], vec![]).unwrap();
                let r = g.reshape(p, vec![b, s, h, dh]).unwrap();
                g.permute(r, vec![0, 2, 1, 3]).unwrap()
            };
            let (q, k, v) = (head(&mut g, sd + 1), head(&mut g, sd + 2), head(&mut g, sd + 3));
            let attn = g.sdpa(q, k, v, true).unwrap();
            let attn = g.permute(attn, vec![0, 2, 1, 3]).unwrap();
            let attn = g.reshape(attn, vec![b * s, dm]).unwrap();
            let wo = con(&mut g, sd + 4, vec![dm, dm]);
            let o = g.dot_general(attn, wo, vec![1], vec![0], vec![], vec![]).unwrap();
            let o = g.reshape(o, vec![b, s, dm]).unwrap();
            x = g.add(x, o).unwrap();
            let n2 = g.rmsnorm(x, 2, 1e-5).unwrap();
            let m2 = g.reshape(n2, vec![b * s, dm]).unwrap();
            let (wg, wu, wd) = (
                con(&mut g, sd + 5, vec![dm, dff]),
                con(&mut g, sd + 6, vec![dm, dff]),
                con(&mut g, sd + 7, vec![dff, dm]),
            );
            let gate = {
                let gp = g.dot_general(m2, wg, vec![1], vec![0], vec![], vec![]).unwrap();
                g.silu(gp)
            };
            let up = g.dot_general(m2, wu, vec![1], vec![0], vec![], vec![]).unwrap();
            let gu = g.mul(gate, up).unwrap();
            let mlp = g.dot_general(gu, wd, vec![1], vec![0], vec![], vec![]).unwrap();
            let mlp = g.reshape(mlp, vec![b, s, dm]).unwrap();
            x = g.add(x, mlp).unwrap();
        }
        let fin = g.rmsnorm(x, 2, 1e-5).unwrap();
        let f2 = g.reshape(fin, vec![b * s, dm]).unwrap();
        let lm = con(&mut g, 999, vec![dm, vocab]);
        let logits = g.dot_general(f2, lm, vec![1], vec![0], vec![], vec![]).unwrap();
        let fwd = bench(&|| {
            metal.eval(&g, logits);
        });
        let toks = (b * s) as f64;
        // rough matmul FLOPs (2 per MAC): per layer 4*dm^2 (attn proj) + 3*dm*dff (swiglu) + lm_head.
        let per_tok =
            (layers as f64) * (4.0 * (dm * dm) as f64 + 3.0 * (dm * dff) as f64) * 2.0 + 2.0 * (dm * vocab) as f64;
        eprintln!(
            "dm={dm} L={layers} {b}x{s}={} tok F16:  fwd {:.2} ms  {:.0} tok/s  ~{:.0} GFLOP/s (matmul)",
            b * s,
            fwd * 1e3,
            toks / fwd,
            per_tok * toks / fwd / 1e9
        );
    }
}
