//! GPT-shape forward throughput benchmarks (incl. the Llama-shape prefill).

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

// Real-model Llama-shape prefill throughput (f16), timed by GPU command-buffer timestamps.
// tok/s = seq / GPU_seconds; shapes set the FLOPs so synthetic constants give an honest number.
// (was benches/llama.rs)
struct LlamaCfg {
    name: &'static str,
    vocab: usize,
    dm: usize,
    heads: usize,
    dh: usize,
    dff: usize,
    layers: usize,
    seq: usize,
}

fn llama_params(c: &LlamaCfg) -> usize {
    2 * c.vocab * c.dm + c.layers * (4 * c.dm * c.dm + 3 * c.dm * c.dff)
}

fn llama_build(g: &mut Graph, c: &LlamaCfg) -> kurumi_core::NodeId {
    use half::f16;
    use kurumi_core::Storage;
    let eps = 1e-5;
    let w = |g: &mut Graph, rows: usize, cols: usize, seed: usize| {
        let d: Vec<f16> = (0..rows * cols).map(|i| f16::from_f32((((i + seed) % 17) as f32) * 0.01 - 0.08)).collect();
        g.const_storage(Storage::F16(d), vec![rows, cols])
    };
    let ids: Vec<i32> = (0..c.seq).map(|i| (i * 7 % c.vocab) as i32).collect();
    let tok = g.const_storage(Storage::I32(ids), vec![1, c.seq]);
    let embed = w(g, c.vocab, c.dm, 9);
    let mut x = g.gather(embed, tok, 0).unwrap();
    for l in 0..c.layers {
        let s = l * 100;
        let hn = g.rmsnorm(x, 2, eps).unwrap();
        let h2d = g.reshape(hn, vec![c.seq, c.dm]).unwrap();
        let proj = |g: &mut Graph, wt| g.dot_general(h2d, wt, vec![1], vec![0], vec![], vec![]).unwrap();
        let heads = |g: &mut Graph, p| {
            let r = g.reshape(p, vec![1, c.seq, c.heads, c.dh]).unwrap();
            g.permute(r, vec![0, 2, 1, 3]).unwrap()
        };
        let (wq, wk, wv, wo) =
            (w(g, c.dm, c.dm, s + 1), w(g, c.dm, c.dm, s + 2), w(g, c.dm, c.dm, s + 3), w(g, c.dm, c.dm, s + 4));
        let (q, k, v) = (proj(g, wq), proj(g, wk), proj(g, wv));
        let (q, k, v) = (heads(g, q), heads(g, k), heads(g, v));
        let attn = g.sdpa(q, k, v, true).unwrap();
        let attn = g.permute(attn, vec![0, 2, 1, 3]).unwrap();
        let attn = g.reshape(attn, vec![c.seq, c.dm]).unwrap();
        let o = g.dot_general(attn, wo, vec![1], vec![0], vec![], vec![]).unwrap();
        let o = g.reshape(o, vec![1, c.seq, c.dm]).unwrap();
        x = g.add(x, o).unwrap();
        let hn2 = g.rmsnorm(x, 2, eps).unwrap();
        let h2 = g.reshape(hn2, vec![c.seq, c.dm]).unwrap();
        let (wg, wu, wd) = (w(g, c.dm, c.dff, s + 5), w(g, c.dm, c.dff, s + 6), w(g, c.dff, c.dm, s + 7));
        let gate = {
            let gp = g.dot_general(h2, wg, vec![1], vec![0], vec![], vec![]).unwrap();
            g.silu(gp)
        };
        let up = g.dot_general(h2, wu, vec![1], vec![0], vec![], vec![]).unwrap();
        let gu = g.mul(gate, up).unwrap();
        let mlp = g.dot_general(gu, wd, vec![1], vec![0], vec![], vec![]).unwrap();
        let mlp = g.reshape(mlp, vec![1, c.seq, c.dm]).unwrap();
        x = g.add(x, mlp).unwrap();
    }
    let fin = g.rmsnorm(x, 2, eps).unwrap();
    let fin2d = g.reshape(fin, vec![c.seq, c.dm]).unwrap();
    let wo_out = w(g, c.dm, c.vocab, 8);
    g.dot_general(fin2d, wo_out, vec![1], vec![0], vec![], vec![]).unwrap()
}

#[test]
#[ignore = "benchmark; run with --release"]
fn llama_prefill_bench() {
    use kurumi_core::Backend;
    unsafe { std::env::set_var("KURUMI_GPUTIME", "1") };
    let Some(be) = MetalBackend::new() else {
        return;
    };
    let run = |be: &MetalBackend, c: &LlamaCfg| {
        let mut g = Graph::new();
        let logits = llama_build(&mut g, c);
        std::hint::black_box(be.eval(&g, logits));
        let _ = MetalContext::take_flush_stats();
        let iters = 10;
        for _ in 0..iters {
            std::hint::black_box(be.eval(&g, logits));
        }
        let (_f, gpu_ms) = MetalContext::take_flush_stats();
        let gpu_s = gpu_ms / 1e3 / iters as f64;
        let gflop = 2.0 * llama_params(c) as f64 * c.seq as f64 / 1e9;
        eprintln!(
            "  {:<6} {:>2}L d{:<4} {:>4.2}B | seq {:<5} | {:>7.0} tok/s  {:>7.0} GFLOPS  ({:.2} ms)",
            c.name,
            c.layers,
            c.dm,
            llama_params(c) as f64 / 1e9,
            c.seq,
            c.seq as f64 / gpu_s,
            gflop / gpu_s,
            gpu_s * 1e3
        );
    };
    eprintln!("Llama-shape prefill (f16, Metal GPU-time) -- {}", be.name());
    for c in &[
        LlamaCfg { name: "125M", vocab: 32000, dm: 768, heads: 12, dh: 64, dff: 2048, layers: 12, seq: 512 },
        LlamaCfg { name: "350M", vocab: 32000, dm: 1024, heads: 16, dh: 64, dff: 2752, layers: 24, seq: 512 },
        LlamaCfg { name: "1.1B", vocab: 32000, dm: 2048, heads: 16, dh: 128, dff: 5504, layers: 22, seq: 512 },
    ] {
        run(&be, c);
    }
    eprintln!("-- attention scaling (350M, seq sweep) --");
    for seq in [512usize, 1024, 2048, 4096] {
        run(&be, &LlamaCfg { name: "350M", vocab: 32000, dm: 1024, heads: 16, dh: 64, dff: 2752, layers: 24, seq });
    }
}

// Flash forward (fused Op::Sdpa) vs the dot+softmax decomposition, isolated to the attention op
// on a GPT-ish causal shape. The flash kernel holds O(dh) per-thread state -- it never allocates
// the [B,H,S,S] score buffer the decomposition must materialize (the memory win, reported). NOTE:
// this is the correct memory-reduced online-softmax kernel (ONE thread per query, scalar ALU),
// NOT a tiled simdgroup-matrix kernel -- so on short seqs the decomposition's two matmuls on
// the matrix units can still beat it; the win is memory (and long-seq scaling).
#[test]
#[ignore = "benchmark; run with --release"]
fn metal_sdpa_flash_vs_decomp_bench() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let bench = |run: &dyn Fn()| {
        run();
        let t = Instant::now();
        for _ in 0..20 {
            run();
        }
        t.elapsed().as_secs_f64() / 20.0
    };
    let (b, h, dh) = (4usize, 8usize, 64usize);
    for s in [128usize, 512, 1024, 2048] {
        let n = b * h * s * dh;
        let mk = |seed: usize| (0..n).map(|i| (((i * 7 + seed) % 23) as f32) * 0.01 - 0.1).collect::<Vec<_>>();
        let mut g = Graph::new();
        let q = g.constant(mk(1), vec![b, h, s, dh]);
        let k = g.constant(mk(2), vec![b, h, s, dh]);
        let v = g.constant(mk(3), vec![b, h, s, dh]);
        let flash = g.sdpa_fused(q, k, v, true).unwrap(); // device flash kernel
        let decomp = g.sdpa_decomposed(q, k, v, true).unwrap(); // MPS matmul + device softmax
        let tf = bench(&|| {
            metal.eval(&g, flash);
        });
        let td = bench(&|| {
            metal.eval(&g, decomp);
        });
        let scores_mb = (b * h * s * s * 4) as f64 / 1e6; // [B,H,S,S] f32 the decomposition materializes
        eprintln!(
            "B={b} H={h} S={s} dh={dh}:  flash {:.3} ms  |  decomp {:.3} ms  |  SxS scores not materialized {:.1} MB  |  decomp/flash {:.2}x",
            tf * 1e3,
            td * 1e3,
            scores_mb,
            td / tf
        );
    }
}
