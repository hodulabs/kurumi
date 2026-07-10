//! Real-model Llama-shape prefill throughput (f16), timed by GPU command-buffer timestamps.
//! (was benches/llama.rs)

use crate::tests::*;

// tok/s = seq / GPU_seconds; shapes set the FLOPs so synthetic constants give an honest number.
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
