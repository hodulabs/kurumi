//! Attention op microbench: fused flash SDPA vs the dot+softmax decomposition.

use crate::tests::*;

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
