//! Weight-only quant matmul benchmarks.

use crate::tests::*;

// weight-only quant throughput: decode (M=1, GEMV) and prefill (M=256, tiled GEMM), int4/int8,
// f32/f16 activation. Decode is memory-bound: GB/s vs the machine's bandwidth is the number to
// watch; prefill reports GFLOP/s. Reads back to host each iter (end-to-end latency).
#[test]
#[ignore = "benchmark"]
fn quant_matmul_bench() {
    use kurumi_core::{Backend, DType, quantize};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (n, k, gs) = (2048usize, 2048, 64);
    let w: Vec<f32> = (0..n * k).map(|i| ((i * 13 % 251) as f32 / 251.0 - 0.5) * 0.5).collect();
    let iters = 30;
    let bench = |run: &dyn Fn()| {
        run();
        let t = Instant::now();
        for _ in 0..iters {
            run();
        }
        t.elapsed().as_secs_f64() / iters as f64
    };
    for bits in [2u8, 4, 8] {
        let q = quantize(&w, n, k, bits, gs, false); // asymmetric (carries mins)
        let wbytes = (n * k * bits as usize / 8) as f64; // packed weight footprint
        eprintln!("int{bits} weight-only quant  [N={n} K={k} G={gs}]:");
        for (label, m) in [("decode  M=1   (GEMV)", 1usize), ("prefill M=256 (GEMM)", 256)] {
            for dt in [DType::F32, DType::F16] {
                let act: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01 - 0.4).collect();
                let mut g = Graph::new();
                let a0 = g.constant(act, vec![m, k]);
                let a = g.cast(a0, dt);
                let qw = g.const_storage(Storage::U8(q.packed.clone()), vec![n, k * bits as usize / 8]);
                let sc = g.const_storage(Storage::F16(q.scales.clone()), vec![n, k / gs]);
                let mn = q.mins.clone().map(|mv| g.const_storage(Storage::F16(mv), vec![n, k / gs]));
                let out = g.quant_matmul(a, qw, sc, mn, bits, gs).unwrap();
                let s = bench(&|| {
                    metal.eval(&g, out);
                });
                let gf = 2.0 * (m * n * k) as f64 / s / 1e9;
                eprintln!(
                    "  {label} {dt:?}:  {:.3} ms  {:>5.0} GFLOP/s  {:>4.0} GB/s decode",
                    s * 1e3,
                    gf,
                    wbytes / s / 1e9
                );
            }
        }
    }
}
