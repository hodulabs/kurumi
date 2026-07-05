//! GEMM / training-step benchmarks (incl. the GPU-time microbench).

use crate::tests::*;

#[test]
#[ignore = "benchmark; CPU matmul is slow in debug"]
fn gpu_vs_cpu_matmul_bench() {
    let Some(ctx) = MetalContext::new() else {
        return;
    };
    let n = 1024usize;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 17) as f32) * 0.01).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.02).collect();
    let flops = 2.0 * (n as f64).powi(3);
    let iters = 30;

    let bench = |run: &dyn Fn()| {
        run(); // warm up
        let t = Instant::now();
        for _ in 0..iters {
            run();
        }
        t.elapsed().as_secs_f64() / iters as f64
    };

    let naive = ctx.gemm_pipeline();
    let gpu_s = bench(&|| {
        ctx.matmul(&naive, &a, n, n, &b, n);
    });
    let t = Instant::now();
    let _ = cpu_matmul(&a, n, n, &b, n);
    let cpu_s = t.elapsed().as_secs_f64();

    let gf = |s: f64| flops / s / 1e9;
    eprintln!("{n}^3 matmul on {}:", ctx.device_name());
    eprintln!("  CPU (gemm)  {:.2} ms  {:.0} GFLOP/s", cpu_s * 1e3, gf(cpu_s));
    eprintln!("  GPU         {:.2} ms  {:.0} GFLOP/s  ({:.1}x CPU)", gpu_s * 1e3, gf(gpu_s), cpu_s / gpu_s);
}

#[test]
#[ignore = "benchmark; run with --release"]
fn metal_training_step_bench() {
    use kurumi_core::{Backend, grad};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let bench = |run: &dyn Fn()| {
        run();
        let t = Instant::now();
        for _ in 0..5 {
            run();
        }
        t.elapsed().as_secs_f64() / 5.0
    };
    for d in [512usize, 1024] {
        let mut g = Graph::new();
        let mut h = g.constant((0..d * d).map(|i| ((i % 17) as f32) * 0.01 - 0.08).collect(), vec![d, d]);
        let w = g.constant((0..d * d).map(|i| ((i % 11) as f32) * 0.005 - 0.02).collect(), vec![d, d]);
        for _ in 0..3 {
            let mm = g.dot_general(h, w, vec![1], vec![0], vec![], vec![]).unwrap();
            h = g.gelu(mm);
        }
        let s = g.sum(h, 1).unwrap();
        let loss = g.sum(s, 0).unwrap();
        let dw = grad(&mut g, loss, &[w]).unwrap()[0]; // forward + full backward (transposed GEMMs)
        let gpu_s = bench(&|| {
            metal.eval(&g, dw);
        });
        let cpu_s = bench(&|| {
            interpret(&g, dw);
        });
        eprintln!(
            "train step (3x matmul+gelu {d}^3, fwd+bwd):  CPU {:.1} ms  |  Metal {:.1} ms  ({:.1}x CPU)",
            cpu_s * 1e3,
            gpu_s * 1e3,
            cpu_s / gpu_s
        );
    }
}

// GPU-time GEMM GFLOPS + elementwise bandwidth via command-buffer timestamps (KURUMI_GPUTIME);
// GPU time excludes host upload/dispatch, so GFLOPS vs the chip f32 roofline signals whether the
// MPS GEMM needs a steel port (near roofline -> skip). (was benches/device.rs)
fn gpu_measure(be: &MetalBackend, g: &Graph, out: kurumi_core::NodeId, iters: u32) -> (f64, f64) {
    use kurumi_core::Backend;
    std::hint::black_box(be.eval(g, out)); // warm: compile pipelines, size buffer cache
    let _ = MetalContext::take_flush_stats();
    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(be.eval(g, out));
    }
    let wall = t.elapsed().as_secs_f64();
    let (_flushes, gpu_ms) = MetalContext::take_flush_stats();
    (gpu_ms, wall)
}

#[test]
#[ignore = "benchmark; run with --release"]
fn gpu_time_microbench() {
    use kurumi_core::Backend;
    // SAFETY: set before any eval; single-threaded bench. Gates the GPU-time path in flush.
    unsafe { std::env::set_var("KURUMI_GPUTIME", "1") };
    let Some(be) = MetalBackend::new() else {
        return;
    };
    eprintln!("Metal microbench -- {} (GPU-time)", be.name());
    for n in [512usize, 1024, 2048] {
        let mut g = Graph::new();
        let a = g.constant(vec![0.01f32; n * n], vec![n, n]);
        let b = g.constant(vec![0.02f32; n * n], vec![n, n]);
        let c = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
        let iters = 30;
        let (gpu_ms, wall) = gpu_measure(&be, &g, c, iters);
        let flops = 2.0 * (n as f64).powi(3) * iters as f64;
        eprintln!(
            "  gemm {n:>4}^3  GPU {:>6.2} ms/it  {:>8.0} GFLOPS | wall {:>8.0} GFLOPS",
            gpu_ms / iters as f64,
            flops / (gpu_ms / 1e3) / 1e9,
            flops / wall / 1e9
        );
    }
    let side = ((64usize * (1 << 20) / 4) as f64).sqrt() as usize; // ~64 MB array
    let mut g = Graph::new();
    let a = g.constant(vec![1.5f32; side * side], vec![side, side]);
    let y = g.neg(a); // read a + write y = 2 arrays
    let (gpu_ms, _) = gpu_measure(&be, &g, y, 100);
    let bytes = 2.0 * (side * side) as f64 * 4.0 * 100.0;
    eprintln!("  neg {:>4} MB  GPU {:>6.0} GB/s", (side * side * 4) >> 20, bytes / (gpu_ms / 1e3) / 1e9);
}
