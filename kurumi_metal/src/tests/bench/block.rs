//! Transformer-block eval benchmarks (f16 vs f32, CPU vs Metal).

use crate::tests::*;

#[test]
#[ignore = "benchmark; run with --release"]
fn metal_f16_vs_f32_block_bench() {
    use half::f16;
    use kurumi_core::{Backend, DType, Storage};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let block = |dt: DType, d: usize| -> (Graph, kurumi_core::NodeId) {
        let cst = |g: &mut Graph, seed: usize| {
            let data: Vec<f32> = (0..d * d).map(|i| ((i + seed) % 17) as f32 * 0.01 - 0.08).collect();
            let s = match dt {
                DType::F16 => Storage::F16(data.iter().map(|&x| f16::from_f32(x)).collect()),
                _ => Storage::F32(data),
            };
            g.const_storage(s, vec![d, d])
        };
        let mut g = Graph::new();
        let mut h = cst(&mut g, 0);
        let w = cst(&mut g, 1);
        for _ in 0..6 {
            let mm = g.dot_general(h, w, vec![1], vec![0], vec![], vec![]).unwrap();
            h = g.gelu(mm);
        }
        (g, h)
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
        let (g32, o32) = block(DType::F32, d);
        let (g16, o16) = block(DType::F16, d);
        let t32 = bench(&|| {
            metal.eval(&g32, o32);
        });
        let t16 = bench(&|| {
            metal.eval(&g16, o16);
        });
        eprintln!(
            "6x(matmul+gelu {d}^3):  f32 {:.1} ms  |  f16 {:.1} ms  ({:.2}x faster, half the memory)",
            t32 * 1e3,
            t16 * 1e3,
            t32 / t16
        );
    }
}

#[test]
#[ignore = "benchmark; run with --release"]
fn metal_block_eval_bench() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let bench = |run: &dyn Fn()| {
        run(); // warm up
        let t = Instant::now();
        for _ in 0..5 {
            run();
        }
        t.elapsed().as_secs_f64() / 5.0
    };
    for d in [256usize, 512, 1024] {
        let mut g = Graph::new();
        let mut h = g.constant((0..d * d).map(|i| ((i % 17) as f32) * 0.01 - 0.08).collect(), vec![d, d]);
        let w = g.constant((0..d * d).map(|i| ((i % 11) as f32) * 0.005 - 0.02).collect(), vec![d, d]);
        for _ in 0..6 {
            let mm = g.dot_general(h, w, vec![1], vec![0], vec![], vec![]).unwrap();
            let ln = g.layernorm(mm, 1, 1e-5).unwrap(); // reduce + broadcast (device)
            h = g.gelu(ln); // elementwise chain (device): whole block stays on GPU
        }
        let gpu_s = bench(&|| {
            metal.eval(&g, h);
        });
        let cpu_s = bench(&|| {
            interpret(&g, h);
        });
        eprintln!(
            "6x (matmul {d}^3 + gelu):  CPU {:.1} ms  |  Metal {:.1} ms  ({:.1}x CPU)",
            cpu_s * 1e3,
            gpu_s * 1e3,
            cpu_s / gpu_s
        );
    }
}
