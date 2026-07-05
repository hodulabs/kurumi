//! Device-resident elementwise / movement / gather / slice-pad vs the CPU oracle.

use crate::tests::*;

// the whole-graph seam: a mixed-op f32 graph runs on Metal (matmul on GPU,
// rest on CPU) and matches the interpreter oracle.
#[test]
fn metal_eval_full_graph_matches_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let mut g = Graph::new();
    let x = g.constant((0..64).map(|i| i as f32 * 0.05).collect(), vec![8, 8]);
    let w = g.constant((0..64).map(|i| (i as f32 * 0.1).sin()).collect(), vec![8, 8]);
    let mm = g.dot_general(x, w, vec![1], vec![0], vec![], vec![]).unwrap(); // GPU
    let relu = {
        let z = g.zeros_like(mm);
        g.max(mm, z).unwrap()
    }; // CPU
    let s = g.softmax(relu, 1).unwrap(); // CPU (decomposed)
    let out = g.dot_general(s, w, vec![1], vec![0], vec![], vec![]).unwrap(); // GPU
    let gpu = metal.eval(&g, out);
    let cpu = interpret(&g, out);
    assert_eq!(gpu.shape, cpu.shape);
    for (a, b) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}

// the GPU dequant-matmul kernels match the CPU quant oracle across int4/int8 x sym/asym,
// float activation dtypes (f32/f16/bf16), and both paths: small M -> GEMV, large M -> tiled
// GEMM. Activation is cast to the target dtype and the result cast back to f32 to compare.
#[test]
fn metal_quant_matmul_matches_cpu() {
    use kurumi_core::{Backend, DType, quantize};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (n, k, gs) = (18usize, 64, 32);
    let w: Vec<f32> = (0..n * k).map(|i| ((i * 13 % 97) as f32 / 97.0 - 0.5) * 4.0).collect();
    for (bits, sym) in [(2u8, false), (2, true), (4, false), (4, true), (8, false), (8, true)] {
        let q = quantize(&w, n, k, bits, gs, sym);
        for &m in &[1usize, 2, 20] {
            // m=1 the scalar-acc decode GEMV, m=2 the array GEMV, m=20 the tiled prefill GEMM.
            for &dt in &[DType::F32, DType::F16, DType::BF16] {
                let act: Vec<f32> = (0..m * k).map(|i| (i * 7 % 53) as f32 / 53.0 - 0.4).collect();
                let mut g = Graph::new();
                let a0 = g.constant(act, vec![m, k]);
                let a = g.cast(a0, dt);
                let qw = g.const_storage(Storage::U8(q.packed.clone()), vec![n, k * bits as usize / 8]);
                let sc = g.const_storage(Storage::F16(q.scales.clone()), vec![n, k / gs]);
                let mn = q.mins.clone().map(|mv| g.const_storage(Storage::F16(mv), vec![n, k / gs]));
                let qm = g.quant_matmul(a, qw, sc, mn, bits, gs).unwrap();
                let out = g.cast(qm, DType::F32);
                let gpu = metal.eval(&g, out);
                let cpu = interpret(&g, out);
                assert_eq!(gpu.shape, cpu.shape);
                for (x, y) in gpu.f32().iter().zip(cpu.f32()) {
                    // gpu and cpu do identical f32-accumulate then dtype-round, so a small
                    // relative bound covers the low-precision (f16/bf16) grids.
                    assert!(
                        (x - y).abs() <= 1e-2 + 3e-2 * x.abs().max(y.abs()),
                        "bits={bits} sym={sym} m={m} dt={dt:?}: {x} vs {y}"
                    );
                }
            }
        }
    }
}

// flip is device-resident via a negative-stride fused view (folds into a consuming
// pointwise, or materializes if it's the output).
#[test]
fn metal_flip_matches_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let mut g = Graph::new();
    let x = g.constant((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);
    // flip two axes, then a pointwise -> the view folds into the fused kernel
    let f = g.flip(x, vec![0, 2]).unwrap();
    let y = {
        let two = g.scalar(f, 2.0);
        g.mul(f, two).unwrap()
    };
    let (gpu, cpu) = (metal.eval(&g, y), interpret(&g, y));
    assert_eq!(gpu.shape, cpu.shape);
    for (a, b) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((a - b).abs() < 1e-4, "flip+ew {a} vs {b}");
    }
    // flip as the direct output (view materialized by one fused kernel)
    let f2 = g.flip(x, vec![1]).unwrap();
    let (gpu2, cpu2) = (metal.eval(&g, f2), interpret(&g, f2));
    for (a, b) in gpu2.f32().iter().zip(cpu2.f32()) {
        assert!((a - b).abs() < 1e-4, "flip out {a} vs {b}");
    }
}

// argmax/argmin on device -> I64 indices, vs the CPU oracle.
#[test]
fn metal_argreduce_matches_cpu() {
    use kurumi_core::{Backend, Storage};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let mut g = Graph::new();
    let x = g.constant(vec![3., 1., 4., 1., 5., 9., 2., 6.], vec![2, 4]);
    let am = g.argmax(x, 1).unwrap();
    let an = g.argmin(x, 1).unwrap();
    let i64s = |t: kurumi_core::TensorVal| match t.storage {
        Storage::I64(v) => v,
        s => panic!("want I64, got {s:?}"),
    };
    for (node, name) in [(am, "argmax"), (an, "argmin")] {
        let gpu = i64s(metal.eval(&g, node));
        let cpu = i64s(interpret(&g, node));
        assert_eq!(gpu, cpu, "{name}");
    }
}

// take_along_dim / gather_along on device vs the CPU oracle.
#[test]
fn metal_gather_along_matches_cpu() {
    use kurumi_core::{Backend, Storage};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let mut g = Graph::new();
    let x = g.constant(vec![10., 20., 30., 40., 50., 60.], vec![2, 3]);
    let idx = g.const_storage(Storage::I64(vec![2, 0, 1, 1]), vec![2, 2]);
    let y = g.gather_along(x, idx, 1).unwrap(); // [[30,10],[50,50]]
    let (gpu, cpu) = (metal.eval(&g, y), interpret(&g, y));
    assert_eq!(gpu.shape, cpu.shape);
    for (a, b) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((a - b).abs() < 1e-4, "gather_along {a} vs {b}");
    }
}

// scatter_along (index_add) on device: atomic add with DUPLICATE indices (the
// embedding-gradient pattern) + Set, vs the CPU oracle.
#[test]
fn metal_scatter_along_matches_cpu() {
    use kurumi_core::{Backend, ScatterOp, Storage};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    // Add with duplicate index 0 -> atomic accumulation
    let mut g = Graph::new();
    let op = g.constant(vec![0., 0., 0.], vec![3]);
    let idx = g.const_storage(Storage::I64(vec![0, 0, 1]), vec![3]);
    let up = g.constant(vec![1., 2., 3.], vec![3]);
    let y = g.scatter_along(op, idx, up, 0, ScatterOp::Add).unwrap();
    let (gpu, cpu) = (metal.eval(&g, y), interpret(&g, y));
    assert_eq!(gpu.f32(), cpu.f32(), "scatter add");
    assert_eq!(gpu.f32(), &[3., 3., 0.]);
    // Set with unique indices
    let mut g = Graph::new();
    let op = g.constant(vec![0., 0., 0.], vec![3]);
    let idx = g.const_storage(Storage::I64(vec![2, 0]), vec![2]);
    let up = g.constant(vec![5., 7.], vec![2]);
    let y = g.scatter_along(op, idx, up, 0, ScatterOp::Set).unwrap();
    let (gpu, cpu) = (metal.eval(&g, y), interpret(&g, y));
    assert_eq!(gpu.f32(), cpu.f32(), "scatter set");
}

// a long f32 elementwise chain (gelu = mul/neg/exp2/recip/add...) feeds off a
// GPU matmul and runs device-resident; the readback must match the CPU oracle.
#[test]
fn metal_device_elementwise_chain_matches_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let mut g = Graph::new();
    let (m, k, n) = (16, 24, 16);
    let x = g.constant((0..m * k).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect(), vec![m, k]);
    let w = g.constant((0..k * n).map(|i| ((i % 7) as f32) * 0.05 - 0.15).collect(), vec![k, n]);
    let mm = g.dot_general(x, w, vec![1], vec![0], vec![], vec![]).unwrap();
    let y = g.gelu(mm); // device-resident elementwise chain
    let gpu = metal.eval(&g, y);
    let cpu = interpret(&g, y);
    assert_eq!(gpu.shape, cpu.shape);
    for (a, b) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((a - b).abs() < 1e-2, "{a} vs {b}");
    }
}

// gather (embedding lookup) runs device-resident: operand on the GPU, indices
// uploaded as i32, so the whole LM (embed -> blocks) stays on device.
#[test]
fn metal_gather_device_matches_cpu() {
    use kurumi_core::{Backend, Storage};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let mut g = Graph::new();
    let table = g.constant((0..6 * 4).map(|i| i as f32 * 0.1 - 0.5).collect(), vec![6, 4]); // [V,D]
    let ids = g.const_storage(Storage::I32(vec![0, 3, 5, 1]), vec![4]);
    let emb = g.gather(table, ids, 0).unwrap(); // [4, 4]
    let y = g.gelu(emb); // device gather -> fused device pointwise
    let gpu = metal.eval(&g, y);
    let cpu = interpret(&g, y);
    assert_eq!(gpu.shape, cpu.shape);
    for (p, q) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((p - q).abs() < 1e-3, "{p} vs {q}");
    }
}

// slice + zero-pad run device-resident (strided gather + the pad kernel),
// composing with fused elementwise, and match the CPU oracle.
#[test]
fn metal_slice_pad_match_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let mut g = Graph::new();
    let x = g.constant((0..24).map(|i| i as f32 * 0.5 - 3.0).collect(), vec![4, 6]);
    let sl = g.slice(x, vec![(1, 3), (2, 5)]).unwrap(); // [2,3]
    let relu = {
        let z = g.zeros_like(sl);
        g.max(sl, z).unwrap()
    };
    let pd = g.pad(relu, vec![(1, 1), (0, 2)]).unwrap(); // [4,5]
    let y = g.mul(pd, pd).unwrap();
    let gpu = metal.eval(&g, y);
    let cpu = interpret(&g, y);
    assert_eq!(gpu.shape, cpu.shape);
    for (p, w) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((p - w).abs() < 1e-4, "{p} vs {w}");
    }
}

#[test]
fn metal_where_cmp_activations_match_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else { return };
    let mut g = Graph::new();
    let x = g.constant((0..24).map(|i| (i as f32) * 0.3 - 3.0).collect(), vec![4, 6]);
    // where-based activations + a comparison-driven select, all device-resident now
    let a = g.elu(x, 1.0);
    let b = g.leaky_relu(a, 0.1);
    let s = g.sign(b);
    let mask = g.cmp_lt(b, a).unwrap(); // device cmp -> bool buffer
    let sel = g.select(mask, s, b).unwrap(); // device where
    let mf = g.masked_fill(sel, mask, 7.0).unwrap();
    let y = g.add(mf, b).unwrap();
    let gpu = metal.eval(&g, y);
    let cpu = interpret(&g, y);
    assert_eq!(gpu.shape, cpu.shape);
    for (p, q) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((p - q).abs() < 1e-4, "{p} vs {q}");
    }
}
