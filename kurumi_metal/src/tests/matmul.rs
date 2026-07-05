//! GPU GEMM correctness: f32/f16/bf16 + all numeric dtypes, cast pairs, dtype/f64 rejects, batched.

use crate::tests::*;

#[test]
fn backend_trait_f32_cpu_and_metal_agree() {
    use kurumi_core::{Backend, CpuBackend, Storage};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    // non-multiple-of-8 sizes exercise the padding path
    let (m, k, n) = (30, 50, 20);
    let a = Storage::F32((0..m * k).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect());
    let b = Storage::F32((0..k * n).map(|i| ((i % 5) as f32) * 0.3 - 0.4).collect());
    let Storage::F32(cpu) = CpuBackend.matmul(&a, m, k, &b, n).unwrap() else { panic!() };
    let Storage::F32(gpu) = metal.matmul(&a, m, k, &b, n).unwrap() else { panic!() };
    assert_eq!(metal.name(), "metal");
    for i in 0..m * n {
        assert!((cpu[i] - gpu[i]).abs() < 1e-2, "i={i}: cpu {} vs metal {}", cpu[i], gpu[i]);
    }
}

#[test]
fn backend_trait_f16_on_metal() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    // f16 GEMM runs natively on the GPU; compare to a CPU f16 reference
    let (m, k, n) = (16, 24, 16);
    let af: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1).collect();
    let bf: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.1).collect();
    let a = Storage::F16(af.iter().map(|&x| f16::from_f32(x)).collect());
    let b = Storage::F16(bf.iter().map(|&x| f16::from_f32(x)).collect());
    let Storage::F16(gpu) = metal.matmul(&a, m, k, &b, n).unwrap() else { panic!("want f16") };
    // CPU f16 reference (same dtype)
    let mut g = Graph::new();
    let na = g.const_storage(a.clone(), vec![m, k]);
    let nb = g.const_storage(b.clone(), vec![k, n]);
    let y = g.dot_general(na, nb, vec![1], vec![0], vec![], vec![]).unwrap();
    let Storage::F16(cpu) = interpret(&g, y).storage else { panic!() };
    for i in 0..m * n {
        assert!((gpu[i].to_f32() - cpu[i].to_f32()).abs() < 0.5, "i={i}: {} vs {}", gpu[i].to_f32(), cpu[i].to_f32());
    }
}

// every numeric dtype except f64 runs matmul on the GPU and matches the CPU
#[test]
fn metal_matmul_all_numeric_dtypes() {
    use kurumi_core::{Storage, interpret};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (m, k, n) = (6, 5, 4);
    let storages: Vec<Storage> = {
        let af: Vec<i64> = (0..(m * k).max(k * n)).map(|i| (i % 5) as i64).collect();
        vec![
            Storage::U8(af.iter().map(|&x| x as u8).collect()),
            Storage::U32(af.iter().map(|&x| x as u32).collect()),
            Storage::I32(af.iter().map(|&x| x as i32).collect()),
            Storage::I64(af.clone()),
            Storage::F16(af.iter().map(|&x| f16::from_f32(x as f32)).collect()),
            Storage::BF16(af.iter().map(|&x| half::bf16::from_f32(x as f32)).collect()),
            Storage::F32(af.iter().map(|&x| x as f32).collect()),
        ]
    };
    for s in &storages {
        let take = |len: usize| slice_storage(s, len);
        let (a, b) = (take(m * k), take(k * n));
        let gpu = metal.matmul(&a, m, k, &b, n).unwrap();
        let mut g = Graph::new();
        let na = g.const_storage(a.clone(), vec![m, k]);
        let nb = g.const_storage(b.clone(), vec![k, n]);
        let y = g.dot_general(na, nb, vec![1], vec![0], vec![], vec![]).unwrap();
        let cpu = interpret(&g, y).storage;
        assert_eq!(gpu, cpu, "dtype {:?}", a.dtype());
    }
}

// every dtype-cast pair (except f64) runs on the GPU and matches the CPU
#[test]
fn metal_cast_all_pairs_match_cpu() {
    use kurumi_core::{CpuBackend, DType};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    // every dtype pair, including f64/fp8/complex (those fall back to the CPU oracle
    // inside metal.cast, so the direct API is all-pairs too).
    use DType::*;
    let dts = [BOOL, U8, U16, U32, U64, I8, I16, I32, I64, F8E4M3, F8E5M2, F16, BF16, F32, F64, C64, C128];
    let base = Storage::F32(vec![0.0, 1.0, 2.0, 3.0]);
    for &from in &dts {
        let src = CpuBackend.cast(&base, from).unwrap();
        for &to in &dts {
            let gpu = metal.cast(&src, to).unwrap();
            let cpu = CpuBackend.cast(&src, to).unwrap();
            assert_eq!(gpu, cpu, "{from:?} -> {to:?}");
        }
    }
}

#[test]
fn metal_rejects_f64_and_mismatch() {
    use kurumi_core::Storage;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f64s = Storage::F64(vec![1.0; 16]); // Apple GPUs have no double
    assert!(metal.matmul(&f64s, 4, 4, &f64s, 4).is_err());
    let b = Storage::BOOL(vec![true; 16]); // matmul on bool is meaningless
    assert!(metal.matmul(&b, 4, 4, &b, 4).is_err());
    let f = Storage::F32(vec![1.0; 16]);
    let h = Storage::F16(vec![f16::ZERO; 16]);
    assert!(metal.matmul(&f, 4, 4, &h, 4).is_err()); // dtype mismatch
}

#[test]
fn gpu_matmul_matches_cpu() {
    let Some(ctx) = MetalContext::new() else {
        return;
    };
    // naive kernel: any multiple of 8
    let (m, k, n) = (64, 128, 96);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.2 - 0.7).collect();
    let gpu = ctx.matmul(&ctx.gemm_pipeline(), &a, m, k, &b, n);
    let cpu = cpu_matmul(&a, m, k, &b, n);
    for i in 0..m * n {
        assert!((gpu[i] - cpu[i]).abs() < 1e-2, "naive i={i}: {} vs {}", gpu[i], cpu[i]);
    }
}

// batched dot_general (canonical + transposed) on the GPU: the enabler for
// attention. Both must match the CPU oracle.
#[test]
fn metal_batched_matmul_matches_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (bsz, m, k, n) = (3usize, 5, 7, 4);
    let mut g = Graph::new();
    let a = g.constant((0..bsz * m * k).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect(), vec![bsz, m, k]);
    let b = g.constant((0..bsz * k * n).map(|i| ((i % 7) as f32) * 0.2 - 0.6).collect(), vec![bsz, k, n]);
    let mm = g.dot_general(a, b, vec![2], vec![1], vec![0], vec![0]).unwrap(); // [bsz,m,n]
    let c = g.constant((0..bsz * n * k).map(|i| ((i % 5) as f32) * 0.3).collect(), vec![bsz, n, k]);
    let mmt = g.dot_general(a, c, vec![2], vec![2], vec![0], vec![0]).unwrap(); // a @ c^T (contract k)
    for &id in &[mm, mmt] {
        let gpu = metal.eval(&g, id);
        let cpu = interpret(&g, id);
        assert_eq!(gpu.shape, cpu.shape);
        for (p, q) in gpu.f32().iter().zip(cpu.f32()) {
            assert!((p - q).abs() < 1e-3, "{p} vs {q}");
        }
    }
}

// f16 runs device-resident end to end: MPS f16 GEMM + fused f16 pointwise (gelu)
// + f16 reduce, matching the CPU f16 oracle. (f16 buffers = half the memory.)
#[test]
fn metal_f16_device_matches_cpu() {
    use half::f16;
    use kurumi_core::{Backend, DType, Storage};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f16v = |xs: Vec<f32>| Storage::F16(xs.into_iter().map(f16::from_f32).collect());
    let to_f32 = |t: &kurumi_core::TensorVal| match &t.storage {
        Storage::F16(v) => v.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        _ => panic!("expected f16"),
    };
    let (m, k, n) = (8usize, 12, 6);
    let mut g = Graph::new();
    let a = g.const_storage(f16v((0..m * k).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect()), vec![m, k]);
    let b = g.const_storage(f16v((0..k * n).map(|i| ((i % 5) as f32) * 0.1 - 0.2).collect()), vec![k, n]);
    let mm = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap(); // MPS f16 GEMM
    let act = g.gelu(mm); // fused f16 pointwise
    let s = g.sum(act, 1).unwrap(); // f16 reduce
    let gpu = metal.eval(&g, s);
    let cpu = interpret(&g, s);
    assert_eq!(gpu.shape, cpu.shape);
    assert_eq!(gpu.dtype(), DType::F16);
    for (p, q) in to_f32(&gpu).iter().zip(to_f32(&cpu)) {
        assert!((p - q).abs() < 5e-2, "{p} vs {q}");
    }
}

// bf16 uses the same device path (MPS BFloat16 + "bfloat" MSL); smoke-check a
// matmul + fused pointwise against the CPU oracle.
#[test]
fn metal_bf16_device_matches_cpu() {
    use half::bf16;
    use kurumi_core::{Backend, DType, Storage};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let bf = |xs: Vec<f32>| Storage::BF16(xs.into_iter().map(bf16::from_f32).collect());
    let to_f32 = |t: &kurumi_core::TensorVal| match &t.storage {
        Storage::BF16(v) => v.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        _ => panic!("expected bf16"),
    };
    let mut g = Graph::new();
    let a = g.const_storage(bf((0..16).map(|i| (i as f32) * 0.1 - 0.8).collect()), vec![4, 4]);
    let b = g.const_storage(bf((0..16).map(|i| (i as f32) * 0.05).collect()), vec![4, 4]);
    let mm = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
    let y = g.silu(mm);
    let gpu = metal.eval(&g, y);
    let cpu = interpret(&g, y);
    assert_eq!(gpu.dtype(), DType::BF16);
    for (p, q) in to_f32(&gpu).iter().zip(to_f32(&cpu)) {
        assert!((p - q).abs() < 5e-2, "{p} vs {q}");
    }
}
