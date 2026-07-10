//! Attention on Metal vs the CPU oracle: single-head block, fused causal SDPA fwd+bwd,
//! and the flash forward across shapes/causal modes.

use crate::tests::*;

// a single-head attention block (Q@K^T -> softmax -> @V) runs fully on the GPU
// (batched matmuls + device softmax) and matches the CPU oracle.
#[test]
fn metal_attention_block_matches_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (bsz, s, dh) = (2usize, 6, 8);
    let mk = |seed: usize| (0..bsz * s * dh).map(|i| (((i + seed) % 13) as f32) * 0.1 - 0.6).collect::<Vec<_>>();
    let mut g = Graph::new();
    let q = g.constant(mk(0), vec![bsz, s, dh]);
    let k = g.constant(mk(3), vec![bsz, s, dh]);
    let v = g.constant(mk(7), vec![bsz, s, dh]);
    let scores = g.dot_general(q, k, vec![2], vec![2], vec![0], vec![0]).unwrap(); // Q@K^T [bsz,s,s]
    let sm = g.softmax(scores, 2).unwrap();
    let out = g.dot_general(sm, v, vec![2], vec![1], vec![0], vec![0]).unwrap(); // @V [bsz,s,dh]
    let gpu = metal.eval(&g, out);
    let cpu = interpret(&g, out);
    assert_eq!(gpu.shape, cpu.shape);
    for (p, q) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((p - q).abs() < 1e-3, "{p} vs {q}");
    }
}

// GPT-style causal multi-head attention (batch=[B,H]) via the fused Op::Sdpa PRIMITIVE:
// forward = the device flash kernel (online softmax), backward = the Op::Sdpa VJP graph
// (transposed batched GEMMs + softmax/mask backward). Built directly with sdpa_fused so the
// flash kernel is exercised regardless of g.sdpa's memory threshold. Both match the CPU oracle.
#[test]
fn metal_causal_sdpa_fwd_bwd_matches_cpu() {
    use kurumi_core::{Backend, grad};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (b, h, s, dh) = (2usize, 2, 5, 8);
    let mk = |seed: usize| (0..b * h * s * dh).map(|i| (((i + seed) % 13) as f32) * 0.1 - 0.6).collect::<Vec<_>>();
    let mut g = Graph::new();
    let q = g.constant(mk(0), vec![b, h, s, dh]);
    let k = g.constant(mk(3), vec![b, h, s, dh]);
    let v = g.constant(mk(7), vec![b, h, s, dh]);
    let out = g.sdpa_fused(q, k, v, true).unwrap(); // primitive directly: flash fwd + VJP bwd (g.sdpa threshold-gates flash)

    let gpu = metal.eval(&g, out);
    let cpu = interpret(&g, out);
    assert_eq!(gpu.shape, cpu.shape);
    for (p, w) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((p - w).abs() < 1e-3, "fwd {p} vs {w}");
    }

    // backward: loss = sum(out) over all axes, grads w.r.t. q, k, v
    let mut acc = out;
    for ax in (0..4).rev() {
        acc = g.sum(acc, ax).unwrap();
    }
    let grads = grad(&mut g, acc, &[q, k, v]).unwrap();
    for &gid in &grads {
        let gg = metal.eval(&g, gid);
        let cg = interpret(&g, gid);
        assert_eq!(gg.shape, cg.shape);
        for (p, w) in gg.f32().iter().zip(cg.f32()) {
            assert!((p - w).abs() < 1e-2, "bwd {p} vs {w}");
        }
    }
}

// The fused Op::Sdpa forward runs the device flash kernel (online softmax, no SxS
// materialization); it must match the CPU interpret oracle (standard row-max softmax) for
// both causal modes across a few [B,H,S,dh] shapes, incl S not a round number (odd tail
// rows / partial causal spans) and dh at 4/8/16 -- guarding online-softmax numerical drift.
#[test]
fn metal_sdpa_flash_forward_matches_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    for (b, h, s, dh) in [(2usize, 2usize, 7usize, 8usize), (1, 3, 5, 16), (2, 1, 9, 4)] {
        let n = b * h * s * dh;
        let mk = |seed: usize| (0..n).map(|i| (((i * 7 + seed) % 29) as f32) * 0.07 - 1.0).collect::<Vec<_>>();
        for causal in [false, true] {
            let mut g = Graph::new();
            let q = g.constant(mk(0), vec![b, h, s, dh]);
            let k = g.constant(mk(5), vec![b, h, s, dh]);
            let v = g.constant(mk(11), vec![b, h, s, dh]);
            let out = g.sdpa_fused(q, k, v, causal).unwrap(); // primitive directly -> flash kernel on Metal (g.sdpa threshold-gates it)
            let gpu = metal.eval(&g, out);
            let cpu = interpret(&g, out);
            assert_eq!(gpu.shape, cpu.shape);
            for (p, w) in gpu.f32().iter().zip(cpu.f32()) {
                assert!((p - w).abs() < 1e-3, "b={b} h={h} s={s} dh={dh} causal={causal}: {p} vs {w}");
            }
        }
    }
}
