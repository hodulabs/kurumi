//! Full transformer blocks on Metal vs the CPU oracle: a 1-block GPT/Llama LM (embed ->
//! pre-norm attention + SwiGLU -> RMSNorm -> logits -> cross-entropy) and a Llama-style block.

use crate::tests::*;

// A full 1-block GPT/Llama LM on the GPU: token embedding (gather) -> pre-norm
// causal-attention + SwiGLU -> final RMSNorm -> logits -> cross-entropy. Forward AND
// grads (w.r.t. the embedding table and output projection) match the CPU oracle.
#[test]
fn metal_gpt_lm_fwd_bwd_matches_cpu() {
    use kurumi_core::{Backend, Storage, grad};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (vocab, bsz, s, hh, dh, dff, eps) = (16usize, 2, 6, 2, 4, 16, 1e-5);
    let dm = hh * dh;
    let mk = |n: usize, seed: usize| (0..n).map(|i| (((i + seed) % 17) as f32) * 0.05 - 0.4).collect::<Vec<_>>();
    let mut g = Graph::new();
    let ids: Vec<i32> = (0..bsz * s).map(|i| (i * 5 % vocab) as i32).collect();
    let tok = g.const_storage(Storage::I32(ids), vec![bsz, s]);
    let embed = g.constant(mk(vocab * dm, 9), vec![vocab, dm]);
    let wo_out = g.constant(mk(dm * vocab, 8), vec![dm, vocab]);
    let (wq, wk, wv, wo) = (
        g.constant(mk(dm * dm, 1), vec![dm, dm]),
        g.constant(mk(dm * dm, 2), vec![dm, dm]),
        g.constant(mk(dm * dm, 3), vec![dm, dm]),
        g.constant(mk(dm * dm, 4), vec![dm, dm]),
    );
    let (wg, wu, wd) = (
        g.constant(mk(dm * dff, 5), vec![dm, dff]),
        g.constant(mk(dm * dff, 6), vec![dm, dff]),
        g.constant(mk(dff * dm, 7), vec![dff, dm]),
    );

    let x0 = g.gather(embed, tok, 0).unwrap(); // [B,S,Dm] token embeddings
    // pre-norm attention sub-block
    let hn = g.rmsnorm(x0, 2, eps).unwrap();
    let h2d = g.reshape(hn, vec![bsz * s, dm]).unwrap();
    let proj = |g: &mut Graph, w| g.dot_general(h2d, w, vec![1], vec![0], vec![], vec![]).unwrap();
    let heads = |g: &mut Graph, p| {
        let r = g.reshape(p, vec![bsz, s, hh, dh]).unwrap();
        g.permute(r, vec![0, 2, 1, 3]).unwrap()
    };
    let (q, k, v) = (proj(&mut g, wq), proj(&mut g, wk), proj(&mut g, wv));
    let (q, k, v) = (heads(&mut g, q), heads(&mut g, k), heads(&mut g, v));
    let attn = g.sdpa(q, k, v, true).unwrap();
    let attn = g.permute(attn, vec![0, 2, 1, 3]).unwrap();
    let attn = g.reshape(attn, vec![bsz * s, dm]).unwrap();
    let o = g.dot_general(attn, wo, vec![1], vec![0], vec![], vec![]).unwrap();
    let o = g.reshape(o, vec![bsz, s, dm]).unwrap();
    let x1 = g.add(x0, o).unwrap();
    // pre-norm SwiGLU MLP sub-block
    let hn2 = g.rmsnorm(x1, 2, eps).unwrap();
    let h2 = g.reshape(hn2, vec![bsz * s, dm]).unwrap();
    let gate = {
        let gp = g.dot_general(h2, wg, vec![1], vec![0], vec![], vec![]).unwrap();
        g.silu(gp)
    };
    let up = g.dot_general(h2, wu, vec![1], vec![0], vec![], vec![]).unwrap();
    let gu = g.mul(gate, up).unwrap();
    let mlp = g.dot_general(gu, wd, vec![1], vec![0], vec![], vec![]).unwrap();
    let mlp = g.reshape(mlp, vec![bsz, s, dm]).unwrap();
    let x2 = g.add(x1, mlp).unwrap();
    // head: final norm -> logits -> cross-entropy
    let fin = g.rmsnorm(x2, 2, eps).unwrap();
    let fin2d = g.reshape(fin, vec![bsz * s, dm]).unwrap();
    let logits = g.dot_general(fin2d, wo_out, vec![1], vec![0], vec![], vec![]).unwrap(); // [B*S, V]
    let mut tgt = vec![0.0f32; bsz * s * vocab];
    for r in 0..bsz * s {
        tgt[r * vocab + ((r + 1) % vocab)] = 1.0; // arbitrary one-hot targets
    }
    let targets = g.constant(tgt, vec![bsz * s, vocab]);
    let ce = g.cross_entropy(logits, targets, 1).unwrap();
    let loss = g.sum(ce, 0).unwrap();

    let grads = grad(&mut g, loss, &[embed, wo_out]).unwrap();
    for &id in &[loss, grads[0], grads[1]] {
        let gpu = metal.eval(&g, id);
        let cpu = interpret(&g, id);
        assert_eq!(gpu.shape, cpu.shape);
        for (p, w) in gpu.f32().iter().zip(cpu.f32()) {
            assert!((p - w).abs() < 2e-2, "{p} vs {w}");
        }
    }
}

// A full Llama-style pre-norm transformer block on the GPU: RMSNorm -> causal
// multi-head attention (head reshape/permute) -> residual -> RMSNorm -> SwiGLU MLP ->
// residual. Forward AND backward must match the CPU oracle.
#[test]
fn metal_transformer_block_fwd_bwd_matches_cpu() {
    use kurumi_core::{Backend, grad};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (bsz, s, h, dh, dff, eps) = (2usize, 4, 2, 4, 16, 1e-5);
    let dm = h * dh;
    let mk = |n: usize, seed: usize| (0..n).map(|i| (((i + seed) % 17) as f32) * 0.05 - 0.4).collect::<Vec<_>>();
    let mut g = Graph::new();
    let x0 = g.constant(mk(bsz * s * dm, 0), vec![bsz, s, dm]);
    let wq = g.constant(mk(dm * dm, 1), vec![dm, dm]);
    let wk = g.constant(mk(dm * dm, 2), vec![dm, dm]);
    let wv = g.constant(mk(dm * dm, 3), vec![dm, dm]);
    let wo = g.constant(mk(dm * dm, 4), vec![dm, dm]);
    let wg = g.constant(mk(dm * dff, 5), vec![dm, dff]);
    let wu = g.constant(mk(dm * dff, 6), vec![dm, dff]);
    let wd = g.constant(mk(dff * dm, 7), vec![dff, dm]);

    // attention sub-block (pre-norm)
    let hn = g.rmsnorm(x0, 2, eps).unwrap();
    let h2d = g.reshape(hn, vec![bsz * s, dm]).unwrap(); // flatten tokens for the linear projections
    let proj = |g: &mut Graph, w| g.dot_general(h2d, w, vec![1], vec![0], vec![], vec![]).unwrap();
    let heads = |g: &mut Graph, p| {
        let r = g.reshape(p, vec![bsz, s, h, dh]).unwrap();
        g.permute(r, vec![0, 2, 1, 3]).unwrap() // [B,H,S,Dh]
    };
    let (q, k, v) = (proj(&mut g, wq), proj(&mut g, wk), proj(&mut g, wv));
    let (q, k, v) = (heads(&mut g, q), heads(&mut g, k), heads(&mut g, v));
    let attn = g.sdpa(q, k, v, true).unwrap(); // [B,H,S,Dh]
    let attn = g.permute(attn, vec![0, 2, 1, 3]).unwrap(); // [B,S,H,Dh]
    let attn = g.reshape(attn, vec![bsz * s, dm]).unwrap();
    let o = g.dot_general(attn, wo, vec![1], vec![0], vec![], vec![]).unwrap();
    let o = g.reshape(o, vec![bsz, s, dm]).unwrap();
    let x1 = g.add(x0, o).unwrap(); // residual

    // SwiGLU MLP sub-block (pre-norm)
    let hn2 = g.rmsnorm(x1, 2, eps).unwrap();
    let h2 = g.reshape(hn2, vec![bsz * s, dm]).unwrap();
    let gp = g.dot_general(h2, wg, vec![1], vec![0], vec![], vec![]).unwrap();
    let gate = g.silu(gp);
    let up = g.dot_general(h2, wu, vec![1], vec![0], vec![], vec![]).unwrap();
    let gu = g.mul(gate, up).unwrap();
    let mlp = g.dot_general(gu, wd, vec![1], vec![0], vec![], vec![]).unwrap();
    let mlp = g.reshape(mlp, vec![bsz, s, dm]).unwrap();
    let out = g.add(x1, mlp).unwrap(); // residual

    let gpu = metal.eval(&g, out);
    let cpu = interpret(&g, out);
    assert_eq!(gpu.shape, cpu.shape);
    for (p, w) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((p - w).abs() < 2e-2, "fwd {p} vs {w}");
    }

    // backward: loss = sum(out), grads w.r.t. an attention + an MLP weight
    let mut acc = out;
    for ax in (0..3).rev() {
        acc = g.sum(acc, ax).unwrap();
    }
    let grads = grad(&mut g, acc, &[wq, wd]).unwrap();
    for &gid in &grads {
        let gg = metal.eval(&g, gid);
        let cg = interpret(&g, gid);
        assert_eq!(gg.shape, cg.shape);
        for (p, w) in gg.f32().iter().zip(cg.f32()) {
            assert!((p - w).abs() < 2e-2, "bwd {p} vs {w}");
        }
    }
}
