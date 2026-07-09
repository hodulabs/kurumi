//! Model-level fwd+bwd vs oracle: softmax/layernorm, attention/SDPA, RoPE, cross-entropy, GPT/transformer, training.

use crate::tests::*;

// softmax + layernorm are now fully device-resident (reduce + broadcast +
// elementwise all on GPU); the readback must still match the CPU oracle.
#[test]
fn metal_softmax_layernorm_match_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let mut g = Graph::new();
    let x = g.constant((0..32 * 16).map(|i| ((i % 23) as f32) * 0.1 - 1.0).collect(), vec![32, 16]);
    let sm = g.softmax(x, 1).unwrap(); // max+sum reduce, broadcast, exp/div
    let ln = g.layernorm(sm, 1, 1e-5).unwrap(); // mean/var reduce, broadcast, elementwise
    let gpu = metal.eval(&g, ln);
    let cpu = interpret(&g, ln);
    assert_eq!(gpu.shape, cpu.shape);
    for (a, b) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}

// a full forward + backward (autograd) training step on Metal: the forward GEMM
// is canonical, the backward GEMMs (dL/dw = x^T @ dy, dL/dx = dy @ w^T) are
// transposed dot_generals: all must run on the GPU and match the CPU oracle.
#[test]
fn metal_forward_backward_matches_cpu() {
    use kurumi_core::{Backend, grad};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let mut g = Graph::new();
    let (m, k, n) = (16, 24, 8);
    let x = g.constant((0..m * k).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect(), vec![m, k]);
    let w = g.constant((0..k * n).map(|i| ((i % 7) as f32) * 0.05 - 0.15).collect(), vec![k, n]);
    let y = g.dot_general(x, w, vec![1], vec![0], vec![], vec![]).unwrap(); // forward GEMM
    let a = g.gelu(y);
    let s = g.sum(a, 1).unwrap();
    let loss = g.sum(s, 0).unwrap(); // scalar
    let grads = grad(&mut g, loss, &[w, x]).unwrap(); // dL/dw, dL/dx (transposed GEMMs)
    for &gid in &grads {
        let gpu = metal.eval(&g, gid);
        let cpu = interpret(&g, gid);
        assert_eq!(gpu.shape, cpu.shape);
        for (p, q) in gpu.f32().iter().zip(cpu.f32()) {
            assert!((p - q).abs() < 1e-2, "{p} vs {q}");
        }
    }
}

// A real SGD training loop learns on Metal. The graph is built ONCE (params/data as
// `Input` nodes); each step feeds current values and evaluates autodiffed grads on the
// GPU -- exercising feeds, matmul fwd+bwd, expand, reduce, fused pointwise, pipeline reuse.
#[test]
fn metal_trains_linear_regression() {
    use kurumi_core::{Backend, DType, Feeds, TensorVal, grad};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (n, d) = (32usize, 4usize);
    // well-conditioned design: each column a distinct frequency (near-orthogonal)
    let xs: Vec<f32> = (0..n).flat_map(|r| (0..d).map(move |c| (r as f32 * 0.3 * (c as f32 + 1.0)).sin())).collect();
    let w_true = [0.5f32, -1.2, 0.3, 2.0];
    let b_true = 0.7f32;
    let ys: Vec<f32> = (0..n).map(|r| (0..d).map(|c| xs[r * d + c] * w_true[c]).sum::<f32>() + b_true).collect();

    // build the model + loss + grads ONCE; params/data are fed each step
    let mut g = Graph::new();
    let x_in = g.input(vec![n, d], DType::F32);
    let y_in = g.input(vec![n, 1], DType::F32);
    let w_in = g.input(vec![d, 1], DType::F32);
    let b_in = g.input(vec![1, 1], DType::F32);
    let xw = g.dot_general(x_in, w_in, vec![1], vec![0], vec![], vec![]).unwrap();
    let bb = g.expand(b_in, vec![n, 1]).unwrap();
    let pred = g.add(xw, bb).unwrap();
    let diff = g.sub(pred, y_in).unwrap();
    let sq = g.mul(diff, diff).unwrap();
    let s1 = g.sum(sq, 0).unwrap();
    let loss = g.sum(s1, 0).unwrap(); // sum of squared error
    let grads = grad(&mut g, loss, &[w_in, b_in]).unwrap();

    let f32v = |shape: Vec<usize>, data: Vec<f32>| TensorVal { shape, storage: Storage::F32(data) };
    let feeds = |w: &[f32], b: f32| -> Feeds {
        Feeds::from([
            (x_in, f32v(vec![n, d], xs.clone())),
            (y_in, f32v(vec![n, 1], ys.clone())),
            (w_in, f32v(vec![d, 1], w.to_vec())),
            (b_in, f32v(vec![1, 1], vec![b])),
        ])
    };

    let mut w = vec![0.0f32; d];
    let mut b = 0.0f32;
    let lr = 0.4 / n as f32;
    let loss0 = metal.eval_with(&g, loss, &feeds(&w, b)).f32()[0];
    for _ in 0..800 {
        let f = feeds(&w, b);
        let gw = metal.eval_with(&g, grads[0], &f).f32().to_vec();
        let gb = metal.eval_with(&g, grads[1], &f).f32()[0];
        for c in 0..d {
            w[c] -= lr * gw[c];
        }
        b -= lr * gb;
    }
    let loss_n = metal.eval_with(&g, loss, &feeds(&w, b)).f32()[0];
    assert!(loss_n < loss0 * 1e-3, "loss did not converge: {loss0} -> {loss_n}");
    for c in 0..d {
        assert!((w[c] - w_true[c]).abs() < 0.1, "w[{c}] = {} (want {})", w[c], w_true[c]);
    }
    assert!((b - b_true).abs() < 0.1, "b = {b} (want {b_true})");
}

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

// cross-entropy loss + its gradient w.r.t. logits run device-resident and match
// the CPU oracle: the training loss for classification / language modeling.
#[test]
fn metal_cross_entropy_fwd_bwd_matches_cpu() {
    use kurumi_core::{Backend, grad};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (n, c) = (6usize, 5usize);
    let mut g = Graph::new();
    let logits = g.constant((0..n * c).map(|i| ((i * 5 % 13) as f32) * 0.2 - 1.0).collect(), vec![n, c]);
    // one-hot targets: example r -> class (r % c)
    let mut t = vec![0.0f32; n * c];
    for r in 0..n {
        t[r * c + (r % c)] = 1.0;
    }
    let targets = g.constant(t, vec![n, c]);
    let ce = g.cross_entropy(logits, targets, 1).unwrap(); // [n]
    let loss = g.sum(ce, 0).unwrap();
    let dlogits = grad(&mut g, loss, &[logits]).unwrap()[0];
    for &id in &[loss, dlogits] {
        let gpu = metal.eval(&g, id);
        let cpu = interpret(&g, id);
        assert_eq!(gpu.shape, cpu.shape);
        for (p, w) in gpu.f32().iter().zip(cpu.f32()) {
            assert!((p - w).abs() < 1e-3, "{p} vs {w}");
        }
    }
}

// RoPE (slice + pad-concat + sin/cos + elementwise) runs device-resident and
// matches the CPU oracle.
#[test]
fn metal_rope_matches_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (b, h, s, d) = (2usize, 2, 5, 8);
    let mut g = Graph::new();
    let x = g.constant((0..b * h * s * d).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect(), vec![b, h, s, d]);
    let y = g.rope(x).unwrap();
    let gpu = metal.eval(&g, y);
    let cpu = interpret(&g, y);
    assert_eq!(gpu.shape, cpu.shape);
    for (p, w) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((p - w).abs() < 1e-3, "{p} vs {w}");
    }
}

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

#[test]
fn metal_conv2d_matches_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else { return };
    let (n, c, h, w, o, k) = (2usize, 3, 8, 8, 4, 3);
    let mut g = Graph::new();
    let gi = g.constant((0..n * c * h * w).map(|i| ((i % 17) as f32) * 0.1 - 0.5).collect(), vec![n, c, h, w]);
    let gw = g.constant((0..o * c * k * k).map(|i| ((i % 7) as f32) * 0.05).collect(), vec![o, c, k, k]);
    let y = g.conv2d(gi, gw, (2, 2), (1, 1), (1, 1)).unwrap();
    let gpu = metal.eval(&g, y);
    let cpu = interpret(&g, y);
    assert_eq!(gpu.shape, cpu.shape);
    for (a, b) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}

// The shared-memo eval_many_with override: several outputs sharing a forward trunk
// (a device matmul + gelu) evaluate in ONE pass -- recycle once up front, then one
// memo across all outputs, so the trunk (reused by the two grads) computes once. The
// batched results must match per-node eval_with exactly: a reused/aliased shared buffer
// that got corrupted mid-pass would diverge, so exact equality pins the shared path.
#[test]
fn metal_eval_many_matches_per_node() {
    use kurumi_core::{Backend, Feeds, grad};
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (m, k, n) = (4usize, 6, 5);
    let mut g = Graph::new();
    let a = g.constant((0..m * k).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect(), vec![m, k]);
    let b = g.constant((0..k * n).map(|i| ((i % 7) as f32) * 0.05 - 0.15).collect(), vec![k, n]);
    let t = {
        let mm = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap(); // shared trunk
        g.gelu(mm)
    };
    let mut y = t;
    for ax in (0..2).rev() {
        y = g.sum(y, ax).unwrap(); // y = sum(gelu(a@b))
    }
    let grads = grad(&mut g, y, &[a, b]).unwrap(); // ga, gb flow back through the shared trunk
    let outs = [y, grads[0], grads[1]];
    let feeds = Feeds::new();
    let many = metal.eval_many_with(&g, &outs, &feeds);
    assert_eq!(many.len(), outs.len());
    for (o, batched) in outs.iter().zip(&many) {
        let single = metal.eval_with(&g, *o, &feeds);
        assert_eq!(batched.shape, single.shape, "shape mismatch for {o:?}");
        assert_eq!(batched.f32(), single.f32(), "eval_many vs eval_with for {o:?}");
    }
}
