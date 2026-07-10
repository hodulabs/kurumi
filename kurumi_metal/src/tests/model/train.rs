//! Training on Metal vs the CPU oracle: forward+backward GEMMs, a real SGD loop, the
//! shared-memo eval_many path, and cross-entropy fwd+bwd.

use crate::tests::*;

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
