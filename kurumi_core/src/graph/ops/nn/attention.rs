//! Attention primitives (RoPE, scaled dot-product attention) as decompositions.

use crate::{DType, Error, Graph, NodeId, Op};

impl Graph {
    /// Rotary position embedding (NeoX/Llama) over `[.., S, D]` (D even, positions 0..S,
    /// base 10000): `x*cos + rotate_half(x)*sin`. A norm-preserving per-position rotation;
    /// pure primitive composition, so it runs device-resident and autodiffs for free.
    pub fn rope(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let xs = self.shape(x);
        let r = xs.len();
        let (s, d) = (xs[r - 2], xs[r - 1]);
        let half = d / 2;
        // angle[s,i] = s * 10000^(-2i/D) = s * exp(i * (-2 ln10000 / D))
        let pos = self.iota(vec![s, half], 0, DType::F32)?;
        let fi = self.iota(vec![s, half], 1, DType::F32)?;
        let c = self.scalar(fi, -2.0 * 10000f32.ln() / d as f32);
        let ic = self.mul(fi, c)?;
        let theta = self.exp(ic);
        let angle = self.mul(pos, theta)?; // [S, half]
        let sin_h = self.sin(angle);
        let hp = self.scalar(angle, std::f32::consts::FRAC_PI_2);
        let cos_arg = self.add(angle, hp)?;
        let cos_h = self.sin(cos_arg); // cos = sin(x + pi/2)
        let cos_full = self.concat_last(cos_h, cos_h, half)?; // [S, D]
        let sin_full = self.concat_last(sin_h, sin_h, half)?;
        // rotate_half(x) = concat(-x[half:], x[:half])
        let x1 = self.slice_last(x, 0, half)?;
        let x2 = self.slice_last(x, half, d)?;
        let nx2 = self.neg(x2);
        let rot = self.concat_last(nx2, x1, half)?; // [.., D]
        let cos_b = self.broadcast_2d_to(cos_full, &xs)?;
        let sin_b = self.broadcast_2d_to(sin_full, &xs)?;
        let a = self.mul(x, cos_b)?;
        let b = self.mul(rot, sin_b)?;
        self.add(a, b)
    }

    // concat two `[.., L]` tensors along the last axis (a then b) via pad+add.
    fn concat_last(&mut self, a: NodeId, b: NodeId, alen: usize) -> Result<NodeId, Error> {
        let (ash, bsh) = (self.shape(a), self.shape(b));
        let r = ash.len();
        let blen = bsh[r - 1];
        let mut pa = vec![(0, 0); r];
        let mut pb = vec![(0, 0); r];
        pa[r - 1] = (0, blen); // a -> [.., alen+blen], zeros on the right
        pb[r - 1] = (alen, 0); // b shifted right by alen
        let ap = self.pad(a, pa)?;
        let bp = self.pad(b, pb)?;
        self.add(ap, bp)
    }

    // slice `[start, end)` on the last axis, keeping all others whole.
    fn slice_last(&mut self, x: NodeId, start: usize, end: usize) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        let r = sh.len();
        let mut ranges: Vec<(usize, usize)> = sh.iter().map(|&d| (0, d)).collect();
        ranges[r - 1] = (start, end);
        self.slice(x, ranges)
    }

    // broadcast a 2D `[A, B]` to `target` whose last two dims are A, B (leading = 1).
    fn broadcast_2d_to(&mut self, t: NodeId, target: &[usize]) -> Result<NodeId, Error> {
        let ts = self.shape(t);
        let r = target.len();
        let mut full = vec![1usize; r];
        full[r - 2] = ts[0];
        full[r - 1] = ts[1];
        let tr = self.reshape(t, full)?;
        self.expand(tr, target.to_vec())
    }

    /// Scaled dot-product attention over the trailing `[S, D]` (leading axes batch/heads).
    /// Auto-selects: the fast MPS dot+softmax decomposition by default; the fused `Op::Sdpa`
    /// flash primitive (device online-softmax forward, no SxS materialization, + VJP backward)
    /// ONLY when self-attention's `[..,S,S]` score buffer would be prohibitively large.
    /// Cross-attention (S_q != S_k) always decomposes. Autodiff free either way.
    pub fn sdpa(&mut self, q: NodeId, k: NodeId, v: NodeId, causal: bool) -> Result<NodeId, Error> {
        self.require("sdpa", q, self.dtype(q).is_float(), "float")?;
        let (qs, ks, vs) = (self.shape(q), self.shape(k), self.shape(v));
        let r = qs.len();
        let (batch, s) = (qs[..r - 2].iter().product::<usize>(), qs[r - 2]);
        let scores_bytes = batch * s * s * 4; // the [..,S,S] f32 buffer the decomposition materializes
        // flash forward is memory-reduced but ~4x slower (scalar one-thread-per-query online
        // softmax), so use it only when the SxS buffer would otherwise be too large; a fast
        // flash would need a tiled simdgroup-matrix kernel.
        if qs == ks && qs == vs && scores_bytes > (1 << 30) {
            return self.sdpa_fused(q, k, v, causal); // huge SxS -> flash (fits, slower but runs)
        }
        self.sdpa_decomposed(q, k, v, causal) // normal S / cross-attn -> fast MPS decomposition
    }

    /// Explicit dot+softmax SDPA: `scores = q@k^T / sqrt(D)` (+ causal -inf bias on future
    /// keys), softmax over keys, then `@v` -> `[.., S, D]`. Two matmuls on MPS, autodiff free.
    /// The cross-attention path of [`Graph::sdpa`] and the fused primitive's correctness
    /// reference (they match bit-close). Generic in S_q vs S_k (rectangular scores).
    pub fn sdpa_decomposed(&mut self, q: NodeId, k: NodeId, v: NodeId, causal: bool) -> Result<NodeId, Error> {
        let qs = self.shape(q);
        let r = qs.len();
        let (s, dh) = (qs[r - 2], qs[r - 1]);
        let batch: Vec<usize> = (0..r - 2).collect();
        let raw = self.dot_general(q, k, vec![r - 1], vec![r - 1], batch.clone(), batch.clone())?;
        let inv = self.scalar(raw, 1.0 / (dh as f32).sqrt());
        let mut scores = self.mul(raw, inv)?;
        if causal {
            let dt = self.dtype(scores);
            let bias = self.causal_bias(s, &self.shape(scores))?;
            let bias = if dt == DType::F32 { bias } else { self.cast(bias, dt) };
            scores = self.add(scores, bias)?;
        }
        let attn = self.softmax(scores, r - 1)?; // over keys (last axis)
        self.dot_general(attn, v, vec![r - 1], vec![r - 2], batch.clone(), batch)
    }

    /// Fused SDPA PRIMITIVE (`Op::Sdpa`): same math as [`sdpa`] but one op the interp oracle
    /// computes directly (the decomposition above is its correctness reference; forward AND
    /// backward match it bit-close). q,k,v must share shape `[..batch.., S, dh]` (dh>0);
    /// out is q's shape. Interpreted by the CPU oracle + a hand-written VJP; the device
    /// kernel is wired separately.
    pub fn sdpa_fused(&mut self, q: NodeId, k: NodeId, v: NodeId, causal: bool) -> Result<NodeId, Error> {
        let (qs, ks, vs) = (self.shape(q), self.shape(k), self.shape(v));
        if qs != ks || qs != vs {
            return Err(Error::shape("sdpa_fused", format!("q/k/v must share shape: {qs:?} {ks:?} {vs:?}")));
        }
        let r = qs.len();
        if r < 2 || qs[r - 1] == 0 {
            return Err(Error::shape("sdpa_fused", format!("need rank>=2 with dh>0, got {qs:?}")));
        }
        self.same_dtype("sdpa_fused", q, k)?;
        self.same_dtype("sdpa_fused", q, v)?;
        Ok(self.push(Op::Sdpa { causal }, vec![q, k, v]))
    }

    // additive causal bias broadcast to `out_shape` (trailing [S,S]): 0 where key j<=i,
    // -inf above the diagonal (softmax zeroes future positions). One host const (static,
    // depends only on S): the old iota+cmp+select form fell to the host backend per layer,
    // fragmenting the GPU command buffer into a sync per call. Grad still flows to scores
    // via the add; the mask is a constant (no learnable), so nothing is lost.
    fn causal_bias(&mut self, s: usize, out_shape: &[usize]) -> Result<NodeId, Error> {
        let mut data = vec![0.0f32; s * s];
        for i in 0..s {
            for j in (i + 1)..s {
                data[i * s + j] = f32::NEG_INFINITY; // future key
            }
        }
        let bias = self.constant(data, vec![s, s]); // [S, S]
        let rank = out_shape.len();
        let mut full = vec![1usize; rank];
        full[rank - 2] = s;
        full[rank - 1] = s;
        let biased = self.reshape(bias, full)?; // [1,..,1,S,S]
        self.expand(biased, out_shape.to_vec()) // broadcast to [.., S, S]
    }
}
