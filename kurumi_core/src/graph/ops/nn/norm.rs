//! Normalization layers (layernorm, rmsnorm, group_norm, instance_norm, lrn). Losses
//! live in `loss.rs`.

use crate::{Error, Graph, NodeId, Op};

impl Graph {
    /// layernorm(x, axis) = (x - mean) / sqrt(var + eps), over `axis` (no affine).
    pub fn layernorm(&mut self, x: NodeId, axis: usize, eps: f32) -> Result<NodeId, Error> {
        let full = self.shape(x);
        let inv_n = 1.0 / full[axis] as f32;

        let sx = self.sum(x, axis)?;
        let invn1 = self.scalar(sx, inv_n);
        let mean = self.mul(sx, invn1)?;
        let mean_b = self.broadcast_back(mean, &full, axis)?;
        let centered = self.sub(x, mean_b)?;

        let sq = self.mul(centered, centered)?;
        let ssq = self.sum(sq, axis)?;
        let invn2 = self.scalar(ssq, inv_n);
        let var = self.mul(ssq, invn2)?;
        let eps_c = self.scalar(var, eps);
        let var_eps = self.add(var, eps_c)?;
        let std = self.sqrt(var_eps);
        let std_b = self.broadcast_back(std, &full, axis)?;
        self.div(centered, std_b)
    }

    /// RMSNorm over `axis`: `x / sqrt(mean(x^2, axis) + eps)` (no centering, no learnable
    /// scale: the frontend multiplies the weight). A fused primitive: the backend runs one
    /// kernel, the interp oracle decomposes. Llama/T5 norm.
    pub fn rmsnorm(&mut self, x: NodeId, axis: usize, eps: f32) -> Result<NodeId, Error> {
        let rank = self.shape(x).len();
        if axis >= rank {
            return Err(Error::shape("rmsnorm", format!("axis {axis} out of range for rank {rank}")));
        }
        self.require("rmsnorm", x, self.dtype(x).is_float(), "float")?;
        Ok(self.push(Op::RmsNorm { axis, eps }, vec![x]))
    }

    /// Group norm on `[N, C, *spatial]`: split `C` into `groups`, normalize each group
    /// (over its channels + all spatial dims) per sample. No affine (frontend scales).
    pub fn group_norm(&mut self, x: NodeId, groups: usize, eps: f32) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        if sh.len() < 2 || groups == 0 || !sh[1].is_multiple_of(groups) {
            return Err(Error::shape("group_norm", "expects [N, C, ..] with C divisible by groups"));
        }
        let (n, c) = (sh[0], sh[1]);
        let rest: usize = sh[2..].iter().product::<usize>() * (c / groups);
        let flat = self.reshape(x, vec![n, groups, rest])?; // [N, G, (C/G)*spatial]
        let normed = self.layernorm(flat, 2, eps)?; // normalize each group
        self.reshape(normed, sh)
    }

    /// Instance norm on `[N, C, *spatial]`: normalize each channel per sample over the
    /// spatial dims (= group norm with `groups = C`).
    pub fn instance_norm(&mut self, x: NodeId, eps: f32) -> Result<NodeId, Error> {
        let c = self.shape(x)[1];
        self.group_norm(x, c, eps)
    }

    /// Local response normalization over the channel axis (dim 1):
    /// `out = x / (k + alpha*sum_{window} x^2)^beta`, window of `size` channels around each.
    pub fn lrn(&mut self, x: NodeId, size: usize, alpha: f32, beta: f32, k: f32) -> Result<NodeId, Error> {
        let sq = self.square(x);
        let sh = self.shape(x);
        let c = sh[1];
        let r = size / 2;
        // pad the channel axis so the sliding window sum keeps length C
        let mut pads = vec![(0, 0); sh.len()];
        pads[1] = (r, size - 1 - r);
        let padded = self.pad(sq, pads)?;
        let psh = self.shape(padded);
        let mut acc: Option<NodeId> = None;
        for j in 0..size {
            let mut ranges: Vec<(usize, usize)> = psh.iter().map(|&d| (0, d)).collect();
            ranges[1] = (j, j + c);
            let win = self.slice(padded, ranges)?;
            acc = Some(match acc {
                None => win,
                Some(a) => self.add(a, win)?,
            });
        }
        let winsum = acc.ok_or_else(|| Error::shape("lrn", "size must be >= 1"))?;
        let al = self.scalar(winsum, alpha);
        let scaled = self.mul(winsum, al)?;
        let kc = self.scalar(scaled, k);
        let base = self.add(scaled, kc)?;
        let be = self.scalar(base, beta);
        let denom = self.pow(base, be)?;
        self.div(x, denom)
    }
}
