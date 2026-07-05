//! N-D pooling (max/avg/min/sum, 1d/2d/3d): each output is a reduction over
//! strided-slice windows, so autodiff & every backend come for free.

use crate::{Error, Graph, NodeId};

impl Graph {
    // Generic windowed reduction over `[N, C, *spatial]` (spatial rank = kernel.len()).
    // No padding. `mode` = "max"|"min"|"sum"|"avg". `dilation` spaces the window taps.
    // Enumerates the kernel-offset grid, slices each strided (dilated) window, reduces.
    fn pool_nd(
        &mut self,
        x: NodeId,
        kernel: &[usize],
        stride: &[usize],
        dilation: &[usize],
        mode: &str,
    ) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        let d = kernel.len();
        if sh.len() != d + 2 {
            return Err(Error::shape("pool", "expects [N, C, *spatial] matching the kernel rank"));
        }
        if stride.len() != d || dilation.len() != d {
            return Err(Error::shape("pool", "stride/dilation rank must match the window"));
        }
        let (n, c) = (sh[0], sh[1]);
        // effective (dilated) window size = dilation*(kernel-1)+1
        let out_sp: Vec<usize> =
            (0..d).map(|i| (sh[2 + i] - (dilation[i] * (kernel[i] - 1) + 1)) / stride[i] + 1).collect();
        let total: usize = kernel.iter().product();
        let mut acc: Option<NodeId> = None;
        for off in 0..total {
            // decode the flat offset into a per-axis kernel index (row-major)
            let mut ki = vec![0usize; d];
            let mut o = off;
            for i in (0..d).rev() {
                ki[i] = o % kernel[i];
                o /= kernel[i];
            }
            let mut ranges = vec![(0, n, 1), (0, c, 1)];
            for i in 0..d {
                let start = ki[i] * dilation[i];
                ranges.push((start, start + (out_sp[i] - 1) * stride[i] + 1, stride[i]));
            }
            let win = self.slice_step(x, ranges)?;
            acc = Some(match acc {
                None => win,
                Some(a) => match mode {
                    "max" => self.max(a, win)?,
                    "min" => self.min(a, win)?,
                    _ => self.add(a, win)?, // sum / avg
                },
            });
        }
        let s = acc.ok_or_else(|| Error::shape("pool", "empty kernel"))?;
        if mode == "avg" {
            let inv = self.scalar(s, 1.0 / total as f32);
            self.mul(s, inv)
        } else {
            Ok(s)
        }
    }

    /// General windowed reduction over `[N, C, *spatial]`: arbitrary `window`/`stride`/
    /// `dilation` (per spatial axis) and `mode` ("max"|"min"|"sum"|"avg"|"mean"). The
    /// fixed-rank `*_pool{1,2,3}d` are dilation-1 wrappers of this.
    pub fn reduce_window(
        &mut self,
        x: NodeId,
        window: &[usize],
        stride: &[usize],
        dilation: &[usize],
        mode: &str,
    ) -> Result<NodeId, Error> {
        let m = if mode == "mean" { "avg" } else { mode };
        if !matches!(m, "max" | "min" | "sum" | "avg") {
            return Err(Error::shape("reduce_window", format!("mode must be max|min|sum|avg|mean, got {mode}")));
        }
        self.pool_nd(x, window, stride, dilation, m)
    }

    /// 1-D pooling `[N, C, L] -> [N, C, Lo]` (no padding).
    pub fn max_pool1d(&mut self, x: NodeId, k: usize, s: usize) -> Result<NodeId, Error> {
        self.pool_nd(x, &[k], &[s], &[1], "max")
    }
    pub fn avg_pool1d(&mut self, x: NodeId, k: usize, s: usize) -> Result<NodeId, Error> {
        self.pool_nd(x, &[k], &[s], &[1], "avg")
    }
    pub fn min_pool1d(&mut self, x: NodeId, k: usize, s: usize) -> Result<NodeId, Error> {
        self.pool_nd(x, &[k], &[s], &[1], "min")
    }
    pub fn sum_pool1d(&mut self, x: NodeId, k: usize, s: usize) -> Result<NodeId, Error> {
        self.pool_nd(x, &[k], &[s], &[1], "sum")
    }

    /// 2-D pooling `[N, C, H, W] -> [N, C, Ho, Wo]` (no padding).
    pub fn max_pool2d(&mut self, x: NodeId, k: (usize, usize), s: (usize, usize)) -> Result<NodeId, Error> {
        self.pool_nd(x, &[k.0, k.1], &[s.0, s.1], &[1, 1], "max")
    }
    pub fn avg_pool2d(&mut self, x: NodeId, k: (usize, usize), s: (usize, usize)) -> Result<NodeId, Error> {
        self.pool_nd(x, &[k.0, k.1], &[s.0, s.1], &[1, 1], "avg")
    }
    pub fn min_pool2d(&mut self, x: NodeId, k: (usize, usize), s: (usize, usize)) -> Result<NodeId, Error> {
        self.pool_nd(x, &[k.0, k.1], &[s.0, s.1], &[1, 1], "min")
    }
    pub fn sum_pool2d(&mut self, x: NodeId, k: (usize, usize), s: (usize, usize)) -> Result<NodeId, Error> {
        self.pool_nd(x, &[k.0, k.1], &[s.0, s.1], &[1, 1], "sum")
    }

    /// 3-D pooling `[N, C, D, H, W] -> [N, C, Do, Ho, Wo]` (no padding).
    pub fn max_pool3d(
        &mut self,
        x: NodeId,
        k: (usize, usize, usize),
        s: (usize, usize, usize),
    ) -> Result<NodeId, Error> {
        self.pool_nd(x, &[k.0, k.1, k.2], &[s.0, s.1, s.2], &[1, 1, 1], "max")
    }
    pub fn avg_pool3d(
        &mut self,
        x: NodeId,
        k: (usize, usize, usize),
        s: (usize, usize, usize),
    ) -> Result<NodeId, Error> {
        self.pool_nd(x, &[k.0, k.1, k.2], &[s.0, s.1, s.2], &[1, 1, 1], "avg")
    }
    pub fn min_pool3d(
        &mut self,
        x: NodeId,
        k: (usize, usize, usize),
        s: (usize, usize, usize),
    ) -> Result<NodeId, Error> {
        self.pool_nd(x, &[k.0, k.1, k.2], &[s.0, s.1, s.2], &[1, 1, 1], "min")
    }
    pub fn sum_pool3d(
        &mut self,
        x: NodeId,
        k: (usize, usize, usize),
        s: (usize, usize, usize),
    ) -> Result<NodeId, Error> {
        self.pool_nd(x, &[k.0, k.1, k.2], &[s.0, s.1, s.2], &[1, 1, 1], "sum")
    }
}
