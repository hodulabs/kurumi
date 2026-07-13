//! Slicing/padding movement: slice/slice_step/flip/pad primitives plus non-constant pad modes
//! (reflect/replicate/circular) decomposed into per-axis slice+flip+concat.

use crate::{Error, Graph, NodeId, Op};

impl Graph {
    // primitives

    /// Slice each axis to `[start, end)` (step 1).
    pub fn slice(&mut self, a: NodeId, ranges: Vec<(usize, usize)>) -> Result<NodeId, Error> {
        let stepped = ranges.into_iter().map(|(s, e)| (s, e, 1)).collect();
        self.slice_step(a, stepped)
    }

    /// Strided slice: each axis `[start, end)` taking every `step`-th element
    /// (`a[start:end:step]`). The downsampling/dilation/striding primitive.
    pub fn slice_step(&mut self, a: NodeId, ranges: Vec<(usize, usize, usize)>) -> Result<NodeId, Error> {
        let s = self.shape(a);
        if ranges.len() != s.len() {
            return Err(Error::shape("slice", format!("rank {} vs {} ranges", s.len(), ranges.len())));
        }
        for (d, &(start, end, step)) in ranges.iter().enumerate() {
            if step == 0 {
                return Err(Error::shape("slice", "step must be >= 1"));
            }
            if start > end || end > s[d] {
                return Err(Error::shape("slice", format!("[{start}, {end}) out of [0, {}]", s[d])));
            }
        }
        Ok(self.push(Op::Slice { ranges }, vec![a]))
    }

    pub fn flip(&mut self, a: NodeId, axes: Vec<usize>) -> Result<NodeId, Error> {
        let rank = self.shape(a).len();
        for (k, &ax) in axes.iter().enumerate() {
            if ax >= rank {
                return Err(Error::shape("flip", format!("axis {ax} out of range for rank {rank}")));
            }
            if axes[..k].contains(&ax) {
                return Err(Error::shape("flip", format!("axis {ax} repeated")));
            }
        }
        Ok(self.push(Op::Flip { axes }, vec![a]))
    }

    pub fn pad(&mut self, a: NodeId, pads: Vec<(usize, usize)>) -> Result<NodeId, Error> {
        let rank = self.shape(a).len();
        if pads.len() != rank {
            return Err(Error::shape("pad", format!("rank {rank} vs {} pads", pads.len())));
        }
        Ok(self.push(Op::Pad { pads }, vec![a]))
    }

    // decompositions

    /// Non-constant padding (`Op::Pad` only zero-fills): `mode` is
    /// `"reflect"` | `"replicate"` | `"circular"`, mirroring `np.pad`/`F.pad`.
    /// Per-axis slice+flip+concat, so autodiff & every backend come for free.
    pub fn pad_mode(&mut self, x: NodeId, pads: Vec<(usize, usize)>, mode: &str) -> Result<NodeId, Error> {
        let rank = self.shape(x).len();
        if pads.len() != rank {
            return Err(Error::shape("pad_mode", "one (before, after) per axis"));
        }
        if !matches!(mode, "reflect" | "replicate" | "circular") {
            return Err(Error::shape("pad_mode", "mode must be reflect|replicate|circular"));
        }
        let mut cur = x;
        for (d, &(before, after)) in pads.iter().enumerate() {
            if before == 0 && after == 0 {
                continue;
            }
            let l = self.shape(cur)[d];
            let mut parts = Vec::new();
            if before > 0 {
                parts.push(self.pad_edge(cur, d, l, before, true, mode)?);
            }
            parts.push(cur);
            if after > 0 {
                parts.push(self.pad_edge(cur, d, l, after, false, mode)?);
            }
            cur = self.concat(&parts, d)?;
        }
        Ok(cur)
    }

    // one padding block for axis `d`, side `is_left`, width `p` (dim length `l`).
    fn pad_edge(
        &mut self,
        x: NodeId,
        d: usize,
        l: usize,
        p: usize,
        is_left: bool,
        mode: &str,
    ) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        let mut r: Vec<(usize, usize)> = sh.iter().map(|&x| (0, x)).collect();
        match mode {
            "replicate" => {
                r[d] = if is_left { (0, 1) } else { (l - 1, l) };
                let edge = self.slice(x, r)?;
                let mut es = sh;
                es[d] = p;
                self.expand(edge, es)
            }
            "reflect" => {
                if p >= l {
                    return Err(Error::shape("pad_mode", "reflect pad must be < dim size"));
                }
                r[d] = if is_left { (1, p + 1) } else { (l - 1 - p, l - 1) };
                let sl = self.slice(x, r)?;
                self.flip(sl, vec![d])
            }
            _ => {
                // circular
                if p > l {
                    return Err(Error::shape("pad_mode", "circular pad must be <= dim size"));
                }
                r[d] = if is_left { (l - p, l) } else { (0, p) };
                self.slice(x, r)
            }
        }
    }
}
