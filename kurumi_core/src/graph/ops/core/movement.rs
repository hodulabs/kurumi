//! Shape/view movement: reshape/permute/expand, slice/flip/pad, squeeze/tile/roll,
//! and non-constant pad modes. (join/split -> join.rs, tril/diagonal -> triangular.rs.)

use crate::{Error, Graph, NodeId, Op};

impl Graph {
    // primitives

    pub fn reshape(&mut self, a: NodeId, shape: Vec<usize>) -> Result<NodeId, Error> {
        let (from, to): (usize, usize) = (self.shape(a).iter().product(), shape.iter().product());
        if from != to {
            return Err(Error::shape("reshape", format!("numel {from} -> {to}")));
        }
        Ok(self.push(Op::Reshape { shape }, vec![a]))
    }

    pub fn permute(&mut self, a: NodeId, perm: Vec<usize>) -> Result<NodeId, Error> {
        let rank = self.shape(a).len();
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        if sorted != (0..rank).collect::<Vec<_>>() {
            return Err(Error::shape("permute", format!("{perm:?} is not a permutation of 0..{rank}")));
        }
        Ok(self.push(Op::Permute { perm }, vec![a]))
    }

    pub fn expand(&mut self, a: NodeId, shape: Vec<usize>) -> Result<NodeId, Error> {
        let from = self.shape(a);
        let ok = from.len() == shape.len() && from.iter().zip(&shape).all(|(&f, &t)| f == t || f == 1);
        if !ok {
            return Err(Error::shape("expand", format!("{from:?} -> {shape:?}")));
        }
        Ok(self.push(Op::Expand { shape }, vec![a]))
    }

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

    /// Swap axes `i` and `j` (generalized transpose).
    pub fn transpose(&mut self, x: NodeId, i: usize, j: usize) -> Result<NodeId, Error> {
        let r = self.shape(x).len();
        let mut perm: Vec<usize> = (0..r).collect();
        perm.swap(i, j);
        self.permute(x, perm)
    }

    /// Transpose the last two axes (2D-style `t`).
    pub fn t(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let r = self.shape(x).len();
        self.transpose(x, r - 2, r - 1)
    }

    /// Flatten to 1-D.
    pub fn flatten(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let n: usize = self.shape(x).iter().product();
        self.reshape(x, vec![n])
    }

    /// Remove `axis` if it has size 1 (else no-op).
    pub fn squeeze(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let mut s = self.shape(x);
        if s[axis] == 1 {
            s.remove(axis);
        }
        self.reshape(x, s)
    }

    /// Insert a size-1 axis at `axis`.
    pub fn unsqueeze(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let mut s = self.shape(x);
        s.insert(axis, 1);
        self.reshape(x, s)
    }

    /// Broadcast to `shape` (NumPy rules): prepend size-1 axes, then expand.
    pub fn broadcast_to(&mut self, x: NodeId, shape: Vec<usize>) -> Result<NodeId, Error> {
        let xs = self.shape(x);
        let x = if xs.len() < shape.len() {
            let mut s = vec![1usize; shape.len() - xs.len()];
            s.extend_from_slice(&xs);
            self.reshape(x, s)?
        } else {
            x
        };
        self.expand(x, shape)
    }

    /// Broadcast `x` to `other`'s shape.
    pub fn broadcast_like(&mut self, x: NodeId, other: NodeId) -> Result<NodeId, Error> {
        let shape = self.shape(other);
        self.broadcast_to(x, shape)
    }

    /// Tile axis `axis` `n` times: `[a,b,c]` -> `[a,b,c,a,b,c]` (insert+expand+merge).
    pub fn tile(&mut self, x: NodeId, axis: usize, n: usize) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        let l = sh[axis];
        let mut s1 = sh.clone();
        s1.insert(axis, 1); // [.., 1, L, ..]
        let r = self.reshape(x, s1.clone())?;
        let mut s2 = s1.clone();
        s2[axis] = n; // [.., n, L, ..]
        let e = self.expand(r, s2)?;
        let mut s3 = sh;
        s3[axis] = l * n; // [.., n*L, ..]
        self.reshape(e, s3)
    }

    /// Repeat each element of `axis` `n` times: `[a,b,c]` -> `[a,a,b,b,c,c]`.
    pub fn repeat_interleave(&mut self, x: NodeId, axis: usize, n: usize) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        let l = sh[axis];
        let mut s1 = sh.clone();
        s1.insert(axis + 1, 1); // [.., L, 1, ..]
        let r = self.reshape(x, s1.clone())?;
        let mut s2 = s1;
        s2[axis + 1] = n; // [.., L, n, ..]
        let e = self.expand(r, s2)?;
        let mut s3 = sh;
        s3[axis] = l * n; // [.., L*n, ..]
        self.reshape(e, s3)
    }

    /// Circular shift along `axis` by `shift`: `concat(x[L-s:], x[:L-s])`.
    pub fn roll(&mut self, x: NodeId, shift: usize, axis: usize) -> Result<NodeId, Error> {
        let l = self.shape(x)[axis];
        let s = shift % l;
        if s == 0 {
            return Ok(x);
        }
        let sh = self.shape(x);
        let whole: Vec<(usize, usize)> = sh.iter().map(|&d| (0, d)).collect();
        let mut ra = whole.clone();
        ra[axis] = (l - s, l);
        let mut rb = whole;
        rb[axis] = (0, l - s);
        let a = self.slice(x, ra)?;
        let b = self.slice(x, rb)?;
        self.concat(&[a, b], axis)
    }

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
