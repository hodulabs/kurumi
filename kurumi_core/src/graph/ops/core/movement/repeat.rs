//! Repetition movement: tile / repeat_interleave / roll, all decomposed into reshape+expand or
//! slice+concat so autodiff & every backend come for free.

use crate::{Error, Graph, NodeId};

impl Graph {
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
}
