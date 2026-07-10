//! Bounded-size dynamic selection (JAX-style static `size`): compress / masked_select /
//! nonzero. cumsum gives each true element its output slot, a scatter places it (masked-out
//! elements route to a discarded dump slot). No dynamic-shape IR; differentiable (scatter's
//! grad gathers back). Static `k` (jnp.nonzero(size=k)): fewer than k hits zero-pad the tail, more drop the surplus.

use crate::{DType, Error, Graph, NodeId, ScatterOp};

impl Graph {
    /// Select up to `k` elements of 1-D `x` where bool `mask` is true (stable order)
    /// into a `[k]` output (zero-padded). `x` and `mask` must be 1-D of equal length.
    pub fn compress(&mut self, mask: NodeId, x: NodeId, k: usize) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        if sh.len() != 1 || self.shape(mask) != sh {
            return Err(Error::shape("compress", "x and mask must be 1-D of equal length"));
        }
        // output slot of each element = (running count of trues) - 1
        let mf = self.cast(mask, DType::F32);
        let cs = self.cumsum(mf, 0)?;
        let one = self.scalar(cs, 1.0);
        let posf = self.sub(cs, one)?;
        let posi = self.cast(posf, DType::I64);
        // route masked-out elements to slot `k` (a dump slot beyond the output)
        let kidx = self.scalar(posi, k as f32);
        let idx = self.select(mask, posi, kidx)?;
        // scatter into a [k+1] buffer (the extra slot absorbs masked/overflow), then trim
        let xdt = self.dtype(x);
        let z = self.constant(vec![0.0; k + 1], vec![k + 1]);
        let z = if xdt == DType::F32 { z } else { self.cast(z, xdt) };
        let scat = self.scatter_along(z, idx, x, 0, ScatterOp::Set)?;
        self.slice(scat, vec![(0, k)])
    }

    /// Flatten `x`/`mask` and select up to `k` masked elements -> `[k]` (torch
    /// `masked_select` with a static size).
    pub fn masked_select(&mut self, x: NodeId, mask: NodeId, k: usize) -> Result<NodeId, Error> {
        let n: usize = self.shape(x).iter().product();
        let xf = self.reshape(x, vec![n])?;
        let mf = self.reshape(mask, vec![n])?;
        self.compress(mf, xf, k)
    }

    /// Flat indices of up to `k` nonzero elements of `x` -> `[k]` I64 (numpy
    /// `nonzero` with a static size).
    pub fn nonzero(&mut self, x: NodeId, k: usize) -> Result<NodeId, Error> {
        let n: usize = self.shape(x).iter().product();
        let xf = self.reshape(x, vec![n])?;
        let z = self.zeros_like(xf);
        let mask = self.ne(xf, z)?;
        let idx = self.iota(vec![n], 0, DType::I64)?;
        self.compress(mask, idx, k)
    }

    /// Sorted unique values of `x` -> `[k]` (numpy `unique` with a static size):
    /// sort, then keep each element that differs from its predecessor (position 0
    /// always kept). Uses roll + `or` to build the mask (concat can't join bool).
    pub fn unique(&mut self, x: NodeId, k: usize) -> Result<NodeId, Error> {
        let n: usize = self.shape(x).iter().product();
        let xf = self.reshape(x, vec![n])?;
        let s = self.sort(xf, 0, false)?; // ascending
        let prev = self.roll(s, 1, 0)?; // prev[i] = s[i-1] (prev[0] wraps, fixed below)
        let neq = self.ne(s, prev)?; // s[i] != s[i-1]
        let idx = self.iota(vec![n], 0, DType::I64)?;
        let zero = self.scalar(idx, 0.0);
        let is0 = self.cmp_eq(idx, zero)?; // force position 0 true
        let mask = self.or(neq, is0)?;
        self.compress(mask, s, k)
    }
}
