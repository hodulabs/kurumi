//! Cumulative scans along an axis: cumsum (triangular-ones contraction), cumprod
//! (log/exp + sign parity), cummax/cummin (triangular-mask reduce).

use crate::{DType, Error, Graph, NodeId, Storage};

impl Graph {
    /// Cumulative sum along `axis`: `out[..,i,..] = sum_{j<=i} x[..,j,..]`.
    /// Contraction with an upper-triangular ones matrix: rides MPS GEMM on Metal,
    /// differentiable for free. O(n^2) materializes an [n,n] mask -- swap in an O(n)
    /// scan primitive if long-axis cumsum ever shows up hot.
    pub fn cumsum(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        let rank = sh.len();
        if axis >= rank {
            return Err(Error::shape("cumsum", "axis out of range"));
        }
        let n = sh[axis];
        // U[j,i] = 1 iff j <= i  ->  out[..,i] = sum_{j<=i} x[..,j]
        let mut data = vec![0.0f32; n * n];
        for j in 0..n {
            for i in j..n {
                data[j * n + i] = 1.0;
            }
        }
        let u = self.constant(data, vec![n, n]);
        let dt = self.dtype(x);
        let u = if dt == DType::F32 { u } else { self.cast(u, dt) };
        let last = rank - 1;
        let xt = if axis == last { x } else { self.transpose(x, axis, last)? };
        let out = self.dot_general(xt, u, vec![last], vec![0], vec![], vec![])?;
        if axis == last { Ok(out) } else { self.transpose(out, axis, last) }
    }

    /// Cumulative product along `axis`. `|out| = exp(cumsum(log|x|))` (log 0 -> -inf
    /// -> exp 0 propagates zeros); sign is `(-1)^(cumulative negative count)` via parity,
    /// handling negatives and zeros.
    pub fn cumprod(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let rdt = self.dtype(x);
        // magnitude. clamp log|x| off -inf (from zeros) so the triangular-matmul
        // cumsum doesn't hit -inf*0 = NaN in its masked terms; exp(-1e30) is still 0.
        let a = self.abs(x);
        let la = self.ln(a);
        let la = self.clamp_min(la, -1e30)?;
        let cs = self.cumsum(la, axis)?;
        let mag = self.exp(cs);
        // sign parity: (-1)^(#negatives so far) = 1 - 2*(cumsum(x<0) mod 2)
        let zero = self.zeros_like(x);
        let neg = self.cmp_lt(x, zero)?;
        let negf = self.cast(neg, rdt);
        let cnt = self.cumsum(negf, axis)?;
        let two = self.scalar(cnt, 2.0);
        let parity = self.rem(cnt, two)?;
        let twop = self.scalar(parity, 2.0);
        let tp = self.mul(twop, parity)?;
        let one = self.scalar(tp, 1.0);
        let sign = self.sub(one, tp)?;
        self.mul(sign, mag)
    }

    /// Cumulative maximum along `axis` (`out[i] = max(x[0..=i])`).
    pub fn cummax(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        self.cumulative_max(x, axis)
    }
    /// Cumulative minimum along `axis`.
    pub fn cummin(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let nx = self.neg(x);
        let c = self.cumulative_max(nx, axis)?;
        Ok(self.neg(c))
    }
    // out[.., i] = max over j<=i of x[.., j]: broadcast to [.., n(i), n(j)], add a
    // lower-triangular 0 / upper -inf mask, reduce_max over j. O(n^2), mirrors cumsum.
    fn cumulative_max(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        let last = sh.len() - 1;
        let xt = if axis == last { x } else { self.transpose(x, axis, last)? };
        let tsh = self.shape(xt);
        let n = tsh[last];
        let mut b1 = tsh.clone();
        b1.insert(last, 1); // [.., 1, n]
        let xr = self.reshape(xt, b1)?;
        let mut full = tsh.clone();
        full.insert(last, n); // [.., n, n]
        let xb = self.broadcast_to(xr, full.clone())?;
        // mask[i,j] = 0 if j<=i else -inf
        let mut m = vec![0f32; n * n];
        for (i, row) in m.chunks_mut(n).enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                if j > i {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        let mut msh = vec![1usize; full.len()];
        msh[full.len() - 2] = n;
        msh[full.len() - 1] = n;
        let mn = self.const_storage(Storage::F32(m), msh);
        let mnc = if self.dtype(xt) == DType::F32 { mn } else { self.cast(mn, self.dtype(xt)) };
        let mb = self.broadcast_to(mnc, full)?;
        let masked = self.add(xb, mb)?;
        let r = self.reduce_max(masked, last + 1)?;
        if axis == last { Ok(r) } else { self.transpose(r, axis, last) }
    }
}
