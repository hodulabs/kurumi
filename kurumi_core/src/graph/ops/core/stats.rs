//! Order & second-moment statistics: median / quantile (via sort), mode, covariance
//! & correlation. Pure decompositions -> autodiff and every backend.

use crate::{DType, Error, Graph, NodeId};

impl Graph {
    /// Median over `axis` (average of the two middle order-statistics for even length).
    pub fn median(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let n = self.shape(x)[axis];
        let s = self.sort(x, axis, false)?;
        if n % 2 == 1 {
            self.pick(s, axis, n / 2)
        } else {
            let a = self.pick(s, axis, n / 2 - 1)?;
            let b = self.pick(s, axis, n / 2)?;
            let sum = self.add(a, b)?;
            let h = self.scalar(sum, 0.5);
            self.mul(sum, h)
        }
    }

    /// The `q`-quantile over `axis` (q in [0,1], linear interpolation between order
    /// statistics: numpy's default).
    pub fn quantile(&mut self, x: NodeId, axis: usize, q: f32) -> Result<NodeId, Error> {
        let n = self.shape(x)[axis];
        let s = self.sort(x, axis, false)?;
        let pos = q.clamp(0.0, 1.0) * (n - 1) as f32;
        let lo = pos.floor() as usize;
        let frac = pos - lo as f32;
        let vlo = self.pick(s, axis, lo)?;
        if frac == 0.0 {
            return Ok(vlo);
        }
        let vhi = self.pick(s, axis, (lo + 1).min(n - 1))?;
        let a = self.scalar(vlo, 1.0 - frac);
        let ta = self.mul(vlo, a)?;
        let b = self.scalar(vhi, frac);
        let tb = self.mul(vhi, b)?;
        self.add(ta, tb)
    }

    // slice one index along `axis` and drop the (now size-1) axis
    fn pick(&mut self, x: NodeId, axis: usize, k: usize) -> Result<NodeId, Error> {
        let mut ranges: Vec<(usize, usize)> = self.shape(x).iter().map(|&d| (0, d)).collect();
        ranges[axis] = (k, k + 1);
        let sl = self.slice(x, ranges)?;
        self.squeeze(sl, axis)
    }

    /// Mode over `axis`: the most frequent value (smallest on ties). O(n^2) pairwise
    /// count over the sorted axis: exact-equality, so for discrete/quantized data.
    pub fn mode(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let s = self.sort(x, axis, false)?; // ascending
        // count[.., i] = sum_j (s[j] == s[i]) via a [.., n, n] pairwise compare on `axis`.
        let si = self.unsqueeze(s, axis)?; // [.., 1, n, ..]
        let sj = self.unsqueeze(s, axis + 1)?; // [.., n, 1, ..]
        let mut full = self.shape(si);
        full[axis] = self.shape(s)[axis]; // [.., n, n, ..]
        let sib = self.broadcast_to(si, full.clone())?;
        let sjb = self.broadcast_to(sj, full)?;
        let eq = self.cmp_eq(sib, sjb)?;
        let eqf = self.cast(eq, DType::F32);
        let count = self.sum(eqf, axis + 1)?; // [.., n, ..]
        let idx = self.argmax(count, axis)?; // first max = smallest value (ascending)
        let idxu = self.unsqueeze(idx, axis)?;
        let m = self.take_along_dim(s, idxu, axis)?;
        self.squeeze(m, axis)
    }

    /// Covariance matrix of `x = [features, observations]` -> `[features, features]`
    /// (rows centered, divided by `observations - 1`).
    pub fn cov(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        if sh.len() != 2 {
            return Err(Error::shape("cov", "expects [features, observations]"));
        }
        let obs = sh[1];
        let m = self.mean(x, 1)?;
        let mb = self.broadcast_back(m, &sh, 1)?;
        let xc = self.sub(x, mb)?;
        let xct = self.transpose(xc, 0, 1)?;
        let prod = self.dot_general(xc, xct, vec![1], vec![0], vec![], vec![])?;
        let inv = self.scalar(prod, 1.0 / (obs.max(2) - 1) as f32);
        self.mul(prod, inv)
    }

    /// Correlation matrix: covariance normalized by the outer product of stddevs.
    pub fn corrcoef(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let c = self.cov(x)?;
        let f = self.shape(c)[0];
        let diag = self.diagonal(c)?; // [f]
        let s = self.sqrt(diag);
        let col = self.reshape(s, vec![f, 1])?;
        let row = self.reshape(s, vec![1, f])?;
        let colb = self.broadcast_to(col, vec![f, f])?;
        let rowb = self.broadcast_to(row, vec![f, f])?;
        let denom = self.mul(colb, rowb)?;
        self.div(c, denom)
    }
}
