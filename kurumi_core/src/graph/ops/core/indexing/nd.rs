//! N-d advanced indexing (TF/ONNX GatherND / ScatterND): a length-K coord tensor selects into
//! x's leading K dims, decomposed to a flat row gather/scatter (so autodiff & every backend get
//! it for free). The axis-wise gather/scatter primitives are in the parent `indexing`.

use crate::{DType, Error, Graph, NodeId, ScatterOp};

impl Graph {
    // Shared core for gather_nd/scatter_nd: fold the length-K coords in `idx` into
    // a flat row index over x's flattened leading K dims. Returns
    // (flat [prod batch], b, d, t, batch dims, trailing dims).
    #[allow(clippy::type_complexity)]
    fn nd_flat(
        &mut self,
        x: NodeId,
        idx: NodeId,
        who: &'static str,
    ) -> Result<(NodeId, usize, usize, usize, Vec<usize>, Vec<usize>), Error> {
        let xsh = self.shape(x);
        let ish = self.shape(idx);
        if ish.is_empty() {
            return Err(Error::shape(who, "index needs a trailing coord axis"));
        }
        let r = xsh.len();
        let last = ish.len() - 1;
        let k = ish[last];
        if k == 0 || k > r {
            return Err(Error::shape(who, "coord length must be in 1..=rank(x)"));
        }
        let idx = if self.dtype(idx) == DType::I64 { idx } else { self.cast(idx, DType::I64) };
        let batch: Vec<usize> = ish[..last].to_vec();
        let trail: Vec<usize> = xsh[k..].to_vec();
        let d: usize = xsh[..k].iter().product();
        let t: usize = trail.iter().product::<usize>().max(1);
        let b: usize = batch.iter().product::<usize>().max(1);
        // flat[grid] = sum_kk idx[grid,kk] * prod(xsh[kk+1..K])
        let mut flat: Option<NodeId> = None;
        let mut stride = 1usize;
        for kk in (0..k).rev() {
            let mut rng: Vec<(usize, usize)> = ish.iter().map(|&d| (0, d)).collect();
            rng[last] = (kk, kk + 1);
            let sl = self.slice(idx, rng)?;
            let coord = self.squeeze(sl, last)?;
            let contrib = if stride == 1 {
                coord
            } else {
                let s = self.scalar(coord, stride as f32);
                self.mul(coord, s)?
            };
            flat = Some(match flat {
                None => contrib,
                Some(f) => self.add(f, contrib)?,
            });
            stride *= xsh[kk];
        }
        Ok((flat.unwrap(), b, d, t, batch, trail))
    }

    /// N-d gather (TF/ONNX `GatherND`, no batch dims): `idx` shape `[.., K]`, K <= rank(x);
    /// each length-K coord selects into x's leading K dims. Output = `idx.shape[:-1] ++
    /// x.shape[K:]`. Decomposes to a flat row-gather (scatter-add backward for free).
    pub fn gather_nd(&mut self, x: NodeId, idx: NodeId) -> Result<NodeId, Error> {
        let (flat, b, d, t, batch, trail) = self.nd_flat(x, idx, "gather_nd")?;
        let xr = self.reshape(x, vec![d, t])?;
        let flat2 = self.reshape(flat, vec![b, 1])?;
        let flatbt = self.expand(flat2, vec![b, t])?;
        let gathered = self.gather_along(xr, flatbt, 0)?;
        let mut out_shape = batch;
        out_shape.extend(trail);
        self.reshape(gathered, out_shape)
    }

    /// N-d scatter (dual of `gather_nd`): write/accumulate `updates` into a copy of
    /// `x` at the coords in `idx`. `updates` shape = `idx.shape[:-1] ++ x.shape[K:]`;
    /// `combine` (Set/Add/...) resolves collisions. Grad flows to both x and updates.
    pub fn scatter_nd(&mut self, x: NodeId, idx: NodeId, updates: NodeId, combine: ScatterOp) -> Result<NodeId, Error> {
        let (flat, b, d, t, batch, trail) = self.nd_flat(x, idx, "scatter_nd")?;
        let mut want = batch;
        want.extend(&trail);
        if self.shape(updates) != want {
            return Err(Error::shape("scatter_nd", "updates shape must be idx.shape[:-1] ++ x.shape[K:]"));
        }
        let xsh = self.shape(x);
        let xr = self.reshape(x, vec![d, t])?;
        let flat2 = self.reshape(flat, vec![b, 1])?;
        let flatbt = self.expand(flat2, vec![b, t])?;
        let upd = self.reshape(updates, vec![b, t])?;
        let out = self.scatter_along(xr, flatbt, upd, 0, combine)?;
        self.reshape(out, xsh)
    }
}
