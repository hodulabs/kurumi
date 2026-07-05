//! Reductions along an axis (keepdim=false) plus whole-tensor and norm variants.

use crate::{ArgKind, DType, Error, Graph, NodeId, Op};

impl Graph {
    // primitives

    pub fn sum(&mut self, a: NodeId, axis: usize) -> Result<NodeId, Error> {
        self.reduce_check_arith("sum", a, axis)?;
        Ok(self.push(Op::Sum { axis }, vec![a]))
    }

    pub fn reduce_max(&mut self, a: NodeId, axis: usize) -> Result<NodeId, Error> {
        self.reduce_check("reduce_max", a, axis)?;
        Ok(self.push(Op::ReduceMax { axis }, vec![a]))
    }

    pub fn prod(&mut self, a: NodeId, axis: usize) -> Result<NodeId, Error> {
        self.reduce_check_arith("prod", a, axis)?;
        Ok(self.push(Op::Prod { axis }, vec![a]))
    }

    /// Index (I64) of the maximum along `axis` (keepdim=false). Non-differentiable.
    pub fn argmax(&mut self, a: NodeId, axis: usize) -> Result<NodeId, Error> {
        self.reduce_check("argmax", a, axis)?;
        Ok(self.push(Op::ArgReduce { axis, kind: ArgKind::Max }, vec![a]))
    }

    /// Index (I64) of the minimum along `axis` (keepdim=false). Non-differentiable.
    pub fn argmin(&mut self, a: NodeId, axis: usize) -> Result<NodeId, Error> {
        self.reduce_check("argmin", a, axis)?;
        Ok(self.push(Op::ArgReduce { axis, kind: ArgKind::Min }, vec![a]))
    }

    // decompositions

    // re-insert a reduced axis as size-1 and broadcast back to `full`
    pub(crate) fn broadcast_back(&mut self, reduced: NodeId, full: &[usize], axis: usize) -> Result<NodeId, Error> {
        let mut keep = self.shape(reduced);
        keep.insert(axis, 1);
        let r = self.reshape(reduced, keep)?;
        self.expand(r, full.to_vec())
    }

    /// Mean over `axis`: `sum(x) / N`.
    pub fn mean(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let n = self.shape(x)[axis] as f32;
        let s = self.sum(x, axis)?;
        let inv = self.scalar(s, 1.0 / n);
        self.mul(s, inv)
    }

    /// Min over `axis` (reduction): `-reduce_max(-x)`.
    pub fn reduce_min(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let nx = self.neg(x);
        let m = self.reduce_max(nx, axis)?;
        Ok(self.neg(m))
    }

    /// Population variance over `axis`: `mean((x - mean)^2)` (correction 0).
    pub fn var(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        self.var_correction(x, axis, 0)
    }

    /// Variance over `axis` with Bessel `correction` (ddof): `sum(x-mean)^2/(N-correction)`.
    /// correction=0 is population variance, correction=1 the unbiased sample estimator
    /// (torch's `unbiased=True` / numpy `ddof=1`).
    pub fn var_correction(&mut self, x: NodeId, axis: usize, correction: usize) -> Result<NodeId, Error> {
        let full = self.shape(x);
        let n = full[axis];
        if correction >= n {
            return Err(Error::shape("var", format!("correction {correction} >= axis length {n}")));
        }
        let m = self.mean(x, axis)?;
        let mb = self.broadcast_back(m, &full, axis)?;
        let c = self.sub(x, mb)?;
        let sq = self.mul(c, c)?;
        let ss = self.sum(sq, axis)?;
        let inv = self.scalar(ss, 1.0 / (n - correction) as f32);
        self.mul(ss, inv)
    }

    /// Standard deviation over `axis`: `sqrt(var)` (correction 0).
    pub fn std(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let v = self.var(x, axis)?;
        Ok(self.sqrt(v))
    }

    /// Standard deviation over `axis` with Bessel `correction`: `sqrt(var_correction)`.
    pub fn std_correction(&mut self, x: NodeId, axis: usize, correction: usize) -> Result<NodeId, Error> {
        let v = self.var_correction(x, axis, correction)?;
        Ok(self.sqrt(v))
    }

    /// L1 norm over `axis`: `sum(|x|)`.
    pub fn l1_norm(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let a = self.abs(x);
        self.sum(a, axis)
    }

    /// L2 norm over `axis`: `sqrt(sum(x^2))`.
    pub fn l2_norm(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let sq = self.mul(x, x)?;
        let s = self.sum(sq, axis)?;
        Ok(self.sqrt(s))
    }

    /// Log of sum over `axis`: `ln(sum(x))`.
    pub fn logsum(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let s = self.sum(x, axis)?;
        Ok(self.ln(s))
    }

    /// Numerically-stable log-sum-exp over `axis`: `m + ln(sum(exp(x - m)))`.
    pub fn logsumexp(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let full = self.shape(x);
        let m = self.reduce_max(x, axis)?;
        let mb = self.broadcast_back(m, &full, axis)?;
        let shifted = self.sub(x, mb)?;
        let e = self.exp(shifted);
        let s = self.sum(e, axis)?;
        let l = self.ln(s);
        self.add(l, m)
    }

    /// True if any element along `axis` is nonzero (returns BOOL).
    pub fn any(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let z = self.zeros_like(x);
        let nz = self.ne(x, z)?;
        let u = self.cast(nz, DType::F32);
        let m = self.reduce_max(u, axis)?;
        let half = self.scalar(m, 0.5);
        self.gt(m, half)
    }

    /// True if all elements along `axis` are nonzero (returns BOOL).
    pub fn all(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let z = self.zeros_like(x);
        let nz = self.ne(x, z)?;
        let u = self.cast(nz, DType::F32);
        let m = self.reduce_min(u, axis)?;
        let half = self.scalar(m, 0.5);
        self.gt(m, half)
    }

    /// Sum over all axes -> scalar.
    pub fn sum_all(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let r = self.shape(x).len();
        let mut y = x;
        for _ in 0..r {
            y = self.sum(y, 0)?;
        }
        Ok(y)
    }

    /// Mean over all elements -> scalar.
    pub fn mean_all(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let total: usize = self.shape(x).iter().product();
        let s = self.sum_all(x)?;
        let inv = self.scalar(s, 1.0 / total as f32);
        self.mul(s, inv)
    }

    /// Product over all axes -> scalar.
    pub fn prod_all(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let r = self.shape(x).len();
        let mut y = x;
        for _ in 0..r {
            y = self.prod(y, 0)?;
        }
        Ok(y)
    }

    /// General p-norm over `axis`: `(sum |x|^p)^(1/p)`.
    pub fn norm_p(&mut self, x: NodeId, p: f32, axis: usize) -> Result<NodeId, Error> {
        let a = self.abs(x);
        let pc = self.scalar(a, p);
        let ap = self.pow(a, pc)?;
        let s = self.sum(ap, axis)?;
        let inv = self.scalar(s, 1.0 / p);
        self.pow(s, inv)
    }
}
