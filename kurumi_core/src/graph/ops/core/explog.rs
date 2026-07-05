//! Exponential / logarithm family: exp/ln/exp10/log10 (base conversions of the
//! exp2/log2 primitives), powers, and the numerically-stable log1p/expm1/logaddexp/
//! xlogy/logit/log_sigmoid. Pure decompositions.

use crate::{Error, Graph, NodeId};

impl Graph {
    /// exp(x) = exp2(x * log2(e))
    pub fn exp(&mut self, x: NodeId) -> NodeId {
        let s = self.scalar(x, std::f32::consts::LOG2_E);
        let xs = self.mul(x, s).expect("scalar shares x's shape");
        self.exp2(xs)
    }

    /// Natural log via the `log2` primitive: `ln(x) = log2(x) * ln(2)`.
    pub fn ln(&mut self, x: NodeId) -> NodeId {
        let l = self.log2(x);
        let c = self.scalar(l, std::f32::consts::LN_2);
        self.mul(l, c).expect("ln: same-shape mul is always valid")
    }

    /// `10^x = exp2(x * log_2 10)`.
    pub fn exp10(&mut self, x: NodeId) -> NodeId {
        let c = self.scalar(x, 10f32.log2());
        let xs = self.mul(x, c).expect("same shape");
        self.exp2(xs)
    }

    /// `log_10(x) = log_2(x) / log_2 10`.
    pub fn log10(&mut self, x: NodeId) -> NodeId {
        let l = self.log2(x);
        let c = self.scalar(l, 1.0 / 10f32.log2());
        self.mul(l, c).expect("same shape")
    }

    /// Power `a^b` for positive base: `exp2(b * log2(a))`.
    pub fn pow(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        let la = self.log2(a);
        let bl = self.mul(b, la)?;
        Ok(self.exp2(bl))
    }

    /// `log(1 + x)`, accurate near 0 (Kahan): `x*ln(u)/(u-1)` with `u = 1+x`, or `x`
    /// where `u` rounds to 1: avoids the cancellation of a naive `ln(1+x)`.
    pub fn log1p(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let one = self.ones_like(x);
        let u = self.add(x, one)?;
        let d = self.sub(u, one)?; // (1+x) - 1, exactly representable
        let eq = self.cmp_eq(u, one)?; // u rounded to 1  <=>  x ~= 0
        let safe = self.select(eq, one, d)?; // guard the divide
        let lu = self.ln(u);
        let xl = self.mul(x, lu)?;
        let ratio = self.div(xl, safe)?;
        self.select(eq, x, ratio)
    }

    /// `exp(x) - 1`, accurate near 0 (Kahan): `(u-1)*x/ln(u)` with `u = exp(x)`, or `x`
    /// where `u` rounds to 1.
    pub fn expm1(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let one = self.ones_like(x);
        let u = self.exp(x);
        let um1 = self.sub(u, one)?;
        let eq = self.cmp_eq(u, one)?;
        let lu = self.ln(u);
        let safe = self.select(eq, one, lu)?;
        let ux = self.mul(um1, x)?;
        let ratio = self.div(ux, safe)?;
        self.select(eq, x, ratio)
    }

    /// `log(exp(a) + exp(b))`, overflow-safe: `max(a,b) + log1p(exp(-|a-b|))`.
    pub fn logaddexp(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        let m = self.max(a, b)?;
        let d = self.sub(a, b)?;
        let ad = self.abs(d);
        let nad = self.neg(ad); // -|a-b| <= 0
        let e = self.exp(nad);
        let l = self.log1p(e)?;
        self.add(m, l)
    }

    /// `x * log(y)`, defined as 0 where `x == 0` (even if `y` is 0/inf): the
    /// cross-entropy / KL convention.
    pub fn xlogy(&mut self, x: NodeId, y: NodeId) -> Result<NodeId, Error> {
        let z = self.zeros_like(x);
        let eq = self.cmp_eq(x, z)?;
        let ly = self.ln(y);
        let xly = self.mul(x, ly)?;
        self.select(eq, z, xly)
    }

    /// Logit `log(x / (1 - x)) = log(x) - log(1 - x)` (inverse of sigmoid).
    pub fn logit(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let one = self.ones_like(x);
        let omx = self.sub(one, x)?;
        let lx = self.ln(x);
        let lo = self.ln(omx);
        self.sub(lx, lo)
    }

    /// `log(sigmoid(x)) = -softplus(-x)` (overflow-safe).
    pub fn log_sigmoid(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let nx = self.neg(x);
        let sp = self.softplus(nx);
        Ok(self.neg(sp))
    }
}
