//! Trigonometric, hyperbolic, and inverse functions (built on the `sin` primitive).

use crate::{Error, Graph, NodeId, Op};

impl Graph {
    // primitive

    pub fn sin(&mut self, a: NodeId) -> NodeId {
        assert!(self.dtype(a).is_float(), "sin requires a float dtype, got {:?}", self.dtype(a));
        self.push(Op::Sin, vec![a])
    }

    // decompositions

    /// `cos(x) = sin(x + pi/2)`.
    pub fn cos(&mut self, x: NodeId) -> NodeId {
        let hp = self.scalar(x, std::f32::consts::FRAC_PI_2);
        let arg = self.add(x, hp).expect("same shape");
        self.sin(arg)
    }

    /// `tan(x) = sin(x) / cos(x)`.
    pub fn tan(&mut self, x: NodeId) -> NodeId {
        let s = self.sin(x);
        let c = self.cos(x);
        self.div(s, c).expect("same shape")
    }

    /// `sinh(x) = (e^x - e^-x) / 2`.
    pub fn sinh(&mut self, x: NodeId) -> NodeId {
        let e = self.exp(x);
        let nx = self.neg(x);
        let en = self.exp(nx);
        let d = self.sub(e, en).expect("same shape");
        let half = self.scalar(x, 0.5);
        self.mul(d, half).expect("same shape")
    }

    /// `cosh(x) = (e^x + e^-x) / 2`.
    pub fn cosh(&mut self, x: NodeId) -> NodeId {
        let e = self.exp(x);
        let nx = self.neg(x);
        let en = self.exp(nx);
        let d = self.add(e, en).expect("same shape");
        let half = self.scalar(x, 0.5);
        self.mul(d, half).expect("same shape")
    }

    /// tanh(x) = 2 * sigmoid(2x) - 1
    pub fn tanh(&mut self, x: NodeId) -> NodeId {
        let two = self.scalar(x, 2.0);
        let x2 = self.mul(x, two).expect("scalar shares x's shape");
        let s = self.sigmoid(x2);
        let two2 = self.scalar(x, 2.0);
        let s2 = self.mul(s, two2).expect("scalar shares x's shape");
        let one = self.scalar(x, 1.0);
        self.sub(s2, one).expect("scalar shares x's shape")
    }

    /// `asinh(x) = ln(x + sqrt(x^2+1))`.
    pub fn asinh(&mut self, x: NodeId) -> NodeId {
        let x2 = self.square(x);
        let one = self.scalar(x, 1.0);
        let s = self.add(x2, one).expect("same shape");
        let r = self.sqrt(s);
        let arg = self.add(x, r).expect("same shape");
        self.ln(arg)
    }

    /// `acosh(x) = ln(x + sqrt(x^2-1))` (x >= 1).
    pub fn acosh(&mut self, x: NodeId) -> NodeId {
        let x2 = self.square(x);
        let one = self.scalar(x, 1.0);
        let s = self.sub(x2, one).expect("same shape");
        let r = self.sqrt(s);
        let arg = self.add(x, r).expect("same shape");
        self.ln(arg)
    }

    /// `atanh(x) = 1/2 ln((1+x)/(1-x))` (|x| < 1).
    pub fn atanh(&mut self, x: NodeId) -> NodeId {
        let one = self.scalar(x, 1.0);
        let num = self.add(one, x).expect("same shape");
        let den = self.sub(one, x).expect("same shape");
        let q = self.div(num, den).expect("same shape");
        let l = self.ln(q);
        let half = self.scalar(x, 0.5);
        self.mul(l, half).expect("same shape")
    }

    /// `atan(x)`: minimax polynomial on |x|<=1 with range reduction
    /// `atan(x) = sign(x)*pi/2 - atan(1/x)` for |x|>1 (max abs error ~1e-5).
    pub fn atan(&mut self, x: NodeId) -> NodeId {
        let a = self.abs(x);
        let one = self.scalar(a, 1.0);
        let flip = self.cmp_lt(one, a).expect("same shape"); // |x| > 1
        let inv = self.recip(a);
        let z = self.select(flip, inv, a).expect("same shape"); // z in [0,1]
        // p(z) = z*(c0 + z^2*(c1 + z^2*(c2 + z^2*(c3 + z^2*c4))))
        let z2 = self.square(z);
        let mut poly = self.scalar(z2, 0.020_835_1);
        for c in [-0.085_133, 0.180_141, -0.330_299_5, 0.999_866] {
            poly = self.mul(poly, z2).expect("same shape");
            let cc = self.scalar(z2, c);
            poly = self.add(poly, cc).expect("same shape");
        }
        let pz = self.mul(poly, z).expect("same shape"); // atan(z), z in [0,1]
        let hp = self.scalar(pz, std::f32::consts::FRAC_PI_2);
        let reduced = self.sub(hp, pz).expect("same shape"); // pi/2 - atan(1/|x|)
        let r = self.select(flip, reduced, pz).expect("same shape");
        let s = self.sign(x);
        self.mul(s, r).expect("same shape")
    }

    /// `asin(x) = atan(x/sqrt(1-x^2))` (|x| < 1).
    pub fn asin(&mut self, x: NodeId) -> NodeId {
        let x2 = self.square(x);
        let one = self.scalar(x2, 1.0);
        let d = self.sub(one, x2).expect("same shape");
        let r = self.sqrt(d);
        let q = self.div(x, r).expect("same shape");
        self.atan(q)
    }

    /// `acos(x) = pi/2 - asin(x)`.
    pub fn acos(&mut self, x: NodeId) -> NodeId {
        let a = self.asin(x);
        let hp = self.scalar(a, std::f32::consts::FRAC_PI_2);
        self.sub(hp, a).expect("same shape")
    }

    /// `atan2(y, x)`: quadrant-correct angle: `atan(y/x) + (x<0 ? +/-pi : 0)`.
    /// (x==0 gives +/-pi/2 via atan(+/-inf).)
    pub fn atan2(&mut self, y: NodeId, x: NodeId) -> Result<NodeId, Error> {
        let ratio = self.div(y, x)?;
        let base = self.atan(ratio);
        let zero = self.zeros_like(x);
        let xlt0 = self.cmp_lt(x, zero)?;
        let ylt0 = self.cmp_lt(y, zero)?;
        let pi = self.scalar(x, std::f32::consts::PI);
        let npi = self.neg(pi);
        let zero2 = self.zeros_like(x);
        let inner = self.select(ylt0, npi, pi)?; // y<0 ? -pi : pi
        let corr = self.select(xlt0, inner, zero2)?; // x<0 ? inner : 0
        self.add(base, corr)
    }

    /// Normalized sinc `sin(pi*x)/(pi*x)`, with `sinc(0) = 1` (numpy convention).
    pub fn sinc(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let z = self.zeros_like(x);
        let eq = self.cmp_eq(x, z)?;
        let px = self.scalar(x, std::f32::consts::PI);
        let pxx = self.mul(px, x)?;
        let s = self.sin(pxx);
        let r = self.div(s, pxx)?;
        let one = self.ones_like(x);
        self.select(eq, one, r)
    }
}
