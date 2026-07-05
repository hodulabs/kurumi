//! Elementwise arithmetic: the core binary/unary primitives + their decompositions
//! (sub/div/abs/sign/min/clamp/rounding/rem). exp/log family lives in `explog.rs`.

use crate::{DType, Error, Graph, NodeId, Op, Storage, cast};

impl Graph {
    // primitives

    pub fn add(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        self.same_shape("add", a, b)?;
        self.same_dtype("add", a, b)?;
        self.require("add", a, self.dtype(a).is_arith(), "numeric or complex")?;
        Ok(self.push(Op::Add, vec![a, b]))
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        self.same_shape("mul", a, b)?;
        self.same_dtype("mul", a, b)?;
        self.require("mul", a, self.dtype(a).is_arith(), "numeric or complex")?;
        Ok(self.push(Op::Mul, vec![a, b]))
    }

    pub fn max(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        self.same_shape("max", a, b)?;
        self.same_dtype("max", a, b)?;
        self.require("max", a, self.dtype(a).is_numeric(), "numeric")?;
        Ok(self.push(Op::Max, vec![a, b]))
    }

    // unary float ops return NodeId (decompositions chain them unwrapped); the dtype
    // class is a debug-time record contract, enforced for users by the frontend (hodu-rs).
    pub fn neg(&mut self, a: NodeId) -> NodeId {
        assert!(
            self.dtype(a).is_signed() || self.dtype(a).is_complex(),
            "neg requires signed/complex, got {:?}",
            self.dtype(a)
        );
        self.push(Op::Neg, vec![a])
    }

    pub fn recip(&mut self, a: NodeId) -> NodeId {
        assert!(
            self.dtype(a).is_float() || self.dtype(a).is_complex(),
            "recip requires float/complex, got {:?}",
            self.dtype(a)
        );
        self.push(Op::Recip, vec![a])
    }
    pub fn sqrt(&mut self, a: NodeId) -> NodeId {
        assert!(
            self.dtype(a).is_float() || self.dtype(a).is_complex(),
            "sqrt requires float/complex, got {:?}",
            self.dtype(a)
        );
        self.push(Op::Sqrt, vec![a])
    }
    pub fn exp2(&mut self, a: NodeId) -> NodeId {
        assert!(
            self.dtype(a).is_float() || self.dtype(a).is_complex(),
            "exp2 requires float/complex, got {:?}",
            self.dtype(a)
        );
        self.push(Op::Exp2, vec![a])
    }
    pub fn log2(&mut self, a: NodeId) -> NodeId {
        assert!(
            self.dtype(a).is_float() || self.dtype(a).is_complex(),
            "log2 requires float/complex, got {:?}",
            self.dtype(a)
        );
        self.push(Op::Log2, vec![a])
    }

    pub fn floor(&mut self, a: NodeId) -> NodeId {
        assert!(self.dtype(a).is_float(), "floor requires a float dtype, got {:?}", self.dtype(a));
        self.push(Op::Floor, vec![a])
    }

    // decompositions

    /// Broadcast a scalar to `like`'s shape via a 1-element const + expand view
    /// (the const buffer stays 1 element; no full tensor is materialized).
    pub fn scalar(&mut self, like: NodeId, v: f32) -> NodeId {
        let shape = self.shape(like);
        // build the constant DIRECTLY in `like`'s dtype (no Cast node): keeps
        // decompositions strict-dtype-correct on f16/bf16/fp8 and, on Metal, avoids
        // a per-scalar cast dispatch. The value is non-differentiable either way.
        let st = match self.dtype(like) {
            DType::F32 => Storage::F32(vec![v]),
            dt => cast(&Storage::F32(vec![v]), dt),
        };
        let c = self.const_storage(st, vec![1; shape.len()]);
        self.expand(c, shape).expect("scalar broadcast is always valid")
    }

    /// f32 tensor of ones/zeros shaped like `like` (autograd seeds & stops).
    pub fn ones_like(&mut self, like: NodeId) -> NodeId {
        self.scalar(like, 1.0)
    }
    pub fn zeros_like(&mut self, like: NodeId) -> NodeId {
        self.scalar(like, 0.0)
    }

    pub fn sub(&mut self, x: NodeId, y: NodeId) -> Result<NodeId, Error> {
        let ny = self.neg(y);
        self.add(x, ny)
    }

    pub fn div(&mut self, x: NodeId, y: NodeId) -> Result<NodeId, Error> {
        let ry = self.recip(y);
        self.mul(x, ry)
    }

    /// Absolute value: `max(x, -x)`.
    pub fn abs(&mut self, x: NodeId) -> NodeId {
        let nx = self.neg(x);
        self.max(x, nx).expect("abs: same-shape max")
    }

    /// Element square: `x * x`.
    pub fn square(&mut self, x: NodeId) -> NodeId {
        self.mul(x, x).expect("square: same-shape mul")
    }

    /// Sign: `1` where `x>0`, `-1` where `x<0`, else `0`.
    pub fn sign(&mut self, x: NodeId) -> NodeId {
        let z = self.zeros_like(x);
        let pos = self.cmp_lt(z, x).expect("same shape"); // 0 < x
        let neg = self.cmp_lt(x, z).expect("same shape"); // x < 0
        let one = self.ones_like(x);
        let neg1 = self.scalar(x, -1.0);
        let p = self.select(pos, one, z).expect("same shape");
        let n = self.select(neg, neg1, z).expect("same shape");
        self.add(p, n).expect("same shape")
    }

    /// Elementwise minimum: `where(a < b, a, b)`. (No `Min` primitive: unlike
    /// `Max`, nothing reduces with it, so the where/cmp decomposition suffices.)
    pub fn min(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        let lt = self.cmp_lt(a, b)?;
        self.select(lt, a, b)
    }

    /// Clamp to `[lo, hi]`: `min(max(x, lo), hi)`.
    pub fn clamp(&mut self, x: NodeId, lo: f32, hi: f32) -> Result<NodeId, Error> {
        let lo_c = self.scalar(x, lo);
        let hi_c = self.scalar(x, hi);
        let up = self.max(x, lo_c)?;
        self.min(up, hi_c)
    }

    /// `max(x, lo)`.
    pub fn clamp_min(&mut self, x: NodeId, lo: f32) -> Result<NodeId, Error> {
        let c = self.scalar(x, lo);
        self.max(x, c)
    }
    /// `min(x, hi)`.
    pub fn clamp_max(&mut self, x: NodeId, hi: f32) -> Result<NodeId, Error> {
        let c = self.scalar(x, hi);
        self.min(x, c)
    }

    /// `ceil(x) = -floor(-x)`.
    pub fn ceil(&mut self, x: NodeId) -> NodeId {
        let nx = self.neg(x);
        let f = self.floor(nx);
        self.neg(f)
    }

    /// `round(x) = floor(x + 0.5)`.
    pub fn round(&mut self, x: NodeId) -> NodeId {
        let half = self.scalar(x, 0.5);
        let s = self.add(x, half).expect("same shape");
        self.floor(s)
    }

    /// Float remainder: `a - floor(a/b)*b`.
    pub fn rem(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        let q = self.div(a, b)?;
        let f = self.floor(q);
        let fb = self.mul(f, b)?;
        self.sub(a, fb)
    }
}
