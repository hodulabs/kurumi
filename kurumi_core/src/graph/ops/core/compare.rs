//! Comparisons, boolean predicates, and select/masking (BOOL-producing ops).

use crate::{DType, Error, Graph, NodeId, Op, Storage};

impl Graph {
    // primitives

    pub fn cmp_lt(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        self.require("cmp_lt", a, !self.dtype(a).is_complex(), "orderable (non-complex)")?;
        self.bin("cmp_lt", Op::CmpLt, a, b) // -> BOOL
    }
    pub fn cmp_eq(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        self.require("cmp_eq", a, !self.dtype(a).is_complex(), "orderable (non-complex)")?;
        self.bin("cmp_eq", Op::CmpEq, a, b) // -> BOOL
    }

    /// `cond ? a : b` elementwise; `cond` must be BOOL, `a`/`b` same shape+dtype.
    pub fn select(&mut self, cond: NodeId, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        self.same_shape("select", cond, a)?;
        self.same_shape("select", a, b)?;
        self.same_dtype("select", a, b)?;
        if self.dtype(cond) != DType::BOOL {
            return Err(Error::shape("select", format!("cond must be BOOL, got {:?}", self.dtype(cond))));
        }
        Ok(self.push(Op::Where, vec![cond, a, b]))
    }

    // decompositions

    /// A BOOL scalar broadcast to `like`'s shape.
    fn bool_scalar(&mut self, like: NodeId, v: bool) -> NodeId {
        let shape = self.shape(like);
        let c = self.const_storage(Storage::BOOL(vec![v]), vec![1; shape.len()]);
        self.expand(c, shape).expect("bool scalar broadcast")
    }

    /// Logical NOT of a BOOL tensor: `xor(b, true)`.
    pub fn logical_not(&mut self, b: NodeId) -> NodeId {
        let t = self.bool_scalar(b, true);
        self.xor(b, t).expect("same shape")
    }

    /// `a > b` = `b < a`.
    pub fn gt(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        self.cmp_lt(b, a)
    }
    /// `a >= b` = `!(a < b)`.
    pub fn ge(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        let lt = self.cmp_lt(a, b)?;
        Ok(self.logical_not(lt))
    }
    /// `a <= b` = `!(b < a)`.
    pub fn le(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        let gt = self.cmp_lt(b, a)?;
        Ok(self.logical_not(gt))
    }
    /// `a != b` = `!(a == b)`.
    pub fn ne(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        let eq = self.cmp_eq(a, b)?;
        Ok(self.logical_not(eq))
    }

    /// `isnan(x)` = `x != x` (NaN is the only value not equal to itself).
    pub fn isnan(&mut self, x: NodeId) -> Result<NodeId, Error> {
        self.ne(x, x)
    }
    /// `isinf(x)` = `|x| == +inf`.
    pub fn isinf(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let a = self.abs(x);
        let inf = self.scalar(x, f32::INFINITY);
        self.cmp_eq(a, inf)
    }
    /// `isfinite(x)` = `!(isnan or isinf)`.
    pub fn isfinite(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let n = self.isnan(x)?;
        let i = self.isinf(x)?;
        let bad = self.or(n, i)?;
        Ok(self.logical_not(bad))
    }

    /// `where(mask, value, x)`: fill positions where `mask` is true with `value`.
    pub fn masked_fill(&mut self, x: NodeId, mask: NodeId, value: f32) -> Result<NodeId, Error> {
        let v = self.scalar(x, value);
        self.select(mask, v, x)
    }
}
