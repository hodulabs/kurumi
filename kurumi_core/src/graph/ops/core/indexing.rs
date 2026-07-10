//! Gather/scatter, argsort, sort/topk, one-hot. N-d advanced indexing (gather_nd/scatter_nd)
//! is in the `nd` submodule.

mod nd;

use crate::{DType, Error, Graph, NodeId, Op, ScatterOp};

impl Graph {
    // primitives

    /// `index` along `axis` (StableHLO iota): arange/positions/one-hot/tril.
    pub fn iota(&mut self, shape: Vec<usize>, axis: usize, dtype: DType) -> Result<NodeId, Error> {
        if axis >= shape.len() {
            return Err(Error::shape("iota", format!("axis {axis} out of range for rank {}", shape.len())));
        }
        Ok(self.push(Op::Iota { shape, axis, dtype }, vec![]))
    }

    /// Gather slices of `operand` along `axis` at `indices` (jnp.take). Output's
    /// `axis` dim is replaced by all of `indices`' dims. OOB indices are clamped.
    pub fn gather(&mut self, operand: NodeId, indices: NodeId, axis: usize) -> Result<NodeId, Error> {
        let rank = self.shape(operand).len();
        if axis >= rank {
            return Err(Error::shape("gather", format!("axis {axis} out of range for rank {rank}")));
        }
        if !self.dtype(indices).is_int() {
            return Err(Error::shape("gather", format!("indices must be integer, got {:?}", self.dtype(indices))));
        }
        Ok(self.push(Op::Gather { axis }, vec![operand, indices]))
    }

    /// Scatter `updates` into a copy of `operand` along `axis` at `indices`
    /// (inverse of gather). OOB indices are dropped. `updates` must have the
    /// gather-output shape; Add/Max combiners require a numeric operand.
    pub fn scatter(
        &mut self,
        operand: NodeId,
        indices: NodeId,
        updates: NodeId,
        axis: usize,
        combine: ScatterOp,
    ) -> Result<NodeId, Error> {
        let op_shape = self.shape(operand);
        if axis >= op_shape.len() {
            return Err(Error::shape("scatter", format!("axis {axis} out of range for rank {}", op_shape.len())));
        }
        if !self.dtype(indices).is_int() {
            return Err(Error::shape("scatter", format!("indices must be integer, got {:?}", self.dtype(indices))));
        }
        self.same_dtype("scatter", operand, updates)?;
        // updates shape must equal operand[..axis] ++ indices ++ operand[axis+1..]
        let idx = self.shape(indices);
        let want: Vec<usize> = op_shape[..axis].iter().chain(&idx).chain(&op_shape[axis + 1..]).copied().collect();
        let got = self.shape(updates);
        if got != want {
            return Err(Error::shape("scatter", format!("updates shape {got:?} != {want:?}")));
        }
        if combine != ScatterOp::Set && !self.dtype(operand).is_int() && !self.dtype(operand).is_float() {
            return Err(Error::shape("scatter", "Add/Max combiner needs a numeric operand"));
        }
        Ok(self.push(Op::Scatter { axis, combine }, vec![operand, indices, updates]))
    }

    /// `take_along_dim`: per-position gather along `axis`. `indices` matches the
    /// output shape (= `operand` shape except `axis`); `out[..,i,..] =
    /// operand[..,indices[..,i,..],..]`. (Pairs with `argmax`/`argsort`.)
    pub fn gather_along(&mut self, operand: NodeId, indices: NodeId, axis: usize) -> Result<NodeId, Error> {
        let op_shape = self.shape(operand);
        if axis >= op_shape.len() {
            return Err(Error::shape("gather_along", format!("axis {axis} out of range for rank {}", op_shape.len())));
        }
        if !self.dtype(indices).is_int() {
            return Err(Error::shape(
                "gather_along",
                format!("indices must be integer, got {:?}", self.dtype(indices)),
            ));
        }
        let idx = self.shape(indices);
        if idx.len() != op_shape.len() || (0..op_shape.len()).any(|d| d != axis && idx[d] != op_shape[d]) {
            return Err(Error::shape(
                "gather_along",
                format!("indices {idx:?} must match operand {op_shape:?} except axis {axis}"),
            ));
        }
        Ok(self.push(Op::GatherAlong { axis }, vec![operand, indices]))
    }

    /// Per-position scatter along `axis` (inverse of `gather_along`; `index_add`-style
    /// with combiners). `indices`/`updates` match each other; OOB dropped.
    pub fn scatter_along(
        &mut self,
        operand: NodeId,
        indices: NodeId,
        updates: NodeId,
        axis: usize,
        combine: ScatterOp,
    ) -> Result<NodeId, Error> {
        let op_shape = self.shape(operand);
        if axis >= op_shape.len() {
            return Err(Error::shape("scatter_along", format!("axis {axis} out of range for rank {}", op_shape.len())));
        }
        if !self.dtype(indices).is_int() {
            return Err(Error::shape(
                "scatter_along",
                format!("indices must be integer, got {:?}", self.dtype(indices)),
            ));
        }
        self.same_dtype("scatter_along", operand, updates)?;
        if self.shape(indices) != self.shape(updates) {
            return Err(Error::shape("scatter_along", "indices and updates must share a shape"));
        }
        Ok(self.push(Op::ScatterAlong { axis, combine }, vec![operand, indices, updates]))
    }

    /// Indices (I64) that sort `x` along `axis` (a per-line permutation, same shape).
    /// Non-differentiable. `sort`/`topk` build on this + `gather_along`.
    pub fn argsort(&mut self, x: NodeId, axis: usize, descending: bool) -> Result<NodeId, Error> {
        self.reduce_check("argsort", x, axis)?;
        Ok(self.push(Op::Argsort { axis, descending }, vec![x]))
    }

    // decompositions

    /// `take_along_dim`: alias for [`Graph::gather_along`].
    pub fn take_along_dim(&mut self, x: NodeId, indices: NodeId, axis: usize) -> Result<NodeId, Error> {
        self.gather_along(x, indices, axis)
    }

    /// Sorted values along `axis`: `gather_along(x, argsort(x))`.
    pub fn sort(&mut self, x: NodeId, axis: usize, descending: bool) -> Result<NodeId, Error> {
        let perm = self.argsort(x, axis, descending)?;
        self.gather_along(x, perm, axis)
    }

    /// Top-`k` along `axis` -> (values, I64 indices). `largest=true` picks the
    /// largest. Built from `argsort` + slice + `gather_along`.
    pub fn topk(&mut self, x: NodeId, k: usize, axis: usize, largest: bool) -> Result<(NodeId, NodeId), Error> {
        let perm = self.argsort(x, axis, largest)?;
        let sh = self.shape(x);
        let mut ranges: Vec<(usize, usize)> = sh.iter().map(|&d| (0, d)).collect();
        ranges[axis] = (0, k);
        let top_idx = self.slice(perm, ranges)?;
        let top_val = self.gather_along(x, top_idx, axis)?;
        Ok((top_val, top_idx))
    }

    /// One-hot encode an integer index tensor: output is `idx.shape ++ [num_classes]`,
    /// `out[.., c] = (idx == c)` as f32.
    pub fn onehot(&mut self, idx: NodeId, num_classes: usize) -> Result<NodeId, Error> {
        let ish = self.shape(idx);
        let r = ish.len();
        let mut out_shape = ish.clone();
        out_shape.push(num_classes);
        let classes = self.iota(out_shape.clone(), r, DType::I64)?; // 0..num_classes along last
        let mut idx_shape = ish;
        idx_shape.push(1);
        let idx_r = self.reshape(idx, idx_shape)?;
        let idx_i = if self.dtype(idx) == DType::I64 { idx_r } else { self.cast(idx_r, DType::I64) };
        let idx_b = self.expand(idx_i, out_shape)?;
        let eq = self.cmp_eq(classes, idx_b)?;
        Ok(self.cast(eq, DType::F32))
    }
}
