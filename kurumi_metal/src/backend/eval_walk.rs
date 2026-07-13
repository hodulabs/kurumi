//! The eval walk: `eval_memo` drives the memoized whole-graph traversal, dispatching
//! each node to a device family (`eval/*`) or the fused core, plus the Val<->device
//! plumbing (to_host/to_dev/materialize/strided_view/as_fused) and the host-oracle seam.

use super::fuse::{FExpr, Leaf, Val, View, fused_msl};
use crate::dtype::{dev_dtype, msl_ty};
use crate::{Buffer, MetalBackend};
use kurumi_core::{DType, Feeds, Graph, NodeId, Op, TensorVal, eval_op};
use std::collections::HashMap;

impl MetalBackend {
    pub(in crate::backend) fn to_host(&self, v: &Val) -> TensorVal {
        match v {
            Val::Host(t) => t.clone(),
            Val::Dev { buf, shape, dt } => {
                self.ctx.flush(); // finish pending GPU work before reading back
                let n = shape.iter().product();
                TensorVal { shape: shape.clone(), storage: self.ctx.download(buf, n, *dt) }
            }
            Val::Fused { shape, leaves, expr, dt } => {
                let buf = self.materialize(shape, leaves, expr, *dt);
                self.ctx.flush();
                let n = shape.iter().product();
                TensorVal { shape: shape.clone(), storage: self.ctx.download(&buf, n, *dt) }
            }
        }
    }
    // get an input as a device buffer; uploads a host value, materializes a fused
    // chain into one kernel. (Only called on f32/f16/bf16 values.)
    pub(in crate::backend) fn to_dev(&self, v: &Val) -> Buffer {
        match v {
            Val::Dev { buf, .. } => buf.clone(),
            Val::Host(t) => self.ctx.upload(&t.storage),
            Val::Fused { shape, leaves, expr, dt } => self.materialize(shape, leaves, expr, *dt),
        }
    }

    // emit ONE kernel for the whole fused pointwise chain (output dtype `dt`); each
    // leaf's view (if any) is baked into the kernel's per-leaf index math.
    pub(in crate::backend) fn materialize(&self, shape: &[usize], leaves: &[Leaf], expr: &FExpr, dt: DType) -> Buffer {
        let n: usize = shape.iter().product();
        let src = fused_msl(expr, leaves, msl_ty(dt));
        let refs: Vec<&Buffer> = leaves.iter().map(|l| &l.buf).collect();
        self.ctx.fused_ew(&src, &refs, n, dt)
    }

    // fold a movement (broadcast/permute/slice) into a strided fused leaf: the input is
    // materialized once, then read through `view` by a pointwise consumer (no strided_dev
    // dispatch, no enlarged intermediate), or by `materialize` for a non-fusable consumer
    // (reduce/matmul/output).
    pub(in crate::backend) fn strided_view(
        &self,
        a: Val,
        base: i64,
        strides: Vec<i64>,
        shape: Vec<usize>,
        dt: DType,
    ) -> Val {
        let buf = self.to_dev(&a);
        let view = View { base, strides, out_shape: shape.clone() };
        Val::Fused { shape, leaves: vec![Leaf { buf, view: Some(view) }], expr: FExpr::Leaf(0), dt }
    }

    // view any device-dtype Val as a fused chain (leaves + expr) for combining.
    pub(in crate::backend) fn as_fused(&self, v: Val) -> (Vec<Leaf>, FExpr) {
        match v {
            Val::Fused { leaves, expr, .. } => (leaves, expr),
            Val::Dev { buf, .. } => (vec![Leaf::plain(buf)], FExpr::Leaf(0)),
            Val::Host(t) => (vec![Leaf::plain(self.ctx.upload(&t.storage))], FExpr::Leaf(0)),
        }
    }

    pub(in crate::backend) fn eval_memo(
        &self,
        g: &Graph,
        id: NodeId,
        feeds: &Feeds,
        memo: &mut HashMap<NodeId, Val>,
    ) -> Val {
        if let Some(v) = memo.get(&id) {
            return v.clone();
        }
        let node = g.node(id);
        let shape = g.shape(id);
        if matches!(node.op, Op::Input { .. }) {
            let v = Val::Host(feeds.get(&id).expect("metal: missing feed for an Input node").clone());
            memo.insert(id, v.clone());
            return v;
        }
        if matches!(node.op, Op::Detach) {
            // identity: pass the input Val straight through (no kernel, no host round-
            // trip). detach only affects autograd, never the forward value.
            let v = self.eval_memo(g, node.src[0], feeds, memo);
            memo.insert(id, v.clone());
            return v;
        }
        let dt = g.dtype(id);
        let dev = dev_dtype(dt);
        if let Op::Const { data, .. } = &node.op
            && dev
        {
            // weight/constant: upload once, keep device-resident across evals (keyed by
            // graph id -> ABA-safe; read-only, so one buffer is shared across consumers/evals).
            let key = (g.id(), id.0);
            let hit = self.const_cache.borrow().get(&key).cloned(); // drop the borrow before borrow_mut
            let buf = hit.unwrap_or_else(|| {
                let b = self.ctx.upload(data);
                self.const_cache.borrow_mut().insert(key, b.clone());
                b
            });
            let v = Val::Dev { buf, shape, dt };
            memo.insert(id, v.clone());
            return v;
        }
        let v = if let Some(v) = self.eval_matmul(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_quant(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_index(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_pointwise(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_complex(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_reduce_arg(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_linalg(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_generate(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_nn(g, node, &shape, dt, feeds, memo) {
            v
        } else {
            self.eval_fused(g, node, &shape, dt, feeds, memo)
        };
        memo.insert(id, v.clone());
        v
    }

    pub(in crate::backend) fn host_op(&self, op: &Op, refs: &[&TensorVal]) -> TensorVal {
        match op {
            // canonical row-major 2D matmul -> GPU (fall back to CPU on a dtype the
            // device can't run, e.g. f64); batched/transposed dots stay on CPU.
            Op::DotGeneral { lhs_contract, rhs_contract, lhs_batch, rhs_batch }
                if lhs_contract.as_slice() == [1]
                    && rhs_contract.as_slice() == [0]
                    && lhs_batch.is_empty()
                    && rhs_batch.is_empty()
                    && refs[0].shape.len() == 2
                    && refs[1].shape.len() == 2 =>
            {
                let (a, b) = (refs[0], refs[1]);
                let (m, k, nn) = (a.shape[0], a.shape[1], b.shape[1]);
                match self.matmul(&a.storage, m, k, &b.storage, nn) {
                    Ok(storage) => TensorVal { shape: vec![m, nn], storage },
                    Err(_) => eval_op(op, refs),
                }
            }
            op => eval_op(op, refs),
        }
    }
}
