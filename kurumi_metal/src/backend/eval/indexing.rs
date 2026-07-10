//! Device gather/scatter: embeddings (`Gather`), `take_along_dim` (`GatherAlong`),
//! and their scatter inverses (`Scatter`/`ScatterAlong`, the gather-VJP hot path).

use crate::MetalBackend;
use crate::backend::eval::Val;
use crate::dtype::dev_dtype;
use kurumi_core::{DType, Feeds, Graph, Node, NodeId, Op, ScatterOp, Storage};
use std::collections::HashMap;

impl MetalBackend {
    pub(in crate::backend) fn eval_index(
        &self,
        g: &Graph,
        node: &Node,
        shape: &[usize],
        dt: DType,
        feeds: &Feeds,
        memo: &mut HashMap<NodeId, Val>,
    ) -> Option<Val> {
        let dev = dev_dtype(dt);
        if let (true, Op::Gather { axis }) = (dev, &node.op) {
            // device gather (embeddings / jnp.take): operand on the GPU, indices
            // uploaded as i32. operand [pre, da, post]; output [pre, idx.., post].
            let axis = *axis;
            let op_shape = g.shape(node.src[0]);
            let post: usize = op_shape[axis + 1..].iter().product();
            let da = op_shape[axis];
            let k: usize = g.shape(node.src[1]).iter().product::<usize>().max(1);
            let n: usize = shape.iter().product();
            let opbuf = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let idx = self.to_host(&self.eval_memo(g, node.src[1], feeds, memo));
            let buf = self.ctx.gather_dev(&opbuf, &storage_i32(&idx.storage), k, post, da, n, dt);
            return Some(Val::Dev { buf, shape: shape.to_vec(), dt });
        }
        if let (true, Op::GatherAlong { axis }) = (dev, &node.op) {
            // device take_along_dim: per-position index into the operand's `axis`.
            let axis = *axis;
            let op_shape = g.shape(node.src[0]);
            let op_axis = op_shape[axis];
            let out_axis = shape[axis];
            let inner: usize = op_shape[axis + 1..].iter().product();
            let n: usize = shape.iter().product();
            let opbuf = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let idx = self.to_host(&self.eval_memo(g, node.src[1], feeds, memo));
            let buf = self.ctx.gather_along_dev(&opbuf, &storage_i32(&idx.storage), op_axis, out_axis, inner, n, dt);
            return Some(Val::Dev { buf, shape: shape.to_vec(), dt });
        }
        if let Op::ScatterAlong { axis, combine } = &node.op
            && scatter_dev_ok(*combine, dt)
        {
            // device scatter_along (index_add): Set any dtype; Add/Max/Min f32 (CAS).
            let axis = *axis;
            let op_shape = g.shape(node.src[0]);
            let upd_shape = g.shape(node.src[2]);
            let op_axis = op_shape[axis];
            let upd_axis = upd_shape[axis];
            let inner: usize = op_shape[axis + 1..].iter().product();
            let op_n: usize = op_shape.iter().product();
            let n_upd: usize = upd_shape.iter().product();
            let operand = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let idx = self.to_host(&self.eval_memo(g, node.src[1], feeds, memo));
            let updates = self.to_dev(&self.eval_memo(g, node.src[2], feeds, memo));
            let cstr = combine_str(*combine);
            let buf = self.ctx.scatter_along_dev(
                &operand,
                &storage_i32(&idx.storage),
                &updates,
                op_axis,
                upd_axis,
                inner,
                op_n,
                n_upd,
                cstr,
                dt,
            );
            return Some(Val::Dev { buf, shape: shape.to_vec(), dt });
        }
        if let Op::Scatter { axis, combine } = &node.op
            && scatter_dev_ok(*combine, dt)
        {
            // device general scatter (jnp.take inverse), f32: the gather-VJP /
            // embedding-backward hot path. idx has one index per axis slot.
            let axis = *axis;
            let op_shape = g.shape(node.src[0]);
            let idx_shape = g.shape(node.src[1]);
            let da = op_shape[axis];
            let post: usize = op_shape[axis + 1..].iter().product();
            let k: usize = idx_shape.iter().product::<usize>().max(1);
            let op_n: usize = op_shape.iter().product();
            let n_upd: usize = g.shape(node.src[2]).iter().product();
            let operand = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let idx = self.to_host(&self.eval_memo(g, node.src[1], feeds, memo));
            let updates = self.to_dev(&self.eval_memo(g, node.src[2], feeds, memo));
            let cstr = combine_str(*combine);
            let buf = self.ctx.scatter_dev(
                &operand,
                &storage_i32(&idx.storage),
                &updates,
                da,
                k,
                post,
                op_n,
                n_upd,
                cstr,
                dt,
            );
            return Some(Val::Dev { buf, shape: shape.to_vec(), dt });
        }
        None
    }
}

// which (combine, dtype) scatters run device-resident: Set for any device dtype
// (direct write); Add/Max/Min for f32 (float-CAS) and i32/u32 (native int atomics).
// f16/bf16/64-bit/small-int combine has no matching Metal atomic -> CPU oracle.
fn scatter_dev_ok(c: ScatterOp, dt: DType) -> bool {
    match c {
        ScatterOp::Set => dev_dtype(dt),
        _ => matches!(dt, DType::F32 | DType::I32 | DType::U32),
    }
}

// scatter combine tag -> the kernel-body selector string.
fn combine_str(c: ScatterOp) -> &'static str {
    match c {
        ScatterOp::Set => "set",
        ScatterOp::Add => "add",
        ScatterOp::Max => "max",
        ScatterOp::Min => "min",
    }
}

// gather indices (i32/i64) -> i32 for the GPU index buffer.
fn storage_i32(s: &Storage) -> Vec<i32> {
    match s {
        Storage::U8(v) => v.iter().map(|&x| x as i32).collect(),
        Storage::U16(v) => v.iter().map(|&x| x as i32).collect(),
        Storage::U32(v) => v.iter().map(|&x| x as i32).collect(),
        Storage::U64(v) => v.iter().map(|&x| x as i32).collect(),
        Storage::I8(v) => v.iter().map(|&x| x as i32).collect(),
        Storage::I16(v) => v.iter().map(|&x| x as i32).collect(),
        Storage::I32(v) => v.clone(),
        Storage::I64(v) => v.iter().map(|&x| x as i32).collect(),
        _ => panic!("gather indices must be integer, got {:?}", s.dtype()),
    }
}
