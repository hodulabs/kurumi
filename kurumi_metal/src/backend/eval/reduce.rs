//! Device index-producing reductions: argmax/argmin (`ArgReduce`) and argsort, both
//! gated on the *operand* dtype (their output is an I64 index buffer, not a float).

// NOTE: value reductions (sum/max/prod, `Ew::Reduce`) live in `eval/fused.rs`; this file is
// the arg/sort index reductions only.

use crate::MetalBackend;
use crate::backend::eval::Val;
use crate::dtype::dev_dtype;
use kurumi_core::{ArgKind, DType, Feeds, Graph, Node, NodeId, Op};
use std::collections::HashMap;

impl MetalBackend {
    pub(in crate::backend) fn eval_reduce_arg(
        &self,
        g: &Graph,
        node: &Node,
        shape: &[usize],
        dt: DType,
        feeds: &Feeds,
        memo: &mut HashMap<NodeId, Val>,
    ) -> Option<Val> {
        if let Op::ArgReduce { axis, kind } = &node.op
            && dev_dtype(g.dtype(node.src[0]))
        {
            // device argmax/argmin -> I64 index buffer (gate on the operand dtype;
            // output is I64, not a device float dtype).
            let in_shape = g.shape(node.src[0]);
            let axis_len = in_shape[*axis];
            let inner: usize = in_shape[axis + 1..].iter().product();
            let out_n: usize = shape.iter().product();
            let a = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let is_max = matches!(kind, ArgKind::Max);
            let buf = self.ctx.argreduce_dev(&a, axis_len, inner, out_n, g.dtype(node.src[0]), is_max);
            return Some(Val::Dev { buf, shape: shape.to_vec(), dt });
        }
        if let Op::Argsort { axis, descending } = &node.op
            && dev_dtype(g.dtype(node.src[0]))
        {
            // device argsort -> I64 permutation (gate on operand dtype; output is I64).
            let in_shape = g.shape(node.src[0]);
            let axis_len = in_shape[*axis];
            let inner: usize = in_shape[axis + 1..].iter().product();
            let out_n: usize = shape.iter().product();
            let n_lines = out_n / axis_len.max(1);
            let a = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let buf = self.ctx.argsort_dev(&a, axis_len, inner, n_lines, out_n, g.dtype(node.src[0]), *descending);
            return Some(Val::Dev { buf, shape: shape.to_vec(), dt });
        }
        None
    }
}
