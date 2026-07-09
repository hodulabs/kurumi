//! Fused nn primitives on the GPU (softmax, ...). Float dtypes (f32/f16/bf16) run device-
//! resident; others fall through to the CPU oracle. Checked against `interp/nn`.

use crate::MetalBackend;
use crate::backend::eval::Val;
use crate::dtype::dev_dtype;
use crate::msl::nn::SDPA_MAX_DH;
use kurumi_core::{DType, Feeds, Graph, Node, NodeId, Op};
use std::collections::HashMap;

impl MetalBackend {
    pub(in crate::backend) fn eval_nn(
        &self,
        g: &Graph,
        node: &Node,
        shape: &[usize],
        dt: DType,
        feeds: &Feeds,
        memo: &mut HashMap<NodeId, Val>,
    ) -> Option<Val> {
        // shared line layout for the axis-wise fused kernels.
        let line = |axis: usize| {
            let in_shape = g.shape(node.src[0]);
            let axis_len = in_shape[axis];
            let inner: usize = in_shape[axis + 1..].iter().product();
            let out_n: usize = shape.iter().product();
            (axis_len, inner, out_n, out_n / axis_len.max(1))
        };
        if let Op::Softmax { axis } = &node.op
            && dev_dtype(dt)
        {
            let (axis_len, inner, out_n, n_lines) = line(*axis);
            let a = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let buf = self.ctx.softmax_dev(&a, axis_len, inner, n_lines, out_n, dt);
            return Some(Val::Dev { buf, shape: shape.to_vec(), dt });
        }
        if let Op::RmsNorm { axis, eps } = &node.op
            && dev_dtype(dt)
        {
            let (axis_len, inner, out_n, n_lines) = line(*axis);
            let a = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let buf = self.ctx.rmsnorm_dev(&a, axis_len, inner, n_lines, out_n, *eps, dt);
            return Some(Val::Dev { buf, shape: shape.to_vec(), dt });
        }
        if let Op::Sdpa { causal } = &node.op
            && dt == DType::F32
        {
            // fused flash-attention forward (online softmax, no SxS materialization). f32 only;
            // dh over the thread-local acc bound -> None, falling to the CPU oracle (eval_fused
            // host path), so correctness is never lost. q,k,v share shape [..batch.., S, dh].
            let qs = g.shape(node.src[0]);
            let r = qs.len();
            let (s, dh) = (qs[r - 2], qs[r - 1]);
            if dh <= SDPA_MAX_DH {
                let batch: usize = qs[..r - 2].iter().product();
                let q = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
                let k = self.to_dev(&self.eval_memo(g, node.src[1], feeds, memo));
                let v = self.to_dev(&self.eval_memo(g, node.src[2], feeds, memo));
                let buf = self.ctx.sdpa_dev(&q, &k, &v, batch, s, dh, *causal);
                return Some(Val::Dev { buf, shape: shape.to_vec(), dt });
            }
        }
        None
    }
}
