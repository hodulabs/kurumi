//! Device scalar-shape ops that aren't part of the fused arithmetic chain: dtype
//! cast, `where` select, comparisons (-> bool), and same-width bitcast (relabel).

use crate::MetalBackend;
use crate::backend::eval::Val;
use crate::dtype::dev_dtype;
use kurumi_core::{DType, Feeds, Graph, Node, NodeId, Op};
use std::collections::HashMap;

impl MetalBackend {
    pub(in crate::backend) fn eval_pointwise(
        &self,
        g: &Graph,
        node: &Node,
        shape: &[usize],
        dt: DType,
        feeds: &Feeds,
        memo: &mut HashMap<NodeId, Val>,
    ) -> Option<Val> {
        let dev = dev_dtype(dt);
        if let (true, Op::Cast { to }) = (dev, &node.op)
            && dev_dtype(g.dtype(node.src[0]))
            && !dt.is_complex()
            && !g.dtype(node.src[0]).is_complex()
        {
            // device dtype cast (f16<->bf16<->f32), e.g. mixed-precision. A non-device
            // source dtype (int/bool/f64) or complex falls through to the host cast.
            let a = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let n: usize = shape.iter().product();
            let buf = self.ctx.cast_dev(&a, n, g.dtype(node.src[0]), *to);
            return Some(Val::Dev { buf, shape: shape.to_vec(), dt });
        }
        if let Op::Cast { to } = &node.op {
            // complex seam cast: f32 -> C64 (imag 0) / C64 -> f32 (real part). Other
            // complex casts (C128, or a non-f32 real side) involve f64 -> host.
            let src_dt = g.dtype(node.src[0]);
            let n: usize = shape.iter().product();
            if *to == DType::C64 && src_dt == DType::F32 {
                let a = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
                return Some(Val::Dev { buf: self.ctx.r2c_dev(&a, n), shape: shape.to_vec(), dt });
            }
            if *to == DType::F32 && src_dt == DType::C64 {
                let a = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
                return Some(Val::Dev { buf: self.ctx.real_dev(&a, n), shape: shape.to_vec(), dt });
            }
        }
        if dev && matches!(node.op, Op::Where) {
            // device select: cond (bool buffer) ? a : b: keeps where/activation
            // chains (sign/elu/leaky_relu/masked_fill/min/clamp) device-resident.
            // C64 (float2) selects too; C128 (dev=false) falls to host.
            let cond = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let a = self.to_dev(&self.eval_memo(g, node.src[1], feeds, memo));
            let b = self.to_dev(&self.eval_memo(g, node.src[2], feeds, memo));
            let n: usize = shape.iter().product();
            return Some(Val::Dev { buf: self.ctx.where_dev(&cond, &a, &b, n, dt), shape: shape.to_vec(), dt });
        }
        if matches!(node.op, Op::CmpLt | Op::CmpEq) && dev_dtype(g.dtype(node.src[0])) {
            // device comparison -> BOOL buffer (output dt is BOOL, so gate on the
            // operand dtype instead of `dev`). Feeds device where without a roundtrip.
            let op = if matches!(node.op, Op::CmpLt) { "<" } else { "==" };
            let a = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let b = self.to_dev(&self.eval_memo(g, node.src[1], feeds, memo));
            let n: usize = shape.iter().product();
            return Some(Val::Dev {
                buf: self.ctx.cmp_dev(&a, &b, op, n, g.dtype(node.src[0])),
                shape: shape.to_vec(),
                dt,
            });
        }
        if let Op::Bitcast { to } = &node.op
            && dev_dtype(*to)
            && dev_dtype(g.dtype(node.src[0]))
        {
            // same-width bit reinterpret: relabel the device buffer's dtype (no kernel).
            let buf = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            return Some(Val::Dev { buf, shape: shape.to_vec(), dt });
        }
        None
    }
}
