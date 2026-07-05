//! The fused-pointwise core (fall-through when no device family matched). Movement
//! (reshape/expand/permute/slice/flip) folds into a strided fused leaf; pad/reduce are
//! device primitives; unary/binary arithmetic accretes into ONE lazy expression,
//! materialized into a single kernel only at a boundary. `None` = host oracle offload.

use crate::backend::eval::{Ew, FExpr, FUSE_CAP, MAX_LEAVES, REDUCE_TG, Val, ew_kind, fused_reduce_msl, leaf_eq};
use crate::dtype::{dev_dtype, msl_ty};
use crate::{Buffer, MetalBackend};
use kurumi_core::{DType, Feeds, Graph, Node, NodeId, Op, TensorVal, row_major_strides};
use std::collections::HashMap;

impl MetalBackend {
    pub(in crate::backend) fn eval_fused(
        &self,
        g: &Graph,
        node: &Node,
        shape: &[usize],
        dt: DType,
        feeds: &Feeds,
        memo: &mut HashMap<NodeId, Val>,
    ) -> Val {
        let dev = dev_dtype(dt);
        // complex (float2) runs the full elementwise/movement/reduce/pad set on device:
        // arithmetic + transcendentals (c* helpers), strided movement, sum/prod (float2
        // acc, complex-mul), and zero-pad (float2 fill).
        match ew_kind(&node.op).filter(|_| dev) {
            Some(Ew::Reshape) => {
                let a = self.eval_memo(g, node.src[0], feeds, memo);
                Val::Dev { buf: self.to_dev(&a), shape: shape.to_vec(), dt } // contiguous: reshape is free
            }
            Some(Ew::Expand) => {
                let in_shape = g.shape(node.src[0]);
                let st = row_major_strides(&in_shape);
                // broadcast: stride 0 on the size-1 (expanded) axes
                let strides: Vec<i64> =
                    (0..shape.len()).map(|ax| if in_shape[ax] == 1 { 0 } else { st[ax] as i64 }).collect();
                let a = self.eval_memo(g, node.src[0], feeds, memo);
                self.strided_view(a, 0, strides, shape.to_vec(), dt)
            }
            Some(Ew::Permute) => {
                let Op::Permute { perm } = &node.op else { unreachable!() };
                let st = row_major_strides(&g.shape(node.src[0]));
                // out axis i reads input axis perm[i] -> its stride is st[perm[i]]
                let strides: Vec<i64> = perm.iter().map(|&p| st[p] as i64).collect();
                let a = self.eval_memo(g, node.src[0], feeds, memo);
                self.strided_view(a, 0, strides, shape.to_vec(), dt)
            }
            Some(Ew::Slice) => {
                let Op::Slice { ranges } = &node.op else { unreachable!() };
                let st = row_major_strides(&g.shape(node.src[0]));
                // base = sum(start*stride); the per-axis read stride scales by `step`.
                let base: i64 = ranges.iter().zip(&st).map(|(&(s, _, _), stride)| (s * stride) as i64).sum();
                let strides: Vec<i64> = ranges.iter().zip(&st).map(|(&(_, _, step), &s)| (s * step) as i64).collect();
                let a = self.eval_memo(g, node.src[0], feeds, memo);
                self.strided_view(a, base, strides, shape.to_vec(), dt)
            }
            Some(Ew::Flip) => {
                // reverse each flipped axis: negative stride + a base at that axis's end.
                let Op::Flip { axes } = &node.op else { unreachable!() };
                let st = row_major_strides(shape);
                let mut base: i64 = 0;
                let strides: Vec<i64> = (0..shape.len())
                    .map(|ax| {
                        if axes.contains(&ax) {
                            base += ((shape[ax] - 1) * st[ax]) as i64;
                            -(st[ax] as i64)
                        } else {
                            st[ax] as i64
                        }
                    })
                    .collect();
                let a = self.eval_memo(g, node.src[0], feeds, memo);
                self.strided_view(a, base, strides, shape.to_vec(), dt)
            }
            Some(Ew::Pad) => {
                let Op::Pad { pads } = &node.op else { unreachable!() };
                let in_shape = g.shape(node.src[0]);
                let st = row_major_strides(&in_shape);
                let lo: Vec<u32> = pads.iter().map(|&(l, _)| l as u32).collect();
                let ins: Vec<u32> = in_shape.iter().map(|&x| x as u32).collect();
                let stride: Vec<u32> = st.iter().map(|&s| s as u32).collect();
                let a = self.eval_memo(g, node.src[0], feeds, memo);
                let buf = self.ctx.pad_dev(&self.to_dev(&a), shape, &lo, &ins, &stride, dt);
                Val::Dev { buf, shape: shape.to_vec(), dt }
            }
            Some(Ew::Reduce { tag, axis }) => {
                let in_shape = g.shape(node.src[0]);
                let out_n: usize = shape.iter().product();
                let a = self.eval_memo(g, node.src[0], feeds, memo);
                // sum/prod over a fused producer -> ONE threadgroup-per-line parallel
                // reduce (computes the producer in parallel, drops the intermediate).
                // max, or a non-fused input, take the plain one-thread-per-output path.
                let buf = match a {
                    // complex reduce takes the materialized path (float2 acc / cmul);
                    // the parallel fused-reduce is real-only.
                    Val::Fused { leaves, expr, .. } if matches!(tag, "sum" | "prod") && !dt.is_complex() => {
                        let src = fused_reduce_msl(tag, &expr, &leaves, msl_ty(dt), &in_shape, axis);
                        let refs: Vec<&Buffer> = leaves.iter().map(|l| &l.buf).collect();
                        self.ctx.reduce_fused(&src, &refs, out_n, REDUCE_TG, dt)
                    }
                    other => {
                        let axis_len = in_shape[axis];
                        let inner: usize = in_shape[axis + 1..].iter().product();
                        self.ctx.reduce_dev(tag, &self.to_dev(&other), axis_len, inner, out_n, dt)
                    }
                };
                Val::Dev { buf, shape: shape.to_vec(), dt }
            }
            Some(Ew::Unary(tag)) => {
                let a = self.eval_memo(g, node.src[0], feeds, memo);
                let (leaves, e) = self.as_fused(a);
                Val::Fused { shape: shape.to_vec(), leaves, expr: FExpr::Un(tag, Box::new(e)), dt }
            }
            Some(Ew::Binary(tag)) => {
                let a = self.eval_memo(g, node.src[0], feeds, memo);
                let b = self.eval_memo(g, node.src[1], feeds, memo);
                let (mut la, ea) = self.as_fused(a);
                let (lb, eb) = self.as_fused(b);
                // dedup b's leaves into la by (buffer, view) identity: a node reused
                // N times (common in autograd) becomes ONE leaf, not N.
                let map: Vec<usize> = lb
                    .into_iter()
                    .map(|leaf| {
                        la.iter().position(|x| leaf_eq(x, &leaf)).unwrap_or_else(|| {
                            la.push(leaf);
                            la.len() - 1
                        })
                    })
                    .collect();
                let expr = FExpr::Bin(tag, Box::new(ea), Box::new(eb.remap(&map)));
                // cap inlining (tree size) and the Metal buffer-arg limit (leaves)
                if expr.size() > FUSE_CAP || la.len() > MAX_LEAVES {
                    Val::Dev { buf: self.materialize(shape, &la, &expr, dt), shape: shape.to_vec(), dt }
                } else {
                    Val::Fused { shape: shape.to_vec(), leaves: la, expr, dt }
                }
            }
            None => {
                // host path: materialize inputs, then GPU matmul-offload or CPU oracle
                let inputs: Vec<TensorVal> =
                    node.src.iter().map(|&s| self.to_host(&self.eval_memo(g, s, feeds, memo))).collect();
                Val::Host(self.host_op(&node.op, &inputs.iter().collect::<Vec<_>>()))
            }
        }
    }
}
