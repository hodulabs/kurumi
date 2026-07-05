//! Weight-only quant matmul on the GPU. Float activations (f32/f16/bf16) run on the device;
//! other act dtypes fall through to the CPU quant oracle. Small M (<= 8, decode/small-batch)
//! uses the GEMV kernel; larger M (prefill) uses the threadgroup-tiled GEMM kernel. Checked
//! against `interp/contract::quant_matmul`.

use crate::MetalBackend;
use crate::backend::eval::Val;
use crate::dtype::dev_dtype;
use kurumi_core::{DType, Feeds, Graph, Node, NodeId, Op};
use std::collections::HashMap;

impl MetalBackend {
    pub(in crate::backend) fn eval_quant(
        &self,
        g: &Graph,
        node: &Node,
        shape: &[usize],
        dt: DType,
        feeds: &Feeds,
        memo: &mut HashMap<NodeId, Val>,
    ) -> Option<Val> {
        let Op::QuantMatmul { bits, group_size, symmetric } = &node.op else {
            return None;
        };
        if !dev_dtype(dt) {
            return None; // float activations (f32/f16/bf16) on the device path; else CPU oracle
        }
        let act = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
        let qw = self.to_dev(&self.eval_memo(g, node.src[1], feeds, memo));
        let scales = self.to_dev(&self.eval_memo(g, node.src[2], feeds, memo));
        // symmetric has no mins; bind a dummy buffer (unused in the kernel).
        let mins = if *symmetric { scales.clone() } else { self.to_dev(&self.eval_memo(g, node.src[3], feeds, memo)) };
        let a = g.shape(node.src[0]);
        let (m, k, n) = (a[0], a[1], g.shape(node.src[1])[0]);
        let (bits, sym) = (*bits, *symmetric);
        // GEMV hoists one group scale per 32-bit word (WPER weights), so it needs the group to
        // not straddle a word; standard group sizes (multiples of 32) always do. The tiled GEMM
        // looks up per element, so it holds for any group size -> route small M there when the
        // GEMV assumption fails (rare) rather than falling back to the CPU oracle.
        let wper = 32 / bits as usize; // weights per 32-bit word: 16 int2 / 8 int4 / 4 int8
        let buf = if m <= 8 && group_size.is_multiple_of(wper) {
            self.ctx.dequant_gemv_dev(&act, &qw, &scales, &mins, m, k, n, *group_size, bits, sym, dt)
        } else {
            self.ctx.dequant_gemm_dev(&act, &qw, &scales, &mins, m, k, n, *group_size, bits, sym, dt)
        };
        Some(Val::Dev { buf, shape: shape.to_vec(), dt })
    }
}
