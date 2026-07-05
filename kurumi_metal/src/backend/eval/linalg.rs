//! Device f32 dense linalg: solve / determinant / Cholesky (one thread per batch
//! matrix). f64 has no Metal double path -> host oracle.

use crate::MetalBackend;
use crate::backend::eval::Val;
use kurumi_core::{DType, Feeds, Graph, Node, NodeId, Op};
use std::collections::HashMap;

impl MetalBackend {
    pub(in crate::backend) fn eval_linalg(
        &self,
        g: &Graph,
        node: &Node,
        shape: &[usize],
        dt: DType,
        feeds: &Feeds,
        memo: &mut HashMap<NodeId, Val>,
    ) -> Option<Val> {
        // Eigvals gates on the INPUT dtype (its output is C64); the rest on the f32
        // output. Gate BEFORE reading shape[r-2] (non-linalg nodes can be rank < 2).
        let is_eigvals = matches!(node.op, Op::Eigvals) && g.dtype(node.src[0]) == DType::F32;
        let is_f32 =
            matches!(node.op, Op::Solve | Op::Det | Op::Cholesky | Op::Eigh | Op::Qr { .. }) && dt == DType::F32;
        if !is_eigvals && !is_f32 {
            return None;
        }
        // Matches the oracle now that linalg is dtype-native (f32 computes in f32).
        let ash = g.shape(node.src[0]);
        let r = ash.len();
        let (m, n) = (ash[r - 2], ash[r - 1]);
        let batch: usize = ash[..r - 2].iter().product();
        let a = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
        if is_eigvals {
            return Some(Val::Dev { buf: self.ctx.eigvals_dev(&a, batch, n), shape: shape.to_vec(), dt });
        }
        let buf = match node.op {
            Op::Solve => {
                let bsh = g.shape(node.src[1]);
                let k = bsh[bsh.len() - 1];
                let b = self.to_dev(&self.eval_memo(g, node.src[1], feeds, memo));
                self.ctx.solve_dev(&a, &b, batch, n, k)
            }
            Op::Det => self.ctx.det_dev(&a, batch, n),
            Op::Cholesky => self.ctx.cholesky_dev(&a, batch, n),
            Op::Eigh => self.ctx.eigh_dev(&a, batch, n),
            Op::Qr { r_factor } => self.ctx.qr_dev(&a, batch, m, n, r_factor),
            _ => unreachable!(),
        };
        Some(Val::Dev { buf, shape: shape.to_vec(), dt })
    }
}
