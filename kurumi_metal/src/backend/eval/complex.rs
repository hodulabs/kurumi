//! Device complex seam: the `Complex`/`Real`/`Imag` primitives on C64 (float2).
//! C128 = double parts -> host (no double on Metal); this unblocks conj/cabs/angle/
//! fft/ifft device-resident.

use crate::MetalBackend;
use crate::backend::eval::Val;
use kurumi_core::{DType, Feeds, Graph, Node, NodeId, Op};
use std::collections::HashMap;

impl MetalBackend {
    pub(in crate::backend) fn eval_complex(
        &self,
        g: &Graph,
        node: &Node,
        shape: &[usize],
        dt: DType,
        feeds: &Feeds,
        memo: &mut HashMap<NodeId, Val>,
    ) -> Option<Val> {
        let n: usize = shape.iter().product();
        match &node.op {
            // (re, im) f32 parts -> C64. F64 parts (-> C128) fall to host.
            Op::Complex if dt == DType::C64 => {
                let re = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
                let im = self.to_dev(&self.eval_memo(g, node.src[1], feeds, memo));
                Some(Val::Dev { buf: self.ctx.complex_dev(&re, &im, n), shape: shape.to_vec(), dt })
            }
            // part extraction on C64 -> F32. C128 source (-> F64) falls to host.
            Op::Real if g.dtype(node.src[0]) == DType::C64 => {
                let z = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
                Some(Val::Dev { buf: self.ctx.real_dev(&z, n), shape: shape.to_vec(), dt })
            }
            Op::Imag if g.dtype(node.src[0]) == DType::C64 => {
                let z = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
                Some(Val::Dev { buf: self.ctx.imag_dev(&z, n), shape: shape.to_vec(), dt })
            }
            _ => None,
        }
    }
}
