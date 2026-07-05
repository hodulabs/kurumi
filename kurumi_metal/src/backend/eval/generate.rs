//! Device generators: `Iota` (index ramp -> arange/positions/eye/tril) and the
//! counter-RNG `RandUniform`.

use crate::MetalBackend;
use crate::backend::eval::Val;
use crate::dtype::dev_dtype;
use kurumi_core::{DType, Feeds, Graph, Node, NodeId, Op, Storage};
use std::collections::HashMap;

impl MetalBackend {
    pub(in crate::backend) fn eval_generate(
        &self,
        g: &Graph,
        node: &Node,
        shape: &[usize],
        dt: DType,
        feeds: &Feeds,
        memo: &mut HashMap<NodeId, Val>,
    ) -> Option<Val> {
        if let Op::Iota { axis, dtype, .. } = &node.op
            && dev_dtype(*dtype)
        {
            // device index generator: arange/positions/eye/tril build on this.
            let stride: usize = shape[axis + 1..].iter().product();
            let axis_len = shape[*axis];
            let n: usize = shape.iter().product();
            return Some(Val::Dev { buf: self.ctx.iota_dev(stride, axis_len, n, dt), shape: shape.to_vec(), dt });
        }
        if matches!(node.op, Op::RandUniform { .. }) {
            // counter RNG -> F32 (a device dtype). seed = src[0] scalar int (host read).
            let seed = self.to_host(&self.eval_memo(g, node.src[0], feeds, memo));
            let seed = match &seed.storage {
                Storage::I64(v) => v[0] as u64,
                Storage::I32(v) => v[0] as u64,
                Storage::U64(v) => v[0],
                Storage::U32(v) => v[0] as u64,
                _ => panic!("metal: rand seed must be integer"),
            };
            let n: usize = shape.iter().product();
            return Some(Val::Dev { buf: self.ctx.rand_dev(seed, n), shape: shape.to_vec(), dt });
        }
        None
    }
}
