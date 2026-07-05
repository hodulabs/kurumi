//! Record-time inference: a node's dtype and shape, derived from its op + inputs
//! and stored on the node (so `shape`/`dtype` lookups are O(1)). Called once per
//! node at `push`. Adding an op means adding its arm here.

use crate::layout::free_axes;
use crate::{DType, Graph, NodeId, Op};

impl Graph {
    // most ops preserve the input dtype; the exceptions carry/derive their own.
    pub(super) fn infer_dtype(&self, op: &Op, src: &[NodeId]) -> DType {
        match op {
            Op::Const { data, .. } => data.dtype(),
            Op::Input { dtype, .. } => *dtype,
            Op::Iota { dtype, .. } => *dtype,
            Op::Cast { to } | Op::Bitcast { to } => *to,
            Op::CmpLt | Op::CmpEq => DType::BOOL,
            Op::ArgReduce { .. } | Op::Argsort { .. } => DType::I64, // indices
            Op::RandUniform { .. } => DType::F32,
            Op::Where => self.node(src[1]).dtype, // value dtype (src[0] is the bool cond)
            // complex construction/extraction
            Op::Complex => {
                if self.node(src[0]).dtype == DType::F64 {
                    DType::C128
                } else {
                    DType::C64
                }
            }
            Op::Real | Op::Imag => self.node(src[0]).dtype.real(),
            Op::Eigvals => {
                if self.node(src[0]).dtype == DType::F64 {
                    DType::C128
                } else {
                    DType::C64
                }
            }
            _ => self.node(src[0]).dtype,
        }
    }

    // Derive a node's shape from its op + inputs' (already-stored) shapes. Called once
    // per node at `push`; reads input shapes in O(1), so the whole graph is O(N).
    pub(super) fn derive_shape(&self, op: &Op, src: &[NodeId]) -> Vec<usize> {
        let ishape = |i: usize| self.node(src[i]).shape.as_slice();
        match op {
            Op::Const { shape, .. } | Op::Input { shape, .. } | Op::Iota { shape, .. } | Op::RandUniform { shape } => {
                shape.clone()
            }
            // elementwise (unary/binary/ternary): output shape = first input's
            Op::Cast { .. }
            | Op::Bitcast { .. }
            | Op::Detach
            | Op::Add
            | Op::Mul
            | Op::Max
            | Op::Neg
            | Op::Recip
            | Op::Sqrt
            | Op::Exp2
            | Op::Log2
            | Op::Sin
            | Op::Floor
            | Op::IDiv
            | Op::And
            | Op::Or
            | Op::Xor
            | Op::Shl
            | Op::Shr
            | Op::CmpLt
            | Op::CmpEq
            | Op::Where
            | Op::Softmax { .. }
            | Op::RmsNorm { .. } => ishape(0).to_vec(),
            Op::Sum { axis } | Op::Prod { axis } | Op::ReduceMax { axis } | Op::ArgReduce { axis, .. } => {
                let mut s = ishape(0).to_vec();
                s.remove(*axis);
                s
            }
            Op::Reshape { shape } | Op::Expand { shape } => shape.clone(),
            Op::Permute { perm } => {
                let s = ishape(0);
                perm.iter().map(|&p| s[p]).collect()
            }
            Op::Slice { ranges } => ranges.iter().map(|&(s, e, st)| (e - s).div_ceil(st)).collect(),
            Op::Flip { .. } => ishape(0).to_vec(),
            Op::Pad { pads } => pads.iter().zip(ishape(0)).map(|(&(lo, hi), &d)| lo + d + hi).collect(),
            // out = batch ++ lhs_free ++ rhs_free
            Op::DotGeneral { lhs_contract, rhs_contract, lhs_batch, rhs_batch } => {
                let (a, b) = (ishape(0), ishape(1));
                let batch = lhs_batch.iter().map(|&i| a[i]);
                let lf = free_axes(a.len(), lhs_batch, lhs_contract).into_iter().map(|i| a[i]);
                let rf = free_axes(b.len(), rhs_batch, rhs_contract).into_iter().map(|i| b[i]);
                batch.chain(lf).chain(rf).collect()
            }
            // operand's `axis` dim is replaced by all of the index tensor's dims
            Op::Gather { axis } => {
                let (op, idx) = (ishape(0), ishape(1));
                op[..*axis].iter().chain(idx).chain(&op[*axis + 1..]).copied().collect()
            }
            Op::GatherAlong { .. } => ishape(1).to_vec(), // output matches the index shape
            Op::Solve => ishape(1).to_vec(),              // X has B's shape
            Op::Cholesky => ishape(0).to_vec(),           // L has A's shape
            Op::Eigh => {
                let mut s = ishape(0).to_vec(); // [.., N, N] -> [.., N, N+1] (vectors | values)
                *s.last_mut().unwrap() += 1;
                s
            }
            Op::Qr { r_factor } => {
                let s = ishape(0);
                let r = s.len();
                let k = s[r - 2].min(s[r - 1]);
                let mut o = s.to_vec();
                if *r_factor {
                    o[r - 2] = k
                } else {
                    o[r - 1] = k
                }
                o
            }
            Op::Eigvals => {
                let s = ishape(0);
                s[..s.len() - 1].to_vec() // [.., N, N] -> [.., N]
            }
            Op::Complex | Op::Real | Op::Imag => ishape(0).to_vec(), // same shape

            Op::Det => {
                let s = ishape(0);
                s[..s.len() - 2].to_vec() // drop the trailing [N, N]
            }
            Op::Scatter { .. } | Op::ScatterAlong { .. } | Op::Argsort { .. } => ishape(0).to_vec(),
            // act[M,K] x dequant(qweight)[N,K]^T -> [M,N]
            Op::QuantMatmul { .. } => vec![ishape(0)[0], ishape(1)[0]],
        }
    }
}
