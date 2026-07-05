//! Device GEMM: MPS float GEMM (batched/transposed) and the naive complex GEMM.
//! `dot_nd` recognizes the dot_general shapes MPS can run as a (batched, optionally
//! transposed) GEMM; anything else falls to the host path in `backend/hostgemm`.

use crate::MetalBackend;
use crate::backend::eval::Val;
use crate::dtype::mps_dtype;
use kurumi_core::{DType, Feeds, Graph, Node, NodeId, Op};
use std::collections::HashMap;

impl MetalBackend {
    pub(in crate::backend) fn eval_matmul(
        &self,
        g: &Graph,
        node: &Node,
        shape: &[usize],
        dt: DType,
        feeds: &Feeds,
        memo: &mut HashMap<NodeId, Val>,
    ) -> Option<Val> {
        if !matches!(node.op, Op::DotGeneral { .. }) {
            return None;
        }
        if let Some(d) = dot_nd(g, node).filter(|_| mps_dtype(dt)) {
            // device-resident (batched) GEMM via MPS: forward, transposed backward,
            // batched attention, rank-N@2D linears. f32/f16 are native; bf16 (no MPS
            // support) upcasts to f32 on-device and back, staying GPU-resident.
            let a = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let b = self.to_dev(&self.eval_memo(g, node.src[1], feeds, memo));
            let buf = if dt == DType::BF16 {
                let (an, bn, cn) = (d.batch * d.a_rc.0 * d.a_rc.1, d.batch * d.b_rc.0 * d.b_rc.1, d.batch * d.m * d.n);
                let af = self.ctx.cast_dev(&a, an, DType::BF16, DType::F32);
                let bf = self.ctx.cast_dev(&b, bn, DType::BF16, DType::F32);
                let cf = self.ctx.mps_matmul_dev(
                    &af,
                    d.a_rc,
                    d.trans_l,
                    &bf,
                    d.b_rc,
                    d.trans_r,
                    d.batch,
                    d.m,
                    d.n,
                    d.k,
                    DType::F32,
                );
                self.ctx.cast_dev(&cf, cn, DType::F32, DType::BF16)
            } else {
                self.ctx.mps_matmul_dev(&a, d.a_rc, d.trans_l, &b, d.b_rc, d.trans_r, d.batch, d.m, d.n, d.k, dt)
            };
            return Some(Val::Dev { buf, shape: shape.to_vec(), dt });
        }
        if let Some(d) = dot_nd(g, node).filter(|_| dt == DType::C64) {
            // device complex GEMM (naive cmul-accumulate): ANY complex dot_general:
            // gate application, batched multi-qubit gates, transposed autograd backward.
            let a = self.to_dev(&self.eval_memo(g, node.src[0], feeds, memo));
            let b = self.to_dev(&self.eval_memo(g, node.src[1], feeds, memo));
            return Some(Val::Dev {
                buf: self.ctx.cmatmul_dev(&a, &b, d.batch, d.m, d.n, d.k, d.trans_l, d.trans_r),
                shape: shape.to_vec(),
                dt,
            });
        }
        None
    }
}

pub(super) struct DotNd {
    pub(super) batch: usize,
    pub(super) a_rc: (usize, usize),
    pub(super) trans_l: bool,
    pub(super) b_rc: (usize, usize),
    pub(super) trans_r: bool,
    pub(super) m: usize,
    pub(super) k: usize,
    pub(super) n: usize,
}

// recognize a dot_general the MPS/complex GEMM path can run as a (batched, optionally
// transposed) GEMM; `None` means it stays on the host contraction.
pub(super) fn dot_nd(g: &Graph, node: &Node) -> Option<DotNd> {
    let Op::DotGeneral { lhs_contract, rhs_contract, lhs_batch, rhs_batch } = &node.op else {
        return None;
    };
    let (a, b) = (g.shape(node.src[0]), g.shape(node.src[1]));
    let (&[cl], &[cr]) = (lhs_contract.as_slice(), rhs_contract.as_slice()) else {
        return None;
    };
    let (r, s) = (a.len(), b.len());

    // batched: leading [0..p] batch dims match on both, then one [rows, cols] each.
    let p = lhs_batch.len();
    if p > 0 {
        let lead: Vec<usize> = (0..p).collect();
        if lhs_batch != &lead || rhs_batch != &lead || a[..p] != b[..p] || r != p + 2 || s != p + 2 {
            return None;
        }
        let (trans_l, trans_r) = (cl == r - 2, cr == s - 1);
        let (m, k) = if trans_l { (a[r - 1], a[r - 2]) } else { (a[r - 2], a[r - 1]) };
        let (kb, n) = if trans_r { (b[s - 1], b[s - 2]) } else { (b[s - 2], b[s - 1]) };
        return (k == kb).then_some(DotNd {
            batch: a[..p].iter().product(),
            a_rc: (a[r - 2], a[r - 1]),
            trans_l,
            b_rc: (b[s - 2], b[s - 1]),
            trans_r,
            m,
            k,
            n,
        });
    }
    if !rhs_batch.is_empty() {
        return None;
    }
    // linear layer: lhs[.., K] (contract last) @ rhs[K, N] (contract first): the
    // leading lhs dims are contiguous, so they flatten into M (one plain GEMM).
    if s == 2 && cl == r - 1 && cr == 0 {
        let (m, k, n) = (a[..r - 1].iter().product(), a[r - 1], b[1]);
        return (k == b[0]).then_some(DotNd {
            batch: 1,
            a_rc: (m, k),
            trans_l: false,
            b_rc: (k, n),
            trans_r: false,
            m,
            k,
            n,
        });
    }
    // general 2D (either contract axis -> optional transpose)
    if r == 2 && s == 2 {
        let (trans_l, trans_r) = (cl == 0, cr == 1);
        let (m, k) = if trans_l { (a[1], a[0]) } else { (a[0], a[1]) };
        let (kb, n) = if trans_r { (b[1], b[0]) } else { (b[0], b[1]) };
        return (k == kb).then_some(DotNd {
            batch: 1,
            a_rc: (a[0], a[1]),
            trans_l,
            b_rc: (b[0], b[1]),
            trans_r,
            m,
            k,
            n,
        });
    }
    None
}
