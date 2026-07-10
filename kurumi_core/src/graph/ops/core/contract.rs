//! Contraction: the `dot_general` primitive and `quant_matmul`. matmul/bmm all lower to
//! dot_general in the frontend; `einsum` (which also lowers to dot_general) is the `einsum`
//! submodule; the interpreter side is `interp/contract.rs`.

mod einsum;

use crate::layout::check_axes;
use crate::{Error, Graph, NodeId, Op};

impl Graph {
    /// Strict contraction (no 1D promotion, no batch broadcast).
    /// Batch/contract dims given as axis indices; paired dims must match in size.
    pub fn dot_general(
        &mut self,
        a: NodeId,
        b: NodeId,
        lhs_contract: Vec<usize>,
        rhs_contract: Vec<usize>,
        lhs_batch: Vec<usize>,
        rhs_batch: Vec<usize>,
    ) -> Result<NodeId, Error> {
        let (sa, sb) = (self.shape(a), self.shape(b));
        self.same_dtype("dot_general", a, b)?;
        self.require("dot_general", a, self.dtype(a).is_arith(), "numeric or complex")?;
        if lhs_contract.len() != rhs_contract.len() {
            return Err(Error::shape("dot_general", "contract count mismatch"));
        }
        if lhs_batch.len() != rhs_batch.len() {
            return Err(Error::shape("dot_general", "batch count mismatch"));
        }
        let a_axes: Vec<usize> = lhs_batch.iter().chain(&lhs_contract).copied().collect();
        let b_axes: Vec<usize> = rhs_batch.iter().chain(&rhs_contract).copied().collect();
        check_axes(&a_axes, sa.len())?;
        check_axes(&b_axes, sb.len())?;
        for (&i, &j) in lhs_batch.iter().zip(&rhs_batch) {
            if sa[i] != sb[j] {
                return Err(Error::shape("dot_general", format!("batch dim {} vs {}", sa[i], sb[j])));
            }
        }
        for (&i, &j) in lhs_contract.iter().zip(&rhs_contract) {
            if sa[i] != sb[j] {
                return Err(Error::shape("dot_general", format!("contract dim {} vs {}", sa[i], sb[j])));
            }
        }
        Ok(self.push(Op::DotGeneral { lhs_contract, rhs_contract, lhs_batch, rhs_batch }, vec![a, b]))
    }

    /// Weight-only quantized matmul: `act[M,K] x dequant(qweight)[N,K]^T -> [M,N]`.
    /// `qweight` is U8-packed ([N, K/2] int4 / [N, K] int8), `scales`/`mins` are F16
    /// [N, K/group_size]; `mins = None` is symmetric. Build the tensors with
    /// [`crate::quantize`]. Frozen weights: no gradient flows through it.
    pub fn quant_matmul(
        &mut self,
        act: NodeId,
        qweight: NodeId,
        scales: NodeId,
        mins: Option<NodeId>,
        bits: u8,
        group_size: usize,
    ) -> Result<NodeId, Error> {
        let (sa, sq) = (self.shape(act), self.shape(qweight));
        let (ra, rq) = (sa.len(), sq.len());
        if ra != 2 || rq != 2 {
            return Err(Error::shape("quant_matmul", format!("act rank {ra}, qweight rank {rq}: both must be rank-2")));
        }
        // Fail fast at record time: an out-of-range bits/group_size or a mis-sized
        // qweight/scales otherwise defers to an eval-time unreachable or silent-wrong dequant.
        let (k, n) = (sa[1], sq[0]); // K = act's contract dim, N = weight rows
        if !matches!(bits, 2 | 4 | 8) {
            return Err(Error::shape("quant_matmul", format!("bits {bits} must be 2, 4, or 8")));
        }
        if group_size == 0 || !k.is_multiple_of(group_size) {
            return Err(Error::shape(
                "quant_matmul",
                format!("K {k} must be a nonzero multiple of group_size {group_size}"),
            ));
        }
        // packed qweight stores 8/bits fields per byte, so it has K*bits/8 cols (no division: exact).
        if sq[1] * 8 != k * bits as usize {
            return Err(Error::shape(
                "quant_matmul",
                format!("qweight cols {} must equal K*bits/8 = {}*{}/8", sq[1], k, bits),
            ));
        }
        // scales (and mins, when asymmetric) hold one entry per [N, K/group_size] group.
        let want = n * (k / group_size);
        if self.shape(scales).iter().product::<usize>() != want {
            return Err(Error::shape("quant_matmul", format!("scales must have N*K/group_size = {want} entries")));
        }
        if let Some(m) = mins
            && self.shape(m).iter().product::<usize>() != want
        {
            return Err(Error::shape("quant_matmul", format!("mins must have N*K/group_size = {want} entries")));
        }
        let symmetric = mins.is_none();
        let mut src = vec![act, qweight, scales];
        src.extend(mins);
        Ok(self.push(Op::QuantMatmul { bits, group_size, symmetric }, src))
    }
}
