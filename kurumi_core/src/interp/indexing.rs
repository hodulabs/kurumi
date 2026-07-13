//! Indexed access along one axis (jnp.take semantics, a pragmatic subset of the
//! full StableHLO gather/scatter). Operand laid out [pre, axis, post] row-major.
//! Gather kernels live in `gather`, scatter kernels in `scatter`; `indices_i64` is shared.

mod gather;
mod scatter;

use crate::{Op, Storage, TensorVal};

pub(super) fn eval(op: &Op, inputs: &[&TensorVal]) -> TensorVal {
    match op {
        Op::Gather { .. } | Op::GatherAlong { .. } => gather::eval(op, inputs),
        Op::Scatter { .. } | Op::ScatterAlong { .. } => scatter::eval(op, inputs),
        _ => unreachable!("indexing::eval: non-indexing op"),
    }
}

// read an integer index tensor as i64
pub(crate) fn indices_i64(s: &Storage) -> Vec<i64> {
    match s {
        Storage::U8(v) => v.iter().map(|&x| x as i64).collect(),
        Storage::U16(v) => v.iter().map(|&x| x as i64).collect(),
        Storage::U32(v) => v.iter().map(|&x| x as i64).collect(),
        Storage::U64(v) => v.iter().map(|&x| x as i64).collect(),
        Storage::I8(v) => v.iter().map(|&x| x as i64).collect(),
        Storage::I16(v) => v.iter().map(|&x| x as i64).collect(),
        Storage::I32(v) => v.iter().map(|&x| x as i64).collect(),
        Storage::I64(v) => v.clone(),
        _ => unreachable!("gather/scatter indices must be integer (record-time validated)"),
    }
}
