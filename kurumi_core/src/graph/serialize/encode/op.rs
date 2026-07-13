//! The exhaustive per-Op encoder: one match arm per `Op` variant writing its tag + attrs. A new
//! Op fails to compile here until its arm exists (the decode side is guarded by the round-trip
//! test). The blob framing and the primitive writers reused below live in the parent `encode.rs`.

use super::{w_argkind, w_bool, w_dtype, w_f32, w_scatter, w_storage, w_u8, w_u32, w_usize, w_vec_usize};
use crate::graph::Op;

pub(crate) fn write_op(o: &mut Vec<u8>, op: &Op) {
    match op {
        Op::Const { data, shape } => {
            w_u8(o, 0);
            w_storage(o, data);
            w_vec_usize(o, shape);
        }
        Op::Input { shape, dtype } => {
            w_u8(o, 1);
            w_vec_usize(o, shape);
            w_dtype(o, *dtype);
        }
        Op::Iota { shape, axis, dtype } => {
            w_u8(o, 2);
            w_vec_usize(o, shape);
            w_usize(o, *axis);
            w_dtype(o, *dtype);
        }
        Op::RandUniform { shape } => {
            w_u8(o, 3);
            w_vec_usize(o, shape);
        }
        Op::Cast { to } => {
            w_u8(o, 4);
            w_dtype(o, *to);
        }
        Op::Bitcast { to } => {
            w_u8(o, 5);
            w_dtype(o, *to);
        }
        Op::Detach => w_u8(o, 6),
        Op::Add => w_u8(o, 7),
        Op::Mul => w_u8(o, 8),
        Op::Max => w_u8(o, 9),
        Op::Neg => w_u8(o, 10),
        Op::IDiv => w_u8(o, 11),
        Op::And => w_u8(o, 12),
        Op::Or => w_u8(o, 13),
        Op::Xor => w_u8(o, 14),
        Op::Shl => w_u8(o, 15),
        Op::Shr => w_u8(o, 16),
        Op::CmpLt => w_u8(o, 17),
        Op::CmpEq => w_u8(o, 18),
        Op::Where => w_u8(o, 19),
        Op::Recip => w_u8(o, 20),
        Op::Sqrt => w_u8(o, 21),
        Op::Exp2 => w_u8(o, 22),
        Op::Log2 => w_u8(o, 23),
        Op::Sin => w_u8(o, 24),
        Op::Floor => w_u8(o, 25),
        Op::Sum { axis } => {
            w_u8(o, 26);
            w_usize(o, *axis);
        }
        Op::Prod { axis } => {
            w_u8(o, 27);
            w_usize(o, *axis);
        }
        Op::ReduceMax { axis } => {
            w_u8(o, 28);
            w_usize(o, *axis);
        }
        Op::ArgReduce { axis, kind } => {
            w_u8(o, 29);
            w_usize(o, *axis);
            w_argkind(o, *kind);
        }
        Op::Softmax { axis } => {
            w_u8(o, 30);
            w_usize(o, *axis);
        }
        Op::RmsNorm { axis, eps } => {
            w_u8(o, 31);
            w_usize(o, *axis);
            w_f32(o, *eps);
        }
        Op::Sdpa { causal } => {
            w_u8(o, 32);
            w_bool(o, *causal);
        }
        Op::Reshape { shape } => {
            w_u8(o, 33);
            w_vec_usize(o, shape);
        }
        Op::Permute { perm } => {
            w_u8(o, 34);
            w_vec_usize(o, perm);
        }
        Op::Expand { shape } => {
            w_u8(o, 35);
            w_vec_usize(o, shape);
        }
        Op::Slice { ranges } => {
            w_u8(o, 36);
            w_u32(o, ranges.len() as u32);
            for &(a, b, c) in ranges {
                w_usize(o, a);
                w_usize(o, b);
                w_usize(o, c);
            }
        }
        Op::Flip { axes } => {
            w_u8(o, 37);
            w_vec_usize(o, axes);
        }
        Op::Pad { pads } => {
            w_u8(o, 38);
            w_u32(o, pads.len() as u32);
            for &(lo, hi) in pads {
                w_usize(o, lo);
                w_usize(o, hi);
            }
        }
        Op::DotGeneral { lhs_contract, rhs_contract, lhs_batch, rhs_batch } => {
            w_u8(o, 39);
            w_vec_usize(o, lhs_contract);
            w_vec_usize(o, rhs_contract);
            w_vec_usize(o, lhs_batch);
            w_vec_usize(o, rhs_batch);
        }
        Op::QuantMatmul { bits, group_size, symmetric } => {
            w_u8(o, 40);
            w_u8(o, *bits);
            w_usize(o, *group_size);
            w_bool(o, *symmetric);
        }
        Op::Solve => w_u8(o, 41),
        Op::Det => w_u8(o, 42),
        Op::Cholesky => w_u8(o, 43),
        Op::Eigh => w_u8(o, 44),
        Op::Qr { r_factor } => {
            w_u8(o, 45);
            w_bool(o, *r_factor);
        }
        Op::Eigvals => w_u8(o, 46),
        Op::Complex => w_u8(o, 47),
        Op::Real => w_u8(o, 48),
        Op::Imag => w_u8(o, 49),
        Op::Gather { axis } => {
            w_u8(o, 50);
            w_usize(o, *axis);
        }
        Op::Scatter { axis, combine } => {
            w_u8(o, 51);
            w_usize(o, *axis);
            w_scatter(o, *combine);
        }
        Op::GatherAlong { axis } => {
            w_u8(o, 52);
            w_usize(o, *axis);
        }
        Op::ScatterAlong { axis, combine } => {
            w_u8(o, 53);
            w_usize(o, *axis);
            w_scatter(o, *combine);
        }
        Op::Argsort { axis, descending } => {
            w_u8(o, 54);
            w_usize(o, *axis);
            w_bool(o, *descending);
        }
    }
}
