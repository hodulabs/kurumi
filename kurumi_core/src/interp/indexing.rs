//! Indexed access along one axis (jnp.take semantics, a pragmatic subset of the
//! full StableHLO gather/scatter). Operand laid out [pre, axis, post] row-major.

use crate::{Num, Op, ScatterOp, Storage, TensorVal, inc, row_major_strides};

// scatter Add/Max/Min combiner dispatch (numeric only; Set uses any_binary! + set_combine).
// `$k` is the scatter kernel (`scatter_k` or `scatter_along_k`).
macro_rules! scatter_num {
    ($k:path, $op:expr, $up:expr, $f:path, $sh:expr, $ax:expr, $idx:expr, $is:expr) => {
        match ($op, $up) {
            (Storage::U8(o), Storage::U8(u)) => Storage::U8($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::U32(o), Storage::U32(u)) => Storage::U32($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::I32(o), Storage::I32(u)) => Storage::I32($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::I64(o), Storage::I64(u)) => Storage::I64($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::F16(o), Storage::F16(u)) => Storage::F16($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::BF16(o), Storage::BF16(u)) => Storage::BF16($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::F32(o), Storage::F32(u)) => Storage::F32($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::F64(o), Storage::F64(u)) => Storage::F64($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::F8E4M3(o), Storage::F8E4M3(u)) => Storage::F8E4M3($k(o, $sh, $ax, $idx, $is, u, $f)),
            (Storage::F8E5M2(o), Storage::F8E5M2(u)) => Storage::F8E5M2($k(o, $sh, $ax, $idx, $is, u, $f)),
            _ => unreachable!("scatter Add/Max/Min on non-numeric dtype"),
        }
    };
}

pub(super) fn eval(op: &Op, inputs: &[&TensorVal]) -> TensorVal {
    match op {
        Op::Gather { axis } => {
            let (operand, ix) = (inputs[0], inputs[1]);
            let idx = indices_i64(&ix.storage);
            let out: Vec<usize> =
                operand.shape[..*axis].iter().chain(&ix.shape).chain(&operand.shape[*axis + 1..]).copied().collect();
            let storage = dispatch!(&operand.storage, |v| gather_k(v, &operand.shape, *axis, &idx, &ix.shape));
            TensorVal { shape: out, storage }
        }
        Op::Scatter { axis, combine } => {
            let (operand, ix, up) = (inputs[0], inputs[1], inputs[2]);
            let idx = indices_i64(&ix.storage);
            let storage = match combine {
                ScatterOp::Set => any_binary!(&operand.storage, &up.storage, |o, u| scatter_k(
                    o,
                    &operand.shape,
                    *axis,
                    &idx,
                    &ix.shape,
                    u,
                    set_combine
                )),
                ScatterOp::Add => {
                    scatter_num!(
                        scatter_k,
                        &operand.storage,
                        &up.storage,
                        Num::add,
                        &operand.shape,
                        *axis,
                        &idx,
                        &ix.shape
                    )
                }
                ScatterOp::Max => {
                    scatter_num!(
                        scatter_k,
                        &operand.storage,
                        &up.storage,
                        Num::max,
                        &operand.shape,
                        *axis,
                        &idx,
                        &ix.shape
                    )
                }
                ScatterOp::Min => {
                    scatter_num!(
                        scatter_k,
                        &operand.storage,
                        &up.storage,
                        Num::min,
                        &operand.shape,
                        *axis,
                        &idx,
                        &ix.shape
                    )
                }
            };
            TensorVal { shape: operand.shape.clone(), storage }
        }
        Op::GatherAlong { axis } => {
            let (operand, ix) = (inputs[0], inputs[1]);
            let idx = indices_i64(&ix.storage);
            let storage = dispatch!(&operand.storage, |v| gather_along_k(v, &operand.shape, *axis, &idx, &ix.shape));
            TensorVal { shape: ix.shape.clone(), storage }
        }
        Op::ScatterAlong { axis, combine } => {
            let (operand, ix, up) = (inputs[0], inputs[1], inputs[2]);
            let idx = indices_i64(&ix.storage);
            let storage = match combine {
                ScatterOp::Set => any_binary!(&operand.storage, &up.storage, |o, u| scatter_along_k(
                    o,
                    &operand.shape,
                    *axis,
                    &idx,
                    &ix.shape,
                    u,
                    set_combine
                )),
                ScatterOp::Add => scatter_num!(
                    scatter_along_k,
                    &operand.storage,
                    &up.storage,
                    Num::add,
                    &operand.shape,
                    *axis,
                    &idx,
                    &ix.shape
                ),
                ScatterOp::Max => scatter_num!(
                    scatter_along_k,
                    &operand.storage,
                    &up.storage,
                    Num::max,
                    &operand.shape,
                    *axis,
                    &idx,
                    &ix.shape
                ),
                ScatterOp::Min => scatter_num!(
                    scatter_along_k,
                    &operand.storage,
                    &up.storage,
                    Num::min,
                    &operand.shape,
                    *axis,
                    &idx,
                    &ix.shape
                ),
            };
            TensorVal { shape: operand.shape.clone(), storage }
        }
        _ => unreachable!("indexing::eval: non-indexing op"),
    }
}

// take_along_dim / scatter_along: per-position index along one axis (torch
// take_along_dim / index_add). The index tensor matches the output shape, so each
// element carries its own axis index (unlike Gather where a `pre` slice shares one).

// out[c] = operand[c with axis <- clamp(idx[c])]; out/idx share `out_shape`.
pub(crate) fn gather_along_k<T: Copy>(
    operand: &[T],
    op_shape: &[usize],
    axis: usize,
    idx: &[i64],
    out_shape: &[usize],
) -> Vec<T> {
    let op_strides = row_major_strides(op_shape);
    let a = op_shape[axis] as i64;
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(out_len);
    let mut coord = vec![0usize; out_shape.len()];
    for &raw in idx.iter().take(out_len) {
        let j = raw.clamp(0, a - 1) as usize;
        let op_flat: usize = (0..op_shape.len()).map(|d| (if d == axis { j } else { coord[d] }) * op_strides[d]).sum();
        out.push(operand[op_flat]);
        inc(&mut coord, out_shape);
    }
    out
}

// copy operand, then for each idx/update position fold its update into the operand
// at [.., idx, ..] (combine), in-bounds only. idx/updates share `idx_shape`.
pub(crate) fn scatter_along_k<T: Copy>(
    operand: &[T],
    op_shape: &[usize],
    axis: usize,
    idx: &[i64],
    idx_shape: &[usize],
    updates: &[T],
    combine: impl Fn(T, T) -> T,
) -> Vec<T> {
    let op_strides = row_major_strides(op_shape);
    let a = op_shape[axis] as i64;
    let mut out = operand.to_vec();
    let upd_len: usize = idx_shape.iter().product::<usize>().max(1);
    let mut coord = vec![0usize; idx_shape.len()];
    for flat in 0..upd_len {
        let j = idx[flat];
        if j >= 0 && j < a {
            let op_flat: usize =
                (0..op_shape.len()).map(|d| (if d == axis { j as usize } else { coord[d] }) * op_strides[d]).sum();
            out[op_flat] = combine(out[op_flat], updates[flat]);
        }
        inc(&mut coord, idx_shape);
    }
    out
}

// read an integer index tensor as i64
pub(crate) fn indices_i64(s: &Storage) -> Vec<i64> {
    match s {
        Storage::U8(v) => v.iter().map(|&x| x as i64).collect(),
        Storage::U32(v) => v.iter().map(|&x| x as i64).collect(),
        Storage::I32(v) => v.iter().map(|&x| x as i64).collect(),
        Storage::I64(v) => v.clone(),
        _ => unreachable!("gather/scatter indices must be integer (record-time validated)"),
    }
}

// operand is laid out [pre, axis, post] row-major; gather the post-slice at each
// (clamped) index. output = [pre, idx_shape, post].
pub(crate) fn gather_k<T: Copy>(
    operand: &[T],
    op_shape: &[usize],
    axis: usize,
    idx: &[i64],
    idx_shape: &[usize],
) -> Vec<T> {
    let pre: usize = op_shape[..axis].iter().product();
    let post: usize = op_shape[axis + 1..].iter().product();
    let da = op_shape[axis] as i64;
    let k: usize = idx_shape.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(pre * k * post);
    for p in 0..pre {
        for &raw in idx.iter().take(k) {
            let a = raw.clamp(0, da - 1) as usize; // OOB clamp
            let base = (p * op_shape[axis] + a) * post;
            out.extend_from_slice(&operand[base..base + post]);
        }
    }
    out
}

pub(crate) fn set_combine<T>(_old: T, new: T) -> T {
    new
}

// copy operand, then write each update post-slice at its (in-bounds) index,
// folding with `combine`. updates is laid out [pre, idx_shape, post].
pub(crate) fn scatter_k<T: Copy>(
    operand: &[T],
    op_shape: &[usize],
    axis: usize,
    idx: &[i64],
    idx_shape: &[usize],
    updates: &[T],
    combine: impl Fn(T, T) -> T,
) -> Vec<T> {
    let pre: usize = op_shape[..axis].iter().product();
    let post: usize = op_shape[axis + 1..].iter().product();
    let da = op_shape[axis] as i64;
    let k: usize = idx_shape.iter().product::<usize>().max(1);
    let mut out = operand.to_vec();
    for p in 0..pre {
        for (ki, &raw) in idx.iter().take(k).enumerate() {
            if raw < 0 || raw >= da {
                continue; // OOB drop
            }
            let dst = (p * op_shape[axis] + raw as usize) * post;
            let src = (p * k + ki) * post;
            for j in 0..post {
                out[dst + j] = combine(out[dst + j], updates[src + j]);
            }
        }
    }
    out
}
