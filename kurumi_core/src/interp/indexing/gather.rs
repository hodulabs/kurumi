//! Gather kernels: jnp.take-style Gather (a `pre` slice shares one index list) and torch
//! take_along_dim-style GatherAlong (each output position carries its own axis index).

use super::indices_i64;
use crate::{Op, Storage, TensorVal, inc, row_major_strides};

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
        Op::GatherAlong { axis } => {
            let (operand, ix) = (inputs[0], inputs[1]);
            let idx = indices_i64(&ix.storage);
            let storage = dispatch!(&operand.storage, |v| gather_along_k(v, &operand.shape, *axis, &idx, &ix.shape));
            TensorVal { shape: ix.shape.clone(), storage }
        }
        _ => unreachable!("gather::eval: non-gather op"),
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
