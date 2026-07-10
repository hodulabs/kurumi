//! Movement kernels: pure element gathers (permute/expand/slice/flip/pad), generic over any
//! `Copy` element. Each walks the output coordinate odometer, mapping it back to a source flat
//! index. (The realize path fuses these as views; here they materialize, as the oracle.)

use crate::{Op, Storage, TensorVal, inc, row_major_strides};

pub(super) fn eval(op: &Op, inputs: &[&TensorVal]) -> TensorVal {
    match op {
        Op::Reshape { shape } => TensorVal { shape: shape.clone(), storage: inputs[0].storage.clone() },
        Op::Permute { perm } => {
            let a = inputs[0];
            let out: Vec<usize> = perm.iter().map(|&p| a.shape[p]).collect();
            TensorVal { shape: out, storage: dispatch!(&a.storage, |v| permute_k(v, &a.shape, perm)) }
        }
        Op::Expand { shape } => {
            let a = inputs[0];
            TensorVal { shape: shape.clone(), storage: dispatch!(&a.storage, |v| expand_k(v, &a.shape, shape)) }
        }
        Op::Slice { ranges } => {
            let a = inputs[0];
            let out: Vec<usize> = ranges.iter().map(|&(s, e, st)| (e - s).div_ceil(st)).collect();
            TensorVal { shape: out, storage: dispatch!(&a.storage, |v| slice_k(v, &a.shape, ranges)) }
        }
        Op::Flip { axes } => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: dispatch!(&a.storage, |v| flip_k(v, &a.shape, axes)) }
        }
        Op::Pad { pads } => {
            let a = inputs[0];
            let out: Vec<usize> = pads.iter().zip(&a.shape).map(|(&(lo, hi), &s)| lo + s + hi).collect();
            TensorVal { shape: out, storage: dispatch!(&a.storage, |v| pad_k(v, &a.shape, pads)) }
        }
        _ => unreachable!("movement::eval: non-movement op"),
    }
}

pub(crate) fn permute_k<T: Copy>(data: &[T], in_shape: &[usize], perm: &[usize]) -> Vec<T> {
    let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
    let in_strides = row_major_strides(in_shape);
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(out_len);
    let mut coord = vec![0usize; out_shape.len()];
    for _ in 0..out_len {
        let in_flat: usize = perm.iter().enumerate().map(|(i, &p)| coord[i] * in_strides[p]).sum();
        out.push(data[in_flat]);
        inc(&mut coord, &out_shape);
    }
    out
}

pub(crate) fn expand_k<T: Copy>(data: &[T], in_shape: &[usize], out_shape: &[usize]) -> Vec<T> {
    let in_strides = row_major_strides(in_shape);
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(out_len);
    let mut coord = vec![0usize; out_shape.len()];
    for _ in 0..out_len {
        let in_flat: usize =
            (0..out_shape.len()).map(|d| if in_shape[d] == 1 { 0 } else { coord[d] * in_strides[d] }).sum();
        out.push(data[in_flat]);
        inc(&mut coord, out_shape);
    }
    out
}

pub(crate) fn slice_k<T: Copy>(data: &[T], in_shape: &[usize], ranges: &[(usize, usize, usize)]) -> Vec<T> {
    let in_strides = row_major_strides(in_shape);
    let out_shape: Vec<usize> = ranges.iter().map(|&(s, e, st)| (e - s).div_ceil(st)).collect();
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(out_len);
    let mut coord = vec![0usize; out_shape.len()];
    for _ in 0..out_len {
        let in_flat: usize = (0..out_shape.len()).map(|d| (ranges[d].0 + coord[d] * ranges[d].2) * in_strides[d]).sum();
        out.push(data[in_flat]);
        inc(&mut coord, &out_shape);
    }
    out
}

pub(crate) fn flip_k<T: Copy>(data: &[T], shape: &[usize], axes: &[usize]) -> Vec<T> {
    let in_strides = row_major_strides(shape);
    let out_len: usize = shape.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(out_len);
    let mut coord = vec![0usize; shape.len()];
    for _ in 0..out_len {
        let in_flat: usize = (0..shape.len())
            .map(|d| {
                let c = if axes.contains(&d) { shape[d] - 1 - coord[d] } else { coord[d] };
                c * in_strides[d]
            })
            .sum();
        out.push(data[in_flat]);
        inc(&mut coord, shape);
    }
    out
}

// pad fills out-of-bounds positions with the element default (0 / false)
pub(crate) fn pad_k<T: Copy + Default>(data: &[T], in_shape: &[usize], pads: &[(usize, usize)]) -> Vec<T> {
    let in_strides = row_major_strides(in_shape);
    let out_shape: Vec<usize> = pads.iter().zip(in_shape).map(|(&(lo, hi), &s)| lo + s + hi).collect();
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(out_len);
    let mut coord = vec![0usize; out_shape.len()];
    for _ in 0..out_len {
        let mut in_flat = 0;
        let mut valid = true;
        for (d, &(lo, _)) in pads.iter().enumerate() {
            if coord[d] < lo || coord[d] >= lo + in_shape[d] {
                valid = false;
                break;
            }
            in_flat += (coord[d] - lo) * in_strides[d];
        }
        out.push(if valid { data[in_flat] } else { T::default() });
        inc(&mut coord, &out_shape);
    }
    out
}
