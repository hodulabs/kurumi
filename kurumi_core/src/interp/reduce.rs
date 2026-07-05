//! Reduction interp kernels: arg-reduce (argmax/argmin), the generic axis fold
//! (`reduce_v`), and sum/prod with f32-promoted accumulation for low-precision floats.

use crate::{ArgKind, DType, Elem, Num, Storage, TensorVal, cast, inc, row_major_strides};

/// Index of the max/min along `axis` (keepdim=false), as I64 indices. Ties take
/// the first (lowest index). Non-differentiable.
pub(crate) fn arg_reduce(tv: &TensorVal, axis: usize, kind: ArgKind) -> TensorVal {
    let min = matches!(kind, ArgKind::Min);
    match &tv.storage {
        Storage::F32(v) => arg_reduce_t(v, &tv.shape, axis, min),
        Storage::F64(v) => arg_reduce_t(v, &tv.shape, axis, min),
        Storage::F16(v) => arg_reduce_t(v, &tv.shape, axis, min),
        Storage::BF16(v) => arg_reduce_t(v, &tv.shape, axis, min),
        Storage::I32(v) => arg_reduce_t(v, &tv.shape, axis, min),
        Storage::I64(v) => arg_reduce_t(v, &tv.shape, axis, min),
        Storage::U32(v) => arg_reduce_t(v, &tv.shape, axis, min),
        Storage::U8(v) => arg_reduce_t(v, &tv.shape, axis, min),
        s => panic!("argreduce on non-orderable dtype {:?}", s.dtype()),
    }
}

fn arg_reduce_t<T: Copy + PartialOrd>(data: &[T], shape: &[usize], axis: usize, min: bool) -> TensorVal {
    let in_strides = row_major_strides(shape);
    let (axis_len, axis_stride) = (shape[axis], in_strides[axis]);
    let out_shape: Vec<usize> = shape.iter().enumerate().filter_map(|(i, &d)| (i != axis).then_some(d)).collect();
    let base_strides: Vec<usize> =
        in_strides.iter().enumerate().filter_map(|(i, &s)| (i != axis).then_some(s)).collect();
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out: Vec<i64> = Vec::with_capacity(out_len);
    let mut coord = vec![0usize; out_shape.len()];
    for _ in 0..out_len {
        let base: usize = coord.iter().zip(&base_strides).map(|(c, s)| c * s).sum();
        let mut best = data[base];
        let mut best_idx = 0usize;
        for k in 1..axis_len {
            let v = data[base + k * axis_stride];
            if if min { v < best } else { v > best } {
                best = v;
                best_idx = k;
            }
        }
        out.push(best_idx as i64);
        inc(&mut coord, &out_shape);
    }
    TensorVal { shape: out_shape, storage: Storage::I64(out) }
}

pub(crate) fn reduce_v<T: Elem>(data: &[T], shape: &[usize], axis: usize, init: T, f: impl Fn(T, T) -> T) -> TensorVal {
    let in_strides = row_major_strides(shape);
    let (axis_len, axis_stride) = (shape[axis], in_strides[axis]);

    let out_shape: Vec<usize> = shape.iter().enumerate().filter_map(|(i, &d)| (i != axis).then_some(d)).collect();
    let base_strides: Vec<usize> =
        in_strides.iter().enumerate().filter_map(|(i, &s)| (i != axis).then_some(s)).collect();

    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(out_len);
    let mut coord = vec![0usize; out_shape.len()];
    for _ in 0..out_len {
        let base: usize = coord.iter().zip(&base_strides).map(|(c, s)| c * s).sum();
        out.push((0..axis_len).fold(init, |acc, k| f(acc, data[base + k * axis_stride])));
        inc(&mut coord, &out_shape);
    }
    TensorVal { shape: out_shape, storage: T::store(out) }
}

// sum/prod reductions promote low-precision floats (f16/bf16/fp8) to f32 to accumulate, then
// round back: matching Metal's f32 accumulator and torch/jax so a long reduction doesn't bleed
// precision to per-step rounding. int/f32/f64/complex fold in their own type (exact / already wide).
pub(crate) fn reduce_sum(s: &Storage, shape: &[usize], axis: usize) -> TensorVal {
    reduce_promoted(s, shape, axis, true)
}
pub(crate) fn reduce_prod(s: &Storage, shape: &[usize], axis: usize) -> TensorVal {
    reduce_promoted(s, shape, axis, false)
}
fn reduce_promoted(s: &Storage, shape: &[usize], axis: usize, is_sum: bool) -> TensorVal {
    let dt = s.dtype();
    let low = matches!(dt, DType::F16 | DType::BF16 | DType::F8E4M3 | DType::F8E5M2);
    let promoted = low.then(|| cast(s, DType::F32));
    let src = promoted.as_ref().unwrap_or(s);
    let r = if is_sum {
        num_reduce!(src, shape, axis, Num::zero, Num::add)
    } else {
        num_reduce!(src, shape, axis, Num::one, Num::mul)
    };
    if low { TensorVal { shape: r.shape, storage: cast(&r.storage, dt) } } else { r }
}
