//! Interp oracles for fused nn primitives. The CPU reference a backend's fused kernel is
//! checked against; each computes the decomposed math directly.

use crate::{DType, Storage, TensorVal, cast, inc, row_major_strides};

// stable softmax over `axis` (shape-preserving): exp(x - rowmax) / rowsum. low-precision
// floats (f16/bf16) promote to f32 for the exp/sum then round back, matching the reduce path
// and the device f32 accumulator; f64 folds in f64.
pub(crate) fn softmax_v(s: &Storage, shape: &[usize], axis: usize) -> TensorVal {
    let dt = s.dtype();
    let storage = match dt {
        DType::F64 => {
            let Storage::F64(v) = s else { unreachable!() };
            Storage::F64(softmax_f64(v, shape, axis))
        }
        _ => {
            let f = cast(s, DType::F32); // f32 directly, f16/bf16 upcast
            let Storage::F32(v) = &f else { unreachable!() };
            let r = Storage::F32(softmax_f32(v, shape, axis));
            if dt == DType::F32 { r } else { cast(&r, dt) }
        }
    };
    TensorVal { shape: shape.to_vec(), storage }
}

macro_rules! softmax_impl {
    ($name:ident, $t:ty) => {
        fn $name(data: &[$t], shape: &[usize], axis: usize) -> Vec<$t> {
            let strides = row_major_strides(shape);
            let (axis_len, axis_stride) = (shape[axis], strides[axis]);
            let out_shape: Vec<usize> =
                shape.iter().enumerate().filter_map(|(i, &d)| (i != axis).then_some(d)).collect();
            let base_strides: Vec<usize> =
                strides.iter().enumerate().filter_map(|(i, &st)| (i != axis).then_some(st)).collect();
            let out_len = out_shape.iter().product::<usize>().max(1);
            let mut out = vec![0 as $t; data.len()];
            let mut coord = vec![0usize; out_shape.len()];
            for _ in 0..out_len {
                let base: usize = coord.iter().zip(&base_strides).map(|(c, st)| c * st).sum();
                let mut m = <$t>::NEG_INFINITY;
                for k in 0..axis_len {
                    m = m.max(data[base + k * axis_stride]);
                }
                let mut sum = 0 as $t;
                for k in 0..axis_len {
                    let e = (data[base + k * axis_stride] - m).exp();
                    out[base + k * axis_stride] = e;
                    sum += e;
                }
                for k in 0..axis_len {
                    out[base + k * axis_stride] /= sum;
                }
                inc(&mut coord, &out_shape);
            }
            out
        }
    };
}
softmax_impl!(softmax_f32, f32);
softmax_impl!(softmax_f64, f64);

// RMSNorm over `axis` (shape-preserving): x / sqrt(mean(x^2) + eps). low-precision floats
// compute in f32 then round back; f64 in f64.
pub(crate) fn rmsnorm_v(s: &Storage, shape: &[usize], axis: usize, eps: f32) -> TensorVal {
    let dt = s.dtype();
    let storage = match dt {
        DType::F64 => {
            let Storage::F64(v) = s else { unreachable!() };
            Storage::F64(rmsnorm_f64(v, shape, axis, eps as f64))
        }
        _ => {
            let f = cast(s, DType::F32);
            let Storage::F32(v) = &f else { unreachable!() };
            let r = Storage::F32(rmsnorm_f32(v, shape, axis, eps));
            if dt == DType::F32 { r } else { cast(&r, dt) }
        }
    };
    TensorVal { shape: shape.to_vec(), storage }
}

macro_rules! rmsnorm_impl {
    ($name:ident, $t:ty) => {
        fn $name(data: &[$t], shape: &[usize], axis: usize, eps: $t) -> Vec<$t> {
            let strides = row_major_strides(shape);
            let (axis_len, axis_stride) = (shape[axis], strides[axis]);
            let out_shape: Vec<usize> =
                shape.iter().enumerate().filter_map(|(i, &d)| (i != axis).then_some(d)).collect();
            let base_strides: Vec<usize> =
                strides.iter().enumerate().filter_map(|(i, &st)| (i != axis).then_some(st)).collect();
            let out_len = out_shape.iter().product::<usize>().max(1);
            let mut out = vec![0 as $t; data.len()];
            let mut coord = vec![0usize; out_shape.len()];
            for _ in 0..out_len {
                let base: usize = coord.iter().zip(&base_strides).map(|(c, st)| c * st).sum();
                let mut ss = 0 as $t;
                for k in 0..axis_len {
                    let x = data[base + k * axis_stride];
                    ss += x * x;
                }
                let rms = (ss / axis_len as $t + eps).sqrt();
                for k in 0..axis_len {
                    out[base + k * axis_stride] = data[base + k * axis_stride] / rms;
                }
                inc(&mut coord, &out_shape);
            }
            out
        }
    };
}
rmsnorm_impl!(rmsnorm_f32, f32);
rmsnorm_impl!(rmsnorm_f64, f64);
