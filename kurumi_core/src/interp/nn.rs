//! Interp oracles for fused nn primitives. The CPU reference a backend's fused kernel is
//! checked against; each computes the decomposed math directly.

use crate::{DType, Op, Storage, TensorVal, cast, inc, row_major_strides};

pub(super) fn eval(op: &Op, inputs: &[&TensorVal]) -> TensorVal {
    match op {
        Op::Softmax { axis } => softmax_v(&inputs[0].storage, &inputs[0].shape, *axis),
        Op::RmsNorm { axis, eps } => rmsnorm_v(&inputs[0].storage, &inputs[0].shape, *axis, *eps),
        Op::Sdpa { causal } => sdpa_v(inputs[0], inputs[1], inputs[2], *causal),
        _ => unreachable!("nn::eval: non-nn op"),
    }
}

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

// SDPA oracle: scores = (q@k^T)/sqrt(dh) [+ causal -inf upper-triangle], softmax over keys,
// then @v. Reuses the exact dot + softmax kernels the decomposition lowers to, so the fused
// primitive matches the decomposition bit-for-bit. q,k,v same shape [.., S, dh] (scores square).
pub(crate) fn sdpa_v(q: &TensorVal, k: &TensorVal, v: &TensorVal, causal: bool) -> TensorVal {
    let r = q.shape.len();
    let (s, dh) = (q.shape[r - 2], q.shape[r - 1]);
    let batch: Vec<usize> = (0..r - 2).collect();
    let raw = super::dot_dispatch(q, k, &[r - 1], &[r - 1], &batch, &batch); // q@k^T -> [.., S, S]
    let scores = scale_causal(&raw, 1.0 / (dh as f32).sqrt(), s, causal);
    let attn = softmax_v(&scores.storage, &scores.shape, r - 1); // over keys (last axis)
    super::dot_dispatch(&attn, v, &[r - 1], &[r - 2], &batch, &batch) // attn@v -> [.., S, dh]
}

// scale by 1/sqrt(dh) and apply the causal -inf mask (key j > query i), preserving dtype.
// low-precision floats compute in f32 (matching the dot/softmax accumulator); f64 native.
fn scale_causal(raw: &TensorVal, scale: f32, s: usize, causal: bool) -> TensorVal {
    let dt = raw.storage.dtype();
    let storage = match dt {
        DType::F64 => {
            let Storage::F64(v) = &raw.storage else { unreachable!() };
            Storage::F64(scale_causal_f64(v, s, scale as f64, causal))
        }
        _ => {
            let f = cast(&raw.storage, DType::F32);
            let Storage::F32(v) = &f else { unreachable!() };
            let out = Storage::F32(scale_causal_f32(v, s, scale, causal));
            if dt == DType::F32 { out } else { cast(&out, dt) }
        }
    };
    TensorVal { shape: raw.shape.clone(), storage }
}

macro_rules! scale_causal_impl {
    ($name:ident, $t:ty) => {
        fn $name(data: &[$t], s: usize, scale: $t, causal: bool) -> Vec<$t> {
            let block = s * s; // one [S, S] score matrix per batch element
            let mut out = vec![0 as $t; data.len()];
            for base in (0..data.len()).step_by(block.max(1)) {
                for i in 0..s {
                    for j in 0..s {
                        let idx = base + i * s + j;
                        out[idx] = if causal && j > i { <$t>::NEG_INFINITY } else { data[idx] * scale };
                    }
                }
            }
            out
        }
    };
}
scale_causal_impl!(scale_causal_f32, f32);
scale_causal_impl!(scale_causal_f64, f64);
