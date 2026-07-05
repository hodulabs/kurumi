//! Argsort kernel: per-line stable sort of indices along an axis (backs sort/topk).

use crate::{Storage, TensorVal, inc, row_major_strides};

/// Indices (I64) that sort `data` along `axis` (stable, ascending or descending).
/// Output has the same shape as the input (a per-line permutation).
pub(crate) fn argsort(tv: &TensorVal, axis: usize, descending: bool) -> TensorVal {
    match &tv.storage {
        Storage::F32(v) => argsort_t(v, &tv.shape, axis, descending),
        Storage::F64(v) => argsort_t(v, &tv.shape, axis, descending),
        Storage::F16(v) => argsort_t(v, &tv.shape, axis, descending),
        Storage::BF16(v) => argsort_t(v, &tv.shape, axis, descending),
        Storage::I32(v) => argsort_t(v, &tv.shape, axis, descending),
        Storage::I64(v) => argsort_t(v, &tv.shape, axis, descending),
        Storage::U32(v) => argsort_t(v, &tv.shape, axis, descending),
        Storage::U8(v) => argsort_t(v, &tv.shape, axis, descending),
        s => panic!("argsort on non-orderable dtype {:?}", s.dtype()),
    }
}

fn argsort_t<T: Copy + PartialOrd>(data: &[T], shape: &[usize], axis: usize, descending: bool) -> TensorVal {
    let strides = row_major_strides(shape);
    let (axis_len, axis_stride) = (shape[axis], strides[axis]);
    let out_shape: Vec<usize> = shape.iter().enumerate().filter_map(|(i, &d)| (i != axis).then_some(d)).collect();
    let base_strides: Vec<usize> = strides.iter().enumerate().filter_map(|(i, &s)| (i != axis).then_some(s)).collect();
    let mut out = vec![0i64; shape.iter().product::<usize>().max(1)];
    let outer: usize = out_shape.iter().product::<usize>().max(1);
    let mut coord = vec![0usize; out_shape.len()];
    for _ in 0..outer {
        let base: usize = coord.iter().zip(&base_strides).map(|(c, s)| c * s).sum();
        let mut perm: Vec<usize> = (0..axis_len).collect();
        // stable sort; ties keep the lower original index
        perm.sort_by(|&i, &j| {
            let (vi, vj) = (data[base + i * axis_stride], data[base + j * axis_stride]);
            let ord = vi.partial_cmp(&vj).unwrap_or(std::cmp::Ordering::Equal);
            if descending { ord.reverse() } else { ord }
        });
        for (rank, &idx) in perm.iter().enumerate() {
            out[base + rank * axis_stride] = idx as i64;
        }
        inc(&mut coord, &out_shape);
    }
    TensorVal { shape: shape.to_vec(), storage: Storage::I64(out) }
}
