//! C ABI: tensor handles & raw tensor IO (f32 and per-dtype bytes; no DLPack yet).

use crate::capi::{KuTensor, dtype_from_u32, raw_slice, set_err, usize_slice};
use kurumi_core::{Storage, TensorVal};
use std::ptr;

/// Build an input tensor of `dtype` from `nbytes` of little-endian element data,
/// row-major over `shape` (every dtype; ku_tensor_f32 is the f32 shortcut).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_tensor_new(
    dtype: u32,
    data: *const u8,
    nbytes: usize,
    shape: *const usize,
    rank: usize,
) -> *mut KuTensor {
    let Some(dt) = dtype_from_u32(dtype) else {
        set_err(format!("ku_tensor_new: bad dtype {dtype}"));
        return ptr::null_mut();
    };
    if data.is_null() && nbytes > 0 {
        set_err("ku_tensor_new: null data".into());
        return ptr::null_mut();
    }
    let storage = Storage::from_bytes(dt, raw_slice(data, nbytes));
    Box::into_raw(Box::new(KuTensor(TensorVal { shape: usize_slice(shape, rank).to_vec(), storage })))
}

/// Byte size of the tensor's storage (the buffer to allocate for ku_tensor_bytes).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_tensor_nbytes(t: *const KuTensor) -> usize {
    if t.is_null() {
        return 0;
    }
    (*t).0.storage.to_bytes().len()
}

/// Copy the tensor's raw little-endian bytes into `out` (>= ku_tensor_nbytes). Any
/// dtype; read ku_tensor_dtype to interpret them.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_tensor_bytes(t: *const KuTensor, out: *mut u8) {
    if t.is_null() || out.is_null() {
        return;
    }
    let bytes = (*t).0.storage.to_bytes();
    ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
}

/// Build an f32 input tensor (row-major over `shape`) to bind via `ku_feeds_set`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_tensor_f32(
    data: *const f32,
    len: usize,
    shape: *const usize,
    rank: usize,
) -> *mut KuTensor {
    if data.is_null() && len > 0 {
        set_err("ku_tensor_f32: null data".into());
        return ptr::null_mut();
    }
    let storage = Storage::F32(raw_slice(data, len).to_vec());
    let shape = usize_slice(shape, rank).to_vec();
    Box::into_raw(Box::new(KuTensor(TensorVal { shape, storage })))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_tensor_free(t: *mut KuTensor) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_tensor_rank(t: *const KuTensor) -> usize {
    if t.is_null() {
        return 0;
    }
    (*t).0.shape.len()
}
/// Write the `rank` dimensions into `out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_tensor_shape(t: *const KuTensor, out: *mut usize) {
    if t.is_null() || out.is_null() {
        return;
    }
    for (i, &d) in (*t).0.shape.iter().enumerate() {
        *out.add(i) = d;
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_tensor_len(t: *const KuTensor) -> usize {
    if t.is_null() {
        return 0;
    }
    (*t).0.storage.len()
}
/// The tensor's dtype as a `KuDType` index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_tensor_dtype(t: *const KuTensor) -> u32 {
    if t.is_null() {
        return 0;
    }
    (*t).0.storage.dtype() as u32
}
/// Copy up to `cap` f32 elements into `out`; returns the count written, or -1 if the
/// tensor is not F32 (or `t`/`out` is null). For any other dtype use ku_tensor_bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_tensor_data_f32(t: *const KuTensor, out: *mut f32, cap: usize) -> isize {
    if t.is_null() || out.is_null() {
        return -1;
    }
    match &(*t).0.storage {
        Storage::F32(v) => {
            let n = v.len().min(cap);
            ptr::copy_nonoverlapping(v.as_ptr(), out, n);
            n as isize
        }
        s => {
            set_err(format!("ku_tensor_data_f32: tensor is {:?}, not F32", s.dtype()));
            -1
        }
    }
}
