//! C ABI: graph node builders -- input / constant / scalar leaves.
use crate::capi::{KU_ERR, KuGraph, build, dtype_from_u32, raw_slice, set_err, usize_slice};
use kurumi_core::{NodeId, Storage};

/// A baked constant of `dtype` from `nbytes` of little-endian element data,
/// row-major over `shape`. Covers every dtype (ku_constant_f32 is the f32 shortcut).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_constant(
    g: *mut KuGraph,
    dtype: u32,
    data: *const u8,
    nbytes: usize,
    shape: *const usize,
    rank: usize,
) -> u32 {
    let Some(dt) = dtype_from_u32(dtype) else {
        set_err(format!("ku_constant: bad dtype {dtype}"));
        return KU_ERR;
    };
    if data.is_null() && nbytes > 0 {
        set_err("ku_constant: null data".into());
        return KU_ERR;
    }
    let storage = Storage::from_bytes(dt, raw_slice(data, nbytes));
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| Ok(gr.const_storage(storage, shape)))
}

/// An `Input` node (fed at eval via `KuFeeds`). `dtype` is a `KuDType` index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_input(g: *mut KuGraph, shape: *const usize, rank: usize, dtype: u32) -> u32 {
    let Some(dt) = dtype_from_u32(dtype) else {
        set_err(format!("ku_input: bad dtype {dtype}"));
        return KU_ERR;
    };
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| Ok(gr.input(shape, dt)))
}

/// A baked f32 constant of `len` elements laid out row-major over `shape`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_constant_f32(
    g: *mut KuGraph,
    data: *const f32,
    len: usize,
    shape: *const usize,
    rank: usize,
) -> u32 {
    if data.is_null() && len > 0 {
        set_err("ku_constant_f32: null data".into());
        return KU_ERR;
    }
    let data = raw_slice(data, len).to_vec();
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| Ok(gr.constant(data, shape)))
}

/// A scalar with the same dtype/shape-broadcastable form as `like`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_scalar(g: *mut KuGraph, like: u32, v: f32) -> u32 {
    build(g, |gr| Ok(gr.scalar(NodeId(like), v)))
}
