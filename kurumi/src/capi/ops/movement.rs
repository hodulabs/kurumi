// Shape / layout movement and dtype reinterpretation (no arithmetic): reshape,
// permute, slice, pad, join/split, roll, triangular masks, cast/bitcast, iota, detach.

use crate::capi::{KU_ERR, KuGraph, build, catch, dtype_from_u32, null_handles, raw_slice, set_err, usize_slice};
use kurumi_core::NodeId;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_reshape(g: *mut KuGraph, x: u32, shape: *const usize, rank: usize) -> u32 {
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| gr.reshape(NodeId(x), shape))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_expand(g: *mut KuGraph, x: u32, shape: *const usize, rank: usize) -> u32 {
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| gr.expand(NodeId(x), shape))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_permute(g: *mut KuGraph, x: u32, perm: *const usize, rank: usize) -> u32 {
    let perm = usize_slice(perm, rank).to_vec();
    build(g, |gr| gr.permute(NodeId(x), perm))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_transpose(g: *mut KuGraph, x: u32, i: usize, j: usize) -> u32 {
    build(g, |gr| gr.transpose(NodeId(x), i, j))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_flip(g: *mut KuGraph, x: u32, axes: *const usize, n: usize) -> u32 {
    let axes = usize_slice(axes, n).to_vec();
    build(g, |gr| gr.flip(NodeId(x), axes))
}
/// Slice per axis: `ranges` is `2*rank` usize (start,end pairs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_slice(g: *mut KuGraph, x: u32, ranges: *const usize, rank: usize) -> u32 {
    let r = usize_slice(ranges, rank * 2);
    let ranges: Vec<(usize, usize)> = (0..rank).map(|i| (r[2 * i], r[2 * i + 1])).collect();
    build(g, |gr| gr.slice(NodeId(x), ranges))
}
/// Strided slice: `ranges` is `3*rank` usize (start, end, step triples).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_slice_step(g: *mut KuGraph, x: u32, ranges: *const usize, rank: usize) -> u32 {
    let r = usize_slice(ranges, rank * 3);
    let ranges: Vec<(usize, usize, usize)> = (0..rank).map(|i| (r[3 * i], r[3 * i + 1], r[3 * i + 2])).collect();
    build(g, |gr| gr.slice_step(NodeId(x), ranges))
}
/// Pad per axis: `pads` is `2*rank` usize (lo,hi pairs), value 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_pad(g: *mut KuGraph, x: u32, pads: *const usize, rank: usize) -> u32 {
    let p = usize_slice(pads, rank * 2);
    let pads: Vec<(usize, usize)> = (0..rank).map(|i| (p[2 * i], p[2 * i + 1])).collect();
    build(g, |gr| gr.pad(NodeId(x), pads))
}
/// Concat `parts` (n node ids) along `axis`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_concat(g: *mut KuGraph, parts: *const u32, n: usize, axis: usize) -> u32 {
    let parts: Vec<NodeId> = raw_slice(parts, n).iter().map(|&i| NodeId(i)).collect();
    build(g, |gr| gr.concat(&parts, axis))
}
/// Stack `parts` (n node ids) along a new `axis`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_stack(g: *mut KuGraph, parts: *const u32, n: usize, axis: usize) -> u32 {
    let parts: Vec<NodeId> = raw_slice(parts, n).iter().map(|&i| NodeId(i)).collect();
    build(g, |gr| gr.stack(&parts, axis))
}
/// Split `x` into `n` pieces of `sizes` along `axis`; writes piece ids to `out`
/// (must hold `n` u32). Returns 0 on success, KU_ERR on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_split(
    g: *mut KuGraph,
    x: u32,
    sizes: *const usize,
    n: usize,
    axis: usize,
    out: *mut u32,
) -> u32 {
    if null_handles(g, out) {
        return KU_ERR;
    }
    let sizes = usize_slice(sizes, n).to_vec();
    let gr = &mut (*g).0;
    catch(KU_ERR, || match gr.split(NodeId(x), &sizes, axis) {
        Ok(v) => unsafe {
            for (i, node) in v.iter().enumerate() {
                *out.add(i) = node.0;
            }
            0
        },
        Err(e) => {
            set_err(format!("{e:?}"));
            KU_ERR
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_tile(g: *mut KuGraph, x: u32, axis: usize, n: usize) -> u32 {
    build(g, |gr| gr.tile(NodeId(x), axis, n))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_repeat_interleave(g: *mut KuGraph, x: u32, axis: usize, n: usize) -> u32 {
    build(g, |gr| gr.repeat_interleave(NodeId(x), axis, n))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_roll(g: *mut KuGraph, x: u32, shift: usize, axis: usize) -> u32 {
    build(g, |gr| gr.roll(NodeId(x), shift, axis))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_broadcast_to(g: *mut KuGraph, x: u32, shape: *const usize, rank: usize) -> u32 {
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| gr.broadcast_to(NodeId(x), shape))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_tril(g: *mut KuGraph, x: u32, diagonal: i64) -> u32 {
    build(g, |gr| gr.tril(NodeId(x), diagonal))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_triu(g: *mut KuGraph, x: u32, diagonal: i64) -> u32 {
    build(g, |gr| gr.triu(NodeId(x), diagonal))
}
/// Identity forward that stops the gradient.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_detach(g: *mut KuGraph, x: u32) -> u32 {
    build(g, |gr| Ok(gr.detach(NodeId(x))))
}
/// Index-along-`axis` tensor of `shape` and `dtype`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_iota(g: *mut KuGraph, shape: *const usize, rank: usize, axis: usize, dtype: u32) -> u32 {
    let Some(dt) = dtype_from_u32(dtype) else {
        set_err(format!("ku_iota: bad dtype {dtype}"));
        return KU_ERR;
    };
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| gr.iota(shape, axis, dt))
}
/// Cast to a KuDType index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_cast(g: *mut KuGraph, x: u32, dtype: u32) -> u32 {
    let Some(dt) = dtype_from_u32(dtype) else {
        set_err(format!("ku_cast: bad dtype {dtype}"));
        return KU_ERR;
    };
    build(g, |gr| Ok(gr.cast(NodeId(x), dt)))
}
/// Reinterpret the bits as `dtype` (same width, no value change).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_bitcast(g: *mut KuGraph, x: u32, dtype: u32) -> u32 {
    let Some(dt) = dtype_from_u32(dtype) else {
        set_err(format!("ku_bitcast: bad dtype {dtype}"));
        return KU_ERR;
    };
    build(g, |gr| gr.bitcast(NodeId(x), dt))
}
