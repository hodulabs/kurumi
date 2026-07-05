// Gather / scatter / sort / mask-based indexing. `scatter_op` (the combiner decode)
// comes from the parent module.

use super::scatter_op;
use crate::capi::{KU_ERR, KuGraph, build, set_err};
use kurumi_core::NodeId;

/// select: cond ? a : b (cond BOOL; a/b same dtype/shape).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_where(g: *mut KuGraph, cond: u32, a: u32, b: u32) -> u32 {
    build(g, |gr| gr.select(NodeId(cond), NodeId(a), NodeId(b)))
}

macro_rules! idx {
    ($($c:ident => $m:ident),* $(,)?) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32, idx: u32, axis: usize) -> u32 {
            build(g, |gr| gr.$m(NodeId(x), NodeId(idx), axis))
        }
    )* };
}
idx! { ku_gather => gather, ku_gather_along => gather_along, ku_take_along_dim => take_along_dim }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_onehot(g: *mut KuGraph, idx: u32, num_classes: usize) -> u32 {
    build(g, |gr| gr.onehot(NodeId(idx), num_classes))
}

// sort / argsort (axis + descending flag)
macro_rules! sort_op {
    ($($c:ident => $m:ident),* $(,)?) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32, axis: usize, descending: u32) -> u32 {
            build(g, |gr| gr.$m(NodeId(x), axis, descending != 0))
        }
    )* };
}
sort_op! { ku_sort => sort, ku_argsort => argsort }

// scatter family. combine: 0=set, 1=add, 2=max, 3=min.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_scatter(
    g: *mut KuGraph,
    operand: u32,
    indices: u32,
    updates: u32,
    axis: usize,
    combine: u32,
) -> u32 {
    let Some(c) = scatter_op(combine) else {
        set_err(format!("ku_scatter: bad combine {combine}"));
        return KU_ERR;
    };
    build(g, |gr| gr.scatter(NodeId(operand), NodeId(indices), NodeId(updates), axis, c))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_scatter_along(
    g: *mut KuGraph,
    operand: u32,
    indices: u32,
    updates: u32,
    axis: usize,
    combine: u32,
) -> u32 {
    let Some(c) = scatter_op(combine) else {
        set_err(format!("ku_scatter_along: bad combine {combine}"));
        return KU_ERR;
    };
    build(g, |gr| gr.scatter_along(NodeId(operand), NodeId(indices), NodeId(updates), axis, c))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_scatter_nd(g: *mut KuGraph, x: u32, idx: u32, updates: u32, combine: u32) -> u32 {
    let Some(c) = scatter_op(combine) else {
        set_err(format!("ku_scatter_nd: bad combine {combine}"));
        return KU_ERR;
    };
    build(g, |gr| gr.scatter_nd(NodeId(x), NodeId(idx), NodeId(updates), c))
}

// masking + dynamic-shape indexing. masked_select/compress/nonzero/unique take an upper-bound
// `k` (the static output length; the IR has static shapes, so the frontend picks a cap).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_masked_fill(g: *mut KuGraph, x: u32, mask: u32, value: f32) -> u32 {
    build(g, |gr| gr.masked_fill(NodeId(x), NodeId(mask), value))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_masked_select(g: *mut KuGraph, x: u32, mask: u32, k: usize) -> u32 {
    build(g, |gr| gr.masked_select(NodeId(x), NodeId(mask), k))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_compress(g: *mut KuGraph, mask: u32, x: u32, k: usize) -> u32 {
    build(g, |gr| gr.compress(NodeId(mask), NodeId(x), k))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_nonzero(g: *mut KuGraph, x: u32, k: usize) -> u32 {
    build(g, |gr| gr.nonzero(NodeId(x), k))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_unique(g: *mut KuGraph, x: u32, k: usize) -> u32 {
    build(g, |gr| gr.unique(NodeId(x), k))
}
