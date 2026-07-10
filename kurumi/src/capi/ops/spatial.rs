// Spatial / vision resize + pixel-shuffle wrappers (mirrors graph/ops/core/spatial.rs).
// `cstr` (resize mode strings) comes from the parent `ops` module.

use crate::capi::ops::cstr;
use crate::capi::{KU_ERR, KuGraph, build, set_err, usize_slice};
use kurumi_core::NodeId;
use std::ffi::c_char;

/// General resize. axes/sizes are `n` usize each; `interp`/`coord` are mode strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_resize(
    g: *mut KuGraph,
    x: u32,
    axes: *const usize,
    na: usize,
    sizes: *const usize,
    ns: usize,
    interp: *const c_char,
    coord: *const c_char,
) -> u32 {
    let (Some(i), Some(c)) = (cstr(interp), cstr(coord)) else {
        set_err("ku_resize: bad interp/coord string".into());
        return KU_ERR;
    };
    let (axes, sizes) = (usize_slice(axes, na).to_vec(), usize_slice(sizes, ns).to_vec());
    build(g, |gr| gr.resize(NodeId(x), &axes, &sizes, i, c))
}
macro_rules! resize2d {
    ($($c:ident => $m:ident),*) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32, out_h: usize, out_w: usize) -> u32 {
            build(g, |gr| gr.$m(NodeId(x), out_h, out_w))
        }
    )* };
}
resize2d! { ku_resize_bilinear => resize_bilinear, ku_resize_bicubic => resize_bicubic }
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_upsample_nearest2d(g: *mut KuGraph, x: u32, factor: usize) -> u32 {
    build(g, |gr| gr.upsample_nearest2d(NodeId(x), factor))
}
macro_rules! depth {
    ($($c:ident => $m:ident),*) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32, r: usize) -> u32 {
            build(g, |gr| gr.$m(NodeId(x), r))
        }
    )* };
}
depth! { ku_space_to_depth => space_to_depth, ku_depth_to_space => depth_to_space }
