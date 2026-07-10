// Pooling wrappers (mirrors graph/ops/nn/pool.rs). k/s (and per-axis for 2-D/3-D)
// window/stride, no padding. `cstr` (reduce_window mode) comes from the `ops` module.

use crate::capi::ops::cstr;
use crate::capi::{KU_ERR, KuGraph, build, set_err, usize_slice};
use kurumi_core::NodeId;
use std::ffi::c_char;

macro_rules! pool1d {
    ($($c:ident => $m:ident),*) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32, k: usize, s: usize) -> u32 {
            build(g, |gr| gr.$m(NodeId(x), k, s))
        }
    )* };
}
pool1d! { ku_max_pool1d => max_pool1d, ku_avg_pool1d => avg_pool1d, ku_min_pool1d => min_pool1d, ku_sum_pool1d => sum_pool1d }
macro_rules! pool2d {
    ($($c:ident => $m:ident),*) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32, kh: usize, kw: usize, sh: usize, sw: usize) -> u32 {
            build(g, |gr| gr.$m(NodeId(x), (kh, kw), (sh, sw)))
        }
    )* };
}
pool2d! { ku_max_pool2d => max_pool2d, ku_avg_pool2d => avg_pool2d, ku_min_pool2d => min_pool2d, ku_sum_pool2d => sum_pool2d }
macro_rules! pool3d {
    ($($c:ident => $m:ident),*) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32, kd: usize, kh: usize, kw: usize, sd: usize, sh: usize, sw: usize) -> u32 {
            build(g, |gr| gr.$m(NodeId(x), (kd, kh, kw), (sd, sh, sw)))
        }
    )* };
}
pool3d! { ku_max_pool3d => max_pool3d, ku_avg_pool3d => avg_pool3d, ku_min_pool3d => min_pool3d, ku_sum_pool3d => sum_pool3d }
/// N-d windowed reduction. window/stride/dilation are `nd` usize each; `mode` = "max"|"sum"|"avg".
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_reduce_window(
    g: *mut KuGraph,
    x: u32,
    window: *const usize,
    nw: usize,
    stride: *const usize,
    ns: usize,
    dilation: *const usize,
    nd: usize,
    mode: *const c_char,
) -> u32 {
    let Some(m) = cstr(mode) else {
        set_err("ku_reduce_window: bad mode string".into());
        return KU_ERR;
    };
    let (window, stride, dilation) =
        (usize_slice(window, nw).to_vec(), usize_slice(stride, ns).to_vec(), usize_slice(dilation, nd).to_vec());
    build(g, |gr| gr.reduce_window(NodeId(x), &window, &stride, &dilation, m))
}
