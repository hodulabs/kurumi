// Statistics: quantile + bias-corrected variance/std.

use crate::capi::{KuGraph, build};
use kurumi_core::NodeId;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_quantile(g: *mut KuGraph, x: u32, axis: usize, q: f32) -> u32 {
    build(g, |gr| gr.quantile(NodeId(x), axis, q))
}
macro_rules! corr {
    ($($c:ident => $m:ident),*) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32, axis: usize, correction: usize) -> u32 {
            build(g, |gr| gr.$m(NodeId(x), axis, correction))
        }
    )* };
}
corr! { ku_std_correction => std_correction, ku_var_correction => var_correction }
