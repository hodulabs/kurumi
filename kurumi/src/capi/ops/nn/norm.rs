// Normalization layers: layernorm/rmsnorm (the `norm!` macro), group/instance norm, LRN, norm_p.

use crate::capi::{KuGraph, build};
use kurumi_core::NodeId;

macro_rules! norm {
    ($($c:ident => $m:ident),* $(,)?) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32, axis: usize, eps: f32) -> u32 {
            build(g, |gr| gr.$m(NodeId(x), axis, eps))
        }
    )* };
}
norm! { ku_layernorm => layernorm, ku_rmsnorm => rmsnorm }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_group_norm(g: *mut KuGraph, x: u32, groups: usize, eps: f32) -> u32 {
    build(g, |gr| gr.group_norm(NodeId(x), groups, eps))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_instance_norm(g: *mut KuGraph, x: u32, eps: f32) -> u32 {
    build(g, |gr| gr.instance_norm(NodeId(x), eps))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_lrn(g: *mut KuGraph, x: u32, size: usize, alpha: f32, beta: f32, k: f32) -> u32 {
    build(g, |gr| gr.lrn(NodeId(x), size, alpha, beta, k))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_norm_p(g: *mut KuGraph, x: u32, p: f32, axis: usize) -> u32 {
    build(g, |gr| gr.norm_p(NodeId(x), p, axis))
}
