// Neural-net layers: parametric activations, normalizations, losses, and attention.
// Convolution and pooling are the `conv`/`pool` submodules.

pub(crate) mod conv;
pub(crate) mod pool;

use crate::capi::{KuGraph, build};
use kurumi_core::NodeId;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_leaky_relu(g: *mut KuGraph, x: u32, slope: f32) -> u32 {
    build(g, |gr| Ok(gr.leaky_relu(NodeId(x), slope)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_elu(g: *mut KuGraph, x: u32, alpha: f32) -> u32 {
    build(g, |gr| Ok(gr.elu(NodeId(x), alpha)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_celu(g: *mut KuGraph, x: u32, alpha: f32) -> u32 {
    build(g, |gr| Ok(gr.celu(NodeId(x), alpha)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_clamp(g: *mut KuGraph, x: u32, lo: f32, hi: f32) -> u32 {
    build(g, |gr| gr.clamp(NodeId(x), lo, hi))
}

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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_cross_entropy(g: *mut KuGraph, logits: u32, targets: u32, axis: usize) -> u32 {
    build(g, |gr| gr.cross_entropy(NodeId(logits), NodeId(targets), axis))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_nll_loss(g: *mut KuGraph, log_probs: u32, target: u32, axis: usize) -> u32 {
    build(g, |gr| gr.nll_loss(NodeId(log_probs), NodeId(target), axis))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_huber_loss(g: *mut KuGraph, pred: u32, target: u32, delta: f32) -> u32 {
    build(g, |gr| gr.huber_loss(NodeId(pred), NodeId(target), delta))
}
/// Scaled dot-product attention; q,k,v are `[.., S, D]`. causal != 0 masks future keys.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_sdpa(g: *mut KuGraph, q: u32, k: u32, v: u32, causal: u32) -> u32 {
    build(g, |gr| gr.sdpa(NodeId(q), NodeId(k), NodeId(v), causal != 0))
}
