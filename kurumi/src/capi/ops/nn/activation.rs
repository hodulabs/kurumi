// Parametric / bounded activations: leaky_relu, elu, celu, clamp.

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
