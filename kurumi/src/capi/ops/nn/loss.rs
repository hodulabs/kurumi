// Classification / regression losses: cross-entropy, NLL, Huber.

use crate::capi::{KuGraph, build};
use kurumi_core::NodeId;

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
