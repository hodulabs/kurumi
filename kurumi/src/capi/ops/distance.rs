// Distance / similarity wrappers (mirrors graph/ops/core/distance.rs).

use crate::capi::{KuGraph, build};
use kurumi_core::NodeId;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_cdist(g: *mut KuGraph, a: u32, b: u32, p: f32) -> u32 {
    build(g, |gr| gr.cdist(NodeId(a), NodeId(b), p))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_pdist(g: *mut KuGraph, a: u32, p: f32) -> u32 {
    build(g, |gr| gr.pdist(NodeId(a), p))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_cosine_similarity(g: *mut KuGraph, a: u32, b: u32, axis: usize) -> u32 {
    build(g, |gr| gr.cosine_similarity(NodeId(a), NodeId(b), axis))
}
