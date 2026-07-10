// Scaled dot-product attention.

use crate::capi::{KuGraph, build};
use kurumi_core::NodeId;

/// Scaled dot-product attention; q,k,v are `[.., S, D]`. causal != 0 masks future keys.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_sdpa(g: *mut KuGraph, q: u32, k: u32, v: u32, causal: u32) -> u32 {
    build(g, |gr| gr.sdpa(NodeId(q), NodeId(k), NodeId(v), causal != 0))
}
