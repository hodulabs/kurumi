// Random generators (seed-based, reproducible). `ku_rand_uniform_keyed` is keyed by a
// runtime seed node instead of a build-time constant.

use crate::capi::{KuGraph, build, usize_slice};
use kurumi_core::NodeId;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_rand_uniform(g: *mut KuGraph, shape: *const usize, rank: usize, seed: u64) -> u32 {
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| Ok(gr.rand_uniform(shape, seed)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_randn(g: *mut KuGraph, shape: *const usize, rank: usize, seed: u64) -> u32 {
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| Ok(gr.randn(shape, seed)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_dropout(g: *mut KuGraph, x: u32, p: f32, seed: u64) -> u32 {
    build(g, |gr| gr.dropout(NodeId(x), p, seed))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_randint(
    g: *mut KuGraph,
    shape: *const usize,
    rank: usize,
    seed: u64,
    lo: i64,
    hi: i64,
) -> u32 {
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| Ok(gr.randint(shape, seed, lo, hi)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_rand_range(
    g: *mut KuGraph,
    shape: *const usize,
    rank: usize,
    seed: u64,
    lo: f32,
    hi: f32,
) -> u32 {
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| Ok(gr.rand_range(shape, seed, lo, hi)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_bernoulli(g: *mut KuGraph, shape: *const usize, rank: usize, seed: u64, p: f32) -> u32 {
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| gr.bernoulli(shape, seed, p))
}
/// Uniform [0,1) keyed by a runtime `seed` node (a scalar int tensor).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_rand_uniform_keyed(g: *mut KuGraph, shape: *const usize, rank: usize, seed: u32) -> u32 {
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| gr.rand_uniform_keyed(shape, NodeId(seed)))
}
