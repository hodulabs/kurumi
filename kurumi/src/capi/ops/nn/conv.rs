// Convolution wrappers (mirrors graph/ops/nn/conv.rs). stride/padding/dilation per
// spatial dim; transpose adds output_padding.

use crate::capi::{KuGraph, build};
use kurumi_core::NodeId;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_conv1d(
    g: *mut KuGraph,
    input: u32,
    weight: u32,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> u32 {
    build(g, |gr| gr.conv1d(NodeId(input), NodeId(weight), stride, padding, dilation))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_conv2d(
    g: *mut KuGraph,
    input: u32,
    weight: u32,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
) -> u32 {
    build(g, |gr| gr.conv2d(NodeId(input), NodeId(weight), (sh, sw), (ph, pw), (dh, dw)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_conv3d(
    g: *mut KuGraph,
    input: u32,
    weight: u32,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    dd: usize,
    dh: usize,
    dw: usize,
) -> u32 {
    build(g, |gr| gr.conv3d(NodeId(input), NodeId(weight), (sd, sh, sw), (pd, ph, pw), (dd, dh, dw)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_conv_transpose1d(
    g: *mut KuGraph,
    input: u32,
    weight: u32,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> u32 {
    build(g, |gr| gr.conv_transpose1d(NodeId(input), NodeId(weight), stride, padding, output_padding, dilation))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_conv_transpose2d(
    g: *mut KuGraph,
    input: u32,
    weight: u32,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    oph: usize,
    opw: usize,
    dh: usize,
    dw: usize,
) -> u32 {
    build(g, |gr| gr.conv_transpose2d(NodeId(input), NodeId(weight), (sh, sw), (ph, pw), (oph, opw), (dh, dw)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_conv_transpose3d(
    g: *mut KuGraph,
    input: u32,
    weight: u32,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    opd: usize,
    oph: usize,
    opw: usize,
    dd: usize,
    dh: usize,
    dw: usize,
) -> u32 {
    build(g, |gr| {
        gr.conv_transpose3d(NodeId(input), NodeId(weight), (sd, sh, sw), (pd, ph, pw), (opd, oph, opw), (dd, dh, dw))
    })
}
