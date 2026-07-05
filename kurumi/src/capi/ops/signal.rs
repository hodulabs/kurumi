// Signal/spectral ops: N-d FFTs, real-inverse FFT, FFT convolution, STFT/ISTFT, and
// window generators.

use crate::capi::{KU_ERR, KuGraph, build, usize_slice};
use kurumi_core::NodeId;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_fftn(g: *mut KuGraph, x: u32, axes: *const usize, n: usize) -> u32 {
    let axes = usize_slice(axes, n).to_vec();
    build(g, |gr| gr.fftn(NodeId(x), &axes))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_ifftn(g: *mut KuGraph, x: u32, axes: *const usize, n: usize) -> u32 {
    let axes = usize_slice(axes, n).to_vec();
    build(g, |gr| gr.ifftn(NodeId(x), &axes))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_irfft(g: *mut KuGraph, x: u32, axis: usize, n: usize) -> u32 {
    build(g, |gr| gr.irfft(NodeId(x), axis, n))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_fft_conv(g: *mut KuGraph, a: u32, b: u32, axis: usize) -> u32 {
    build(g, |gr| gr.fft_conv(NodeId(a), NodeId(b), axis))
}
/// STFT; pass `window = KU_ERR` for no window.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_stft(g: *mut KuGraph, x: u32, n_fft: usize, hop: usize, window: u32) -> u32 {
    let w = (window != KU_ERR).then_some(NodeId(window));
    build(g, |gr| gr.stft(NodeId(x), n_fft, hop, w))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_istft(g: *mut KuGraph, frames: u32, hop: usize, window: u32) -> u32 {
    let w = (window != KU_ERR).then_some(NodeId(window));
    build(g, |gr| gr.istft(NodeId(frames), hop, w))
}
macro_rules! window {
    ($($c:ident => $m:ident),*) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, n: usize) -> u32 {
            build(g, |gr| Ok(gr.$m(n)))
        }
    )* };
}
window! { ku_hann_window => hann_window, ku_hamming_window => hamming_window, ku_blackman_window => blackman_window, ku_bartlett_window => bartlett_window }
