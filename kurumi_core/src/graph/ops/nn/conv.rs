//! Convolution & transposed convolution, decomposed from strided-slice + dot_general. No
//! conv/im2col primitive: each kernel offset is one strided-slice window, and autodiff
//! gives the conv backward + weight-gradient for free. (An im2col fast-op could replace
//! this if perf demands; semantics stay identical.) Direct conv is in `conv/direct.rs`,
//! transposed conv in `conv/transpose.rs`.

mod direct;
mod transpose;
