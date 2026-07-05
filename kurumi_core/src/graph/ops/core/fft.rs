//! Discrete Fourier transform via the DFT matrix (complex matmul). Pure decomposition
//! (autodiff & every backend for free). O(n^2): materializes an [n,n] complex matrix;
//! swap in a Cooley-Tukey O(n log n) primitive if long transforms run hot.

use crate::{DType, Error, Graph, NodeId, Storage};

impl Graph {
    /// 1-D DFT along `axis`: `X[j] = sum_k x[k]*e^{-2pi i kj/n}`. Real input is promoted
    /// to complex (C64/C128). Output is complex; differentiable.
    pub fn fft(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        self.dft(x, axis, false)
    }

    /// Inverse 1-D DFT along `axis`: `x[k] = (1/n) sum_j X[j]*e^{+2pi i kj/n}`.
    pub fn ifft(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        self.dft(x, axis, true)
    }

    fn dft(&mut self, x: NodeId, axis: usize, inverse: bool) -> Result<NodeId, Error> {
        let dt = self.dtype(x);
        // promote a real signal to complex (imag 0)
        let (x, cdt) = if dt.is_complex() {
            (x, dt)
        } else if dt == DType::F64 {
            (self.cast(x, DType::C128), DType::C128)
        } else {
            (self.cast(x, DType::C64), DType::C64)
        };
        let sh = self.shape(x);
        if axis >= sh.len() {
            return Err(Error::shape("fft", "axis out of range"));
        }
        let n = sh[axis];
        // DFT matrix W[k,j] = e^{-/+ 2pi i kj/n} (*1/n for the inverse). Symmetric in k,j.
        let sign = if inverse { 1.0 } else { -1.0 };
        let scale = if inverse { 1.0 / n as f64 } else { 1.0 };
        let (mut re, mut im) = (vec![0f64; n * n], vec![0f64; n * n]);
        for k in 0..n {
            for j in 0..n {
                let ang = sign * 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                re[k * n + j] = ang.cos() * scale;
                im[k * n + j] = ang.sin() * scale;
            }
        }
        let rdt = cdt.real();
        let mk = |v: Vec<f64>| -> Storage {
            if rdt == DType::F64 { Storage::F64(v) } else { Storage::F32(v.iter().map(|&x| x as f32).collect()) }
        };
        let wr = self.const_storage(mk(re), vec![n, n]);
        let wi = self.const_storage(mk(im), vec![n, n]);
        let w = self.complex(wr, wi)?;
        // contract W with x along `axis` (move to last, dot_general, move back).
        let last = sh.len() - 1;
        let xt = if axis == last { x } else { self.transpose(x, axis, last)? };
        let out = self.dot_general(xt, w, vec![last], vec![0], vec![], vec![])?;
        if axis == last { Ok(out) } else { self.transpose(out, axis, last) }
    }
}
