//! Signal processing built on the `fft` primitive-decomposition: N-D transforms, real FFT,
//! FFT convolution, and the analytic (Hilbert) signal. All pure decompositions -> autodiff &
//! every backend for free. Window functions are in `windows`, STFT/ISTFT in `stft`.

use crate::{Error, Graph, NodeId, Storage};

mod stft;
mod windows;

impl Graph {
    /// FFT along each axis in `axes` (applied in order). Real input promotes to complex.
    pub fn fftn(&mut self, x: NodeId, axes: &[usize]) -> Result<NodeId, Error> {
        let mut cur = x;
        for &ax in axes {
            cur = self.fft(cur, ax)?;
        }
        Ok(cur)
    }
    /// Inverse FFT along each axis in `axes`.
    pub fn ifftn(&mut self, x: NodeId, axes: &[usize]) -> Result<NodeId, Error> {
        let mut cur = x;
        for &ax in axes {
            cur = self.ifft(cur, ax)?;
        }
        Ok(cur)
    }
    /// 2-D FFT over the last two axes.
    pub fn fft2(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let r = self.shape(x).len();
        if r < 2 {
            return Err(Error::shape("fft2", "needs rank >= 2"));
        }
        self.fftn(x, &[r - 2, r - 1])
    }
    /// Inverse 2-D FFT over the last two axes.
    pub fn ifft2(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let r = self.shape(x).len();
        if r < 2 {
            return Err(Error::shape("ifft2", "needs rank >= 2"));
        }
        self.ifftn(x, &[r - 2, r - 1])
    }

    /// Real FFT along `axis`: the non-redundant half of `fft(x)`, length `n/2 + 1`
    /// (Hermitian symmetry makes the rest redundant for real input).
    pub fn rfft(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        if axis >= sh.len() {
            return Err(Error::shape("rfft", "axis out of range"));
        }
        let n = sh[axis];
        let f = self.fft(x, axis)?;
        let mut ranges: Vec<(usize, usize)> = self.shape(f).iter().map(|&d| (0, d)).collect();
        ranges[axis] = (0, n / 2 + 1);
        self.slice(f, ranges)
    }

    /// Inverse real FFT along `axis`: a length-`n/2+1` complex half-spectrum back to a
    /// real length-`n` signal, rebuilding the mirror by Hermitian symmetry.
    pub fn irfft(&mut self, x: NodeId, axis: usize, n: usize) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        if axis >= sh.len() {
            return Err(Error::shape("irfft", "axis out of range"));
        }
        let m = n / 2 + 1;
        if sh[axis] != m {
            return Err(Error::shape("irfft", format!("axis length {} != n/2+1 = {m}", sh[axis])));
        }
        let n_mirror = n - m; // conj-mirror of x[1 ..= n_mirror]
        let full = if n_mirror == 0 {
            x
        } else {
            let mut ranges: Vec<(usize, usize)> = sh.iter().map(|&d| (0, d)).collect();
            ranges[axis] = (1, 1 + n_mirror);
            let mid = self.slice(x, ranges)?;
            let flipped = self.flip(mid, vec![axis])?;
            let mirror = self.conj(flipped)?;
            self.concat(&[x, mirror], axis)?
        };
        let inv = self.ifft(full, axis)?;
        self.real(inv)
    }

    /// Circular convolution along `axis` via FFT: `ifft(fft(a)*fft(b))`. Returns the
    /// real part for real inputs, else complex.
    pub fn fft_conv(&mut self, a: NodeId, b: NodeId, axis: usize) -> Result<NodeId, Error> {
        let real_in = !self.dtype(a).is_complex();
        let fa = self.fft(a, axis)?;
        let fb = self.fft(b, axis)?;
        let prod = self.mul(fa, fb)?;
        let conv = self.ifft(prod, axis)?;
        if real_in { self.real(conv) } else { Ok(conv) }
    }

    /// Analytic signal along `axis` (Hilbert transform): `ifft(fft(x)*H)` with the
    /// one-sided spectral multiplier `H`. Output is complex; its imaginary part is the
    /// Hilbert transform of `x`.
    pub fn hilbert(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        if axis >= sh.len() {
            return Err(Error::shape("hilbert", "axis out of range"));
        }
        let n = sh[axis];
        let f = self.fft(x, axis)?; // complex, shape == sh
        // H: 1 at DC (and Nyquist if n even), 2 on the positive freqs, 0 on the negative.
        let mut h = vec![0f32; n];
        h[0] = 1.0;
        if n.is_multiple_of(2) {
            h[n / 2] = 1.0;
            h.iter_mut().take(n / 2).skip(1).for_each(|v| *v = 2.0);
        } else {
            h.iter_mut().take(n.div_ceil(2)).skip(1).for_each(|v| *v = 2.0);
        }
        let cdt = self.dtype(f); // C64 or C128
        let mut wsh = vec![1usize; sh.len()];
        wsh[axis] = n;
        let hn = self.const_storage(Storage::F32(h), wsh);
        let hc = self.cast(hn, cdt); // real -> complex (imag 0)
        let hb = self.broadcast_to(hc, sh)?;
        let af = self.mul(f, hb)?;
        self.ifft(af, axis)
    }
}
