//! Short-time Fourier transform + its inverse (overlap-add). Pure decompositions over the
//! `fft` primitive; the window functions they apply are in the sibling `windows`.

use crate::{Error, Graph, NodeId};

impl Graph {
    /// Short-time FFT over the last axis: frame the length-`L` signal into
    /// `(L - n_fft)/hop + 1` windows of `n_fft`, optionally multiply by `window`
    /// (length `n_fft`), and FFT each frame -> `[.., n_frames, n_fft]` complex.
    pub fn stft(&mut self, x: NodeId, n_fft: usize, hop: usize, window: Option<NodeId>) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        let last = sh.len() - 1;
        let l = sh[last];
        if n_fft == 0 || hop == 0 || l < n_fft {
            return Err(Error::shape("stft", "need n_fft>0, hop>0, signal length >= n_fft"));
        }
        let n_frames = (l - n_fft) / hop + 1;
        // window (if any) reshaped/broadcast to a frame's shape [.., n_fft], once.
        let mut fsh = sh.clone();
        fsh[last] = n_fft;
        let win = match window {
            Some(w) => {
                let mut wsh = vec![1usize; sh.len()];
                wsh[last] = n_fft;
                let wr = self.reshape(w, wsh)?;
                Some(self.broadcast_to(wr, fsh.clone())?)
            }
            None => None,
        };
        let mut frames = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let mut ranges: Vec<(usize, usize)> = sh.iter().map(|&d| (0, d)).collect();
            ranges[last] = (i * hop, i * hop + n_fft);
            let frame = self.slice(x, ranges)?;
            let frame = match win {
                Some(w) => self.mul(frame, w)?,
                None => frame,
            };
            frames.push(frame);
        }
        // [.., n_frames, n_fft], FFT the last (n_fft) axis
        let stacked = self.stack(&frames, last)?;
        self.fft(stacked, last + 1)
    }

    /// Inverse STFT (overlap-add) of complex `[.., n_frames, n_fft]` -> real `[.., L]`,
    /// `L = (n_frames-1)*hop + n_fft`. Applies the synthesis `window` and normalizes by
    /// the overlap-added window^2, so `istft(stft(x, w), w)` reconstructs `x` (COLA).
    pub fn istft(&mut self, frames: NodeId, hop: usize, window: Option<NodeId>) -> Result<NodeId, Error> {
        let sh = self.shape(frames);
        let r = sh.len();
        if r < 2 || hop == 0 {
            return Err(Error::shape("istft", "expects [.., n_frames, n_fft] and hop>0"));
        }
        let (nf, n_fft) = (sh[r - 2], sh[r - 1]);
        let l = (nf - 1) * hop + n_fft;
        let time = {
            let inv = self.ifft(frames, r - 1)?;
            self.real(inv)?
        }; // [.., n_frames, n_fft]
        // synthesis window (broadcast over frames) and window^2 for the denominator
        let mut wsh = vec![1usize; r];
        wsh[r - 1] = n_fft;
        let (time, w2) = match window {
            Some(w) => {
                let wr = self.reshape(w, wsh.clone())?;
                let wb = self.broadcast_to(wr, self.shape(time))?;
                let t = self.mul(time, wb)?;
                let w2 = self.mul(w, w)?; // [n_fft]
                (t, w2)
            }
            None => (time, self.constant(vec![1.0; n_fft], vec![n_fft])),
        };
        // overlap-add a per-frame [.., n_fft] slab into [.., L] at offset i*hop
        let mut num: Option<NodeId> = None;
        for i in 0..nf {
            let mut ranges: Vec<(usize, usize)> = self.shape(time).iter().map(|&d| (0, d)).collect();
            ranges[r - 2] = (i, i + 1);
            let fr = self.slice(time, ranges)?;
            let fr = self.squeeze(fr, r - 2)?; // [.., n_fft]
            let mut pads = vec![(0, 0); r - 1];
            pads[r - 2] = (i * hop, l - i * hop - n_fft);
            let padded = self.pad(fr, pads)?; // [.., L]
            num = Some(match num {
                None => padded,
                Some(a) => self.add(a, padded)?,
            });
        }
        let num = num.ok_or_else(|| Error::shape("istft", "no frames"))?;
        // denominator: overlap-added window^2 -> [L], broadcast + divide (eps guards gaps)
        let mut den: Option<NodeId> = None;
        for i in 0..nf {
            let padded = self.pad(w2, vec![(i * hop, l - i * hop - n_fft)])?; // [L]
            den = Some(match den {
                None => padded,
                Some(a) => self.add(a, padded)?,
            });
        }
        let den = den.unwrap();
        let eps = self.scalar(den, 1e-8);
        let den = self.add(den, eps)?; // [L]
        let numsh = self.shape(num);
        let mut dsh = vec![1usize; numsh.len()];
        *dsh.last_mut().unwrap() = l;
        let denr = self.reshape(den, dsh)?; // [1.., L]
        let denb = self.broadcast_to(denr, numsh)?;
        self.div(num, denb)
    }
}
