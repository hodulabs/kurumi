//! Direct convolution (1/2/3-D): each kernel offset is one strided-slice window fed to
//! dot_general; autodiff gives the backward + weight-gradient for free.

use crate::{Error, Graph, NodeId};

impl Graph {
    /// 1-D convolution. `input` `[N, C, L]`, `weight` `[O, C, K]` -> `[N, O, Lo]`.
    pub fn conv1d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<NodeId, Error> {
        let ish = self.shape(input);
        let wsh = self.shape(weight);
        let (n, c, l) = (ish[0], ish[1], ish[2]);
        let (o, k) = (wsh[0], wsh[2]);
        let lo = (l + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
        let padded = self.pad(input, vec![(0, 0), (0, 0), (padding, padding)])?;
        let mut acc: Option<NodeId> = None;
        for ki in 0..k {
            let ls = ki * dilation;
            let window =
                self.slice_step(padded, vec![(0, n, 1), (0, c, 1), (ls, ls + (lo - 1) * stride + 1, stride)])?;
            let wsl = self.slice(weight, vec![(0, o), (0, c), (ki, ki + 1)])?;
            let wsl = self.reshape(wsl, vec![o, c])?;
            let term = self.dot_general(window, wsl, vec![1], vec![1], vec![], vec![])?; // [N, Lo, O]
            let term = self.permute(term, vec![0, 2, 1])?; // [N, O, Lo]
            acc = Some(match acc {
                None => term,
                Some(a) => self.add(a, term)?,
            });
        }
        acc.ok_or_else(|| Error::shape("conv1d", "empty kernel"))
    }

    /// 2-D convolution. `input` `[N, C, H, W]`, `weight` `[O, C, Kh, Kw]` ->
    /// `[N, O, Ho, Wo]`. `stride`/`padding`/`dilation` are `(h, w)` pairs.
    pub fn conv2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<NodeId, Error> {
        let ish = self.shape(input);
        let wsh = self.shape(weight);
        let (n, c, h, w) = (ish[0], ish[1], ish[2], ish[3]);
        let (o, kh, kw) = (wsh[0], wsh[2], wsh[3]);
        let ((sh, sw), (ph, pw), (dh, dw)) = (stride, padding, dilation);
        let ho = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        let wo = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
        let padded = self.pad(input, vec![(0, 0), (0, 0), (ph, ph), (pw, pw)])?;
        let mut acc: Option<NodeId> = None;
        for ki in 0..kh {
            for kj in 0..kw {
                let (hs, ws) = (ki * dh, kj * dw);
                let window = self.slice_step(
                    padded,
                    vec![(0, n, 1), (0, c, 1), (hs, hs + (ho - 1) * sh + 1, sh), (ws, ws + (wo - 1) * sw + 1, sw)],
                )?; // [N, C, Ho, Wo]
                let wsl = self.slice(weight, vec![(0, o), (0, c), (ki, ki + 1), (kj, kj + 1)])?;
                let wsl = self.reshape(wsl, vec![o, c])?; // [O, C]
                // contract C -> [N, Ho, Wo, O], then -> [N, O, Ho, Wo]
                let term = self.dot_general(window, wsl, vec![1], vec![1], vec![], vec![])?;
                let term = self.permute(term, vec![0, 3, 1, 2])?;
                acc = Some(match acc {
                    None => term,
                    Some(a) => self.add(a, term)?,
                });
            }
        }
        acc.ok_or_else(|| Error::shape("conv2d", "empty kernel"))
    }

    /// 3-D convolution. `input` `[N,C,D,H,W]`, `weight` `[O,C,Kd,Kh,Kw]` ->
    /// `[N,O,Do,Ho,Wo]`. `stride`/`padding`/`dilation` are `(d,h,w)` triples.
    pub fn conv3d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
    ) -> Result<NodeId, Error> {
        let ish = self.shape(input);
        let wsh = self.shape(weight);
        let (n, c) = (ish[0], ish[1]);
        let (dd, hh, ww) = (ish[2], ish[3], ish[4]);
        let (o, kd, kh, kw) = (wsh[0], wsh[2], wsh[3], wsh[4]);
        let ((sd, sh, sw), (pd, ph, pw), (dld, dlh, dlw)) = (stride, padding, dilation);
        let dout = (dd + 2 * pd - dld * (kd - 1) - 1) / sd + 1;
        let hout = (hh + 2 * ph - dlh * (kh - 1) - 1) / sh + 1;
        let wout = (ww + 2 * pw - dlw * (kw - 1) - 1) / sw + 1;
        let padded = self.pad(input, vec![(0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)])?;
        let mut acc: Option<NodeId> = None;
        for ka in 0..kd {
            for ki in 0..kh {
                for kj in 0..kw {
                    let (ds, hs, ws) = (ka * dld, ki * dlh, kj * dlw);
                    let window = self.slice_step(
                        padded,
                        vec![
                            (0, n, 1),
                            (0, c, 1),
                            (ds, ds + (dout - 1) * sd + 1, sd),
                            (hs, hs + (hout - 1) * sh + 1, sh),
                            (ws, ws + (wout - 1) * sw + 1, sw),
                        ],
                    )?; // [N,C,Do,Ho,Wo]
                    let wsl = self.slice(weight, vec![(0, o), (0, c), (ka, ka + 1), (ki, ki + 1), (kj, kj + 1)])?;
                    let wsl = self.reshape(wsl, vec![o, c])?;
                    let term = self.dot_general(window, wsl, vec![1], vec![1], vec![], vec![])?; // [N,Do,Ho,Wo,O]
                    let term = self.permute(term, vec![0, 4, 1, 2, 3])?; // [N,O,Do,Ho,Wo]
                    acc = Some(match acc {
                        None => term,
                        Some(a) => self.add(a, term)?,
                    });
                }
            }
        }
        acc.ok_or_else(|| Error::shape("conv3d", "empty kernel"))
    }
}
