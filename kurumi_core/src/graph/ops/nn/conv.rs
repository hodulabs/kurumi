//! Convolution & transposed convolution, decomposed from strided-slice + dot_general. No
//! conv/im2col primitive: each kernel offset is one strided-slice window, and autodiff
//! gives the conv backward + weight-gradient for free. (An im2col fast-op could replace
//! this if perf demands; semantics stay identical.)

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

    // insert `factor-1` zeros after each element along `axis` (k -> (k-1)*factor+1).
    fn dilate_axis(&mut self, x: NodeId, axis: usize, factor: usize) -> Result<NodeId, Error> {
        if factor <= 1 {
            return Ok(x);
        }
        let sh = self.shape(x);
        let k = sh[axis];
        let mut s1 = sh.clone();
        s1.insert(axis + 1, 1);
        let r1 = self.reshape(x, s1)?;
        let mut pads = vec![(0, 0); sh.len() + 1];
        pads[axis + 1] = (0, factor - 1);
        let p = self.pad(r1, pads)?;
        let mut s2 = sh.clone();
        s2[axis] = k * factor;
        let m = self.reshape(p, s2)?;
        let want = (k - 1) * factor + 1;
        let mut ranges: Vec<(usize, usize)> = self.shape(m).iter().map(|&d| (0, d)).collect();
        ranges[axis] = (0, want);
        self.slice(m, ranges)
    }

    /// 1-D transposed convolution. `input` `[N,C,L]`, `weight` `[C,O,K]` -> `[N,O,Lo]`.
    pub fn conv_transpose1d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> Result<NodeId, Error> {
        let k = self.shape(weight)[2];
        let xd = self.dilate_axis(input, 2, stride)?;
        let wt = self.permute(weight, vec![1, 0, 2])?; // [O, C, K]
        let wt = self.flip(wt, vec![2])?;
        let cp = dilation * (k - 1) - padding;
        let xp = self.pad(xd, vec![(0, 0), (0, 0), (cp, cp + output_padding)])?;
        self.conv1d(xp, wt, 1, 0, dilation)
    }

    /// 2-D transposed (fractionally-strided) convolution. `input` `[N,C,H,W]`, `weight`
    /// `[C,O,Kh,Kw]` -> `[N,O,Ho,Wo]` where `Ho = (H-1)*s - 2p + d*(Kh-1) + output_padding + 1`.
    /// Decomposed as: dilate the input by `stride`, a stride-1 conv with the C/O-swapped
    /// spatially-flipped weight and padding `d*(K-1)-p`, then `output_padding` on the bottom/right.
    pub fn conv_transpose2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<NodeId, Error> {
        let wsh = self.shape(weight); // [C, O, Kh, Kw]
        let (kh, kw) = (wsh[2], wsh[3]);
        let ((sh, sw), (ph, pw), (oph, opw), (dh, dw)) = (stride, padding, output_padding, dilation);
        let xd = self.dilate_axis(input, 2, sh)?;
        let xd = self.dilate_axis(xd, 3, sw)?;
        let wt = self.permute(weight, vec![1, 0, 2, 3])?; // [O, C, Kh, Kw]
        let wt = self.flip(wt, vec![2, 3])?; // spatial flip
        // asymmetric pad: `cp` each side, plus `output_padding` extra on the high side
        // (output_padding extends the valid output region, not just trailing zeros).
        let (cph, cpw) = (dh * (kh - 1) - ph, dw * (kw - 1) - pw);
        let xp = self.pad(xd, vec![(0, 0), (0, 0), (cph, cph + oph), (cpw, cpw + opw)])?;
        self.conv2d(xp, wt, (1, 1), (0, 0), (dh, dw))
    }

    /// 3-D transposed convolution. `input` `[N,C,D,H,W]`, `weight` `[C,O,Kd,Kh,Kw]`
    /// -> `[N,O,Do,Ho,Wo]`. Same decomposition as `conv_transpose2d`, three spatial axes.
    pub fn conv_transpose3d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
    ) -> Result<NodeId, Error> {
        let wsh = self.shape(weight); // [C, O, Kd, Kh, Kw]
        let (kd, kh, kw) = (wsh[2], wsh[3], wsh[4]);
        let ((sd, sh, sw), (pd, ph, pw), (opd, oph, opw), (dd, dh, dw)) = (stride, padding, output_padding, dilation);
        let xd = self.dilate_axis(input, 2, sd)?;
        let xd = self.dilate_axis(xd, 3, sh)?;
        let xd = self.dilate_axis(xd, 4, sw)?;
        let wt = self.permute(weight, vec![1, 0, 2, 3, 4])?; // [O, C, Kd, Kh, Kw]
        let wt = self.flip(wt, vec![2, 3, 4])?;
        let (cpd, cph, cpw) = (dd * (kd - 1) - pd, dh * (kh - 1) - ph, dw * (kw - 1) - pw);
        let xp = self.pad(xd, vec![(0, 0), (0, 0), (cpd, cpd + opd), (cph, cph + oph), (cpw, cpw + opw)])?;
        self.conv3d(xp, wt, (1, 1, 1), (0, 0, 0), (dd, dh, dw))
    }
}
