//! Transposed (fractionally-strided) convolution (1/2/3-D): dilate the input by `stride`,
//! then a stride-1 direct conv with the C/O-swapped spatially-flipped weight.

use crate::{Error, Graph, NodeId};

impl Graph {
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
