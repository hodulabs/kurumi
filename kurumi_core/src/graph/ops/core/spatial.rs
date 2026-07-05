//! Spatial resampling & rearrangement: resize, upsample, pixel-shuffle. All pure
//! decompositions (gather / reshape / permute) -> differentiable, every backend.

use crate::{Error, Graph, NodeId, Storage};

// Output index -> source coordinate (float), per coord-transform mode.
fn resize_src(o: usize, in_sz: usize, out_sz: usize, coord: &str) -> f32 {
    let (o, inf, outf) = (o as f32, in_sz as f32, out_sz as f32);
    match coord {
        "align_corners" => {
            if out_sz > 1 {
                o * (inf - 1.0) / (outf - 1.0)
            } else {
                0.0
            }
        }
        "asymmetric" => o * inf / outf,
        "pytorch_half_pixel" => {
            if out_sz > 1 {
                (o + 0.5) * inf / outf - 0.5
            } else {
                0.0
            }
        }
        _ => (o + 0.5) * inf / outf - 0.5, // half_pixel (default; validated upstream)
    }
}

// Catmull-Rom cubic convolution kernel (a = -0.5).
fn cubic_w(s: f32) -> f32 {
    let (a, s) = (-0.5f32, s.abs());
    if s <= 1.0 {
        (a + 2.0) * s * s * s - (a + 3.0) * s * s + 1.0
    } else if s < 2.0 {
        a * s * s * s - 5.0 * a * s * s + 8.0 * a * s - 4.0 * a
    } else {
        0.0
    }
}

impl Graph {
    /// General separable resize along `axes` to `sizes`. `interp` in nearest|linear|cubic;
    /// `coord` (coordinate transform) in half_pixel|align_corners|asymmetric|pytorch_half_pixel.
    /// Per-axis gather+weight decomposition -> differentiable, every backend, 1-D..N-D; the
    /// 2-D bilinear/bicubic wrappers below call this.
    pub fn resize(
        &mut self,
        x: NodeId,
        axes: &[usize],
        sizes: &[usize],
        interp: &str,
        coord: &str,
    ) -> Result<NodeId, Error> {
        if axes.len() != sizes.len() {
            return Err(Error::shape("resize", "axes/sizes length mismatch"));
        }
        if !matches!(interp, "nearest" | "linear" | "cubic") {
            return Err(Error::shape("resize", format!("interp must be nearest|linear|cubic, got {interp}")));
        }
        if !matches!(coord, "half_pixel" | "align_corners" | "asymmetric" | "pytorch_half_pixel") {
            return Err(Error::shape("resize", format!("unknown coord mode {coord}")));
        }
        let rank = self.shape(x).len();
        let mut cur = x;
        for (&ax, &sz) in axes.iter().zip(sizes) {
            if ax >= rank {
                return Err(Error::shape("resize", "axis out of range"));
            }
            cur = self.resize_axis(cur, ax, sz, interp, coord)?;
        }
        Ok(cur)
    }

    // Resize one axis: gather the interp taps along `axis`, weight, sum. Indices/weights
    // are static (shape-determined). Nearest is a single un-weighted gather (works on any
    // dtype); linear/cubic mix with f32 weights (float inputs).
    fn resize_axis(
        &mut self,
        x: NodeId,
        axis: usize,
        out_size: usize,
        interp: &str,
        coord: &str,
    ) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        let in_sz = sh[axis] as i64;
        let clamp = |p: i64| p.clamp(0, in_sz - 1);
        if interp == "nearest" {
            let ix: Vec<i64> =
                (0..out_size).map(|o| clamp(resize_src(o, sh[axis], out_size, coord).round() as i64)).collect();
            let idxn = self.const_storage(Storage::I64(ix), vec![out_size]);
            return self.gather(x, idxn, axis);
        }
        // (index array, weight array) per tap: linear = 2 taps, cubic = 4.
        let taps: Vec<(Vec<i64>, Vec<f32>)> = if interp == "linear" {
            let (mut lo, mut hi, mut wlo, mut whi) =
                (vec![0i64; out_size], vec![0i64; out_size], vec![0f32; out_size], vec![0f32; out_size]);
            for o in 0..out_size {
                let s = resize_src(o, sh[axis], out_size, coord);
                let f = s.floor();
                lo[o] = clamp(f as i64);
                hi[o] = clamp(f as i64 + 1);
                whi[o] = s - f;
                wlo[o] = 1.0 - whi[o];
            }
            vec![(lo, wlo), (hi, whi)]
        } else {
            // cubic: 4 taps at base-1..base+2
            let mut taps = vec![(vec![0i64; out_size], vec![0f32; out_size]); 4];
            for o in 0..out_size {
                let s = resize_src(o, sh[axis], out_size, coord);
                let base = s.floor();
                let t = s - base;
                for (tap, (ix, w)) in taps.iter_mut().enumerate() {
                    ix[o] = clamp(base as i64 - 1 + tap as i64);
                    w[o] = cubic_w(t + 1.0 - tap as f32); // distance src -> tap position
                }
            }
            taps
        };
        let full: Vec<usize> = sh.iter().enumerate().map(|(i, &d)| if i == axis { out_size } else { d }).collect();
        let mut wsh = vec![1usize; sh.len()];
        wsh[axis] = out_size;
        let mut acc: Option<NodeId> = None;
        for (ix, w) in taps {
            let idxn = self.const_storage(Storage::I64(ix), vec![out_size]);
            let g = self.gather(x, idxn, axis)?;
            let wn = self.const_storage(Storage::F32(w), wsh.clone());
            let wf = self.broadcast_to(wn, full.clone())?;
            let term = self.mul(g, wf)?;
            acc = Some(match acc {
                None => term,
                Some(a) => self.add(a, term)?,
            });
        }
        acc.ok_or_else(|| Error::shape("resize", "no taps"))
    }

    /// Bilinear resize of `[N, C, H, W]` to `[N, C, out_h, out_w]` (align_corners=False).
    pub fn resize_bilinear(&mut self, x: NodeId, out_h: usize, out_w: usize) -> Result<NodeId, Error> {
        if self.shape(x).len() != 4 {
            return Err(Error::shape("resize_bilinear", "expects [N, C, H, W]"));
        }
        self.resize(x, &[2, 3], &[out_h, out_w], "linear", "half_pixel")
    }

    /// Bicubic resize of `[N, C, H, W]` to `[N, C, out_h, out_w]` (align_corners=False,
    /// Catmull-Rom a=-0.5).
    pub fn resize_bicubic(&mut self, x: NodeId, out_h: usize, out_w: usize) -> Result<NodeId, Error> {
        if self.shape(x).len() != 4 {
            return Err(Error::shape("resize_bicubic", "expects [N, C, H, W]"));
        }
        self.resize(x, &[2, 3], &[out_h, out_w], "cubic", "half_pixel")
    }

    /// Nearest-neighbor 2x upsample by an integer `factor`: `[N, C, H, W] ->
    /// [N, C, H*f, W*f]` (each pixel repeated along H then W).
    pub fn upsample_nearest2d(&mut self, x: NodeId, factor: usize) -> Result<NodeId, Error> {
        let a = self.repeat_interleave(x, 2, factor)?;
        self.repeat_interleave(a, 3, factor)
    }

    /// Pixel shuffle `[N, C*r^2, H, W] -> [N, C, H*r, W*r]` (reshape + permute).
    pub fn depth_to_space(&mut self, x: NodeId, r: usize) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        if sh.len() != 4 || !sh[1].is_multiple_of(r * r) {
            return Err(Error::shape("depth_to_space", "expects [N, C*r^2, H, W]"));
        }
        let (n, h, w) = (sh[0], sh[2], sh[3]);
        let c = sh[1] / (r * r);
        let a = self.reshape(x, vec![n, c, r, r, h, w])?;
        let b = self.permute(a, vec![0, 1, 4, 2, 5, 3])?; // [N, C, H, r, W, r]
        self.reshape(b, vec![n, c, h * r, w * r])
    }

    /// Inverse pixel shuffle `[N, C, H*r, W*r] -> [N, C*r^2, H, W]`.
    pub fn space_to_depth(&mut self, x: NodeId, r: usize) -> Result<NodeId, Error> {
        let sh = self.shape(x);
        if sh.len() != 4 || !sh[2].is_multiple_of(r) || !sh[3].is_multiple_of(r) {
            return Err(Error::shape("space_to_depth", "expects [N, C, H*r, W*r]"));
        }
        let (n, c) = (sh[0], sh[1]);
        let (h, w) = (sh[2] / r, sh[3] / r);
        let a = self.reshape(x, vec![n, c, h, r, w, r])?;
        let b = self.permute(a, vec![0, 1, 3, 5, 2, 4])?; // [N, C, r, r, H, W]
        self.reshape(b, vec![n, c * r * r, h, w])
    }
}
