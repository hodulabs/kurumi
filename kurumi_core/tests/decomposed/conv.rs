use kurumi_core::*;

// reference NCHW conv2d (no groups), stride/pad/dilation.
#[allow(clippy::too_many_arguments)]
fn ref_conv2d(
    inp: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    wt: &[f32],
    o: usize,
    kh: usize,
    kw: usize,
    s: usize,
    p: usize,
    d: usize,
) -> (Vec<f32>, usize, usize) {
    let ho = (h + 2 * p - d * (kh - 1) - 1) / s + 1;
    let wo = (w + 2 * p - d * (kw - 1) - 1) / s + 1;
    let mut out = vec![0f32; n * o * ho * wo];
    for ni in 0..n {
        for oi in 0..o {
            for i in 0..ho {
                for j in 0..wo {
                    let mut acc = 0.0;
                    for ci in 0..c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = (i * s + ki * d) as isize - p as isize;
                                let iw = (j * s + kj * d) as isize - p as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let iv = inp[((ni * c + ci) * h + ih as usize) * w + iw as usize];
                                    let wv = wt[((oi * c + ci) * kh + ki) * kw + kj];
                                    acc += iv * wv;
                                }
                            }
                        }
                    }
                    out[((ni * o + oi) * ho + i) * wo + j] = acc;
                }
            }
        }
    }
    (out, ho, wo)
}

#[test]
fn conv2d_matches_reference() {
    let cases = [
        // (H,W,C,O,Kh,Kw,stride,pad,dilation)
        (3usize, 3usize, 1usize, 1usize, 2usize, 2usize, 1usize, 0usize, 1usize),
        (5, 5, 2, 3, 3, 3, 1, 1, 1), // padded
        (7, 7, 2, 2, 3, 3, 2, 0, 1), // strided
        (7, 7, 1, 1, 3, 3, 1, 2, 2), // dilated + padded
    ];
    for (h, w, c, o, kh, kw, s, p, d) in cases {
        let n = 2;
        let inp: Vec<f32> = (0..n * c * h * w).map(|i| ((i * 7 % 13) as f32) * 0.3 - 1.0).collect();
        let wt: Vec<f32> = (0..o * c * kh * kw).map(|i| ((i * 5 % 11) as f32) * 0.2 - 0.5).collect();
        let mut g = Graph::new();
        let gi = g.constant(inp.clone(), vec![n, c, h, w]);
        let gw = g.constant(wt.clone(), vec![o, c, kh, kw]);
        let y = g.conv2d(gi, gw, (s, s), (p, p), (d, d)).unwrap();
        let got = interpret(&g, y);
        let (want, ho, wo) = ref_conv2d(&inp, n, c, h, w, &wt, o, kh, kw, s, p, d);
        assert_eq!(got.shape, vec![n, o, ho, wo], "shape for {:?}", (h, w, c, o, kh, kw, s, p, d));
        for (a, b) in got.f32().iter().zip(&want) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b} for {:?}", (h, w, c, o, kh, kw, s, p, d));
        }
    }
}

#[test]
fn conv2d_backward_shapes() {
    // gradient flows to both input and weight (weight-grad for free via autodiff)
    let mut g = Graph::new();
    let gi = g.constant((0..2 * 4 * 4).map(|i| i as f32 * 0.1).collect(), vec![1, 2, 4, 4]);
    let gw = g.constant((0..3 * 2 * 3 * 3).map(|i| i as f32 * 0.05).collect(), vec![3, 2, 3, 3]);
    let y = g.conv2d(gi, gw, (1, 1), (1, 1), (1, 1)).unwrap();
    let loss = {
        let s1 = g.sum(y, 3).unwrap();
        let s2 = g.sum(s1, 2).unwrap();
        let s3 = g.sum(s2, 1).unwrap();
        g.sum(s3, 0).unwrap()
    };
    let grads = grad(&mut g, loss, &[gi, gw]).unwrap();
    assert_eq!(interpret(&g, grads[0]).shape, vec![1, 2, 4, 4]);
    assert_eq!(interpret(&g, grads[1]).shape, vec![3, 2, 3, 3]);
}

#[test]
fn pool2d_basic() {
    let mut g = Graph::new();
    // 1x1x4x4 ramp
    let x = g.constant((0..16).map(|i| i as f32).collect(), vec![1, 1, 4, 4]);
    let mp = g.max_pool2d(x, (2, 2), (2, 2)).unwrap();
    let omp = interpret(&g, mp);
    assert_eq!(omp.shape, vec![1, 1, 2, 2]);
    assert_eq!(omp.f32(), &[5., 7., 13., 15.]); // max of each 2x2 block
    let ap = g.avg_pool2d(x, (2, 2), (2, 2)).unwrap();
    let oap = interpret(&g, ap);
    assert_eq!(oap.f32(), &[2.5, 4.5, 10.5, 12.5]);
}

#[test]
fn conv3d_matches_reference() {
    // 3D conv: small case, compare to an explicit loop.
    let (n, c, dd, hh, ww, o, k) = (1usize, 2, 4, 4, 4, 2, 2);
    let (s, p, dil) = (1usize, 0usize, 1usize);
    let inp: Vec<f32> = (0..n * c * dd * hh * ww).map(|i| ((i % 13) as f32) * 0.2 - 1.0).collect();
    let wt: Vec<f32> = (0..o * c * k * k * k).map(|i| ((i % 7) as f32) * 0.1).collect();
    let mut g = Graph::new();
    let gi = g.constant(inp.clone(), vec![n, c, dd, hh, ww]);
    let gw = g.constant(wt.clone(), vec![o, c, k, k, k]);
    let y = g.conv3d(gi, gw, (s, s, s), (p, p, p), (dil, dil, dil)).unwrap();
    let (dout, hout, wout) = ((dd - k) / s + 1, (hh - k) / s + 1, (ww - k) / s + 1);
    let mut want = vec![0f32; n * o * dout * hout * wout];
    for oi in 0..o {
        for a in 0..dout {
            for i in 0..hout {
                for j in 0..wout {
                    let mut acc = 0.0;
                    for ci in 0..c {
                        for ka in 0..k {
                            for ki in 0..k {
                                for kj in 0..k {
                                    let iv = inp[(((ci) * dd + a + ka) * hh + i + ki) * ww + j + kj];
                                    let wv = wt[(((oi * c + ci) * k + ka) * k + ki) * k + kj];
                                    acc += iv * wv;
                                }
                            }
                        }
                    }
                    want[((oi * dout + a) * hout + i) * wout + j] = acc;
                }
            }
        }
    }
    let got = interpret(&g, y);
    assert_eq!(got.shape, vec![n, o, dout, hout, wout]);
    for (a, b) in got.f32().iter().zip(&want) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}

// reference conv_transpose2d via the scatter (overlap-add) definition.
#[allow(clippy::too_many_arguments)]
fn ref_convt2d(
    inp: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    wt: &[f32],
    o: usize,
    kh: usize,
    kw: usize,
    s: usize,
    p: usize,
    op: usize,
    d: usize,
) -> (Vec<f32>, usize, usize) {
    let ho = (h - 1) * s - 2 * p + d * (kh - 1) + op + 1;
    let wo = (w - 1) * s - 2 * p + d * (kw - 1) + op + 1;
    let mut y = vec![0f32; n * o * ho * wo];
    for ni in 0..n {
        for ci in 0..c {
            for ih in 0..h {
                for iw in 0..w {
                    let xv = inp[((ni * c + ci) * h + ih) * w + iw];
                    for oi in 0..o {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let oh = (ih * s + ki * d) as isize - p as isize;
                                let ow = (iw * s + kj * d) as isize - p as isize;
                                if oh >= 0 && oh < ho as isize && ow >= 0 && ow < wo as isize {
                                    let wv = wt[((ci * o + oi) * kh + ki) * kw + kj];
                                    y[((ni * o + oi) * ho + oh as usize) * wo + ow as usize] += xv * wv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (y, ho, wo)
}

#[test]
fn conv_transpose2d_matches_reference() {
    let cases = [
        (4usize, 4usize, 1usize, 1usize, 3usize, 3usize, 1usize, 0usize, 0usize, 1usize),
        (3, 3, 2, 2, 3, 3, 2, 1, 1, 1), // strided + pad + output_pad
        (5, 5, 1, 2, 2, 2, 2, 0, 0, 1), // upsample
    ];
    for (h, w, c, o, kh, kw, s, p, op, d) in cases {
        let n = 2;
        let inp: Vec<f32> = (0..n * c * h * w).map(|i| ((i * 7 % 13) as f32) * 0.3 - 1.0).collect();
        let wt: Vec<f32> = (0..c * o * kh * kw).map(|i| ((i * 5 % 11) as f32) * 0.2 - 0.5).collect();
        let mut g = Graph::new();
        let gi = g.constant(inp.clone(), vec![n, c, h, w]);
        let gw = g.constant(wt.clone(), vec![c, o, kh, kw]);
        let y = g.conv_transpose2d(gi, gw, (s, s), (p, p), (op, op), (d, d)).unwrap();
        let got = interpret(&g, y);
        let (want, ho, wo) = ref_convt2d(&inp, n, c, h, w, &wt, o, kh, kw, s, p, op, d);
        assert_eq!(got.shape, vec![n, o, ho, wo], "shape for {:?}", (h, w, c, o, kh, kw, s, p, op, d));
        for (a, b) in got.f32().iter().zip(&want) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b} for {:?}", (h, w, c, o, kh, kw, s, p, op, d));
        }
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn conv_transpose1d_and_backward() {
    // conv_transpose1d upsampling, vs a 1D scatter reference
    let (n, c, l, o, k, s, p, op, d) = (1usize, 1, 3, 1, 2, 2, 0, 0, 1);
    let inp = vec![1.0f32, 2.0, 3.0];
    let wt = vec![1.0f32, 0.5]; // [C=1,O=1,K=2]
    let mut g = Graph::new();
    let gi = g.constant(inp.clone(), vec![n, c, l]);
    let gw = g.constant(wt.clone(), vec![c, o, k]);
    let y = g.conv_transpose1d(gi, gw, s, p, op, d).unwrap();
    let lo = (l - 1) * s - 2 * p + d * (k - 1) + op + 1;
    let mut want = vec![0f32; lo];
    for il in 0..l {
        for ki in 0..k {
            let ol = (il * s + ki * d) as isize - p as isize;
            if ol >= 0 && (ol as usize) < lo {
                want[ol as usize] += inp[il] * wt[ki];
            }
        }
    }
    let got = interpret(&g, y);
    assert_eq!(got.shape, vec![n, o, lo]);
    for (a, b) in got.f32().iter().zip(&want) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }

    // backward flows to input and weight
    let mut g2 = Graph::new();
    let xi = g2.constant((0..2 * 5 * 5).map(|i| i as f32 * 0.1).collect(), vec![1, 2, 5, 5]);
    let wi = g2.constant((0..2 * 3 * 3 * 3).map(|i| i as f32 * 0.05).collect(), vec![2, 3, 3, 3]);
    let yt = g2.conv_transpose2d(xi, wi, (2, 2), (1, 1), (1, 1), (1, 1)).unwrap();
    let loss = {
        let a = g2.sum(yt, 3).unwrap();
        let b = g2.sum(a, 2).unwrap();
        let c = g2.sum(b, 1).unwrap();
        g2.sum(c, 0).unwrap()
    };
    let grads = grad(&mut g2, loss, &[xi, wi]).unwrap();
    assert_eq!(interpret(&g2, grads[0]).shape, vec![1, 2, 5, 5]);
    assert_eq!(interpret(&g2, grads[1]).shape, vec![2, 3, 3, 3]);
}

#[test]
fn max_pool1d_basics() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 3., 2., 5., 4., 0.], vec![1, 1, 6]);
    let mp = g.max_pool1d(x, 2, 2).unwrap(); // windows [1,3][2,5][4,0] -> [3,5,4]
    assert_eq!(interpret(&g, mp).shape, vec![1, 1, 3]);
    assert_eq!(interpret(&g, mp).f32(), &[3., 5., 4.]);
}

#[test]
fn conv_transpose3d_identity() {
    // 1-channel 1x1x1 kernel (value 1), stride 1, no pad -> transpose conv = identity
    let mut g = Graph::new();
    let x = g.constant((1..=8).map(|i| i as f32).collect(), vec![1, 1, 2, 2, 2]);
    let w = g.constant(vec![1.], vec![1, 1, 1, 1, 1]); // [C=1, O=1, Kd=Kh=Kw=1]
    let y = g.conv_transpose3d(x, w, (1, 1, 1), (0, 0, 0), (0, 0, 0), (1, 1, 1)).unwrap();
    assert_eq!(interpret(&g, y).shape, vec![1, 1, 2, 2, 2]);
    assert_eq!(interpret(&g, y).f32(), interpret(&g, x).f32());
    // stride 2 upsamples the spatial dims: Do = (2-1)*2 + 1 = 3
    let y2 = g.conv_transpose3d(x, w, (2, 2, 2), (0, 0, 0), (0, 0, 0), (1, 1, 1)).unwrap();
    assert_eq!(interpret(&g, y2).shape, vec![1, 1, 3, 3, 3]);
}

#[test]
fn resize_bilinear_basics() {
    let mut g = Graph::new();
    // 1x1x1x2 [1,2] resized to width 4 (align_corners=False): [1, 1.25, 1.75, 2]
    let x = g.constant(vec![1., 2.], vec![1, 1, 1, 2]);
    let r = g.resize_bilinear(x, 1, 4).unwrap();
    assert_eq!(interpret(&g, r).shape, vec![1, 1, 1, 4]);
    let rv = interpret(&g, r).f32().to_vec();
    for (got, want) in rv.iter().zip([1., 1.25, 1.75, 2.]) {
        assert!((got - want).abs() < 1e-5, "resize {rv:?}");
    }
    // bilinear of a constant field is constant
    let cst = g.constant(vec![3.0; 8], vec![1, 2, 2, 2]);
    let rc = g.resize_bilinear(cst, 4, 4).unwrap();
    assert_eq!(interpret(&g, rc).shape, vec![1, 2, 4, 4]);
    assert!(interpret(&g, rc).f32().iter().all(|&v| (v - 3.0).abs() < 1e-5), "const resize");
}

#[test]
fn resize_general_and_reduce_window() {
    let mut g = Graph::new();
    // dilated max reduce_window: [1..=6], window=2 stride=1 dilation=2 -> eff window 3,
    // out=4, taps at o and o+2: [max(1,3),max(2,4),max(3,5),max(4,6)] = [3,4,5,6]
    let x = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![1, 1, 6]);
    let rw = g.reduce_window(x, &[2], &[1], &[2], "max").unwrap();
    assert_eq!(interpret(&g, rw).shape, vec![1, 1, 4]);
    assert_eq!(interpret(&g, rw).f32(), &[3., 4., 5., 6.]);
    // linear resize, align_corners: [1,2,3] axis2 -> 5, src=o*0.5 -> [1,1.5,2,2.5,3]
    let a = g.constant(vec![1., 2., 3.], vec![1, 1, 3]);
    let ra = g.resize(a, &[2], &[5], "linear", "align_corners").unwrap();
    assert_eq!(interpret(&g, ra).shape, vec![1, 1, 5]);
    let rav = interpret(&g, ra).f32().to_vec();
    for (got, want) in rav.iter().zip([1., 1.5, 2., 2.5, 3.]) {
        assert!((got - want).abs() < 1e-5, "align_corners {rav:?}");
    }
    // nearest resize (asymmetric): [1,2,3] axis2 -> 6, src=o*0.5 rounded (half away): idx
    // 0,1,1,2,2,2 -> [1,2,2,3,3,3]
    let nn = g.resize(a, &[2], &[6], "nearest", "asymmetric").unwrap();
    assert_eq!(interpret(&g, nn).f32(), &[1., 2., 2., 3., 3., 3.]);
    // 3-D trilinear on a constant field stays constant; [1,1,2,2,2] -> [1,1,3,3,3]
    let cst = g.constant(vec![7.0; 8], vec![1, 1, 2, 2, 2]);
    let tri = g.resize(cst, &[2, 3, 4], &[3, 3, 3], "linear", "half_pixel").unwrap();
    assert_eq!(interpret(&g, tri).shape, vec![1, 1, 3, 3, 3]);
    assert!(interpret(&g, tri).f32().iter().all(|&v| (v - 7.0).abs() < 1e-5), "trilinear const");
}
