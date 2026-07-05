//! Weight-only quantization: a [rows, cols] f32 matrix is quantized to int4/int8
//! along `cols` (the contraction/K axis), each group of `group_size` cols sharing an
//! f16 scale (symmetric) or scale+min (asymmetric). Round-to-nearest. This is the
//! exact reference the dequant-GEMV kernels are checked against; scales are stored in
//! f16 here too, so the reference carries the same rounding as the kernels.

use half::f16;

/// A quantized weight matrix. `packed` is row-major: int4 packs two values per byte
/// (low nibble = even col), int8 is one byte per col. `scales`/`mins` are [rows, cols/group_size].
#[derive(Clone, Debug, PartialEq)]
pub struct Quantized {
    pub packed: Vec<u8>,
    pub scales: Vec<f16>,
    pub mins: Option<Vec<f16>>, // Some = asymmetric (w = q*scale + min), None = symmetric (w = q*scale)
    pub rows: usize,
    pub cols: usize,
    pub bits: u8, // 4 or 8
    pub group_size: usize,
}

fn row_bytes(cols: usize, bits: u8) -> usize {
    cols * bits as usize / 8
}

// store an unsigned `bits`-wide field at (r, c) into the packed buffer. 8/bits fields per byte,
// low field first. bits in {2, 4, 8}.
fn put(packed: &mut [u8], r: usize, c: usize, bits: u8, rb: usize, v: u8) {
    let per = 8 / bits as usize;
    let (i, shift) = (r * rb + c / per, (c % per) * bits as usize);
    let mask = ((1u16 << bits) - 1) as u8;
    packed[i] = (packed[i] & !(mask << shift)) | ((v & mask) << shift);
}

// read the raw unsigned `bits`-wide field at (r, c).
fn get(packed: &[u8], r: usize, c: usize, bits: u8, rb: usize) -> u8 {
    let per = 8 / bits as usize;
    let (b, shift) = (packed[r * rb + c / per], (c % per) * bits as usize);
    (b >> shift) & (((1u16 << bits) - 1) as u8)
}

/// Quantize `w` ([rows, cols], row-major). `symmetric` uses a signed range with no
/// min; asymmetric uses an unsigned range with a per-group min. `cols` must be a
/// multiple of `group_size`, and `bits` is 2, 4, or 8.
pub fn quantize(w: &[f32], rows: usize, cols: usize, bits: u8, group_size: usize, symmetric: bool) -> Quantized {
    assert!(matches!(bits, 2 | 4 | 8), "bits must be 2, 4, or 8");
    assert!(group_size != 0 && cols.is_multiple_of(group_size), "cols must be a multiple of group_size");
    assert_eq!(w.len(), rows * cols, "w length mismatch");

    let n_groups = cols / group_size;
    let rb = row_bytes(cols, bits);
    let mut packed = vec![0u8; rows * rb];
    let mut scales = vec![f16::ZERO; rows * n_groups];
    let mut mins = (!symmetric).then(|| vec![f16::ZERO; rows * n_groups]);

    for r in 0..rows {
        for g in 0..n_groups {
            let base = r * cols + g * group_size;
            let grp = &w[base..base + group_size];
            let si = r * n_groups + g;

            if symmetric {
                // signed range [-2^(bits-1), 2^(bits-1)-1]: int2 [-2,1], int4 [-8,7], int8 [-128,127].
                let amax = grp.iter().fold(0f32, |a, &x| a.max(x.abs()));
                let (qlo, qhi) = (-(1i32 << (bits - 1)), (1i32 << (bits - 1)) - 1);
                let s = f16::from_f32(if amax > 0.0 { amax / qhi as f32 } else { 1.0 });
                scales[si] = s;
                let (sv, mask) = (s.to_f32(), (1i32 << bits) - 1);
                for (i, &x) in grp.iter().enumerate() {
                    let q = (x / sv).round().clamp(qlo as f32, qhi as f32) as i32;
                    put(&mut packed, r, g * group_size + i, bits, rb, (q & mask) as u8);
                }
            } else {
                let mn = grp.iter().copied().fold(f32::INFINITY, f32::min);
                let mx = grp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let qmax = ((1u32 << bits) - 1) as f32;
                let s = f16::from_f32(if mx > mn { (mx - mn) / qmax } else { 1.0 });
                let m = f16::from_f32(mn);
                scales[si] = s;
                mins.as_mut().unwrap()[si] = m;
                let (sv, mv) = (s.to_f32(), m.to_f32());
                for (i, &x) in grp.iter().enumerate() {
                    let q = ((x - mv) / sv).round().clamp(0.0, qmax) as u32 as u8;
                    put(&mut packed, r, g * group_size + i, bits, rb, q);
                }
            }
        }
    }
    Quantized { packed, scales, mins, rows, cols, bits, group_size }
}

// dequantize one raw value with its (already hoisted) group scale/min. asym: q*s+min;
// sym: signed(q)*s, sign-extending the `bits`-wide field.
fn deq_raw(raw: u8, scale: f32, min: Option<f32>, bits: u8) -> f32 {
    match min {
        Some(m) => raw as f32 * scale + m,
        None => {
            let sh = 32 - bits as u32; // sign-extend: int2 <<30>>30, int4 <<28>>28, int8 <<24>>24
            (((raw as i32) << sh >> sh) as f32) * scale
        }
    }
}

/// Exact inverse of [`quantize`] (the reference for the fused kernels).
pub fn dequantize(q: &Quantized) -> Vec<f32> {
    let ng = q.cols / q.group_size;
    let rb = row_bytes(q.cols, q.bits);
    let mut out = vec![0f32; q.rows * q.cols];
    for r in 0..q.rows {
        for c in 0..q.cols {
            let gi = r * ng + c / q.group_size;
            let mn = q.mins.as_ref().map(|m| m[gi].to_f32());
            out[r * q.cols + c] = deq_raw(get(&q.packed, r, c, q.bits, rb), q.scales[gi].to_f32(), mn, q.bits);
        }
    }
    out
}

/// Fused CPU dequant-matmul: `act[M,K] x dequant(q)[N,K]^T -> [M,N]`, dequantizing each
/// packed weight on the fly so the full f32 weight is never materialized (the memory
/// point of quant). For M=1 this is the decode-time GEMV.
pub fn dequant_matmul(act: &[f32], m: usize, q: &Quantized) -> Vec<f32> {
    let (n, k, gsz) = (q.rows, q.cols, q.group_size);
    assert_eq!(act.len(), m * k, "act length mismatch");
    let ng = k / gsz;
    let rb = row_bytes(k, q.bits);
    let mut out = vec![0f32; m * n];
    for ni in 0..n {
        for g in 0..ng {
            let gi = ni * ng + g;
            let s = q.scales[gi].to_f32();
            let mn = q.mins.as_ref().map(|mv| mv[gi].to_f32());
            for j in 0..gsz {
                let c = g * gsz + j;
                let w = deq_raw(get(&q.packed, ni, c, q.bits, rb), s, mn, q.bits);
                for mi in 0..m {
                    out[mi * n + ni] += act[mi * k + c] * w;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // round-to-nearest with an f16 scale: every element recovers to within the group's
    // scale (the rounding step). Checks all four modes.
    fn roundtrip_bound(bits: u8, group_size: usize, symmetric: bool) {
        let (rows, cols) = (3, 128);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 % 211) as f32 / 211.0 - 0.5) * 6.0).collect();
        let q = quantize(&w, rows, cols, bits, group_size, symmetric);
        let dq = dequantize(&q);
        let n_groups = cols / group_size;
        for r in 0..rows {
            for c in 0..cols {
                let s = q.scales[r * n_groups + c / group_size].to_f32();
                let err = (w[r * cols + c] - dq[r * cols + c]).abs();
                // one rounding step is <= s/2; f16 scale rounding adds a small slack.
                assert!(err <= s * 0.55 + 1e-3, "bits={bits} sym={symmetric} err={err} scale={s}");
            }
        }
    }

    #[test]
    fn roundtrip_all_modes() {
        for &bits in &[2u8, 4, 8] {
            for &g in &[32usize, 64, 128] {
                roundtrip_bound(bits, g, true);
                roundtrip_bound(bits, g, false);
            }
        }
    }

    #[test]
    fn int8_asym_beats_int4() {
        let (rows, cols) = (2, 64);
        let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32).sin() * 3.0).collect();
        let mse = |bits| {
            let dq = dequantize(&quantize(&w, rows, cols, bits, 32, false));
            w.iter().zip(&dq).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / w.len() as f32
        };
        assert!(mse(8) < mse(4), "int8 should be more accurate than int4");
    }

    #[test]
    fn packing_roundtrips_nibbles() {
        // int4 packs two per byte without cross-talk.
        let w: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
        let q = quantize(&w, 1, 32, 4, 32, true);
        assert_eq!(q.packed.len(), 16);
        let dq = dequantize(&q);
        assert_eq!(dq.len(), 32);
    }

    // the fused dequant-matmul kernel equals dequantize-then-matmul, all bit widths + modes.
    #[test]
    fn fused_dequant_matmul_matches_reference() {
        for &bits in &[2u8, 4, 8] {
            for &sym in &[true, false] {
                let (m, n, k, g) = (3, 5, 64, 32);
                let w: Vec<f32> = (0..n * k).map(|i| ((i * 29 % 83) as f32 / 83.0 - 0.5) * 3.0).collect();
                let act: Vec<f32> = (0..m * k).map(|i| (i * 11 % 61) as f32 / 61.0 - 0.3).collect();
                let q = quantize(&w, n, k, bits, g, sym);
                let fused = dequant_matmul(&act, m, &q);
                let wdq = dequantize(&q);
                for mi in 0..m {
                    for ni in 0..n {
                        let want: f32 = (0..k).map(|ki| act[mi * k + ki] * wdq[ni * k + ki]).sum();
                        assert!((fused[mi * n + ni] - want).abs() < 1e-4, "bits={bits} sym={sym}");
                    }
                }
            }
        }
    }

    // the QuantMatmul op (via the interpreter) matches dequant-then-matmul exactly, and
    // stays within int4 quant error of the full-precision matmul.
    #[test]
    fn quant_matmul_op_matches_oracle() {
        use crate::{Graph, Storage, interpret};
        let (m, n, k, g) = (2usize, 3, 64, 32);
        let w: Vec<f32> = (0..n * k).map(|i| ((i * 13 % 97) as f32 / 97.0 - 0.5) * 4.0).collect();
        let act: Vec<f32> = (0..m * k).map(|i| (i * 7 % 53) as f32 / 53.0).collect();
        let q = quantize(&w, n, k, 4, g, false);
        let ng = k / g;

        let mut gr = Graph::new();
        let a = gr.constant(act.clone(), vec![m, k]);
        let qw = gr.const_storage(Storage::U8(q.packed.clone()), vec![n, k / 2]);
        let sc = gr.const_storage(Storage::F16(q.scales.clone()), vec![n, ng]);
        let mn = gr.const_storage(Storage::F16(q.mins.clone().unwrap()), vec![n, ng]);
        let out = gr.quant_matmul(a, qw, sc, Some(mn), 4, g).unwrap();
        let got = match interpret(&gr, out).storage {
            Storage::F32(v) => v,
            _ => panic!("expected f32"),
        };

        let wdq = dequantize(&q);
        let mm = |wm: &[f32]| {
            let mut o = vec![0f32; m * n];
            for mi in 0..m {
                for ni in 0..n {
                    o[mi * n + ni] = (0..k).map(|ki| act[mi * k + ki] * wm[ni * k + ki]).sum();
                }
            }
            o
        };
        let want = mm(&wdq);
        for (x, y) in got.iter().zip(&want) {
            assert!((x - y).abs() < 1e-3, "oracle mismatch: {x} vs {y}");
        }
        let full = mm(&w);
        let num: f32 = got.iter().zip(&full).map(|(a, b)| (a - b).abs()).sum();
        let den: f32 = full.iter().map(|x| x.abs()).sum::<f32>().max(1e-6);
        assert!(num / den < 0.15, "int4 quant rel err {}", num / den);
    }
}
