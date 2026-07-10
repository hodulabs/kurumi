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
mod tests;
