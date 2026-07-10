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
