// Contraction ops: matmul, general contraction, weight-only quantized matmul, and
// the standalone weight quantizer. `cstr` (einsum) comes from the parent module.

use super::cstr;
use crate::capi::{KU_ERR, KuGraph, build, raw_slice, set_err, usize_slice};
use kurumi_core::{NodeId, quantize};
use std::ffi::c_char;

/// 2-D matmul `a @ b` (contract a's last axis with b's first).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_matmul(g: *mut KuGraph, a: u32, b: u32) -> u32 {
    build(g, |gr| gr.dot_general(NodeId(a), NodeId(b), vec![1], vec![0], vec![], vec![]))
}

/// General contraction: contract `lc`/`rc` axes, batch `lb`/`rb` axes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_dot_general(
    g: *mut KuGraph,
    a: u32,
    b: u32,
    lc: *const usize,
    nlc: usize,
    rc: *const usize,
    nrc: usize,
    lb: *const usize,
    nlb: usize,
    rb: *const usize,
    nrb: usize,
) -> u32 {
    let (lc, rc, lb, rb) = (
        usize_slice(lc, nlc).to_vec(),
        usize_slice(rc, nrc).to_vec(),
        usize_slice(lb, nlb).to_vec(),
        usize_slice(rb, nrb).to_vec(),
    );
    build(g, |gr| gr.dot_general(NodeId(a), NodeId(b), lc, rc, lb, rb))
}

/// Weight-only quantized matmul: `act[M,K] x dequant(qweight)[N,K]^T -> [M,N]`. `qweight`
/// is a U8-packed constant, `scales`/`mins` are F16 [N, K/group_size]; pass `mins = KU_ERR`
/// for symmetric. Build the operands with ku_quantize + ku_constant.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_quant_matmul(
    g: *mut KuGraph,
    act: u32,
    qweight: u32,
    scales: u32,
    mins: u32,
    bits: u8,
    group_size: usize,
) -> u32 {
    let mins = (mins != KU_ERR).then_some(NodeId(mins));
    build(g, |gr| gr.quant_matmul(NodeId(act), NodeId(qweight), NodeId(scales), mins, bits, group_size))
}

/// Quantize an f32 weight `[rows, cols]` (row-major) for ku_quant_matmul. Writes
/// `out_packed` (rows*cols*bits/8 bytes), `out_scales` (rows*cols/group_size f16
/// bit-patterns), and `out_mins` (same count, or pass NULL when `symmetric != 0`).
/// `bits` is 2, 4, or 8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_quantize(
    w: *const f32,
    rows: usize,
    cols: usize,
    bits: u8,
    group_size: usize,
    symmetric: u32,
    out_packed: *mut u8,
    out_scales: *mut u16,
    out_mins: *mut u16,
) {
    if w.is_null() || out_packed.is_null() || out_scales.is_null() {
        set_err("ku_quantize: null w/out_packed/out_scales".into());
        return;
    }
    let q = quantize(raw_slice(w, rows * cols), rows, cols, bits, group_size, symmetric != 0);
    std::ptr::copy_nonoverlapping(q.packed.as_ptr(), out_packed, q.packed.len());
    let scales: Vec<u16> = q.scales.iter().map(|x| x.to_bits()).collect();
    std::ptr::copy_nonoverlapping(scales.as_ptr(), out_scales, scales.len());
    if let Some(mins) = q.mins.filter(|_| !out_mins.is_null()) {
        let mins: Vec<u16> = mins.iter().map(|x| x.to_bits()).collect();
        std::ptr::copy_nonoverlapping(mins.as_ptr(), out_mins, mins.len());
    }
}

/// Einstein summation, e.g. "ij,jk->ik". `operands` = `n` node ids.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_einsum(g: *mut KuGraph, equation: *const c_char, operands: *const u32, n: usize) -> u32 {
    let Some(eq) = cstr(equation) else {
        set_err("ku_einsum: bad equation string".into());
        return KU_ERR;
    };
    let ops: Vec<NodeId> = raw_slice(operands, n).iter().map(|&i| NodeId(i)).collect();
    build(g, |gr| gr.einsum(eq, &ops))
}
