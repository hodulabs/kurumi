//! Row-tiled tape executor: the innermost (contiguous) axis runs a whole row at a time; each
//! instruction is one monomorphic loop over the row (auto-vectorized), only `depth` scratch
//! rows live (L1-resident). Bit-identical to the per-element walk.

use crate::realize::tape::kernels::{apply_binary_from, apply_unary_from};
use crate::realize::tape::{Instr, Leaf};
use std::cell::RefCell;

thread_local! {
    // reused across run() calls; grows to the largest scratch any fused group
    // needs, then never reallocates: per-group scratch alloc amortizes to 0.
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

// Row-tiled tape executor. The innermost (contiguous) axis runs a whole row at a time;
// each instruction is one monomorphic loop over the row (auto-vectorized), only `depth`
// scratch rows live (L1-resident), no full-tensor intermediates. Two things hold a
// memory-bound op near the streaming-copy ceiling (not 3x below): a contiguous leaf row
// reads in place (buffer slice, no copy), and the terminal op writes straight to `out`
// (so `neg` = one read+write pass). `src[level]` tracks each live operand's `inner` floats
// (buffer slice or scratch row). Bit-identical to the per-element walk: same scalar ops,
// same operand order.
pub(super) fn run_into(leaves: &[Leaf], tape: &[Instr], shape: &[usize], out: &mut Vec<f32>) {
    let rank = shape.len();
    let total: usize = shape.iter().product();
    if total == 0 {
        out.clear();
        return;
    }
    // reused buffers keep their allocation; resize is a no-op when the size is
    // unchanged, and every element is fully overwritten by a terminal op below.
    out.resize(total, 0.0);
    let inner = if rank == 0 { 1 } else { shape[rank - 1] };
    let outer = total / inner;
    let depth = max_stack_depth(tape);
    let need = depth * inner;
    let last = tape.len() - 1;

    // odometer over the OUTER axes only; the inner axis is the row index `i`
    let mut coord = vec![0usize; rank.saturating_sub(1)];
    let mut base: Vec<i64> = leaves.iter().map(|l| l.base).collect();

    SCRATCH.with_borrow_mut(|pool| {
        if pool.len() < need {
            pool.resize(need, 0.0);
        }
        let scratch = &mut pool[..need];
        let mut src: Vec<*const f32> = vec![std::ptr::null(); depth.max(1)];
        for row in 0..outer {
            let out_row = row * inner;
            let mut sp = 0usize;
            for (ii, instr) in tape.iter().enumerate() {
                match instr {
                    Instr::Load(li) => {
                        let li = *li as usize;
                        let l = &leaves[li];
                        let ic = if rank == 0 { 0 } else { l.coeffs[rank - 1] };
                        if l.guards.is_empty() && ic == 1 {
                            // contiguous row = a slice of the buffer: read in place.
                            // SAFETY: base..base+inner is a valid contiguous row (ic == 1).
                            src[sp] = unsafe { l.buf.as_ptr().add(base[li] as usize) };
                        } else {
                            let r = &mut scratch[sp * inner..sp * inner + inner];
                            load_row(r, l, base[li], inner, rank, &coord);
                            src[sp] = r.as_ptr();
                        }
                        sp += 1;
                        if ii == last {
                            // bare-load root (no op above it): copy the loaded row out.
                            // SAFETY: src[sp-1] covers `inner` valid f32.
                            let s = unsafe { std::slice::from_raw_parts(src[sp - 1], inner) };
                            out[out_row..out_row + inner].copy_from_slice(s);
                        }
                    }
                    Instr::Unary(op) => {
                        if ii == last {
                            apply_unary_from(*op, src[sp - 1], &mut out[out_row..out_row + inner]);
                        } else {
                            let lvl = sp - 1;
                            let dst = &mut scratch[lvl * inner..lvl * inner + inner];
                            apply_unary_from(*op, src[lvl], dst);
                            src[lvl] = dst.as_ptr();
                        }
                    }
                    Instr::Binary(op) => {
                        let (a, b) = (src[sp - 2], src[sp - 1]);
                        if ii == last {
                            apply_binary_from(*op, a, b, &mut out[out_row..out_row + inner]);
                        } else {
                            let lvl = sp - 2;
                            let dst = &mut scratch[lvl * inner..lvl * inner + inner];
                            apply_binary_from(*op, a, b, dst);
                            src[lvl] = dst.as_ptr();
                        }
                        sp -= 1;
                    }
                }
            }

            // advance the outer odometer, updating each leaf row-base incrementally
            for i in (0..coord.len()).rev() {
                if coord[i] + 1 < shape[i] {
                    coord[i] += 1;
                    for (li, l) in leaves.iter().enumerate() {
                        base[li] += l.coeffs[i];
                    }
                    break;
                }
                let back = shape[i] as i64 - 1;
                for (li, l) in leaves.iter().enumerate() {
                    base[li] -= l.coeffs[i] * back;
                }
                coord[i] = 0;
            }
        }
    });
}

// max simultaneous live rows = scratch depth (Load +1, Unary 0, Binary -1)
fn max_stack_depth(tape: &[Instr]) -> usize {
    let (mut sp, mut max) = (0usize, 0usize);
    for instr in tape {
        match instr {
            Instr::Load(_) => {
                sp += 1;
                max = max.max(sp);
            }
            Instr::Unary(_) => {}
            Instr::Binary(_) => sp -= 1,
        }
    }
    max
}

// fill a scratch row with leaf `l`'s values at this outer position (`base` = the
// leaf offset at inner index 0). `ic` is the inner-axis stride: 1 = contiguous
// (memcpy), 0 = broadcast (splat), else strided gather. Guards (pad) mask to 0.
fn load_row(row: &mut [f32], l: &Leaf, base: i64, inner: usize, rank: usize, coord: &[usize]) {
    let ic = if rank == 0 { 0 } else { l.coeffs[rank - 1] };
    if l.guards.is_empty() {
        if ic == 1 {
            let b = base as usize;
            row[..inner].copy_from_slice(&l.buf[b..b + inner]);
        } else if ic == 0 {
            row[..inner].fill(l.buf[base as usize]);
        } else {
            for (i, r) in row[..inner].iter_mut().enumerate() {
                *r = l.buf[(base + ic * i as i64) as usize];
            }
        }
        return;
    }
    let inner_axis = rank - 1; // a guard implies a padded axis, so rank >= 1
    for (i, r) in row[..inner].iter_mut().enumerate() {
        let ok = l.guards.iter().all(|g| {
            let c = if g.var as usize == inner_axis { i as i64 } else { coord[g.var as usize] as i64 };
            g.lo <= c && c < g.hi
        });
        *r = if ok { l.buf[(base + ic * i as i64) as usize] } else { 0.0 };
    }
}
