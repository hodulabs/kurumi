//! Monomorphic per-op row kernels. The match is hoisted out of the loop so each arm
//! vectorizes; transcendentals stay scalar libm (same as the oracle), so every result is
//! bit-identical. `apply_*_from` handle the pointer operands the executor threads through.

use crate::realize::expr::{BinOp, UnOp};

// monomorphic per-op loops: the match is hoisted out so each arm vectorizes. Arithmetic
// ops auto-vectorize; transcendentals stay scalar libm (same as the oracle), so every
// result is bit-identical.
fn apply_unary_row(op: UnOp, row: &mut [f32]) {
    match op {
        UnOp::Neg => row.iter_mut().for_each(|x| *x = -*x),
        UnOp::Recip => row.iter_mut().for_each(|x| *x = 1.0 / *x),
        UnOp::Sqrt => row.iter_mut().for_each(|x| *x = x.sqrt()),
        UnOp::Exp2 => row.iter_mut().for_each(|x| *x = x.exp2()),
        UnOp::Log2 => row.iter_mut().for_each(|x| *x = x.log2()),
        UnOp::Sin => row.iter_mut().for_each(|x| *x = x.sin()),
    }
}

fn apply_binary_row(op: BinOp, a: &mut [f32], b: &[f32]) {
    match op {
        BinOp::Add => a.iter_mut().zip(b).for_each(|(x, y)| *x += *y),
        BinOp::Mul => a.iter_mut().zip(b).for_each(|(x, y)| *x *= *y),
        BinOp::Max => a.iter_mut().zip(b).for_each(|(x, y)| *x = x.max(*y)),
    }
}

// dst[j] = op(a[j]). `a` points to dst.len() valid f32 (buffer slice or scratch row) and
// may alias dst (in-place scratch step). The pointer-equality branch keeps the aliasing
// case to one &mut (no overlapping slices); the disjoint case zips two slices to vectorize.
pub(super) fn apply_unary_from(op: UnOp, a: *const f32, dst: &mut [f32]) {
    if a == dst.as_ptr() {
        return apply_unary_row(op, dst);
    }
    // SAFETY: a covers dst.len() valid f32, disjoint from dst on this branch.
    let a = unsafe { std::slice::from_raw_parts(a, dst.len()) };
    match op {
        UnOp::Neg => dst.iter_mut().zip(a).for_each(|(o, x)| *o = -*x),
        UnOp::Recip => dst.iter_mut().zip(a).for_each(|(o, x)| *o = 1.0 / *x),
        UnOp::Sqrt => dst.iter_mut().zip(a).for_each(|(o, x)| *o = x.sqrt()),
        UnOp::Exp2 => dst.iter_mut().zip(a).for_each(|(o, x)| *o = x.exp2()),
        UnOp::Log2 => dst.iter_mut().zip(a).for_each(|(o, x)| *o = x.log2()),
        UnOp::Sin => dst.iter_mut().zip(a).for_each(|(o, x)| *o = x.sin()),
    }
}

// dst[j] = a[j] OP b[j]. `b` is disjoint from dst; `a` may alias dst. All ops are
// commutative, so the in-place branch (op(dst, b) = op(a, b)) is exact.
pub(super) fn apply_binary_from(op: BinOp, a: *const f32, b: *const f32, dst: &mut [f32]) {
    // SAFETY: b covers dst.len() valid f32, disjoint from dst.
    let b = unsafe { std::slice::from_raw_parts(b, dst.len()) };
    if a == dst.as_ptr() {
        return apply_binary_row(op, dst, b);
    }
    // SAFETY: a covers dst.len() valid f32, disjoint from dst on this branch.
    let a = unsafe { std::slice::from_raw_parts(a, dst.len()) };
    match op {
        BinOp::Add => dst.iter_mut().zip(a).zip(b).for_each(|((o, x), y)| *o = *x + *y),
        BinOp::Mul => dst.iter_mut().zip(a).zip(b).for_each(|((o, x), y)| *o = *x * *y),
        BinOp::Max => dst.iter_mut().zip(a).zip(b).for_each(|((o, x), y)| *o = x.max(*y)),
    }
}
