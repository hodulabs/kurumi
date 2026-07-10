//! Compiled fused kernel: a lazy elementwise `Expr` (from the realize scheduler) compiles
//! to a flat tape over leaves with affine source offsets (`compile`), run row-by-row (`run`,
//! one monomorphic loop per op -> auto-vectorized, kernels in `kernels`). Scalar per-element
//! walk (`eval_expr`) is the fallback for the rare non-affine leaf.

mod compile;
mod kernels;
mod run;

use crate::lower::index::{self, Guard};
use crate::realize::expr::{BinOp, Expr, UnOp};
use std::rc::Rc;

fn eval_expr(e: &Expr, coord: &[usize]) -> f32 {
    match e {
        Expr::Load { buf, view } => index::load_at(buf, view, coord),
        Expr::Unary(op, a) => op.apply(eval_expr(a, coord)),
        Expr::Add(a, b) => eval_expr(a, coord) + eval_expr(b, coord),
        Expr::Mul(a, b) => eval_expr(a, coord) * eval_expr(b, coord),
        Expr::Max(a, b) => eval_expr(a, coord).max(eval_expr(b, coord)),
    }
}

// compiled fused kernel: a flat tape over leaves with affine offsets. Movement views are
// affine, so each leaf's source offset is precomputed once and advanced incrementally
// (no per-element Sym/Expr tree walk).

struct Leaf {
    buf: Rc<[f32]>,
    coeffs: Vec<i64>, // source-offset coefficient per output axis
    base: i64,
    guards: Vec<Guard>,
}

enum Instr {
    Load(u32),
    Unary(UnOp),
    Binary(BinOp),
}

pub(super) fn eval_fused(expr: &Expr, shape: &[usize]) -> Vec<f32> {
    let mut out = Vec::new();
    eval_fused_into(expr, shape, &mut out);
    out
}

// Fill `out` with the fused result (reusing its allocation: the eval-loop path,
// no per-call output alloc/page-fault). `out` is fully overwritten.
pub(super) fn eval_fused_into(expr: &Expr, shape: &[usize], out: &mut Vec<f32>) {
    crate::realize::bump_kernel(); // one fused elementwise group = one pass
    let rank = shape.len();
    let (mut leaves, mut tape) = (Vec::new(), Vec::new());
    if compile::compile(expr, rank, &mut leaves, &mut tape) {
        run::run_into(&leaves, &tape, shape, out);
        return;
    }
    // fallback (non-affine leaf, shouldn't occur): per-element tree walk
    let len = shape.iter().product::<usize>().max(1);
    out.clear();
    out.reserve(len);
    let mut coord = vec![0usize; rank];
    for _ in 0..len {
        out.push(eval_expr(expr, &coord));
        crate::inc(&mut coord, shape);
    }
}
