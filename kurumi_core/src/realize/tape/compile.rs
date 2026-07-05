//! Compile a lazy elementwise `Expr` into a flat tape over leaves with affine source offsets.
//! Movement views are affine, so each leaf's source offset is precomputed once and advanced
//! incrementally at run time (no per-element Sym/Expr tree walk).

use super::{Instr, Leaf};
use crate::realize::repr::{BinOp, Expr};

pub(super) fn compile(e: &Expr, rank: usize, leaves: &mut Vec<Leaf>, tape: &mut Vec<Instr>) -> bool {
    match e {
        Expr::Load { buf, view } => {
            let mut coeffs = vec![0i64; rank];
            let mut base = 0i64;
            if !view.offset.affine(&mut coeffs, &mut base, 1) {
                return false;
            }
            tape.push(Instr::Load(leaves.len() as u32));
            leaves.push(Leaf { buf: buf.clone(), coeffs, base, guards: view.guards.clone() });
            true
        }
        Expr::Unary(op, a) => compile(a, rank, leaves, tape) && push(tape, Instr::Unary(*op)),
        Expr::Add(a, b) => compile_bin(a, b, BinOp::Add, rank, leaves, tape),
        Expr::Mul(a, b) => compile_bin(a, b, BinOp::Mul, rank, leaves, tape),
        Expr::Max(a, b) => compile_bin(a, b, BinOp::Max, rank, leaves, tape),
    }
}

fn compile_bin(a: &Expr, b: &Expr, op: BinOp, rank: usize, leaves: &mut Vec<Leaf>, tape: &mut Vec<Instr>) -> bool {
    compile(a, rank, leaves, tape) && compile(b, rank, leaves, tape) && push(tape, Instr::Binary(op))
}

fn push(tape: &mut Vec<Instr>, i: Instr) -> bool {
    tape.push(i);
    true
}
