//! Elementwise interp: the dispatch for every unary/binary/compare/select/cast primitive,
//! plus the generic zip/map/compare/select kernels it drives (written once over a `Copy`
//! type and monomorphized by the dispatch macros). Reductions are in `reduce.rs`.

use crate::{Bitwise, Float, Int, Num, Op, Signed, Storage, TensorVal, bitcast, cast};

pub(super) fn eval(op: &Op, inputs: &[&TensorVal]) -> TensorVal {
    match op {
        Op::Cast { to } => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: cast(&a.storage, *to) }
        }
        Op::Bitcast { to } => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: bitcast(&a.storage, *to) }
        }
        Op::Add => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: num_binary!(&a.storage, &b.storage, Num::add) }
        }
        Op::Mul => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: num_binary!(&a.storage, &b.storage, Num::mul) }
        }
        Op::Max => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: num_binary!(&a.storage, &b.storage, Num::max) }
        }
        Op::IDiv => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: int_binary!(&a.storage, &b.storage, Int::idiv) }
        }
        Op::Shl => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: int_binary!(&a.storage, &b.storage, Int::shl) }
        }
        Op::Shr => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: int_binary!(&a.storage, &b.storage, Int::shr) }
        }
        Op::And => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: bitwise_binary!(&a.storage, &b.storage, Bitwise::and) }
        }
        Op::Or => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: bitwise_binary!(&a.storage, &b.storage, Bitwise::or) }
        }
        Op::Xor => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: bitwise_binary!(&a.storage, &b.storage, Bitwise::xor) }
        }
        Op::CmpLt => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: cmp_binary!(&a.storage, &b.storage, PartialOrd::lt) }
        }
        Op::CmpEq => {
            let (a, b) = (inputs[0], inputs[1]);
            TensorVal { shape: a.shape.clone(), storage: cmp_binary!(&a.storage, &b.storage, PartialEq::eq) }
        }
        Op::Where => {
            let (c, a, b) = (inputs[0], inputs[1], inputs[2]);
            let cond = match &c.storage {
                Storage::BOOL(v) => v,
                _ => unreachable!("where cond must be BOOL"),
            };
            let storage = any_binary!(&a.storage, &b.storage, |x, y| select_k(cond, x, y));
            TensorVal { shape: a.shape.clone(), storage }
        }
        Op::Neg => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: signed_unary!(&a.storage, Signed::neg) }
        }
        Op::Recip => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: float_unary!(&a.storage, Float::recip) }
        }
        Op::Sqrt => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: float_unary!(&a.storage, Float::sqrt) }
        }
        Op::Exp2 => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: float_unary!(&a.storage, Float::exp2) }
        }
        Op::Log2 => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: float_unary!(&a.storage, Float::log2) }
        }
        Op::Sin => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: float_unary!(&a.storage, Float::sin) }
        }
        Op::Floor => {
            let a = inputs[0];
            TensorVal { shape: a.shape.clone(), storage: float_unary!(&a.storage, Float::floor) }
        }
        _ => unreachable!("elementwise::eval: non-elementwise op"),
    }
}

pub(crate) fn zip_map<T: Copy>(x: &[T], y: &[T], f: impl Fn(T, T) -> T) -> Vec<T> {
    x.iter().zip(y).map(|(&a, &b)| f(a, b)).collect()
}

pub(crate) fn map1<T: Copy>(v: &[T], f: impl Fn(T) -> T) -> Vec<T> {
    v.iter().map(|&x| f(x)).collect()
}

// comparison -> bool (f takes refs: PartialOrd::lt / PartialEq::eq)
pub(crate) fn cmp_map<T>(x: &[T], y: &[T], f: impl Fn(&T, &T) -> bool) -> Vec<bool> {
    x.iter().zip(y).map(|(a, b)| f(a, b)).collect()
}

// where: cond ? a : b, elementwise
pub(crate) fn select_k<T: Copy>(cond: &[bool], a: &[T], b: &[T]) -> Vec<T> {
    (0..a.len()).map(|i| if cond[i] { a[i] } else { b[i] }).collect()
}
