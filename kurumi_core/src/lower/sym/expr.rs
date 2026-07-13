//! The symbolic index expression: the `Sym` term tree, its variable/range types, builder
//! constructors + operators, and the pure queries over it (value bounds, affine accumulation,
//! variable substitution). The simplifier that rewrites these lives in `simplify`.

use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Rem};

pub type VarId = u32;

/// Per-variable inclusive range [min, max]. Loop indices are >= 0.
pub type Ranges = HashMap<VarId, (i64, i64)>;

/// Div/Mod are by a positive constant: strides and dim sizes are constants in the
/// static-shape case.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Sym {
    Const(i64),
    Var(VarId),
    Add(Vec<Sym>),
    Mul(i64, Box<Sym>),
    Div(Box<Sym>, i64),
    Mod(Box<Sym>, i64),
}

pub fn c(v: i64) -> Sym {
    Sym::Const(v)
}

pub fn var(id: VarId) -> Sym {
    Sym::Var(id)
}

// Build expressions with operators: Sym + Sym, Sym * c, Sym / c, Sym % c.
impl Add for Sym {
    type Output = Sym;
    fn add(self, o: Sym) -> Sym {
        Sym::Add(vec![self, o])
    }
}
impl Mul<i64> for Sym {
    type Output = Sym;
    fn mul(self, k: i64) -> Sym {
        Sym::Mul(k, Box::new(self))
    }
}
impl Div<i64> for Sym {
    type Output = Sym;
    fn div(self, k: i64) -> Sym {
        Sym::Div(Box::new(self), k)
    }
}
impl Rem<i64> for Sym {
    type Output = Sym;
    fn rem(self, k: i64) -> Sym {
        Sym::Mod(Box::new(self), k)
    }
}

impl Sym {
    /// Inclusive value range under the given variable ranges (saturating).
    pub fn bounds(&self, r: &Ranges) -> (i64, i64) {
        match self {
            Sym::Const(v) => (*v, *v),
            Sym::Var(v) => r.get(v).copied().unwrap_or((i64::MIN, i64::MAX)),
            Sym::Add(ts) => ts.iter().fold((0, 0), |(lo, hi), t| {
                let (l, h) = t.bounds(r);
                (lo.saturating_add(l), hi.saturating_add(h))
            }),
            Sym::Mul(k, e) => {
                let (l, h) = e.bounds(r);
                let (a, b) = (k.saturating_mul(l), k.saturating_mul(h));
                if *k >= 0 { (a, b) } else { (b, a) }
            }
            Sym::Div(e, k) => {
                let (l, h) = e.bounds(r);
                (l.div_euclid(*k), h.div_euclid(*k))
            }
            Sym::Mod(e, k) => {
                let (l, h) = e.bounds(r);
                // within one block => exact, else full [0, k-1]
                if l >= 0 && h - l < *k && l.div_euclid(*k) == h.div_euclid(*k) {
                    (l.rem_euclid(*k), h.rem_euclid(*k))
                } else {
                    (0, *k - 1)
                }
            }
        }
    }

    /// Accumulate this expression as affine into `coeffs[var]` and `base`,
    /// scaled by `scale`. Returns false if it contains div/mod (not affine):
    /// movement views are affine, so this lets the scheduler precompute offsets.
    pub(crate) fn affine(&self, coeffs: &mut [i64], base: &mut i64, scale: i64) -> bool {
        match self {
            Sym::Const(c) => {
                *base += scale * c;
                true
            }
            Sym::Var(v) => {
                coeffs[*v as usize] += scale;
                true
            }
            Sym::Add(ts) => ts.iter().all(|t| t.affine(coeffs, base, scale)),
            Sym::Mul(k, e) => e.affine(coeffs, base, scale * k),
            Sym::Div(..) | Sym::Mod(..) => false,
        }
    }

    /// Substitute variables with expressions (used to compose movement views).
    pub fn subst(&self, m: &HashMap<VarId, Sym>) -> Sym {
        match self {
            Sym::Const(_) => self.clone(),
            Sym::Var(v) => m.get(v).cloned().unwrap_or_else(|| self.clone()),
            Sym::Add(ts) => Sym::Add(ts.iter().map(|t| t.subst(m)).collect()),
            Sym::Mul(k, e) => Sym::Mul(*k, Box::new(e.subst(m))),
            Sym::Div(e, k) => Sym::Div(Box::new(e.subst(m)), *k),
            Sym::Mod(e, k) => Sym::Mod(Box::new(e.subst(m)), *k),
        }
    }
}
