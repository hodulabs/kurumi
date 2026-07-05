//! Integer symbolic simplifier for index expressions.
//!
//! Movement ops lower to address arithmetic over loop indices; reshape emits `/` and `%`.
//! This reduces those using value ranges + algebra so movement chains fuse
//! (provably-identity index => no copy). Failure is safe: the div/mod stays and the
//! scheduler falls back to a `contiguous` copy.

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

    pub fn simplify(&self, r: &Ranges) -> Sym {
        match self {
            Sym::Const(_) | Sym::Var(_) => self.clone(),
            Sym::Add(ts) => {
                let mut flat = Vec::new();
                let mut konst = 0i64;
                for t in ts {
                    match t.simplify(r) {
                        Sym::Const(c) => konst += c,
                        Sym::Add(inner) => flat.extend(inner),
                        s => flat.push(s),
                    }
                }
                if konst != 0 {
                    flat.push(Sym::Const(konst));
                }
                sum(flat)
            }
            Sym::Mul(k, e) => match e.simplify(r) {
                _ if *k == 0 => Sym::Const(0),
                s if *k == 1 => s,
                Sym::Const(c) => Sym::Const(k * c),
                Sym::Mul(k2, e2) => Sym::Mul(k * k2, e2),
                s => Sym::Mul(*k, Box::new(s)),
            },
            Sym::Div(e, k) => simplify_div(&e.simplify(r), *k, r),
            Sym::Mod(e, k) => simplify_mod(&e.simplify(r), *k, r),
        }
    }
}

// (sum exact_multiples_of_k + rest) / k = sum(mult/k) + rest/k    [floor div]
fn simplify_div(e: &Sym, k: i64, r: &Ranges) -> Sym {
    if k == 1 {
        return e.clone();
    }
    if let Sym::Const(v) = e {
        return Sym::Const(v.div_euclid(k));
    }
    let (lo, hi) = e.bounds(r);
    if lo >= 0 && hi < k {
        return Sym::Const(0); // 0 <= x < k  =>  x / k = 0
    }
    let (mut quotient, mut rem) = (Vec::new(), Vec::new());
    for t in as_terms(e) {
        match as_exact_multiple(&t, k) {
            Some(q) => quotient.push(q),
            None => rem.push(t),
        }
    }
    if !quotient.is_empty() {
        quotient.push(Sym::Div(Box::new(sum(rem).simplify(r)), k));
        return sum(quotient).simplify(r);
    }
    Sym::Div(Box::new(e.clone()), k)
}

// (sum exact_multiples_of_k + rest) % k = rest % k
fn simplify_mod(e: &Sym, k: i64, r: &Ranges) -> Sym {
    if k == 1 {
        return Sym::Const(0);
    }
    if let Sym::Const(v) = e {
        return Sym::Const(v.rem_euclid(k));
    }
    let (lo, hi) = e.bounds(r);
    if lo >= 0 && hi < k {
        return e.clone(); // 0 <= x < k  =>  x % k = x
    }
    let terms = as_terms(e);
    let rem: Vec<Sym> = terms.iter().filter(|t| as_exact_multiple(t, k).is_none()).cloned().collect();
    if rem.len() < terms.len() {
        return simplify_mod(&sum(rem).simplify(r), k, r);
    }
    Sym::Mod(Box::new(e.clone()), k)
}

fn as_exact_multiple(t: &Sym, k: i64) -> Option<Sym> {
    match t {
        Sym::Const(v) if v % k == 0 => Some(Sym::Const(v / k)),
        Sym::Mul(coef, inner) if coef % k == 0 => {
            Some(if coef / k == 1 { (**inner).clone() } else { Sym::Mul(coef / k, inner.clone()) })
        }
        _ => None,
    }
}

fn as_terms(e: &Sym) -> Vec<Sym> {
    match e {
        Sym::Add(ts) => ts.clone(),
        _ => vec![e.clone()],
    }
}

fn sum(mut terms: Vec<Sym>) -> Sym {
    match terms.len() {
        0 => Sym::Const(0),
        1 => terms.pop().unwrap(),
        _ => Sym::Add(terms),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges(pairs: &[(VarId, (i64, i64))]) -> Ranges {
        pairs.iter().copied().collect()
    }

    #[test]
    fn const_fold() {
        let r = Ranges::new();
        assert_eq!((c(7) / 2).simplify(&r), c(3));
        assert_eq!((c(7) % 2).simplify(&r), c(1));
    }

    #[test]
    #[allow(clippy::modulo_one)] // deliberately exercising the x % 1 => 0 rule
    fn trivial_divisor() {
        let r = Ranges::new();
        assert_eq!((var(0) / 1).simplify(&r), var(0));
        assert_eq!((var(0) % 1).simplify(&r), c(0));
    }

    #[test]
    fn range_collapse() {
        // j in [0,2] (a size-3 dim)
        let r = ranges(&[(1, (0, 2))]);
        assert_eq!((var(1) / 3).simplify(&r), c(0));
        assert_eq!((var(1) % 3).simplify(&r), var(1));
    }

    #[test]
    fn reshape_cancel() {
        // (3*i + j) with i in [0,1], j in [0,2]:  /3 -> i,  %3 -> j
        let r = ranges(&[(0, (0, 1)), (1, (0, 2))]);
        let expr = || var(0) * 3 + var(1);
        assert_eq!((expr() / 3).simplify(&r), var(0));
        assert_eq!((expr() % 3).simplify(&r), var(1));
    }

    #[test]
    fn fallback_when_unprovable() {
        // unknown range: cannot prove, div/mod stays (scheduler will copy)
        let r = Ranges::new();
        assert!(matches!((var(0) / 3).simplify(&r), Sym::Div(..)));
        assert!(matches!((var(0) % 3).simplify(&r), Sym::Mod(..)));
    }
}
