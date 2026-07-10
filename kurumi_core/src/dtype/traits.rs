//! The element trait hierarchy: one trait per op category (Elem/Num/Signed/Int/
//! Bitwise/Float). This file holds only the trait DEFS; the universal `Elem` impls are in
//! `elem`, the per-type-class impls in `int`/`float`/`complex`. Adding a dtype = adding one
//! type to those impls.

mod complex;
mod elem;
mod float;
mod int;

use crate::Storage;

/// Every storable element: typed access + f64 round-trip (for cast).
pub trait Elem: Copy + Default + 'static {
    fn slice(s: &Storage) -> &[Self];
    fn store(v: Vec<Self>) -> Storage;
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
}

// numeric (excludes BOOL): add/mul/max + reduce identities
pub(crate) trait Num: Elem {
    fn add(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn max(self, o: Self) -> Self;
    fn min(self, o: Self) -> Self;
    fn zero() -> Self;
    fn one() -> Self; // reduce-product identity
    fn lowest() -> Self; // reduce-max identity
}
// signed negation (excludes unsigned)
pub(crate) trait Signed: Elem {
    fn neg(self) -> Self;
}
// integer-only ops (excludes bool + float)
pub(crate) trait Int: Elem {
    fn idiv(self, o: Self) -> Self;
    fn shl(self, o: Self) -> Self;
    fn shr(self, o: Self) -> Self;
}
// bitwise/logical (bool + integers)
pub(crate) trait Bitwise: Elem {
    fn and(self, o: Self) -> Self;
    fn or(self, o: Self) -> Self;
    fn xor(self, o: Self) -> Self;
}
// float-only transcendentals (f16/bf16 compute in f32 then round)
pub(crate) trait Float: Num {
    fn recip(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp2(self) -> Self;
    fn log2(self) -> Self;
    fn sin(self) -> Self;
    fn floor(self) -> Self;
}
