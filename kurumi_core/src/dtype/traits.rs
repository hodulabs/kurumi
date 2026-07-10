//! The element trait hierarchy: one trait per op category (Elem/Num/Signed/Int/
//! Bitwise/Float). Trait defs + the `Elem` impls live here; the per-type-class impls are in
//! the submodules (`int`/`float`/`complex`). Adding a dtype = adding one type to these impls.

mod complex;
mod float;
mod int;

use crate::{DType, Storage};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};

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

macro_rules! impl_elem {
    ($t:ty, $variant:ident, $to:expr, $from:expr) => {
        impl Elem for $t {
            fn slice(s: &Storage) -> &[Self] {
                match s {
                    Storage::$variant(v) => v,
                    _ => panic!("dtype mismatch: want {:?}, got {:?}", DType::$variant, s.dtype()),
                }
            }
            fn store(v: Vec<Self>) -> Storage {
                Storage::$variant(v)
            }
            fn to_f64(self) -> f64 {
                $to(self)
            }
            fn from_f64(x: f64) -> Self {
                $from(x)
            }
        }
    };
}

impl_elem!(bool, BOOL, |v| if v { 1.0 } else { 0.0 }, |x: f64| x != 0.0);
impl_elem!(u8, U8, |v| v as f64, |x: f64| x as u8);
impl_elem!(u16, U16, |v| v as f64, |x: f64| x as u16);
impl_elem!(u32, U32, |v| v as f64, |x: f64| x as u32);
impl_elem!(u64, U64, |v| v as f64, |x: f64| x as u64);
impl_elem!(i8, I8, |v| v as f64, |x: f64| x as i8);
impl_elem!(i16, I16, |v| v as f64, |x: f64| x as i16);
impl_elem!(i32, I32, |v| v as f64, |x: f64| x as i32);
impl_elem!(i64, I64, |v| v as f64, |x: f64| x as i64);
impl_elem!(f16, F16, |v: f16| v.to_f64(), |x: f64| f16::from_f64(x));
impl_elem!(bf16, BF16, |v: bf16| v.to_f64(), |x: f64| bf16::from_f64(x));
impl_elem!(F8E4M3, F8E4M3, |v: F8E4M3| v.to_f32() as f64, |x: f64| F8E4M3::from_f32(x as f32));
impl_elem!(F8E5M2, F8E5M2, |v: F8E5M2| v.to_f32() as f64, |x: f64| F8E5M2::from_f32(x as f32));
impl_elem!(f32, F32, |v| v as f64, |x: f64| x as f32);
impl_elem!(f64, F64, |v| v, |x: f64| x);
