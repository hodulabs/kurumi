//! Complex element impls (C64/C128): numeric (arithmetic + complex-valued transcendentals)
//! but NOT ordered. max/min/lowest/floor panic: the builder gates every order-based op
//! (max/reduce_max/cmp/argmax/sort/floor) away from complex, so these are unreachable in
//! practice. to_f64 = real part (lossy); the real->complex cast uses from_f64 (im=0), and
//! complex->complex casts are special-cased in convert.rs to keep the imaginary part. Trait
//! defs are in the parent `traits`.

use super::{Elem, Float, Num, Signed};
use crate::{DType, Storage};
use num_complex::Complex;

macro_rules! impl_complex {
    ($variant:ident, $r:ty) => {
        impl Elem for Complex<$r> {
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
                self.re as f64
            }
            fn from_f64(x: f64) -> Self {
                Complex::new(x as $r, 0.0)
            }
        }
        impl Num for Complex<$r> {
            fn add(self, o: Self) -> Self {
                self + o
            }
            fn mul(self, o: Self) -> Self {
                self * o
            }
            fn max(self, _o: Self) -> Self {
                panic!("complex dtype has no total order (max)")
            }
            fn min(self, _o: Self) -> Self {
                panic!("complex dtype has no total order (min)")
            }
            fn zero() -> Self {
                Complex::new(0.0, 0.0)
            }
            fn one() -> Self {
                Complex::new(1.0, 0.0)
            }
            fn lowest() -> Self {
                panic!("complex dtype has no total order (reduce-max)")
            }
        }
        impl Signed for Complex<$r> {
            fn neg(self) -> Self {
                -self
            }
        }
        impl Float for Complex<$r> {
            fn recip(self) -> Self {
                self.inv()
            }
            fn sqrt(self) -> Self {
                Complex::sqrt(self)
            }
            fn exp2(self) -> Self {
                (self * (2.0 as $r).ln()).exp()
            }
            fn log2(self) -> Self {
                self.ln() / (2.0 as $r).ln()
            }
            fn sin(self) -> Self {
                Complex::sin(self)
            }
            fn floor(self) -> Self {
                panic!("complex dtype has no floor")
            }
        }
    };
}
impl_complex!(C64, f32);
impl_complex!(C128, f64);
