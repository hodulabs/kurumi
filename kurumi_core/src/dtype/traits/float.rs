//! Float element impls: Num, Signed, and Float transcendentals for f32/f64/f16/bf16/FP8.
//! f16/bf16/FP8 upcast to f32, compute, round back. Trait defs are in the parent `traits`.

use crate::dtype::traits::{Float, Num, Signed};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};

macro_rules! impl_num_float {
    ($t:ty, $zero:expr, $one:expr, $low:expr, $max:expr, $min:expr) => {
        impl Num for $t {
            fn add(self, o: Self) -> Self {
                self + o
            }
            fn mul(self, o: Self) -> Self {
                self * o
            }
            fn max(self, o: Self) -> Self {
                $max(self, o)
            }
            fn min(self, o: Self) -> Self {
                $min(self, o)
            }
            fn zero() -> Self {
                $zero
            }
            fn one() -> Self {
                $one
            }
            fn lowest() -> Self {
                $low
            }
        }
    };
}
// f32/f64 max = the IEEE max the row executor uses, so realize == interpret bit-for-bit
impl_num_float!(f32, 0.0, 1.0, f32::NEG_INFINITY, f32::max, f32::min);
impl_num_float!(f64, 0.0, 1.0, f64::NEG_INFINITY, f64::max, f64::min);
impl_num_float!(
    f16,
    f16::ZERO,
    f16::ONE,
    f16::NEG_INFINITY,
    |a: f16, b: f16| f16::from_f32(a.to_f32().max(b.to_f32())),
    |a: f16, b: f16| f16::from_f32(a.to_f32().min(b.to_f32()))
);
impl_num_float!(
    bf16,
    bf16::ZERO,
    bf16::ONE,
    bf16::NEG_INFINITY,
    |a: bf16, b: bf16| bf16::from_f32(a.to_f32().max(b.to_f32())),
    |a: bf16, b: bf16| bf16::from_f32(a.to_f32().min(b.to_f32()))
);
// FP8: arithmetic upcasts to f32 then rounds (no native FP8 ALU). `lowest` uses
// f32::MIN, which saturates to the format's most-negative finite value (no inf in
// e4m3): a valid reduce-max identity.
impl_num_float!(
    F8E4M3,
    F8E4M3::from_f32(0.0),
    F8E4M3::from_f32(1.0),
    F8E4M3::from_f32(f32::MIN),
    |a: F8E4M3, b: F8E4M3| F8E4M3::from_f32(a.to_f32().max(b.to_f32())),
    |a: F8E4M3, b: F8E4M3| F8E4M3::from_f32(a.to_f32().min(b.to_f32()))
);
impl_num_float!(
    F8E5M2,
    F8E5M2::from_f32(0.0),
    F8E5M2::from_f32(1.0),
    F8E5M2::from_f32(f32::MIN),
    |a: F8E5M2, b: F8E5M2| F8E5M2::from_f32(a.to_f32().max(b.to_f32())),
    |a: F8E5M2, b: F8E5M2| F8E5M2::from_f32(a.to_f32().min(b.to_f32()))
);

impl Signed for f32 {
    fn neg(self) -> Self {
        -self
    }
}
impl Signed for f64 {
    fn neg(self) -> Self {
        -self
    }
}
impl Signed for F8E4M3 {
    fn neg(self) -> Self {
        -self
    }
}
impl Signed for F8E5M2 {
    fn neg(self) -> Self {
        -self
    }
}
impl Signed for f16 {
    fn neg(self) -> Self {
        -self
    }
}
impl Signed for bf16 {
    fn neg(self) -> Self {
        -self
    }
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            fn recip(self) -> Self {
                1.0 / self
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn exp2(self) -> Self {
                <$t>::exp2(self)
            }
            fn log2(self) -> Self {
                <$t>::log2(self)
            }
            fn sin(self) -> Self {
                <$t>::sin(self)
            }
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
        }
    };
}
impl_float!(f32);
impl_float!(f64);

// f16/bf16: upcast to f32, compute, round back (no CPU perf win; real perf is
// on Metal, this only provides correct storage + semantics)
macro_rules! impl_float_half {
    ($t:ty) => {
        impl Float for $t {
            fn recip(self) -> Self {
                <$t>::from_f32(self.to_f32().recip())
            }
            fn sqrt(self) -> Self {
                <$t>::from_f32(self.to_f32().sqrt())
            }
            fn exp2(self) -> Self {
                <$t>::from_f32(self.to_f32().exp2())
            }
            fn log2(self) -> Self {
                <$t>::from_f32(self.to_f32().log2())
            }
            fn sin(self) -> Self {
                <$t>::from_f32(self.to_f32().sin())
            }
            fn floor(self) -> Self {
                <$t>::from_f32(self.to_f32().floor())
            }
        }
    };
}
impl_float_half!(f16);
impl_float_half!(bf16);
impl_float_half!(F8E4M3); // FP8 transcendentals upcast to f32 then round
impl_float_half!(F8E5M2);
