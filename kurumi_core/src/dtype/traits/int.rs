//! Integer element impls: numeric (wrapping) + Int (idiv/shl/shr) + Bitwise, plus bool
//! Bitwise and integer Signed. Trait defs are in the parent `traits`.

use super::{Bitwise, Int, Num, Signed};

// int numerics wrap on overflow (defined behaviour for tensor arithmetic)
macro_rules! impl_num_int {
    ($t:ty) => {
        impl Num for $t {
            fn add(self, o: Self) -> Self {
                self.wrapping_add(o)
            }
            fn mul(self, o: Self) -> Self {
                self.wrapping_mul(o)
            }
            fn max(self, o: Self) -> Self {
                Ord::max(self, o)
            }
            fn min(self, o: Self) -> Self {
                Ord::min(self, o)
            }
            fn zero() -> Self {
                0
            }
            fn one() -> Self {
                1
            }
            fn lowest() -> Self {
                <$t>::MIN
            }
        }
        impl Int for $t {
            // x/0 = 0 (defined, panic-free); wrapping_div also handles MIN/-1
            fn idiv(self, o: Self) -> Self {
                if o == 0 { 0 } else { self.wrapping_div(o) }
            }
            fn shl(self, o: Self) -> Self {
                self.wrapping_shl(o as u32)
            }
            fn shr(self, o: Self) -> Self {
                self.wrapping_shr(o as u32)
            }
        }
        impl Bitwise for $t {
            fn and(self, o: Self) -> Self {
                self & o
            }
            fn or(self, o: Self) -> Self {
                self | o
            }
            fn xor(self, o: Self) -> Self {
                self ^ o
            }
        }
    };
}
impl_num_int!(u8);
impl_num_int!(u16);
impl_num_int!(u32);
impl_num_int!(u64);
impl_num_int!(i8);
impl_num_int!(i16);
impl_num_int!(i32);
impl_num_int!(i64);

impl Bitwise for bool {
    fn and(self, o: Self) -> Self {
        self && o
    }
    fn or(self, o: Self) -> Self {
        self || o
    }
    fn xor(self, o: Self) -> Self {
        self ^ o
    }
}

impl Signed for i8 {
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}
impl Signed for i16 {
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}
impl Signed for i32 {
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}
impl Signed for i64 {
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}
