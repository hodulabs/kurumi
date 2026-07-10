//! The universal `Elem` impls: typed storage access + f64 round-trip, for all 16 dtypes.
//! The trait hierarchy is declared in the parent (`traits.rs`); the per-class ops live in
//! `int`/`float`/`complex`. Adding a dtype = one `impl_elem!` line here plus its class impl.
use crate::dtype::traits::Elem;
use crate::{DType, Storage};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};

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
