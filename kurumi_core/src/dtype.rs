//! Element dtypes and typed storage. One generic kernel per op, dispatched
//! to every dtype at a single match (dtype add = one mapping).
//! Submodules: `dispatch` (the dtype-dispatch macros, re-exported crate-wide via the
//! `#[macro_use]` chain), `traits` (the element trait hierarchy) and `convert` (cast/
//! bitcast/iota).

#[macro_use]
mod dispatch;
mod convert;
mod traits;

pub(crate) use convert::{bitcast, cast, iota_storage};
pub(crate) use traits::{Bitwise, Elem, Float, Int, Num, Signed};

use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};
use num_complex::Complex;

/// Native element types. Reserved (quant track): F8E4M3 F8E5M2 F8E8M0 F4(MX)
/// + Quant descriptor: those need the quantization subsystem, not just a tag.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DType {
    BOOL,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F8E4M3, // 8-bit float (4-exp, 3-mant): FP8 weights/activations
    F8E5M2, // 8-bit float (5-exp, 2-mant): FP8 gradients (wider range)
    F16,
    BF16,
    F32,
    F64,
    C64,  // complex<f32> (two f32: re, im)
    C128, // complex<f64>
}

impl DType {
    pub fn is_float(self) -> bool {
        matches!(self, DType::F8E4M3 | DType::F8E5M2 | DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }
    pub fn is_int(self) -> bool {
        matches!(
            self,
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64
        )
    }
    /// Complex dtypes are numeric but NOT ordered: max/min/cmp/floor/argmax/sort
    /// reject them (see the builder gates); their real counterpart for real()/imag().
    pub fn is_complex(self) -> bool {
        matches!(self, DType::C64 | DType::C128)
    }
    pub fn is_numeric(self) -> bool {
        self.is_int() || self.is_float()
    }
    /// Dtypes that support field arithmetic (+,*,-,/): numeric or complex. (Order-
    /// based ops like max/min/cmp/argmax/sort/floor stay `is_numeric`, excluding complex.)
    pub fn is_arith(self) -> bool {
        self.is_numeric() || self.is_complex()
    }
    pub fn is_signed(self) -> bool {
        self.is_float() || matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64)
    }
    /// The real component dtype of a complex dtype (C64 -> F32, C128 -> F64).
    pub fn real(self) -> DType {
        match self {
            DType::C64 => DType::F32,
            DType::C128 => DType::F64,
            _ => self,
        }
    }
    /// Size in bytes of one element (bitcast requires matching widths).
    pub fn width(self) -> usize {
        match self {
            DType::BOOL | DType::U8 | DType::I8 | DType::F8E4M3 | DType::F8E5M2 => 1,
            DType::U16 | DType::I16 | DType::F16 | DType::BF16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 | DType::C64 => 8,
            DType::C128 => 16,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Storage {
    BOOL(Vec<bool>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F8E4M3(Vec<F8E4M3>),
    F8E5M2(Vec<F8E5M2>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    C64(Vec<Complex<f32>>),
    C128(Vec<Complex<f64>>),
}

impl Storage {
    pub fn dtype(&self) -> DType {
        match self {
            Storage::BOOL(_) => DType::BOOL,
            Storage::U8(_) => DType::U8,
            Storage::U16(_) => DType::U16,
            Storage::U32(_) => DType::U32,
            Storage::U64(_) => DType::U64,
            Storage::I8(_) => DType::I8,
            Storage::I16(_) => DType::I16,
            Storage::I32(_) => DType::I32,
            Storage::I64(_) => DType::I64,
            Storage::F8E4M3(_) => DType::F8E4M3,
            Storage::F8E5M2(_) => DType::F8E5M2,
            Storage::F16(_) => DType::F16,
            Storage::BF16(_) => DType::BF16,
            Storage::F32(_) => DType::F32,
            Storage::F64(_) => DType::F64,
            Storage::C64(_) => DType::C64,
            Storage::C128(_) => DType::C128,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Storage::BOOL(v) => v.len(),
            Storage::U8(v) => v.len(),
            Storage::U16(v) => v.len(),
            Storage::U32(v) => v.len(),
            Storage::U64(v) => v.len(),
            Storage::I8(v) => v.len(),
            Storage::I16(v) => v.len(),
            Storage::I32(v) => v.len(),
            Storage::I64(v) => v.len(),
            Storage::F8E4M3(v) => v.len(),
            Storage::F8E5M2(v) => v.len(),
            Storage::F16(v) => v.len(),
            Storage::BF16(v) => v.len(),
            Storage::F32(v) => v.len(),
            Storage::F64(v) => v.len(),
            Storage::C64(v) => v.len(),
            Storage::C128(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    // f32 accessors for the f32-only realize fast path (panic on other dtypes)
    pub(crate) fn as_f32(&self) -> &[f32] {
        f32::slice(self)
    }
    pub(crate) fn into_f32(self) -> Vec<f32> {
        match self {
            Storage::F32(v) => v,
            _ => panic!("expected F32, got {:?}", self.dtype()),
        }
    }
    /// Reinterpret little-endian `bytes` as a storage of `dtype` (bitwise; len must
    /// be a multiple of the element width). For the C ABI / any raw-buffer exchange.
    pub fn from_bytes(dtype: DType, bytes: &[u8]) -> Storage {
        convert::storage_from_bytes(bytes, dtype)
    }
    /// This storage's little-endian byte representation.
    pub fn to_bytes(&self) -> Vec<u8> {
        convert::storage_to_bytes(self)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct TensorVal {
    pub shape: Vec<usize>,
    pub storage: Storage,
}

impl TensorVal {
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }
    /// f32 view (panics on other dtypes): convenience for the f32 hot path/tests.
    pub fn f32(&self) -> &[f32] {
        f32::slice(&self.storage)
    }
}
