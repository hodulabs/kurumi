//! Dtype conversions: value cast (through an f64 intermediate), bit reinterpret
//! (bitcast, via native bytes), and the iota generator. All routed through the
//! `Elem` f64 round-trip so a new dtype needs no new conversion code.

use crate::dtype::Elem;
use crate::{DType, Storage};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};
use num_complex::Complex;

/// Convert between dtypes through an f64 intermediate.
// i64 magnitudes > 2^53 lose precision through f64; revisit if exact wide-int
// casts are needed (none in the float-centric paths today).
pub(crate) fn cast(src: &Storage, to: DType) -> Storage {
    // complex -> complex: convert both parts (the f64 real-part path drops the imag).
    match (src, to) {
        (Storage::C64(v), DType::C128) => {
            return Storage::C128(v.iter().map(|z| Complex::new(z.re as f64, z.im as f64)).collect());
        }
        (Storage::C128(v), DType::C64) => {
            return Storage::C64(v.iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect());
        }
        (Storage::C64(_), DType::C64) | (Storage::C128(_), DType::C128) => return src.clone(),
        _ => {}
    }
    // otherwise via f64: real->complex sets imag 0 (from_f64); complex->real takes re.
    let f: Vec<f64> = match src {
        Storage::BOOL(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::U8(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::U16(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::U32(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::U64(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::I8(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::I16(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::I32(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::I64(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::F8E4M3(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::F8E5M2(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::F16(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::BF16(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::F32(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::F64(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::C64(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
        Storage::C128(v) => v.iter().map(|&x| Elem::to_f64(x)).collect(),
    };
    f64_to_storage(f, to)
}

// build a storage of `to` from f64 values (cast target / iota generator)
fn f64_to_storage(f: Vec<f64>, to: DType) -> Storage {
    match to {
        DType::BOOL => Storage::BOOL(f.iter().map(|&x| bool::from_f64(x)).collect()),
        DType::U8 => Storage::U8(f.iter().map(|&x| u8::from_f64(x)).collect()),
        DType::U16 => Storage::U16(f.iter().map(|&x| u16::from_f64(x)).collect()),
        DType::U32 => Storage::U32(f.iter().map(|&x| u32::from_f64(x)).collect()),
        DType::U64 => Storage::U64(f.iter().map(|&x| u64::from_f64(x)).collect()),
        DType::I8 => Storage::I8(f.iter().map(|&x| i8::from_f64(x)).collect()),
        DType::I16 => Storage::I16(f.iter().map(|&x| i16::from_f64(x)).collect()),
        DType::I32 => Storage::I32(f.iter().map(|&x| i32::from_f64(x)).collect()),
        DType::I64 => Storage::I64(f.iter().map(|&x| i64::from_f64(x)).collect()),
        DType::F8E4M3 => Storage::F8E4M3(f.iter().map(|&x| Elem::from_f64(x)).collect()),
        DType::F8E5M2 => Storage::F8E5M2(f.iter().map(|&x| Elem::from_f64(x)).collect()),
        DType::F16 => Storage::F16(f.iter().map(|&x| f16::from_f64(x)).collect()),
        DType::BF16 => Storage::BF16(f.iter().map(|&x| bf16::from_f64(x)).collect()),
        DType::F32 => Storage::F32(f.iter().map(|&x| f32::from_f64(x)).collect()),
        DType::F64 => Storage::F64(f),
        DType::C64 => Storage::C64(f.iter().map(|&x| Complex::new(x as f32, 0.0)).collect()),
        DType::C128 => Storage::C128(f.iter().map(|&x| Complex::new(x, 0.0)).collect()),
    }
}

// iota: value at each position = its index along `axis` (StableHLO iota).
// arange/one-hot/tril/position-ids all build on this (O(1) in the IR).
pub(crate) fn iota_storage(shape: &[usize], axis: usize, dtype: DType) -> Storage {
    let len: usize = shape.iter().product();
    let stride: usize = shape[axis + 1..].iter().product();
    let axis_len = shape[axis];
    let idx: Vec<f64> = (0..len).map(|flat| ((flat / stride) % axis_len) as f64).collect();
    f64_to_storage(idx, dtype)
}

// raw native bytes of each element (for bitcast)
pub(crate) fn storage_to_bytes(s: &Storage) -> Vec<u8> {
    match s {
        Storage::BOOL(v) => v.iter().map(|&x| x as u8).collect(),
        Storage::U8(v) => v.clone(),
        Storage::U16(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        Storage::U32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        Storage::U64(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        Storage::I8(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        Storage::I16(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        Storage::I32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        Storage::I64(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        Storage::F32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        Storage::F64(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        Storage::F16(v) => v.iter().flat_map(|x| x.to_bits().to_ne_bytes()).collect(),
        Storage::BF16(v) => v.iter().flat_map(|x| x.to_bits().to_ne_bytes()).collect(),
        Storage::F8E4M3(v) => v.iter().map(|x| x.to_bits()).collect(),
        Storage::F8E5M2(v) => v.iter().map(|x| x.to_bits()).collect(),
        Storage::C64(v) => v.iter().flat_map(|z| [z.re.to_ne_bytes(), z.im.to_ne_bytes()].concat()).collect(),
        Storage::C128(v) => v.iter().flat_map(|z| [z.re.to_ne_bytes(), z.im.to_ne_bytes()].concat()).collect(),
    }
}

pub(crate) fn storage_from_bytes(b: &[u8], to: DType) -> Storage {
    fn chunks<const N: usize>(b: &[u8]) -> impl Iterator<Item = [u8; N]> + '_ {
        b.chunks_exact(N).map(|c| c.try_into().unwrap())
    }
    match to {
        DType::BOOL => Storage::BOOL(b.iter().map(|&x| x != 0).collect()),
        DType::U8 => Storage::U8(b.to_vec()),
        DType::U16 => Storage::U16(chunks(b).map(u16::from_ne_bytes).collect()),
        DType::U32 => Storage::U32(chunks(b).map(u32::from_ne_bytes).collect()),
        DType::U64 => Storage::U64(chunks(b).map(u64::from_ne_bytes).collect()),
        DType::I8 => Storage::I8(chunks(b).map(i8::from_ne_bytes).collect()),
        DType::I16 => Storage::I16(chunks(b).map(i16::from_ne_bytes).collect()),
        DType::I32 => Storage::I32(chunks(b).map(i32::from_ne_bytes).collect()),
        DType::I64 => Storage::I64(chunks(b).map(i64::from_ne_bytes).collect()),
        DType::F32 => Storage::F32(chunks(b).map(f32::from_ne_bytes).collect()),
        DType::F64 => Storage::F64(chunks(b).map(f64::from_ne_bytes).collect()),
        DType::F16 => Storage::F16(chunks(b).map(|c| f16::from_bits(u16::from_ne_bytes(c))).collect()),
        DType::BF16 => Storage::BF16(chunks(b).map(|c| bf16::from_bits(u16::from_ne_bytes(c))).collect()),
        DType::F8E4M3 => Storage::F8E4M3(b.iter().map(|&x| F8E4M3::from_bits(x)).collect()),
        DType::F8E5M2 => Storage::F8E5M2(b.iter().map(|&x| F8E5M2::from_bits(x)).collect()),
        DType::C64 => Storage::C64(
            chunks::<8>(b)
                .map(|c| {
                    Complex::new(
                        f32::from_ne_bytes(c[..4].try_into().unwrap()),
                        f32::from_ne_bytes(c[4..].try_into().unwrap()),
                    )
                })
                .collect(),
        ),
        DType::C128 => Storage::C128(
            chunks::<16>(b)
                .map(|c| {
                    Complex::new(
                        f64::from_ne_bytes(c[..8].try_into().unwrap()),
                        f64::from_ne_bytes(c[8..].try_into().unwrap()),
                    )
                })
                .collect(),
        ),
    }
}

/// Reinterpret the bits as another dtype (no value conversion). Same-width only
/// at the builder, so the shape is unchanged.
pub(crate) fn bitcast(src: &Storage, to: DType) -> Storage {
    storage_from_bytes(&storage_to_bytes(src), to)
}
