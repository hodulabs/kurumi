//! DType -> Metal representation: MSL type name, MPS data type, device-path gates,
//! reduce accumulator type. Single source of truth (was duplicated across context/backend).

use kurumi_core::DType;
use objc2_metal_performance_shaders::MPSDataType;

// device-path element dtype -> MSL type name. Every Metal-native dtype maps
// (f32/f16/bf16 + bool + 8 int widths); FP8/f64/C128 never reach the device path.
pub(crate) fn msl_ty(dt: DType) -> &'static str {
    match dt {
        DType::BOOL | DType::U8 => "uchar",
        DType::U16 => "ushort",
        DType::U32 => "uint",
        DType::U64 => "ulong",
        DType::I8 => "char",
        DType::I16 => "short",
        DType::I32 => "int",
        DType::I64 => "long",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        DType::C64 => "float2", // complex<f32> = (re, im); C128 has no device path (no double)
        _ => "float",           // F32 (F64/FP8/C128 never reach the device path)
    }
}

// elementwise / reduce / movement / gather / cmp / where / cast run device-resident
// on every Metal-native dtype: f32/f16/bf16 + bool + the 8 integer widths. (FP8/f64/
// complex have no native Metal ALU -> CPU oracle.)
pub(crate) fn dev_dtype(dt: DType) -> bool {
    matches!(
        dt,
        DType::F32
            | DType::F16
            | DType::BF16
            | DType::BOOL
            | DType::U8
            | DType::U16
            | DType::U32
            | DType::U64
            | DType::I8
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::C64 // complex<f32> as float2: arithmetic (cmul/crecip), transcendentals
                         // (cexp2/clog2/csqrt/csin), reduce (sum/prod), pad, where, matmul,
                         // and the f32<->C64 cast seam all run device. C128 = host (no double).
    )
}

// dtypes MPS GEMM can run (float only; integer matmul has no simdgroup/MPS path and
// stays on the naive host-offload kernel).
pub(crate) fn mps_dtype(dt: DType) -> bool {
    matches!(dt, DType::F32 | DType::F16 | DType::BF16)
}

// dtype -> MPS data type for the MPS GEMM path (floats only reach here).
pub(crate) fn mps_ty(dt: DType) -> MPSDataType {
    match dt {
        DType::F16 => MPSDataType::Float16,
        DType::BF16 => MPSDataType::BFloat16,
        _ => MPSDataType::Float32,
    }
}

// reduce accumulator type: floats fold in `float`; ints in their own type (exact);
// complex in float2 (component-add / complex-multiply).
pub(crate) fn reduce_acc_ty(dt: DType) -> &'static str {
    if dt.is_int() {
        msl_ty(dt)
    } else if dt.is_complex() {
        "float2"
    } else {
        "float"
    }
}
