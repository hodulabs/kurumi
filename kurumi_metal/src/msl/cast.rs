//! Dtype cast kernel source. Matches Rust `as`: `!= 0` to bool, saturating truncation
//! float->int, plain C cast otherwise. Launched by `context/dispatch/cast.rs`.

use crate::dtype::msl_ty;
use kurumi_core::DType;

// `cast_k`: in[i] -> out[i] casting `src_dt` to `dst_dt`.
pub(crate) fn cast_msl(src_dt: DType, dst_dt: DType) -> String {
    let sty = msl_ty(src_dt);
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void cast_k(device const {sty}* in [[buffer(0)]], device {}* out [[buffer(1)]],\n\
                            uint i [[thread_position_in_grid]]) {{ {} }}",
        msl_ty(dst_dt),
        cast_body(src_dt, dst_dt)
    )
}

// Cast kernel body `in[i] -> out[i]`, matching Rust `as`:
//   - to BOOL: `!= 0` (nonzero -> 1), not value-preserving truncation.
//   - float -> int: truncate toward zero + saturate to range + NaN -> 0.
//   - otherwise: plain C cast (int<->int wrap/narrow, int/float<->float).
fn cast_body(src_dt: DType, dst_dt: DType) -> String {
    let dty = msl_ty(dst_dt);
    if dst_dt == DType::BOOL {
        return "out[i] = (in[i] != 0) ? 1 : 0;".to_string();
    }
    // float -> {8,16,32}-bit int: truncate toward zero + saturate to range + NaN -> 0.
    // 64-bit ints skip saturation: bounds aren't f32-representable, truncation is exact
    // in-range, and out-of-range f32->i64 is a negligible corner.
    if let (true, Some((lo, hi))) = (src_dt.is_float(), int_bounds(dst_dt)) {
        return format!(
            "float x = (float)in[i]; {dty} r;\n\
             if (isnan(x)) r = 0; else if (x >= {hi}.0f) r = ({dty}){hi}; else if (x <= {lo}.0f) r = ({dty}){lo}; else r = ({dty})x;\n\
             out[i] = r;"
        );
    }
    format!("out[i] = ({dty})in[i];")
}

// (min, max) decimal literals for float->int saturation; None where skipped
// (64-bit ints, non-int dtypes).
fn int_bounds(dt: DType) -> Option<(&'static str, &'static str)> {
    match dt {
        DType::U8 => Some(("0", "255")),
        DType::U16 => Some(("0", "65535")),
        DType::U32 => Some(("0", "4294967295")),
        DType::I8 => Some(("-128", "127")),
        DType::I16 => Some(("-32768", "32767")),
        DType::I32 => Some(("-2147483648", "2147483647")),
        _ => None, // I64/U64 (f32-imprecise bounds) + non-int
    }
}
