//! Reduction kernel sources: axis reduce, argmax/argmin, argsort. Fold semantics match the
//! CPU oracle. Launched by `context/dispatch/reduce.rs` (the fused parallel-reduce source is
//! generated in `backend/fuse.rs`). Layout is outer x axis_len x inner.

use crate::dtype::{msl_ty, reduce_acc_ty};
use kurumi_core::DType;

// `reduce_k`: serial fold `acc = op(acc, x)` along an axis (keepdim=false). `tag` =
// "sum"|"max"|"prod". Floats accumulate in `float`, ints in their own type (exact); max seeds
// acc with the first element (no per-dtype type-min sentinel). C64 = float2: sum is
// component-add, prod is complex multiply (max is rejected at the builder).
pub(crate) fn reduce_msl(tag: &str, dt: DType) -> String {
    let ty = msl_ty(dt);
    let acc = reduce_acc_ty(dt);
    let cx = dt.is_complex();
    // (init expr, first loop index, fold body); max is seeded by in[base] at k=0.
    let (init, k0, op) = match (cx, tag) {
        (true, "sum") => ("float2(0.0f)".to_string(), 0, "acc + x"),
        (true, "prod") => ("float2(1.0f, 0.0f)".to_string(), 0, "cmul(acc, x)"),
        (false, "sum") => (format!("({acc})0"), 0, "acc + x"),
        (false, "prod") => (format!("({acc})1"), 0, "acc * x"),
        (false, "max") => (format!("({acc})in[base]"), 1, "max(acc, x)"),
        _ => unreachable!("reduce tag {tag} (complex: sum/prod only)"),
    };
    let helpers = if cx {
        "static inline float2 cmul(float2 a, float2 b){ return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }\n"
    } else {
        ""
    };
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{helpers}\
         kernel void reduce_k(device const {ty}* in [[buffer(0)]],\n\
                            device {ty}* out [[buffer(1)]],\n\
                            constant uint& axis_len [[buffer(2)]],\n\
                            constant uint& inner [[buffer(3)]],\n\
                            uint gid [[thread_position_in_grid]]) {{\n\
             uint base = (gid / inner) * axis_len * inner + (gid % inner);\n\
             {acc} acc = {init};\n\
             for (uint k = {k0}; k < axis_len; k++) {{ {acc} x = ({acc})in[base + k * inner]; acc = {op}; }}\n\
             out[gid] = ({ty})acc;\n}}"
    )
}

// `argreduce_k`: argmax/argmin along an axis -> I64 index. `is_max` picks the test; ties keep
// the first occurrence (matches the CPU oracle).
pub(crate) fn argreduce_msl(ty: &str, is_max: bool) -> String {
    let cmp = if is_max { ">" } else { "<" };
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void argreduce_k(device const {ty}* in [[buffer(0)]],\n\
                            device long* out [[buffer(1)]],\n\
                            constant uint& axis_len [[buffer(2)]],\n\
                            constant uint& inner [[buffer(3)]],\n\
                            uint gid [[thread_position_in_grid]]) {{\n\
             uint base = (gid / inner) * axis_len * inner + (gid % inner);\n\
             {ty} best = in[base]; long bi = 0;\n\
             for (uint k = 1; k < axis_len; k++) {{ {ty} x = in[base + k * inner]; if (x {cmp} best) {{ best = x; bi = (long)k; }} }}\n\
             out[gid] = bi;\n}}"
    )
}

// `argsort_k`: argsort along an axis -> I64 permutation. One thread per line runs a stable
// in-place insertion sort of the index array. Strict comparison keeps ties in original order
// (matches the oracle); `descending` flips it. O(L^2) per line (bitonic if long axes get hot).
pub(crate) fn argsort_msl(ty: &str, descending: bool) -> String {
    let cmp = if descending { "fv < ev" } else { "fv > ev" };
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void argsort_k(device const {ty}* in [[buffer(0)]], device long* out [[buffer(1)]],\n\
                            constant uint& axis_len [[buffer(2)]], constant uint& inner [[buffer(3)]],\n\
                            uint gid [[thread_position_in_grid]]) {{\n\
             uint base = (gid / inner) * axis_len * inner + (gid % inner);\n\
             for (uint k = 0; k < axis_len; k++) out[base + k * inner] = (long)k;\n\
             for (uint i = 1; i < axis_len; i++) {{\n\
                 long e = out[base + i * inner]; {ty} ev = in[base + (uint)e * inner]; uint jj = i;\n\
                 while (jj > 0) {{ long f = out[base + (jj - 1) * inner]; {ty} fv = in[base + (uint)f * inner];\n\
                     if (!({cmp})) break; out[base + jj * inner] = f; jj--; }}\n\
                 out[base + jj * inner] = e; }} }}"
    )
}
