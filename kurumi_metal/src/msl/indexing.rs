//! Gather / scatter kernel sources launched by `context/dispatch/indexing.rs`. `ty` = the
//! operand/output MSL element type; indices are i32.

use kurumi_core::DType;

// Gather (jnp.take): operand laid out [pre, da, post], indices flattened to length k; output
// [pre, k, post]. out[p,j,t] = operand[p, clamp(idx[j]), t].
pub(crate) fn gather_msl(ty: &str) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void gather_k(device const {ty}* operand [[buffer(0)]],\n\
                              device const int* idx [[buffer(1)]],\n\
                              device {ty}* out [[buffer(2)]],\n\
                              constant uint& k [[buffer(3)]],\n\
                              constant uint& post [[buffer(4)]],\n\
                              constant uint& da [[buffer(5)]],\n\
                              uint gid [[thread_position_in_grid]]) {{\n\
             uint kp = k * post; uint p = gid / kp; uint rem = gid - p * kp;\n\
             uint j = rem / post; uint t = rem - j * post;\n\
             int a = idx[j]; uint au = a < 0 ? 0u : (uint(a) >= da ? da - 1u : uint(a));\n\
             out[gid] = operand[(p * da + au) * post + t];\n}}"
    )
}

// gather_along / take_along_dim: out[..i..] = operand[.. clamp(idx[..i..]) ..] along an axis.
// `idx` has the output shape; `inner` = product of dims after the axis.
pub(crate) fn gather_along_msl(ty: &str) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void gather_along_k(device const {ty}* operand [[buffer(0)]],\n\
                            device const int* idx [[buffer(1)]],\n\
                            device {ty}* out [[buffer(2)]],\n\
                            constant uint& op_axis [[buffer(3)]],\n\
                            constant uint& out_axis [[buffer(4)]],\n\
                            constant uint& inner [[buffer(5)]],\n\
                            uint gid [[thread_position_in_grid]]) {{\n\
             uint post = gid % inner;\n\
             uint pre = gid / (inner * out_axis);\n\
             int a = idx[gid]; a = max(0, min(a, (int)op_axis - 1));\n\
             out[gid] = operand[pre * (op_axis * inner) + (uint)a * inner + post];\n}}"
    )
}

// scatter_along (index_add dual of gather_along): each update j combines into out[.. idx[j] ..]
// along the axis. OOB dropped. `combine`/`dt` pick the combine body (see `scatter_combine_body`).
pub(crate) fn scatter_along_msl(ty: &str, combine: &str, dt: DType) -> String {
    let body = scatter_combine_body(combine, dt);
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void scatter_along_k(device const int* idx [[buffer(0)]], device const {ty}* updates [[buffer(1)]],\n\
                            device {ty}* out [[buffer(2)]], constant uint& op_axis [[buffer(3)]],\n\
                            constant uint& upd_axis [[buffer(4)]], constant uint& inner [[buffer(5)]],\n\
                            uint gid [[thread_position_in_grid]]) {{\n\
             uint post = gid % inner; uint pre = gid / (inner * upd_axis); int a = idx[gid];\n\
             if (a < 0 || a >= (int)op_axis) return;\n\
             uint dst = pre * (op_axis * inner) + (uint)a * inner + post; {ty} v = updates[gid];\n\
             {body}\n}}"
    )
}

// General scatter (jnp.take inverse): operand [pre, da, post], `idx` length k, updates
// [pre, k, post]; each update combines into out[.. idx[ki] ..]. OOB dropped.
pub(crate) fn scatter_msl(ty: &str, combine: &str, dt: DType) -> String {
    let body = scatter_combine_body(combine, dt);
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void scatter_k(device const int* idx [[buffer(0)]], device const {ty}* updates [[buffer(1)]],\n\
                            device {ty}* out [[buffer(2)]], constant uint& da [[buffer(3)]],\n\
                            constant uint& k [[buffer(4)]], constant uint& post [[buffer(5)]],\n\
                            uint gid [[thread_position_in_grid]]) {{\n\
             uint j = gid % post; uint ki = (gid / post) % k; uint pre = gid / (post * k);\n\
             int a = idx[ki]; if (a < 0 || a >= (int)da) return;\n\
             uint dst = (pre * da + (uint)a) * post + j; {ty} v = updates[gid];\n\
             {body}\n}}"
    )
}

// device memcpy `out[gid] = in[gid]` (scatter starts from an operand copy).
pub(crate) fn copy_msl(ty: &str) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void copy_k(device const {ty}* in [[buffer(0)]], device {ty}* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{ out[gid] = in[gid]; }}"
    )
}

// MSL for one scatter-combine at `out[dst]` given `v` (dtype `dt`). Set: direct write (any
// dtype); i32/u32: native integer atomics (exact); f32: CAS loop on the reinterpreted uint
// (Metal has no native f32 atomic add/max/min). Shared by both scatter kernels; gated by `scatter_dev_ok`.
fn scatter_combine_body(combine: &str, dt: DType) -> String {
    if combine == "set" {
        return "out[dst] = v;".to_string();
    }
    // native integer atomics: exact, no CAS loop.
    if matches!(dt, DType::I32 | DType::U32) {
        let aty = if dt == DType::I32 { "atomic_int" } else { "atomic_uint" };
        return format!("atomic_fetch_{combine}_explicit((device {aty}*)(out + dst), v, memory_order_relaxed);");
    }
    // f32: float-CAS on the reinterpreted uint word.
    let r = match combine {
        "add" => "f + v",
        "max" => "max(f, v)",
        "min" => "min(f, v)",
        _ => unreachable!("scatter combine {combine}"),
    };
    format!(
        "device atomic_uint* p = (device atomic_uint*)(out + dst);\n\
         uint old = atomic_load_explicit(p, memory_order_relaxed);\n\
         uint nv; do {{ float f = as_type<float>(old); nv = as_type<uint>({r}); }}\n\
         while (!atomic_compare_exchange_weak_explicit(p, &old, nv, memory_order_relaxed, memory_order_relaxed));"
    )
}
