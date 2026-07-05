//! Pointwise / movement-at-the-boundary kernel sources launched by
//! `context/dispatch/pointwise.rs` (elementwise fusion itself lives in `backend/fuse.rs`).

// Zero-pad (element type `ty`): out[gid] = in[mapped] if the output coord is inside
// the original [lo, lo+in_shape) region on every axis, else 0.
pub(crate) fn pad_msl(ty: &str) -> String {
    // zero fill for out-of-bounds; float2 (complex) needs the vector constructor.
    let zero = if ty == "float2" { "float2(0.0)".to_string() } else { format!("({ty})0") };
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void pad_k(device const {ty}* in [[buffer(0)]],\n\
                           device {ty}* out [[buffer(1)]],\n\
                           constant uint& rank [[buffer(2)]],\n\
                           constant uint* out_shape [[buffer(3)]],\n\
                           constant uint* lo [[buffer(4)]],\n\
                           constant uint* in_shape [[buffer(5)]],\n\
                           constant uint* in_stride [[buffer(6)]],\n\
                           uint gid [[thread_position_in_grid]]) {{\n\
             uint idx = gid, in_idx = 0; bool inb = true;\n\
             for (uint a = 0; a < rank; a++) {{ uint ax = rank-1-a; uint c = idx % out_shape[ax]; idx /= out_shape[ax]; if (c < lo[ax] || c >= lo[ax] + in_shape[ax]) inb = false; else in_idx += (c - lo[ax]) * in_stride[ax]; }}\n\
             out[gid] = inb ? in[in_idx] : {zero};\n}}"
    )
}

// `cmp_k`: a[i] OP b[i] -> BOOL (uchar 0/1); `op` = "<" or "==". Keeps compares device-resident
// so where/select chains don't fall to host.
pub(crate) fn cmp_msl(ty: &str, op: &str) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void cmp_k(device const {ty}* a [[buffer(0)]], device const {ty}* b [[buffer(1)]],\n\
                           device uchar* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {{\n\
             out[i] = (a[i] {op} b[i]) ? 1 : 0; }}"
    )
}

// `where_k`: select cond ? a : b (cond BOOL/uchar; a/b/out are `ty`).
pub(crate) fn where_msl(ty: &str) -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void where_k(device const uchar* c [[buffer(0)]], device const {ty}* a [[buffer(1)]],\n\
                             device const {ty}* b [[buffer(2)]], device {ty}* out [[buffer(3)]],\n\
                             uint i [[thread_position_in_grid]]) {{ out[i] = c[i] ? a[i] : b[i]; }}"
    )
}
