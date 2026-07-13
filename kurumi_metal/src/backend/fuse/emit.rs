//! MSL string emission for the fused pointwise and fused-reduce kernels.

use super::{FExpr, Leaf, REDUCE_TG};

impl FExpr {
    // `bits` = element bit-width (shift-mask); `cx` = complex (float2) chain -> mul/recip
    // and the transcendentals use the complex helpers (cmul/crecip/csqrt/cexp2/clog2/csin);
    // add/neg are native float2 ops. Complex reduce takes the materialized path in eval_fused.
    fn emit(&self, bits: u32, cx: bool) -> String {
        let m = bits.saturating_sub(1); // shift-amount mask (wrapping_shl semantics)
        match self {
            FExpr::Leaf(i) => format!("v{i}"),
            FExpr::Un("neg", e) => format!("(-{})", e.emit(bits, cx)),
            FExpr::Un("recip", e) if cx => format!("crecip({})", e.emit(bits, cx)),
            FExpr::Un("recip", e) => format!("(1.0f / {})", e.emit(bits, cx)),
            // complex transcendentals -> the c* helpers; real -> the MSL intrinsic.
            FExpr::Un("sqrt", e) if cx => format!("csqrt({})", e.emit(bits, cx)),
            FExpr::Un("exp2", e) if cx => format!("cexp2({})", e.emit(bits, cx)),
            FExpr::Un("log2", e) if cx => format!("clog2({})", e.emit(bits, cx)),
            FExpr::Un("sin", e) if cx => format!("csin({})", e.emit(bits, cx)),
            FExpr::Un(f, e) => format!("{f}({})", e.emit(bits, cx)), // sqrt/exp2/log2/sin (real)
            FExpr::Bin("add", a, b) => format!("({} + {})", a.emit(bits, cx), b.emit(bits, cx)),
            FExpr::Bin("mul", a, b) if cx => format!("cmul({}, {})", a.emit(bits, cx), b.emit(bits, cx)),
            FExpr::Bin("mul", a, b) => format!("({} * {})", a.emit(bits, cx), b.emit(bits, cx)),
            FExpr::Bin("max", a, b) => format!("max({}, {})", a.emit(bits, cx), b.emit(bits, cx)),
            // idiv: x/0 = 0 (matches the CPU oracle). INT_MIN/-1 is UB on the
            // GPU (CPU wraps to INT_MIN): a corner not worth a second branch.
            FExpr::Bin("idiv", a, b) => {
                let (a, b) = (a.emit(bits, cx), b.emit(bits, cx));
                format!("({b} == 0 ? 0 : {a} / {b})")
            }
            FExpr::Bin("and", a, b) => format!("({} & {})", a.emit(bits, cx), b.emit(bits, cx)),
            FExpr::Bin("or", a, b) => format!("({} | {})", a.emit(bits, cx), b.emit(bits, cx)),
            FExpr::Bin("xor", a, b) => format!("({} ^ {})", a.emit(bits, cx), b.emit(bits, cx)),
            // shift amount masked to the width (wrapping_shl/shr); small types promote to
            // int then truncate on the outer store, matching wrapping semantics.
            FExpr::Bin("shl", a, b) => format!("({} << ({} & {m}))", a.emit(bits, cx), b.emit(bits, cx)),
            FExpr::Bin("shr", a, b) => format!("({} >> ({} & {m}))", a.emit(bits, cx), b.emit(bits, cx)),
            FExpr::Bin(op, ..) => unreachable!("unknown fused binary {op}"),
        }
    }
}

// emit `{ty} v{i} = l{i}[...]` for each leaf, read at flat index `idx` (a uint expr)
// decomposed over each leaf's view; a viewless leaf reads `l{i}[idx]` directly.
fn emit_leaf_reads(leaves: &[Leaf], ty: &str, idx: &str) -> String {
    let mut s = String::new();
    for (i, leaf) in leaves.iter().enumerate() {
        match &leaf.view {
            None => s += &format!("    {ty} v{i} = l{i}[{idx}];\n"),
            Some(vw) => {
                // decompose `idx` over out_shape (last axis first), accumulate the
                // strided source index; drop the size-1/stride-0 (broadcast) terms.
                // signed (int) so flip's negative strides work.
                s += &format!("    int src{i} = {}; {{ uint t = {idx};\n", vw.base);
                for ax in (0..vw.out_shape.len()).rev() {
                    let ext = vw.out_shape[ax];
                    if vw.strides[ax] != 0 {
                        s += &format!("        src{i} += (int)(t % {ext}u) * ({});\n", vw.strides[ax]);
                    }
                    s += &format!("        t /= {ext}u;\n");
                }
                s += &format!("    }}\n    {ty} v{i} = l{i}[src{i}];\n");
            }
        }
    }
    s
}

// complex helpers, prepended to a float2 (C64) fused kernel. mul/recip are exact;
// exp2/log2/sqrt/sin mirror num_complex's identities (tolerance vs CPU, like real
// transcendentals). exp2(z)=e^(z ln2); log2(z)=ln(z)/ln2; principal sqrt; sin.
const CX_HELPERS: &str = "static inline float2 cmul(float2 a, float2 b){ return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }\n\
     static inline float2 crecip(float2 a){ float d = a.x*a.x + a.y*a.y; return float2(a.x/d, -a.y/d); }\n\
     static inline float2 cexp2(float2 z){ float2 w = float2(z.x*0.69314718f, z.y*0.69314718f); float e = exp(w.x); return float2(e*cos(w.y), e*sin(w.y)); }\n\
     static inline float2 clog2(float2 z){ float2 l = float2(0.5f*log(z.x*z.x + z.y*z.y), atan2(z.y, z.x)); return float2(l.x*1.44269504f, l.y*1.44269504f); }\n\
     static inline float2 csqrt(float2 z){ float r = sqrt(z.x*z.x + z.y*z.y); float re = sqrt(max(0.0f,(r+z.x)*0.5f)); float im = sqrt(max(0.0f,(r-z.x)*0.5f)); return float2(re, z.y < 0.0f ? -im : im); }\n\
     static inline float2 csin(float2 z){ return float2(sin(z.x)*cosh(z.y), cos(z.x)*sinh(z.y)); }\n";

pub(in crate::backend) fn fused_msl(expr: &FExpr, leaves: &[Leaf], ty: &str) -> String {
    let n = leaves.len();
    let cx = ty == "float2"; // complex chain: emit cmul/crecip + prepend the helpers
    let mut s = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    if cx {
        s += CX_HELPERS;
    }
    s += "kernel void fused_k(\n";
    for i in 0..n {
        s += &format!("    device const {ty}* l{i} [[buffer({i})]],\n");
    }
    s += &format!("    device {ty}* out [[buffer({n})]],\n");
    s += "    uint gid [[thread_position_in_grid]]) {\n";
    s += &emit_leaf_reads(leaves, ty, "gid");
    s += &format!("    out[gid] = ({ty})({});\n}}", expr.emit(msl_bits(ty), cx));
    s
}

/// Fused parallel reduce (`tag` = "sum" | "prod") along an axis, reading input through
/// a fused pointwise expr. ONE threadgroup per output line; its `REDUCE_TG` threads
/// compute the expr over their strided slice IN PARALLEL (a heavy producer like exp
/// isn't serialized), then a shared-memory tree reduce folds the line. Beats
/// materialize+reduce: drops the intermediate, keeps parallelism. `in_shape` is the
/// fused chain's shape (what leaf views map over).
pub(in crate::backend) fn fused_reduce_msl(
    tag: &str,
    expr: &FExpr,
    leaves: &[Leaf],
    ty: &str,
    in_shape: &[usize],
    axis: usize,
) -> String {
    let n = leaves.len();
    let acc = if matches!(ty, "half" | "bfloat" | "float") { "float" } else { ty };
    let (init, op) = match tag {
        "sum" => ("0", "+"),
        "prod" => ("1", "*"),
        _ => unreachable!("fused parallel reduce only for sum/prod, got {tag}"),
    };
    let axis_len = in_shape[axis];
    let inner: usize = in_shape[axis + 1..].iter().product();
    let val = expr.emit(msl_bits(ty), false); // reduce is gated to real dtypes (complex -> host)
    let reads = emit_leaf_reads(leaves, ty, "p"); // per-element read at the strided index p
    let tg = REDUCE_TG;
    let mut s = String::from("#include <metal_stdlib>\nusing namespace metal;\nkernel void reduce_k(\n");
    for i in 0..n {
        s += &format!("    device const {ty}* l{i} [[buffer({i})]],\n");
    }
    s += &format!("    device {ty}* out [[buffer({n})]],\n");
    s += "    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {\n";
    s += &format!("    threadgroup {acc} scratch[{tg}];\n");
    s += &format!("    uint base = (tgid / {inner}u) * {axis_len}u * {inner}u + (tgid % {inner}u);\n");
    s += &format!("    {acc} part = ({acc}){init};\n");
    s += &format!("    for (uint k = tid; k < {axis_len}u; k += {tg}u) {{\n        uint p = base + k * {inner}u;\n");
    s += &reads;
    s += &format!("        part = part {op} ({acc})({ty})({val});\n    }}\n");
    s += "    scratch[tid] = part;\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    s += &format!("    for (uint s = {}u; s > 0u; s >>= 1) {{\n", tg / 2);
    s += &format!("        if (tid < s) scratch[tid] = scratch[tid] {op} scratch[tid + s];\n");
    s += "        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n";
    s += &format!("    if (tid == 0u) out[tgid] = ({ty})scratch[0];\n}}");
    s
}

// element bit-width by MSL type name (for the shift-amount mask). Only int chains
// carry shifts; floats never do, so their width is irrelevant.
fn msl_bits(ty: &str) -> u32 {
    match ty {
        "uchar" | "char" => 8,
        "ushort" | "short" | "half" | "bfloat" => 16,
        "ulong" | "long" => 64,
        _ => 32, // uint/int/float
    }
}
