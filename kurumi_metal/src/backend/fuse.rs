//! The device fusion model: same-shape pointwise ops defer into one `FExpr`
//! tree (`Val::Fused`) and lower to a single MSL kernel (`fused_msl`) at a
//! materialization boundary. `ew_kind` classifies an op for this path.

use crate::Buffer;
use kurumi_core::{DType, Op, TensorVal};
use objc2::rc::Retained;

pub(super) enum Ew {
    Unary(&'static str),
    Binary(&'static str),
    Reshape,                                   // f32 reshape = same contiguous buffer, new shape (zero copy)
    Expand,                                    // broadcast size-1 axes on-device (no host materialization)
    Permute,                                   // axis permutation via strided gather (no host transpose)
    Slice,                                     // sub-region via strided gather + start offset
    Flip,                                      // axis reversal via negative strides
    Pad,                                       // zero-pad
    Reduce { tag: &'static str, axis: usize }, // sum/max/prod along an axis (keepdim=false)
}

pub(super) fn ew_kind(op: &Op) -> Option<Ew> {
    Some(match op {
        Op::Add => Ew::Binary("add"),
        Op::Mul => Ew::Binary("mul"),
        Op::Max => Ew::Binary("max"),
        // integer / bitwise (single-dtype chains; `emit` bakes div0-guard + shift-mask)
        Op::IDiv => Ew::Binary("idiv"),
        Op::And => Ew::Binary("and"),
        Op::Or => Ew::Binary("or"),
        Op::Xor => Ew::Binary("xor"),
        Op::Shl => Ew::Binary("shl"),
        Op::Shr => Ew::Binary("shr"),
        Op::Neg => Ew::Unary("neg"),
        Op::Recip => Ew::Unary("recip"),
        Op::Sqrt => Ew::Unary("sqrt"),
        Op::Exp2 => Ew::Unary("exp2"),
        Op::Log2 => Ew::Unary("log2"),
        Op::Sin => Ew::Unary("sin"),
        Op::Floor => Ew::Unary("floor"),
        Op::Reshape { .. } => Ew::Reshape,
        Op::Expand { .. } => Ew::Expand,
        Op::Permute { .. } => Ew::Permute,
        Op::Slice { .. } => Ew::Slice,
        Op::Flip { .. } => Ew::Flip,
        Op::Pad { .. } => Ew::Pad,
        Op::Sum { axis } => Ew::Reduce { tag: "sum", axis: *axis },
        Op::ReduceMax { axis } => Ew::Reduce { tag: "max", axis: *axis },
        Op::Prod { axis } => Ew::Reduce { tag: "prod", axis: *axis },
        _ => return None,
    })
}

#[derive(Clone)]
pub(super) enum Val {
    Host(TensorVal),
    Dev { buf: Buffer, shape: Vec<usize>, dt: DType },
    Fused { shape: Vec<usize>, leaves: Vec<Leaf>, expr: FExpr, dt: DType },
}

// a fused-kernel input buffer, read contiguously at the output coordinate
// (`view == None`) or via a strided index map (`view == Some`). The map folds a
// movement (broadcast/permute/slice) into the consumer read: no strided_dev
// dispatch, no materialized intermediate.
#[derive(Clone)]
pub(super) struct Leaf {
    pub buf: Buffer,
    pub view: Option<View>,
}

// source index = base + sum_ax coord_ax(gid) * strides[ax], over `out_shape`.
// broadcast: stride 0 on size-1 axes; permute: permuted strides; slice: base +
// step-scaled strides. `strides` are into the (contiguous) source buffer.
#[derive(Clone, PartialEq)]
pub(super) struct View {
    pub base: i64, // signed: flip contributes a negative per-axis stride
    pub strides: Vec<i64>,
    pub out_shape: Vec<usize>,
}

impl Leaf {
    pub(super) fn plain(buf: Buffer) -> Self {
        Leaf { buf, view: None }
    }
}

// fused pointwise expression over same-shape leaves (each read at the output
// coordinate `gid`). Tags match `ew_kind`; `emit` lowers them to MSL.
#[derive(Clone)]
pub(super) enum FExpr {
    Leaf(usize),
    Un(&'static str, Box<FExpr>),
    Bin(&'static str, Box<FExpr>, Box<FExpr>),
}

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
    // remap leaf indices (used when merging a second operand's deduped leaves)
    pub(super) fn remap(self, map: &[usize]) -> FExpr {
        match self {
            FExpr::Leaf(i) => FExpr::Leaf(map[i]),
            FExpr::Un(f, e) => FExpr::Un(f, Box::new(e.remap(map))),
            FExpr::Bin(op, a, b) => FExpr::Bin(op, Box::new(a.remap(map)), Box::new(b.remap(map))),
        }
    }
    pub(super) fn size(&self) -> usize {
        match self {
            FExpr::Leaf(_) => 1,
            FExpr::Un(_, e) => 1 + e.size(),
            FExpr::Bin(_, a, b) => 1 + a.size() + b.size(),
        }
    }
}

// guard against pathological inlining: a fused tree past this many nodes (or
// distinct leaves: Metal allows ~31 buffer args) is materialized first.
// chains here are short (reduces/matmuls break them).
pub(super) const FUSE_CAP: usize = 64;
pub(super) const MAX_LEAVES: usize = 24;

pub(super) fn same_buf(a: &Buffer, b: &Buffer) -> bool {
    std::ptr::addr_eq(Retained::as_ptr(a), Retained::as_ptr(b))
}

// two leaves read the same data iff same buffer AND same view (a plain read and a
// strided read of one buffer are different values -> must not dedup).
pub(super) fn leaf_eq(a: &Leaf, b: &Leaf) -> bool {
    same_buf(&a.buf, &b.buf) && a.view == b.view
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

pub(super) fn fused_msl(expr: &FExpr, leaves: &[Leaf], ty: &str) -> String {
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

// threads per reduction threadgroup (power of 2 for the tree reduce; extra threads
// past axis_len fold the identity harmlessly).
pub(super) const REDUCE_TG: usize = 128;

/// Fused parallel reduce (`tag` = "sum" | "prod") along an axis, reading input through
/// a fused pointwise expr. ONE threadgroup per output line; its `REDUCE_TG` threads
/// compute the expr over their strided slice IN PARALLEL (a heavy producer like exp
/// isn't serialized), then a shared-memory tree reduce folds the line. Beats
/// materialize+reduce: drops the intermediate, keeps parallelism. `in_shape` is the
/// fused chain's shape (what leaf views map over).
pub(super) fn fused_reduce_msl(
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
