//! The device fusion model: same-shape pointwise ops defer into one `FExpr`
//! tree (`Val::Fused`) and lower to a single MSL kernel (`fused_msl`) at a
//! materialization boundary. `ew_kind` classifies an op for this path. The MSL
//! string emission lives in `emit`.

mod emit;

use crate::Buffer;
use kurumi_core::{DType, Op, TensorVal};
use objc2::rc::Retained;

pub(super) use emit::{fused_msl, fused_reduce_msl};

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

// threads per reduction threadgroup (power of 2 for the tree reduce; extra threads
// past axis_len fold the identity harmlessly).
pub(super) const REDUCE_TG: usize = 128;
