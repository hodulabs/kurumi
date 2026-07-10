//! Encoding half of the graph IR codec: serialize_graph / serialize_reachable, the
//! reachable-cone prune + dense remap, and write_op (the exhaustive encode match every new
//! Op must extend).
use crate::graph::serialize::{InputBinding, InputRole, MAGIC, VERSION};
use crate::graph::{ArgKind, Graph, NodeId, Op, ScatterOp};
use crate::{DType, Storage};
use std::collections::{HashMap, HashSet};

/// Serialize a graph and its output/input bindings into a self-contained blob. The whole
/// node arena is written in id order (id-preserving). See [`serialize_reachable`] to write
/// only the live cone.
pub fn serialize_graph(g: &Graph, outputs: &[NodeId], inputs: &[InputBinding]) -> Vec<u8> {
    let nodes: Vec<(&Op, Vec<u32>)> = g.nodes.iter().map(|n| (&n.op, n.src.iter().map(|s| s.0).collect())).collect();
    let outs: Vec<u32> = outputs.iter().map(|o| o.0).collect();
    let ins: Vec<(u32, InputRole, &str)> = inputs.iter().map(|b| (b.node.0, b.role, b.name.as_str())).collect();
    write_blob(&nodes, &outs, &ins)
}

/// Serialize only the nodes reachable from `outputs`, remapped to a dense id range. Arena
/// nodes no output depends on -- backward nodes from a training `grad()`, dead builder
/// scratch -- are dropped, so a live graph exports a clean inference program. Input bindings
/// whose node is unreachable are omitted. The blob reads back via [`deserialize_graph`] just
/// like a whole-arena one (it is already dense and topologically ordered).
pub fn serialize_reachable(g: &Graph, outputs: &[NodeId], inputs: &[InputBinding]) -> Vec<u8> {
    let order = reachable_multi(g, outputs);
    let remap: HashMap<NodeId, u32> = order.iter().enumerate().map(|(i, &id)| (id, i as u32)).collect();
    let nodes: Vec<(&Op, Vec<u32>)> = order
        .iter()
        .map(|&id| {
            let n = g.node(id);
            (&n.op, n.src.iter().map(|s| remap[s]).collect())
        })
        .collect();
    let outs: Vec<u32> = outputs.iter().map(|o| remap[o]).collect();
    let ins: Vec<(u32, InputRole, &str)> =
        inputs.iter().filter_map(|b| remap.get(&b.node).map(|&id| (id, b.role, b.name.as_str()))).collect();
    write_blob(&nodes, &outs, &ins)
}

// nodes reachable from any of `roots`, topologically ordered (each after its sources). one
// shared seen set across roots so the order is a valid dense-replay sequence.
fn reachable_multi(g: &Graph, roots: &[NodeId]) -> Vec<NodeId> {
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    let mut stack: Vec<(NodeId, bool)> = roots.iter().rev().map(|&r| (r, false)).collect();
    while let Some((id, expanded)) = stack.pop() {
        if expanded {
            order.push(id);
            continue;
        }
        if !seen.insert(id) {
            continue;
        }
        stack.push((id, true));
        for &s in &g.node(id).src {
            if !seen.contains(&s) {
                stack.push((s, false));
            }
        }
    }
    order
}

// shared blob encoder: nodes already in dense id order, src as new-id u32s.
fn write_blob(nodes: &[(&Op, Vec<u32>)], outputs: &[u32], inputs: &[(u32, InputRole, &str)]) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend_from_slice(MAGIC);
    w_u8(&mut o, VERSION);
    w_u32(&mut o, nodes.len() as u32);
    for (op, src) in nodes {
        write_op(&mut o, op);
        w_u32(&mut o, src.len() as u32);
        for &s in src {
            w_u32(&mut o, s);
        }
    }
    w_u32(&mut o, outputs.len() as u32);
    for &out in outputs {
        w_u32(&mut o, out);
    }
    w_u32(&mut o, inputs.len() as u32);
    for &(node, role, name) in inputs {
        w_u32(&mut o, node);
        let r = match role {
            InputRole::Weight => 0,
            InputRole::Runtime => 1,
        };
        w_u8(&mut o, r);
        w_str(&mut o, name);
    }
    o
}

// op codec

pub(super) fn write_op(o: &mut Vec<u8>, op: &Op) {
    match op {
        Op::Const { data, shape } => {
            w_u8(o, 0);
            w_storage(o, data);
            w_vec_usize(o, shape);
        }
        Op::Input { shape, dtype } => {
            w_u8(o, 1);
            w_vec_usize(o, shape);
            w_dtype(o, *dtype);
        }
        Op::Iota { shape, axis, dtype } => {
            w_u8(o, 2);
            w_vec_usize(o, shape);
            w_usize(o, *axis);
            w_dtype(o, *dtype);
        }
        Op::RandUniform { shape } => {
            w_u8(o, 3);
            w_vec_usize(o, shape);
        }
        Op::Cast { to } => {
            w_u8(o, 4);
            w_dtype(o, *to);
        }
        Op::Bitcast { to } => {
            w_u8(o, 5);
            w_dtype(o, *to);
        }
        Op::Detach => w_u8(o, 6),
        Op::Add => w_u8(o, 7),
        Op::Mul => w_u8(o, 8),
        Op::Max => w_u8(o, 9),
        Op::Neg => w_u8(o, 10),
        Op::IDiv => w_u8(o, 11),
        Op::And => w_u8(o, 12),
        Op::Or => w_u8(o, 13),
        Op::Xor => w_u8(o, 14),
        Op::Shl => w_u8(o, 15),
        Op::Shr => w_u8(o, 16),
        Op::CmpLt => w_u8(o, 17),
        Op::CmpEq => w_u8(o, 18),
        Op::Where => w_u8(o, 19),
        Op::Recip => w_u8(o, 20),
        Op::Sqrt => w_u8(o, 21),
        Op::Exp2 => w_u8(o, 22),
        Op::Log2 => w_u8(o, 23),
        Op::Sin => w_u8(o, 24),
        Op::Floor => w_u8(o, 25),
        Op::Sum { axis } => {
            w_u8(o, 26);
            w_usize(o, *axis);
        }
        Op::Prod { axis } => {
            w_u8(o, 27);
            w_usize(o, *axis);
        }
        Op::ReduceMax { axis } => {
            w_u8(o, 28);
            w_usize(o, *axis);
        }
        Op::ArgReduce { axis, kind } => {
            w_u8(o, 29);
            w_usize(o, *axis);
            w_argkind(o, *kind);
        }
        Op::Softmax { axis } => {
            w_u8(o, 30);
            w_usize(o, *axis);
        }
        Op::RmsNorm { axis, eps } => {
            w_u8(o, 31);
            w_usize(o, *axis);
            w_f32(o, *eps);
        }
        Op::Sdpa { causal } => {
            w_u8(o, 32);
            w_bool(o, *causal);
        }
        Op::Reshape { shape } => {
            w_u8(o, 33);
            w_vec_usize(o, shape);
        }
        Op::Permute { perm } => {
            w_u8(o, 34);
            w_vec_usize(o, perm);
        }
        Op::Expand { shape } => {
            w_u8(o, 35);
            w_vec_usize(o, shape);
        }
        Op::Slice { ranges } => {
            w_u8(o, 36);
            w_u32(o, ranges.len() as u32);
            for &(a, b, c) in ranges {
                w_usize(o, a);
                w_usize(o, b);
                w_usize(o, c);
            }
        }
        Op::Flip { axes } => {
            w_u8(o, 37);
            w_vec_usize(o, axes);
        }
        Op::Pad { pads } => {
            w_u8(o, 38);
            w_u32(o, pads.len() as u32);
            for &(lo, hi) in pads {
                w_usize(o, lo);
                w_usize(o, hi);
            }
        }
        Op::DotGeneral { lhs_contract, rhs_contract, lhs_batch, rhs_batch } => {
            w_u8(o, 39);
            w_vec_usize(o, lhs_contract);
            w_vec_usize(o, rhs_contract);
            w_vec_usize(o, lhs_batch);
            w_vec_usize(o, rhs_batch);
        }
        Op::QuantMatmul { bits, group_size, symmetric } => {
            w_u8(o, 40);
            w_u8(o, *bits);
            w_usize(o, *group_size);
            w_bool(o, *symmetric);
        }
        Op::Solve => w_u8(o, 41),
        Op::Det => w_u8(o, 42),
        Op::Cholesky => w_u8(o, 43),
        Op::Eigh => w_u8(o, 44),
        Op::Qr { r_factor } => {
            w_u8(o, 45);
            w_bool(o, *r_factor);
        }
        Op::Eigvals => w_u8(o, 46),
        Op::Complex => w_u8(o, 47),
        Op::Real => w_u8(o, 48),
        Op::Imag => w_u8(o, 49),
        Op::Gather { axis } => {
            w_u8(o, 50);
            w_usize(o, *axis);
        }
        Op::Scatter { axis, combine } => {
            w_u8(o, 51);
            w_usize(o, *axis);
            w_scatter(o, *combine);
        }
        Op::GatherAlong { axis } => {
            w_u8(o, 52);
            w_usize(o, *axis);
        }
        Op::ScatterAlong { axis, combine } => {
            w_u8(o, 53);
            w_usize(o, *axis);
            w_scatter(o, *combine);
        }
        Op::Argsort { axis, descending } => {
            w_u8(o, 54);
            w_usize(o, *axis);
            w_bool(o, *descending);
        }
    }
}

// primitive writers

fn w_u8(o: &mut Vec<u8>, v: u8) {
    o.push(v);
}
fn w_u32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn w_u64(o: &mut Vec<u8>, v: u64) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn w_usize(o: &mut Vec<u8>, v: usize) {
    w_u64(o, v as u64);
}
fn w_f32(o: &mut Vec<u8>, v: f32) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn w_bool(o: &mut Vec<u8>, v: bool) {
    o.push(v as u8);
}
fn w_vec_usize(o: &mut Vec<u8>, v: &[usize]) {
    w_u32(o, v.len() as u32);
    for &x in v {
        w_usize(o, x);
    }
}
fn w_str(o: &mut Vec<u8>, s: &str) {
    w_u32(o, s.len() as u32);
    o.extend_from_slice(s.as_bytes());
}
fn w_dtype(o: &mut Vec<u8>, d: DType) {
    w_u8(o, dtype_tag(d));
}
fn w_storage(o: &mut Vec<u8>, s: &Storage) {
    w_dtype(o, s.dtype());
    let bytes = s.to_bytes();
    w_u64(o, bytes.len() as u64);
    o.extend_from_slice(&bytes);
}
fn w_scatter(o: &mut Vec<u8>, s: ScatterOp) {
    let tag = match s {
        ScatterOp::Set => 0,
        ScatterOp::Add => 1,
        ScatterOp::Max => 2,
        ScatterOp::Min => 3,
    };
    w_u8(o, tag);
}
fn w_argkind(o: &mut Vec<u8>, k: ArgKind) {
    let tag = match k {
        ArgKind::Max => 0,
        ArgKind::Min => 1,
    };
    w_u8(o, tag);
}

// dtype tag: stable numbering, identical to the C ABI (kurumi.h) so the graph blob and
// every other kurumi surface agree. Explicit (not `as u8`) so a reorder can't shift it.
fn dtype_tag(d: DType) -> u8 {
    match d {
        DType::BOOL => 0,
        DType::U8 => 1,
        DType::U16 => 2,
        DType::U32 => 3,
        DType::U64 => 4,
        DType::I8 => 5,
        DType::I16 => 6,
        DType::I32 => 7,
        DType::I64 => 8,
        DType::F8E4M3 => 9,
        DType::F8E5M2 => 10,
        DType::F16 => 11,
        DType::BF16 => 12,
        DType::F32 => 13,
        DType::F64 => 14,
        DType::C64 => 15,
        DType::C128 => 16,
    }
}
