//! Encoding half of the graph IR codec: serialize_graph / serialize_reachable / serialize_multi,
//! the reachable-cone prune + dense remap, the blob framing, and the primitive byte writers. The
//! exhaustive per-Op encode match (write_op) lives in `op` and is re-exported here.

mod op;

pub(super) use op::write_op;

use crate::graph::inspect::reachable_multi;
use crate::graph::serialize::{InputBinding, InputRole, MAGIC, VERSION};
use crate::graph::{ArgKind, Graph, NodeId, Op, ScatterOp};
use crate::{DType, Storage};
use std::collections::HashMap;

// one entry's roots + bindings, already remapped to the blob's dense id range.
struct EncEntry<'a> {
    name: &'a str,
    outputs: Vec<u32>,
    inputs: Vec<(u32, InputRole, &'a str)>,
}

/// Serialize a graph and its output/input bindings into a self-contained blob. The whole
/// node arena is written in id order (id-preserving); the bindings become the blob's single
/// entry (name ""). See [`serialize_reachable`] to write only the live cone, [`serialize_multi`]
/// for N named entries.
pub fn serialize_graph(g: &Graph, outputs: &[NodeId], inputs: &[InputBinding]) -> Vec<u8> {
    let nodes: Vec<(&Op, Vec<u32>)> = g.nodes.iter().map(|n| (&n.op, n.src.iter().map(|s| s.0).collect())).collect();
    let e = EncEntry {
        name: "",
        outputs: outputs.iter().map(|o| o.0).collect(),
        inputs: inputs.iter().map(|b| (b.node.0, b.role, b.name.as_str())).collect(),
    };
    write_blob(&nodes, &[e])
}

/// Serialize only the nodes reachable from `outputs`, remapped to a dense id range. Arena
/// nodes no output depends on -- backward nodes from a training `grad()`, dead builder
/// scratch -- are dropped, so a live graph exports a clean inference program. Input bindings
/// whose node is unreachable are omitted; the bindings become the blob's single entry (name "").
/// The blob reads back via [`deserialize_graph`] just like a whole-arena one (it is already
/// dense and topologically ordered).
pub fn serialize_reachable(g: &Graph, outputs: &[NodeId], inputs: &[InputBinding]) -> Vec<u8> {
    write_reachable(g, outputs, &[("", outputs, inputs)])
}

/// Serialize N named entries sharing one arena: prune to the UNION of every entry's output
/// cone, dense-remap once, then write each entry's remapped outputs + inputs (an entry input
/// whose node is unreachable is dropped, as in [`serialize_reachable`]). One artifact can thus
/// carry e.g. "forward" and "forward_backward". Read back with [`deserialize_multi`];
/// [`deserialize_graph`] returns entry 0.
pub fn serialize_multi(g: &Graph, entries: &[(&str, &[NodeId], &[InputBinding])]) -> Vec<u8> {
    let roots: Vec<NodeId> = entries.iter().flat_map(|&(_, outs, _)| outs.iter().copied()).collect();
    write_reachable(g, &roots, entries)
}

// prune to the cone of `roots`, dense-remap once, and write each entry over the remapped ids.
fn write_reachable(g: &Graph, roots: &[NodeId], entries: &[(&str, &[NodeId], &[InputBinding])]) -> Vec<u8> {
    let order = reachable_multi(g, roots);
    let remap: HashMap<NodeId, u32> = order.iter().enumerate().map(|(i, &id)| (id, i as u32)).collect();
    let nodes: Vec<(&Op, Vec<u32>)> = order
        .iter()
        .map(|&id| {
            let n = g.node(id);
            (&n.op, n.src.iter().map(|s| remap[s]).collect())
        })
        .collect();
    let enc: Vec<EncEntry> = entries
        .iter()
        .map(|&(name, outs, ins)| EncEntry {
            name,
            outputs: outs.iter().map(|o| remap[o]).collect(),
            inputs: ins.iter().filter_map(|b| remap.get(&b.node).map(|&id| (id, b.role, b.name.as_str()))).collect(),
        })
        .collect();
    write_blob(&nodes, &enc)
}

// shared blob encoder: nodes already in dense id order (src as new-id u32s), then the entry
// table -- each entry's outputs/inputs use the same ids.
fn write_blob(nodes: &[(&Op, Vec<u32>)], entries: &[EncEntry]) -> Vec<u8> {
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
    w_u32(&mut o, entries.len() as u32);
    for e in entries {
        w_str(&mut o, e.name);
        w_u32(&mut o, e.outputs.len() as u32);
        for &out in &e.outputs {
            w_u32(&mut o, out);
        }
        w_u32(&mut o, e.inputs.len() as u32);
        for &(node, role, name) in &e.inputs {
            w_u32(&mut o, node);
            let r = match role {
                InputRole::Weight => 0,
                InputRole::Runtime => 1,
            };
            w_u8(&mut o, r);
            w_str(&mut o, name);
        }
    }
    o
}

// primitive writers (shared by write_blob above and the per-Op match in `op`)

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
