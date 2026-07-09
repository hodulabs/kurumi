//! Binary serialization of the graph IR: the closed op set + src edges + output/input
//! bindings, as a self-contained "runnable graph" blob. Reconstruction replays node
//! construction in id order and re-infers shape/dtype -- a node holds nothing that isn't
//! derivable from (op, src), so only {op + attrs, src ids} reach disk. The encode `match`
//! is exhaustive, so a new Op fails to compile until its arm exists; the decode side is
//! guarded by the all-ops round-trip test.

use super::{ArgKind, Graph, NodeId, Op, ScatterOp};
use crate::{DType, Error, Storage};
use std::collections::{HashMap, HashSet};

const MAGIC: &[u8] = b"KGPH";
const VERSION: u8 = 1;

/// Whether a serialized `Input` binds a stored weight (by name) or is fed by the caller.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InputRole {
    Weight,
    Runtime,
}

/// A serialized graph `Input`: its node, whether it's a weight or runtime feed, and its name.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InputBinding {
    pub node: NodeId,
    pub role: InputRole,
    pub name: String,
}

/// A deserialized runnable graph: the rebuilt graph plus its output nodes and input bindings.
pub struct Runnable {
    pub graph: Graph,
    pub outputs: Vec<NodeId>,
    pub inputs: Vec<InputBinding>,
}

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

/// Rebuild a graph from a blob by replaying `push` in id order (shape/dtype re-inferred).
/// This is a trust boundary: a corrupt length/tag is a clean error, never a panic. A blob
/// that is structurally valid but semantically bogus (a src referencing a later node) can
/// still panic in inference -- that is the same contract the builder has for live graphs.
pub fn deserialize_graph(bytes: &[u8]) -> Result<Runnable, Error> {
    let mut r = Reader::new(bytes);
    if r.take(4)? != MAGIC {
        return Err(err("bad magic (not a KGPH graph blob)"));
    }
    let ver = r.u8()?;
    if ver != VERSION {
        return Err(err(format!("unsupported graph blob version {ver}")));
    }
    let n = r.u32()? as usize;
    let mut g = Graph::new();
    for _ in 0..n {
        let op = read_op(&mut r)?;
        let sl = r.u32()? as usize;
        let mut src = Vec::new();
        for _ in 0..sl {
            src.push(r.node_id()?);
        }
        g.push(op, src);
    }
    let outputs = r.node_id_vec()?;
    let ni = r.u32()? as usize;
    let mut inputs = Vec::new();
    for _ in 0..ni {
        let node = r.node_id()?;
        let role = match r.u8()? {
            0 => InputRole::Weight,
            1 => InputRole::Runtime,
            x => return Err(err(format!("bad input role {x}"))),
        };
        let name = r.str()?;
        inputs.push(InputBinding { node, role, name });
    }
    Ok(Runnable { graph: g, outputs, inputs })
}

// op codec

fn write_op(o: &mut Vec<u8>, op: &Op) {
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

fn read_op(r: &mut Reader) -> Result<Op, Error> {
    let tag = r.u8()?;
    Ok(match tag {
        0 => Op::Const { data: r.storage()?, shape: r.vec_usize()? },
        1 => Op::Input { shape: r.vec_usize()?, dtype: r.dtype()? },
        2 => Op::Iota { shape: r.vec_usize()?, axis: r.usize()?, dtype: r.dtype()? },
        3 => Op::RandUniform { shape: r.vec_usize()? },
        4 => Op::Cast { to: r.dtype()? },
        5 => Op::Bitcast { to: r.dtype()? },
        6 => Op::Detach,
        7 => Op::Add,
        8 => Op::Mul,
        9 => Op::Max,
        10 => Op::Neg,
        11 => Op::IDiv,
        12 => Op::And,
        13 => Op::Or,
        14 => Op::Xor,
        15 => Op::Shl,
        16 => Op::Shr,
        17 => Op::CmpLt,
        18 => Op::CmpEq,
        19 => Op::Where,
        20 => Op::Recip,
        21 => Op::Sqrt,
        22 => Op::Exp2,
        23 => Op::Log2,
        24 => Op::Sin,
        25 => Op::Floor,
        26 => Op::Sum { axis: r.usize()? },
        27 => Op::Prod { axis: r.usize()? },
        28 => Op::ReduceMax { axis: r.usize()? },
        29 => Op::ArgReduce { axis: r.usize()?, kind: r.argkind()? },
        30 => Op::Softmax { axis: r.usize()? },
        31 => Op::RmsNorm { axis: r.usize()?, eps: r.f32()? },
        32 => Op::Sdpa { causal: r.bool()? },
        33 => Op::Reshape { shape: r.vec_usize()? },
        34 => Op::Permute { perm: r.vec_usize()? },
        35 => Op::Expand { shape: r.vec_usize()? },
        36 => Op::Slice { ranges: r.triples()? },
        37 => Op::Flip { axes: r.vec_usize()? },
        38 => Op::Pad { pads: r.pairs()? },
        39 => Op::DotGeneral {
            lhs_contract: r.vec_usize()?,
            rhs_contract: r.vec_usize()?,
            lhs_batch: r.vec_usize()?,
            rhs_batch: r.vec_usize()?,
        },
        40 => Op::QuantMatmul { bits: r.u8()?, group_size: r.usize()?, symmetric: r.bool()? },
        41 => Op::Solve,
        42 => Op::Det,
        43 => Op::Cholesky,
        44 => Op::Eigh,
        45 => Op::Qr { r_factor: r.bool()? },
        46 => Op::Eigvals,
        47 => Op::Complex,
        48 => Op::Real,
        49 => Op::Imag,
        50 => Op::Gather { axis: r.usize()? },
        51 => Op::Scatter { axis: r.usize()?, combine: r.scatter()? },
        52 => Op::GatherAlong { axis: r.usize()? },
        53 => Op::ScatterAlong { axis: r.usize()?, combine: r.scatter()? },
        54 => Op::Argsort { axis: r.usize()?, descending: r.bool()? },
        _ => return Err(err(format!("unknown op tag {tag}"))),
    })
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

// reader

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        match end {
            Some(end) => {
                let s = &self.buf[self.pos..end];
                self.pos = end;
                Ok(s)
            }
            None => Err(err("unexpected end of graph blob")),
        }
    }
    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn usize(&mut self) -> Result<usize, Error> {
        Ok(self.u64()? as usize)
    }
    fn f32(&mut self) -> Result<f32, Error> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn bool(&mut self) -> Result<bool, Error> {
        Ok(self.u8()? != 0)
    }
    fn node_id(&mut self) -> Result<NodeId, Error> {
        Ok(NodeId(self.u32()?))
    }
    fn vec_usize(&mut self) -> Result<Vec<usize>, Error> {
        let n = self.u32()? as usize;
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(self.usize()?);
        }
        Ok(v)
    }
    fn node_id_vec(&mut self) -> Result<Vec<NodeId>, Error> {
        let n = self.u32()? as usize;
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(self.node_id()?);
        }
        Ok(v)
    }
    fn triples(&mut self) -> Result<Vec<(usize, usize, usize)>, Error> {
        let n = self.u32()? as usize;
        let mut v = Vec::new();
        for _ in 0..n {
            v.push((self.usize()?, self.usize()?, self.usize()?));
        }
        Ok(v)
    }
    fn pairs(&mut self) -> Result<Vec<(usize, usize)>, Error> {
        let n = self.u32()? as usize;
        let mut v = Vec::new();
        for _ in 0..n {
            v.push((self.usize()?, self.usize()?));
        }
        Ok(v)
    }
    fn str(&mut self) -> Result<String, Error> {
        let n = self.u32()? as usize;
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| err("invalid utf8 in name"))
    }
    fn dtype(&mut self) -> Result<DType, Error> {
        Ok(match self.u8()? {
            0 => DType::BOOL,
            1 => DType::U8,
            2 => DType::U16,
            3 => DType::U32,
            4 => DType::U64,
            5 => DType::I8,
            6 => DType::I16,
            7 => DType::I32,
            8 => DType::I64,
            9 => DType::F8E4M3,
            10 => DType::F8E5M2,
            11 => DType::F16,
            12 => DType::BF16,
            13 => DType::F32,
            14 => DType::F64,
            15 => DType::C64,
            16 => DType::C128,
            x => return Err(err(format!("unknown dtype tag {x}"))),
        })
    }
    fn storage(&mut self) -> Result<Storage, Error> {
        let dt = self.dtype()?;
        let nbytes = self.u64()? as usize;
        Ok(Storage::from_bytes(dt, self.take(nbytes)?))
    }
    fn scatter(&mut self) -> Result<ScatterOp, Error> {
        Ok(match self.u8()? {
            0 => ScatterOp::Set,
            1 => ScatterOp::Add,
            2 => ScatterOp::Max,
            3 => ScatterOp::Min,
            x => return Err(err(format!("unknown scatter op {x}"))),
        })
    }
    fn argkind(&mut self) -> Result<ArgKind, Error> {
        Ok(match self.u8()? {
            0 => ArgKind::Max,
            1 => ArgKind::Min,
            x => return Err(err(format!("unknown argkind {x}"))),
        })
    }
}

fn err(msg: impl Into<String>) -> Error {
    Error::shape("deserialize_graph", msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Feeds, TensorVal, interpret_with};

    // every Op variant, with structurally valid attrs (encoding needs no valid graph).
    fn all_ops() -> Vec<Op> {
        vec![
            Op::Const { data: Storage::F32(vec![1.5, -2.5]), shape: vec![2] },
            Op::Input { shape: vec![2, 3], dtype: DType::F32 },
            Op::Iota { shape: vec![4], axis: 0, dtype: DType::I64 },
            Op::RandUniform { shape: vec![2, 2] },
            Op::Cast { to: DType::F16 },
            Op::Bitcast { to: DType::U32 },
            Op::Detach,
            Op::Add,
            Op::Mul,
            Op::Max,
            Op::Neg,
            Op::IDiv,
            Op::And,
            Op::Or,
            Op::Xor,
            Op::Shl,
            Op::Shr,
            Op::CmpLt,
            Op::CmpEq,
            Op::Where,
            Op::Recip,
            Op::Sqrt,
            Op::Exp2,
            Op::Log2,
            Op::Sin,
            Op::Floor,
            Op::Sum { axis: 1 },
            Op::Prod { axis: 0 },
            Op::ReduceMax { axis: 2 },
            Op::ArgReduce { axis: 1, kind: ArgKind::Min },
            Op::Softmax { axis: 0 },
            Op::RmsNorm { axis: 1, eps: 1e-5 },
            Op::Sdpa { causal: true },
            Op::Reshape { shape: vec![6] },
            Op::Permute { perm: vec![1, 0] },
            Op::Expand { shape: vec![2, 3, 4] },
            Op::Slice { ranges: vec![(0, 2, 1), (1, 3, 2)] },
            Op::Flip { axes: vec![0, 2] },
            Op::Pad { pads: vec![(1, 1), (0, 2)] },
            Op::DotGeneral { lhs_contract: vec![1], rhs_contract: vec![0], lhs_batch: vec![], rhs_batch: vec![] },
            Op::QuantMatmul { bits: 4, group_size: 64, symmetric: false },
            Op::Solve,
            Op::Det,
            Op::Cholesky,
            Op::Eigh,
            Op::Qr { r_factor: true },
            Op::Eigvals,
            Op::Complex,
            Op::Real,
            Op::Imag,
            Op::Gather { axis: 0 },
            Op::Scatter { axis: 1, combine: ScatterOp::Add },
            Op::GatherAlong { axis: 2 },
            Op::ScatterAlong { axis: 0, combine: ScatterOp::Max },
            Op::Argsort { axis: 1, descending: true },
        ]
    }

    // exhaustive codec guard: every op tag encodes and decodes back to an identical op.
    // (Op has no PartialEq, so compare its Debug form -- which includes Const bytes.)
    #[test]
    fn every_op_round_trips() {
        let ops = all_ops();
        assert_eq!(ops.len(), 55, "all 55 op variants must be covered");
        for op in &ops {
            let mut buf = Vec::new();
            write_op(&mut buf, op);
            let mut r = Reader::new(&buf);
            let back = read_op(&mut r).expect("decode");
            assert_eq!(r.pos, buf.len(), "trailing bytes for {op:?}");
            assert_eq!(format!("{op:?}"), format!("{back:?}"));
        }
    }

    // whole-blob path: a real graph serializes, replays (re-inferring shapes), and computes
    // the identical value; the output/input metadata round-trips too.
    #[test]
    fn graph_blob_round_trips() {
        let mut g = Graph::new();
        let x = g.input(vec![2, 3], DType::F32);
        let w = g.constant(vec![2.0; 6], vec![2, 3]);
        let a = g.push(Op::Add, vec![x, w]);
        let m = g.push(Op::Mul, vec![a, w]);
        let out = g.push(Op::Sum { axis: 1 }, vec![m]);

        let outputs = vec![out];
        let inputs = vec![InputBinding { node: x, role: InputRole::Runtime, name: "x".into() }];
        let blob = serialize_graph(&g, &outputs, &inputs);

        let r = deserialize_graph(&blob).expect("deserialize");
        assert_eq!(r.outputs, outputs);
        assert_eq!(r.inputs, inputs);

        // replay is id-preserving, so the same feed (keyed by NodeId) drives both graphs.
        let mut feeds = Feeds::new();
        feeds.insert(x, TensorVal { shape: vec![2, 3], storage: Storage::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) });
        let want = interpret_with(&g, out, &feeds);
        let got = interpret_with(&r.graph, r.outputs[0], &feeds);
        assert_eq!(want, got);
    }

    // serialize_reachable drops arena nodes no output depends on, remaps to a dense range,
    // and still computes the same value.
    #[test]
    fn reachable_prunes_dead_nodes() {
        let mut g = Graph::new();
        let x = g.input(vec![2, 3], DType::F32);
        let w = g.constant(vec![2.0; 6], vec![2, 3]);
        let _dead = g.push(Op::Neg, vec![w]); // reachable from w, but not an ancestor of `out`
        let a = g.push(Op::Add, vec![x, w]);
        let out = g.push(Op::Sum { axis: 1 }, vec![a]);

        let outputs = vec![out];
        let inputs = vec![InputBinding { node: x, role: InputRole::Runtime, name: "x".into() }];

        let full = serialize_graph(&g, &outputs, &inputs);
        let pruned = serialize_reachable(&g, &outputs, &inputs);
        assert!(pruned.len() < full.len(), "the dead Neg node must be dropped");

        let r = deserialize_graph(&pruned).expect("deserialize");
        assert_eq!(r.inputs.len(), 1);
        assert_eq!(r.inputs[0].name, "x");

        // ids were remapped, so feed via the returned bindings, not the original NodeIds.
        let mut feeds = Feeds::new();
        feeds.insert(
            r.inputs[0].node,
            TensorVal { shape: vec![2, 3], storage: Storage::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) },
        );
        let got = interpret_with(&r.graph, r.outputs[0], &feeds);
        assert_eq!(got.f32().to_vec(), vec![12.0, 21.0]);
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(deserialize_graph(b"XXXX\x01").is_err());
        assert!(deserialize_graph(&[]).is_err());
    }
}
