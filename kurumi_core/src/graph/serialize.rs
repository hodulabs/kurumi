//! Binary serialization of the graph IR: the closed op set + src edges + output/input
//! bindings, as a self-contained "runnable graph" blob. Reconstruction replays node
//! construction in id order and re-infers shape/dtype -- a node holds nothing that isn't
//! derivable from (op, src), so only {op + attrs, src ids} reach disk. The encode `match`
//! is exhaustive, so a new Op fails to compile until its arm exists; the decode side is
//! guarded by the all-ops round-trip test. Encoding lives in `encode`, decoding in `decode`.

mod decode;
mod encode;

pub use decode::deserialize_graph;
pub use encode::{serialize_graph, serialize_reachable};

use crate::graph::{Graph, NodeId};

const MAGIC: &[u8] = b"KGPH";
const VERSION: u8 = 1;

// The blob is little-endian everywhere: headers/attrs write LE explicitly, and Const payloads
// go through `Storage::to_bytes`, which is native-endian -- equal to LE only on a little-endian
// host. Every real target (x86/ARM/wasm) is LE; refuse to build on a big-endian host rather than
// silently emit a blob other targets can't read (02-artifact-format: one artifact, every target).
#[cfg(target_endian = "big")]
compile_error!("kurumi graph serialization assumes a little-endian host (Const payload is native-endian)");

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

#[cfg(test)]
mod tests;
