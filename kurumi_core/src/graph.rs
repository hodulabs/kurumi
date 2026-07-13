//! IR core: the closed primitive `Op` set, the arena builder, and record-time shape/dtype
//! inference. Op *builders* live in `graph/ops/<category>.rs`; this file holds the arena,
//! node accessors, and leaf builders (const/input/cast). The record-time validation guards
//! every builder calls are in `check`.

mod check;
mod infer;
mod inspect;
mod op;
mod ops;
mod pass;
mod serialize;

pub use inspect::{dump, node_count, reachable};
pub use op::{ArgKind, Node, NodeId, Op, ScatterOp};
pub use pass::{amp, simplify};
pub use serialize::{
    Entry, InputBinding, InputRole, MultiRunnable, Runnable, deserialize_graph, deserialize_multi, serialize_graph,
    serialize_multi, serialize_reachable,
};

use crate::{DType, Error, Storage};
use std::sync::atomic::{AtomicU64, Ordering};

// process-unique graph ids, so a backend can key a per-graph cache (e.g. uploaded
// constants) without ABA: a dropped graph's NodeIds never collide with a new graph's.
static GRAPH_COUNTER: AtomicU64 = AtomicU64::new(1);

pub struct Graph {
    nodes: Vec<Node>,
    id: u64,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), id: GRAPH_COUNTER.fetch_add(1, Ordering::Relaxed) }
    }

    /// A process-unique id assigned at construction. Backends key per-graph caches on
    /// it (uploaded weights stay device-resident across evals of the same graph).
    pub fn id(&self) -> u64 {
        self.id
    }

    pub(in crate::graph) fn push(&mut self, op: Op, src: Vec<NodeId>) -> NodeId {
        let dtype = self.infer_dtype(&op, &src);
        let shape = self.derive_shape(&op, &src);
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node { op, dtype, src, shape });
        id
    }

    /// The IR node at `id` (op + dtype + sources): backends walk the graph with this.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// Dtype is stored on the node (inferred at record time).
    pub fn dtype(&self, id: NodeId) -> DType {
        self.node(id).dtype
    }

    /// Shape, stored on the node (derived once at record time: O(1) lookup).
    pub fn shape(&self, id: NodeId) -> Vec<usize> {
        self.node(id).shape.clone()
    }

    /// f32 constant (convenience). For other dtypes use [`Graph::const_storage`].
    pub fn constant(&mut self, data: Vec<f32>, shape: Vec<usize>) -> NodeId {
        self.const_storage(Storage::F32(data), shape)
    }

    pub fn const_storage(&mut self, data: Storage, shape: Vec<usize>) -> NodeId {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        self.push(Op::Const { data, shape }, vec![])
    }

    /// An input slot fed at eval time (model params / data): build the graph once, then feed
    /// each input's value per step via the backend's feed map. Grad treats it as a leaf, so
    /// params are just inputs you take the gradient w.r.t.
    pub fn input(&mut self, shape: Vec<usize>, dtype: DType) -> NodeId {
        self.push(Op::Input { shape, dtype }, vec![])
    }

    /// Convert to another dtype. Auto-promotion (mixed-dtype binary ops) is the
    /// frontend's job: it inserts these casts per a promotion table.
    pub fn cast(&mut self, a: NodeId, to: DType) -> NodeId {
        self.push(Op::Cast { to }, vec![a])
    }

    /// Identity value, but the gradient stops here (stop_gradient / no-backprop).
    pub fn detach(&mut self, a: NodeId) -> NodeId {
        self.push(Op::Detach, vec![a])
    }

    /// Reinterpret the bits as `to` (same width, shape unchanged): no conversion.
    pub fn bitcast(&mut self, a: NodeId, to: DType) -> Result<NodeId, Error> {
        let from = self.dtype(a);
        if from.width() != to.width() {
            return Err(Error::shape(
                "bitcast",
                format!("width {:?}({}) != {:?}({})", from, from.width(), to, to.width()),
            ));
        }
        Ok(self.push(Op::Bitcast { to }, vec![a]))
    }
}
