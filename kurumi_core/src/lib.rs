//! kurumi engine core: closed-primitive IR, dtype system, reference
//! interpreter, and the view-fused realize path. Modules are flat files
//! (foo.rs), no mod.rs.

pub mod lower;
pub mod quant;
pub mod realize;
pub mod rng;
pub mod tensor;

#[macro_use]
mod dtype;
mod backend;
mod error;
mod grad;
mod graph;
mod interp;
mod layout;

pub use backend::{Backend, CpuBackend, Feeds};
pub(crate) use dtype::{Bitwise, Elem, Float, Int, Num, Signed, bitcast, cast, iota_storage};
pub use dtype::{DType, Storage, TensorVal};
pub use error::Error;
pub use grad::grad;
pub use graph::{
    ArgKind, Graph, InputBinding, InputRole, Node, NodeId, Op, Runnable, ScatterOp, amp, deserialize_graph, dump,
    node_count, reachable, serialize_graph, serialize_reachable, simplify,
};
pub(crate) use interp::{dot_dispatch, dot_general, reduce_v};
pub use interp::{eval_op, interpret, interpret_many, interpret_with};
pub use layout::row_major_strides;
pub(crate) use layout::{free_axes, inc};
pub use quant::{Quantized, dequant_matmul, dequantize, quantize};
pub use rng::Key;
pub use tensor::{Ctx, Tensor};
