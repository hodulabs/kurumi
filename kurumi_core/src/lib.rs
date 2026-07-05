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

pub use backend::{Backend, CpuBackend};
pub(crate) use dtype::{Bitwise, Elem, Float, Int, Num, Signed, bitcast, cast, iota_storage};
pub use dtype::{DType, Storage, TensorVal};
pub use error::Error;
pub use grad::grad;
pub use graph::{ArgKind, Graph, Node, NodeId, Op, ScatterOp, amp, dump, node_count, reachable, simplify};
pub use interp::{Feeds, eval_op, interpret, interpret_with};
pub(crate) use interp::{dot_dispatch, dot_general, reduce_v};
pub use layout::row_major_strides;
pub(crate) use layout::{free_axes, inc};
pub use quant::{Quantized, dequant_matmul, dequantize, quantize};
pub use rng::Key;
pub use tensor::{Ctx, Tensor};
