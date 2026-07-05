//! MSL kernel sources, one file per dispatch family (mirrors `context/dispatch/`). Pure
//! strings / per-dtype generators; the device layer (context.rs) and backend compile them on
//! demand. Elementwise fusion generates its MSL in `backend/fuse.rs` (it is codegen, not a
//! fixed kernel), so it has no file here.

pub(crate) mod cast;
pub(crate) mod complex;
pub(crate) mod generate;
pub(crate) mod hostgemm;
pub(crate) mod indexing;
pub(crate) mod linalg;
pub(crate) mod matmul;
pub(crate) mod nn;
pub(crate) mod pointwise;
pub(crate) mod quant;
pub(crate) mod reduce;
