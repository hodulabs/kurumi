//! Device-op evaluation, by family (roughly mirrors `context/dispatch/*` and
//! `graph/ops/*`, but not 1:1 - Cast folds into `pointwise`, hostgemm into `matmul`).
//! Each `eval_*` returns `Some(Val)` if it runs the op device-resident, else `None` to
//! fall through to the next family, then the fused-pointwise core. Called in sequence
//! from `MetalBackend::eval_memo`. New device ops drop into the matching file.

mod complex;
mod fused;
mod generate;
mod indexing;
mod linalg;
mod matmul;
mod nn;
mod pointwise;
mod quant;
mod reduce;

// shared surface for the family submodules (defined in parent `backend`).
pub(super) use crate::backend::fuse::{
    Ew, FExpr, FUSE_CAP, MAX_LEAVES, REDUCE_TG, Val, ew_kind, fused_reduce_msl, leaf_eq,
};
pub(super) use crate::backend::{combine_str, scatter_dev_ok, storage_i32};
