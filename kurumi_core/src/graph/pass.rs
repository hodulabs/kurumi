//! IR->IR rewrite passes: each rebuilds the subgraph rooted at a node (bottom-up,
//! `reachable` + a remap) and returns the new root. They share that shape, so new passes
//! (DCE, constant folding, canonicalization, fusion) drop in here. Read-only analysis
//! lives in the sibling `inspect`.

mod amp;
mod simplify;

pub use amp::amp;
pub use simplify::simplify;
