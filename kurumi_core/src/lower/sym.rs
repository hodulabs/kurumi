//! Integer symbolic simplifier for index expressions.
//!
//! Movement ops lower to address arithmetic over loop indices; reshape emits `/` and `%`.
//! This reduces those using value ranges + algebra so movement chains fuse
//! (provably-identity index => no copy). Failure is safe: the div/mod stays and the
//! scheduler falls back to a `contiguous` copy.
//!
//! The `Sym` term tree + its queries live in `expr`; the rewrite driver + rules in `simplify`.

mod expr;
mod simplify;

pub use expr::{Ranges, Sym, VarId, c, var};
