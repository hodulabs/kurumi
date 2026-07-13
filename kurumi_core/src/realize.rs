//! View-fused evaluator. Movement only rewrites the read view over a shared source
//! buffer (Rc, 0 copies); elementwise ops fuse into one lazy expression read at the
//! output coordinate. Materializes only at a boundary (reduce, contraction, output,
//! movement on a fused result, multi-consumer node), so a movement+elementwise subtree
//! runs in ONE pass and a shared node computes once. Checked against `interpret`.
//! Submodules: expr (types), tape (executor), plan (compile-once replay), sched (the
//! scheduler walk graph -> `Realized`), eval (one-shot force entries + the fused-path gate).

mod eval;
mod expr;
mod plan;
mod sched;
mod tape;

pub use eval::{force, force_counted, force_into};
pub use expr::Realized;
pub use plan::Plan;
pub use sched::realize;

pub(crate) use eval::{bump_kernel, fused_supported};
pub(crate) use sched::{Sched, consumer_counts, go};

#[cfg(test)]
mod fuzz;
#[cfg(test)]
mod tests;
