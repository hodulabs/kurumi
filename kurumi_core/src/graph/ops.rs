//! Op builders on `impl Graph`, grouped by domain. `core` is the base tensor substrate
//! (elementwise/reduce/movement/indexing/linalg/...); `nn` is the ML fast-op layer
//! (activation/norm/attention/conv), composed down onto core. New flat domains (math,
//! quantum, signal, ...) become siblings here, depending on `core` the same one-way. The
//! closed primitives live in the engine core (op.rs/interp/grad/dtype), not here, so they
//! stay domain-agnostic.

mod core;
mod nn;
