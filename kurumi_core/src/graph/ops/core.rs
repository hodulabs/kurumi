//! Base tensor substrate: domain-agnostic op builders every higher layer composes on.
//! Primitives (push one `Op`) and their decompositions live together per kind.

mod bitwise;
mod compare;
mod complex;
mod contract;
mod distance;
mod einsum;
mod elementwise;
mod explog;
mod fft;
mod indexing;
mod join;
mod linalg;
mod masked;
mod movement;
mod random;
mod reduce;
mod scan;
mod signal;
mod spatial;
mod special;
mod stats;
mod triangular;
mod trig;
