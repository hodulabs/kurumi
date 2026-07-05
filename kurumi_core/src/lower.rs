//! Movement lowering (the RANGEIFY layer): a chain of movement ops becomes ONE index
//! expression over the source buffer, and an integer symbolic simplifier collapses the
//! div/mod that reshape emits (provably-identity index => no copy). Used by realize.

pub mod index;
pub mod sym;
