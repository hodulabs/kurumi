//! Movement lowering: a chain of movement ops becomes ONE index expression over the
//! source buffer (no intermediate copies). reshape on a non-contiguous view emits div/mod;
//! if the simplifier can't collapse them we signal a contiguous-copy fallback (never wrong).
//! pad adds validity guards (masked load: out-of-range coords read the pad value, not the
//! source). The RANGEIFY model that replaces a separate view algebra.
//!
//! The movement algebra (building a `View`) lives in `movement`; the reader that evaluates a
//! finished view against a buffer lives in `eval`.

mod eval;
mod movement;

pub(crate) use eval::load_at;
pub use eval::read;

use crate::lower::sym::{Sym, VarId};

/// Validity guard on an output loop var: the read is valid iff `lo <= var < hi`,
/// otherwise it yields the pad value (0). Produced by `pad`.
#[derive(Clone, Debug)]
pub struct Guard {
    pub var: VarId,
    pub lo: i64,
    pub hi: i64,
}

/// A read pattern into a source buffer: for output coordinate (var(0)..var(n-1),
/// each over [0, shape[i])), `offset` gives the flat source index; `guards` mask
/// out-of-range coords to the pad value.
#[derive(Clone, Debug)]
pub struct View {
    pub shape: Vec<usize>,
    pub offset: Sym,
    pub contiguous: bool,
    pub guards: Vec<Guard>,
}
