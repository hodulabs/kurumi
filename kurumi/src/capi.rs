//! C ABI: graph-builder surface, exposed from the cdylib/staticlib. Opaque handles +
//! a thread-local last-error; builder ops return a node id (`KU_ERR` = error, then
//! `ku_last_error`). A `KuBackend` is created once and reused across evals so the Metal
//! pipeline/const caches survive a training loop. Tensor exchange is a raw copy (f32 or
//! per-dtype little-endian bytes); no DLPack zero-copy path.

#![allow(clippy::missing_safety_doc)] // every fn's contract is documented in kurumi.h
#![allow(unsafe_op_in_unsafe_fn)] // FFI glue: each `extern "C"` fn is unsafe by contract

mod eval;
mod graph;
mod ops;
mod tensor;
#[cfg(test)]
mod tests;

use kurumi_core::realize::Plan;
use kurumi_core::{Backend, DType, Feeds, Graph, NodeId, TensorVal};
use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::ptr;

/// Sentinel returned by builder ops on error (real node ids are `< u32::MAX`).
pub const KU_ERR: u32 = u32::MAX;

// opaque handles (heap Box, freed by the matching ku_*_free)
pub struct KuGraph(Graph);
pub struct KuTensor(TensorVal);
pub struct KuFeeds(Feeds);
pub struct KuBackend(Box<dyn Backend>);
pub struct KuPlan(Plan);

thread_local! {
    static LAST_ERR: RefCell<Option<CString>> = const { RefCell::new(None) };
}
pub(crate) fn set_err(msg: String) {
    LAST_ERR.with(|e| *e.borrow_mut() = CString::new(msg).ok());
}
// map a builder Result to a node id, stashing the message on error.
pub(crate) fn ok_node(r: Result<NodeId, kurumi_core::Error>) -> u32 {
    match r {
        Ok(n) => n.0,
        Err(e) => {
            set_err(format!("{e:?}"));
            KU_ERR
        }
    }
}

/// Last error message on this thread (NUL-terminated), or NULL if none. Valid until
/// the next failing call on the same thread; copy it if you need to keep it.
#[unsafe(no_mangle)]
pub extern "C" fn ku_last_error() -> *const c_char {
    LAST_ERR.with(|e| e.borrow().as_ref().map_or(ptr::null(), |s| s.as_ptr()))
}

// DType <-> the C enum (index = declaration order; see KuDType in kurumi.h).
pub(crate) fn dtype_from_u32(u: u32) -> Option<DType> {
    use DType::*;
    Some(match u {
        0 => BOOL,
        1 => U8,
        2 => U16,
        3 => U32,
        4 => U64,
        5 => I8,
        6 => I16,
        7 => I32,
        8 => I64,
        9 => F8E4M3,
        10 => F8E5M2,
        11 => F16,
        12 => BF16,
        13 => F32,
        14 => F64,
        15 => C64,
        16 => C128,
        _ => return None,
    })
}

// Run f at the FFI boundary; a panic (bad node id index-out-of-bounds, internal
// invariant) is turned into `on_err` with the message stashed in last_error. Catches
// panics only, NOT segfaults - callers must null-check pointers before dereferencing.
pub(crate) fn catch<T>(on_err: T, f: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(e) => {
            let msg = e
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic at FFI boundary".to_string());
            set_err(format!("panic: {msg}"));
            on_err
        }
    }
}

// Single-output builder boundary: null graph, a panic, or a builder Err all map to
// KU_ERR (message via set_err / ok_node). AssertUnwindSafe is sound here: a bad-id panic
// fires during shape/dtype inference before the node is pushed, so the graph stays
// consistent, and the frontend drops the graph on KU_ERR anyway.
pub(crate) fn build(g: *mut KuGraph, f: impl FnOnce(&mut Graph) -> Result<NodeId, kurumi_core::Error>) -> u32 {
    if g.is_null() {
        set_err("null graph handle".into());
        return KU_ERR;
    }
    catch(KU_ERR, || ok_node(f(unsafe { &mut (*g).0 })))
}

// null-check a multi-output op's g + out pointers; true (with err set) means bail. A null
// deref would segfault (uncatchable). Caller: `if null_handles(g, out) { return KU_ERR; }`.
pub(crate) fn null_handles(g: *mut KuGraph, out: *mut u32) -> bool {
    if g.is_null() || out.is_null() {
        set_err("null graph/out handle".into());
        return true;
    }
    false
}

// null/empty-safe view over a C array: null ptr or n==0 -> empty slice (from_raw_parts
// on a null ptr is UB, and an uncatchable segfault, so guard it here).
pub(crate) unsafe fn raw_slice<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 || p.is_null() { &[] } else { std::slice::from_raw_parts(p, n) }
}
pub(crate) unsafe fn usize_slice<'a>(p: *const usize, n: usize) -> &'a [usize] {
    raw_slice(p, n)
}
