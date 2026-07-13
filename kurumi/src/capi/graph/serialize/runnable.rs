//! C ABI: the `ku_runnable_*` accessors over a deserialized runnable graph -- output/input
//! metadata (entry 0 aliases + entry-scoped), taking the rebuilt graph, and freeing the handle.
//! The serialize/deserialize entry points live in the parent `serialize.rs`.

use super::write_bytes;
use crate::capi::{KU_ERR, KuGraph, KuRunnable, set_err};
use kurumi_core::{Graph, InputRole};

/// Move the rebuilt graph out of the runnable into its own handle (call once). The runnable
/// keeps its output/input metadata; free the graph with `ku_graph_free` and the runnable
/// with `ku_runnable_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_take_graph(h: *mut KuRunnable) -> *mut KuGraph {
    if h.is_null() {
        set_err("ku_runnable_take_graph: null handle".into());
        return std::ptr::null_mut();
    }
    let g = std::mem::replace(&mut (*h).0.graph, Graph::new());
    Box::into_raw(Box::new(KuGraph(g)))
}

// entry-0 aliases: the runnable's single/forward entry. Kept so an old consumer (hodu-py's
// `_deserialize`) reads the forward entry unchanged; each just forwards to the entry accessor
// with entry index 0.

/// Number of output nodes in the runnable (entry 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_output_count(h: *const KuRunnable) -> usize {
    ku_runnable_entry_output_count(h, 0)
}

/// The i-th output NodeId (entry 0); KU_ERR if the handle is null or `i` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_output(h: *const KuRunnable, i: usize) -> u32 {
    ku_runnable_entry_output(h, 0, i)
}

/// Number of input bindings in the runnable (entry 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_input_count(h: *const KuRunnable) -> usize {
    ku_runnable_entry_input_count(h, 0)
}

/// The i-th input's NodeId (entry 0); KU_ERR if the handle is null or `i` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_input_node(h: *const KuRunnable, i: usize) -> u32 {
    ku_runnable_entry_input_node(h, 0, i)
}

/// The i-th input's role (entry 0): 0 = weight (bound by name), 1 = runtime feed. KU_ERR if the
/// handle is null or `i` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_input_role(h: *const KuRunnable, i: usize) -> u32 {
    ku_runnable_entry_input_role(h, 0, i)
}

/// Write up to `cap` bytes of the i-th input's name (entry 0, UTF-8, no trailing NUL) into
/// `out`; returns the full length. Size-then-write: pass cap=0 first. 0 on a null handle/bad index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_input_name(h: *const KuRunnable, i: usize, out: *mut u8, cap: usize) -> usize {
    ku_runnable_entry_input_name(h, 0, i, out, cap)
}

// entry-scoped accessors: reach every named entry (i indexes the entry, j the output/input).

/// Number of entry points in the runnable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_entry_count(h: *const KuRunnable) -> usize {
    if h.is_null() {
        return 0;
    }
    let r = &(*h).0;
    r.entries.len()
}

/// Write up to `cap` bytes of the i-th entry's name (UTF-8, no trailing NUL) into `out`;
/// returns the full length. Size-then-write: pass cap=0 first. 0 on a null handle/bad index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_entry_name(h: *const KuRunnable, i: usize, out: *mut u8, cap: usize) -> usize {
    if h.is_null() {
        return 0;
    }
    let r = &(*h).0;
    let Some(e) = r.entries.get(i) else {
        return 0;
    };
    write_bytes(e.name.as_bytes(), out, cap)
}

/// Number of outputs of the i-th entry.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_entry_output_count(h: *const KuRunnable, i: usize) -> usize {
    if h.is_null() {
        return 0;
    }
    let r = &(*h).0;
    r.entries.get(i).map_or(0, |e| e.outputs.len())
}

/// The j-th output NodeId of the i-th entry; KU_ERR on a null handle or bad index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_entry_output(h: *const KuRunnable, i: usize, j: usize) -> u32 {
    if h.is_null() {
        return KU_ERR;
    }
    let r = &(*h).0;
    r.entries.get(i).and_then(|e| e.outputs.get(j)).map_or(KU_ERR, |n| n.0)
}

/// Number of input bindings of the i-th entry.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_entry_input_count(h: *const KuRunnable, i: usize) -> usize {
    if h.is_null() {
        return 0;
    }
    let r = &(*h).0;
    r.entries.get(i).map_or(0, |e| e.inputs.len())
}

/// The j-th input's NodeId of the i-th entry; KU_ERR on a null handle or bad index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_entry_input_node(h: *const KuRunnable, i: usize, j: usize) -> u32 {
    if h.is_null() {
        return KU_ERR;
    }
    let r = &(*h).0;
    r.entries.get(i).and_then(|e| e.inputs.get(j)).map_or(KU_ERR, |b| b.node.0)
}

/// The j-th input's role of the i-th entry: 0 = weight, 1 = runtime feed. KU_ERR on a null
/// handle or bad index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_entry_input_role(h: *const KuRunnable, i: usize, j: usize) -> u32 {
    if h.is_null() {
        return KU_ERR;
    }
    let r = &(*h).0;
    match r.entries.get(i).and_then(|e| e.inputs.get(j)) {
        Some(b) => match b.role {
            InputRole::Weight => 0,
            InputRole::Runtime => 1,
        },
        None => KU_ERR,
    }
}

/// Write up to `cap` bytes of the j-th input's name of the i-th entry (UTF-8, no trailing NUL)
/// into `out`; returns the full length. Size-then-write. 0 on a null handle/bad index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_entry_input_name(
    h: *const KuRunnable,
    i: usize,
    j: usize,
    out: *mut u8,
    cap: usize,
) -> usize {
    if h.is_null() {
        return 0;
    }
    let r = &(*h).0;
    let Some(b) = r.entries.get(i).and_then(|e| e.inputs.get(j)) else {
        return 0;
    };
    write_bytes(b.name.as_bytes(), out, cap)
}

/// Free a runnable handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_free(h: *mut KuRunnable) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}
