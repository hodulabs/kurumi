//! C ABI: runnable-graph serialization (the .hodu graph section) -- the serialize/deserialize
//! entry points. The `ku_runnable_*` accessors over the rebuilt graph's output/input metadata
//! live in the `runnable` submodule.

mod runnable;

// C ABI symbols are exported straight from `runnable` (no_mangle); this re-export only keeps the
// Rust path `serialize::ku_runnable_*` stable for the glob-importing tests.
#[cfg(test)]
pub use runnable::*;

use crate::capi::{KU_ERR, KuGraph, KuRunnable, catch, raw_slice, set_err};
use kurumi_core::{
    InputBinding, InputRole, NodeId, deserialize_multi, serialize_graph, serialize_multi, serialize_reachable,
};
use std::ffi::{CStr, c_char};

// a C string pointer -> owned String (NULL -> empty). Borrowed only for the call's duration.
unsafe fn cstr(p: *const c_char) -> String {
    if p.is_null() { String::new() } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
}

// size-then-write a byte slice into a C out buffer (see the name accessors): copy up to `cap`
// bytes, always return the full length.
unsafe fn write_bytes(src: &[u8], out: *mut u8, cap: usize) -> usize {
    let n = src.len().min(cap);
    if n > 0 && !out.is_null() {
        std::ptr::copy_nonoverlapping(src.as_ptr(), out, n);
    }
    src.len()
}

// shared marshaling for ku_graph_serialize[_reachable]: build the input bindings from the C
// arrays, serialize (whole arena or reachable cone), then size-then-write into `out`.
unsafe fn serialize_impl(
    reachable: bool,
    g: *const KuGraph,
    outputs: *const u32,
    n_out: usize,
    in_nodes: *const u32,
    in_roles: *const u8,
    in_names: *const *const c_char,
    n_in: usize,
    out: *mut u8,
    cap: usize,
) -> usize {
    if g.is_null() {
        set_err("ku_graph_serialize: null graph".into());
        return 0;
    }
    let gr = &(*g).0;
    let out_ids: Vec<NodeId> = raw_slice(outputs, n_out).iter().map(|&i| NodeId(i)).collect();
    let nodes = raw_slice(in_nodes, n_in);
    let roles = raw_slice(in_roles, n_in);
    let names = raw_slice(in_names, n_in);
    let mut inputs = Vec::with_capacity(n_in);
    for i in 0..n_in {
        let role = if roles.get(i).copied() == Some(0) { InputRole::Weight } else { InputRole::Runtime };
        let name = match names.get(i) {
            Some(&p) if !p.is_null() => CStr::from_ptr(p).to_string_lossy().into_owned(),
            _ => String::new(),
        };
        let node = NodeId(nodes.get(i).copied().unwrap_or(KU_ERR));
        inputs.push(InputBinding { node, role, name });
    }
    let blob =
        if reachable { serialize_reachable(gr, &out_ids, &inputs) } else { serialize_graph(gr, &out_ids, &inputs) };
    let n = blob.len().min(cap);
    if n > 0 && !out.is_null() {
        std::ptr::copy_nonoverlapping(blob.as_ptr(), out, n);
    }
    blob.len()
}

/// Serialize the graph plus its output nodes and input bindings into a self-contained blob.
/// `outputs` (n_out node ids) are the roots to eval; `in_nodes`/`in_roles`/`in_names` (n_in
/// each) bind each Input node to a name and a role (0 = weight bound by name, 1 = runtime
/// feed); `in_names` is an array of NUL-terminated UTF-8 strings. Size-then-write: pass
/// cap=0 with out=NULL to get the length, then a buffer of that size. Returns the full length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_graph_serialize(
    g: *const KuGraph,
    outputs: *const u32,
    n_out: usize,
    in_nodes: *const u32,
    in_roles: *const u8,
    in_names: *const *const c_char,
    n_in: usize,
    out: *mut u8,
    cap: usize,
) -> usize {
    serialize_impl(false, g, outputs, n_out, in_nodes, in_roles, in_names, n_in, out, cap)
}

/// Like [`ku_graph_serialize`] but writes only the nodes reachable from `outputs`, remapped
/// to a dense id range -- dropping backward/dead arena nodes so a training graph exports a
/// clean inference program. Unreachable input bindings are omitted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_graph_serialize_reachable(
    g: *const KuGraph,
    outputs: *const u32,
    n_out: usize,
    in_nodes: *const u32,
    in_roles: *const u8,
    in_names: *const *const c_char,
    n_in: usize,
    out: *mut u8,
    cap: usize,
) -> usize {
    serialize_impl(true, g, outputs, n_out, in_nodes, in_roles, in_names, n_in, out, cap)
}

/// Serialize N named entry points sharing one arena into a self-contained multi-entry blob (so
/// one artifact holds e.g. "forward" and "forward_backward"). `names` holds `n_entries`
/// NUL-terminated entry names. Outputs are flattened: `out_counts[e]` gives entry `e`'s output
/// count, `outputs` concatenates all entries' output ids. Inputs likewise: `in_counts[e]` is
/// entry `e`'s input count, and `in_nodes`/`in_roles`/`in_names` concatenate all entries'
/// bindings (role 0 = weight, 1 = runtime feed). Only the union of all entries' output cones is
/// written (remapped dense); an entry input on an unreachable node is dropped. Size-then-write:
/// cap=0 with out=NULL returns the length. Read back with `ku_graph_deserialize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_graph_serialize_multi(
    g: *const KuGraph,
    n_entries: usize,
    names: *const *const c_char,
    out_counts: *const u32,
    outputs: *const u32,
    in_counts: *const u32,
    in_nodes: *const u32,
    in_roles: *const u8,
    in_names: *const *const c_char,
    out: *mut u8,
    cap: usize,
) -> usize {
    if g.is_null() {
        set_err("ku_graph_serialize_multi: null graph".into());
        return 0;
    }
    let gr = &(*g).0;
    let name_ptrs = raw_slice(names, n_entries);
    let oc = raw_slice(out_counts, n_entries);
    let ic = raw_slice(in_counts, n_entries);
    let all_out = raw_slice(outputs, oc.iter().map(|&c| c as usize).sum());
    let total_in: usize = ic.iter().map(|&c| c as usize).sum();
    let all_in_nodes = raw_slice(in_nodes, total_in);
    let all_in_roles = raw_slice(in_roles, total_in);
    let all_in_names = raw_slice(in_names, total_in);

    // build owned per-entry outputs/inputs, then a slice of (name, &outs, &ins) tuples over them.
    let names_owned: Vec<String> = name_ptrs.iter().map(|&p| cstr(p)).collect();
    let mut out_vecs: Vec<Vec<NodeId>> = Vec::with_capacity(n_entries);
    let mut in_vecs: Vec<Vec<InputBinding>> = Vec::with_capacity(n_entries);
    let (mut o_off, mut i_off) = (0usize, 0usize);
    for e in 0..n_entries {
        let no = oc[e] as usize;
        out_vecs.push(all_out[o_off..o_off + no].iter().map(|&i| NodeId(i)).collect());
        o_off += no;
        let ni = ic[e] as usize;
        let mut binds = Vec::with_capacity(ni);
        for k in i_off..i_off + ni {
            let role = if all_in_roles.get(k).copied() == Some(0) { InputRole::Weight } else { InputRole::Runtime };
            let name = all_in_names.get(k).map(|&p| cstr(p)).unwrap_or_default();
            let node = NodeId(all_in_nodes.get(k).copied().unwrap_or(KU_ERR));
            binds.push(InputBinding { node, role, name });
        }
        i_off += ni;
        in_vecs.push(binds);
    }
    let tuples: Vec<(&str, &[NodeId], &[InputBinding])> =
        (0..n_entries).map(|e| (names_owned[e].as_str(), out_vecs[e].as_slice(), in_vecs[e].as_slice())).collect();
    let blob = serialize_multi(gr, &tuples);
    write_bytes(&blob, out, cap)
}

/// Deserialize a blob (single- or multi-entry) into a runnable handle: the rebuilt shared graph
/// plus every entry's outputs/input bindings. The non-`entry` accessors below expose entry 0
/// (the single/forward entry); `ku_runnable_entry_*` reach all entries. Returns NULL on a
/// malformed or entry-less blob (see `ku_last_error`). Free with `ku_runnable_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_graph_deserialize(bytes: *const u8, len: usize) -> *mut KuRunnable {
    // Untrusted input crosses here: a blob that trips a panic during shape/dtype re-inference
    // must not unwind across `extern "C"` (that is an abort/UB). Deref the raw slice outside the
    // guard (like the inspection fns above), then run deserialize under `catch`.
    let data = raw_slice(bytes, len);
    catch(std::ptr::null_mut(), || match deserialize_multi(data) {
        Ok(r) if r.entries.is_empty() => {
            set_err("ku_graph_deserialize: graph blob has no entries".into());
            std::ptr::null_mut()
        }
        Ok(r) => Box::into_raw(Box::new(KuRunnable(r))),
        Err(e) => {
            set_err(format!("ku_graph_deserialize: {e:?}"));
            std::ptr::null_mut()
        }
    })
}
