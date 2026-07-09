//! C ABI: graph lifecycle, passes/inspection, and runnable-graph serialization. Node
//! builders (input / constant / scalar) live in graph/build.rs.

use crate::capi::{KU_ERR, KuGraph, KuRunnable, build, catch, raw_slice, set_err};
use kurumi_core::{
    Graph, InputBinding, InputRole, NodeId, amp, deserialize_graph, dump, node_count, serialize_graph,
    serialize_reachable, simplify,
};
use std::ffi::{CStr, c_char};

pub(crate) mod build;

#[unsafe(no_mangle)]
pub extern "C" fn ku_graph_new() -> *mut KuGraph {
    Box::into_raw(Box::new(KuGraph(Graph::new())))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_graph_free(g: *mut KuGraph) {
    if !g.is_null() {
        drop(Box::from_raw(g));
    }
}
// passes & inspection

/// Algebraic simplification of the graph reachable from `root`; returns the new root.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_simplify(g: *mut KuGraph, root: u32) -> u32 {
    build(g, |gr| Ok(simplify(gr, NodeId(root))))
}

/// Automatic mixed precision: insert casts so matmuls run in f16; returns new root.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_amp(g: *mut KuGraph, root: u32) -> u32 {
    build(g, |gr| Ok(amp(gr, NodeId(root))))
}

/// Number of distinct nodes reachable from `root`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_node_count(g: *const KuGraph, root: u32) -> usize {
    if g.is_null() {
        return 0;
    }
    let gr = &(*g).0;
    catch(0, || node_count(gr, NodeId(root)))
}

/// Write up to `cap` bytes of the human-readable graph dump into `out` (UTF-8, no
/// trailing NUL); returns the full length. Pass cap=0 to size the buffer first.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_dump(g: *const KuGraph, root: u32, out: *mut u8, cap: usize) -> usize {
    if g.is_null() {
        return 0;
    }
    let gr = &(*g).0;
    catch(0, || {
        let s = dump(gr, NodeId(root));
        let n = s.len().min(cap);
        if n > 0 && !out.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), out, n) };
        }
        s.len()
    })
}

/// Rank (number of dims) of `node`'s shape; SIZE_MAX on a null graph or invalid node.
/// Lets a frontend query shapes to insert broadcast/cast before the strict builder ops.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_node_rank(g: *const KuGraph, node: u32) -> usize {
    if g.is_null() {
        return usize::MAX;
    }
    let gr = &(*g).0;
    catch(usize::MAX, || gr.shape(NodeId(node)).len())
}

/// Write `node`'s shape (`ku_node_rank` size_t values) into `out`; no-op on a null
/// graph/out or invalid node. Size the buffer with `ku_node_rank` first.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_node_shape(g: *const KuGraph, node: u32, out: *mut usize) {
    if g.is_null() || out.is_null() {
        return;
    }
    let gr = &(*g).0;
    catch((), || {
        let sh = gr.shape(NodeId(node));
        unsafe { std::ptr::copy_nonoverlapping(sh.as_ptr(), out, sh.len()) };
    });
}

/// dtype index (matches KuDType) of `node`; KU_ERR on a null graph or invalid node.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_node_dtype(g: *const KuGraph, node: u32) -> u32 {
    if g.is_null() {
        return KU_ERR;
    }
    let gr = &(*g).0;
    catch(KU_ERR, || gr.dtype(NodeId(node)) as u32)
}

// runnable-graph serialization (the .hodu graph section)

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

/// Deserialize a blob from [`ku_graph_serialize`] into a runnable handle (rebuilt graph +
/// output nodes + input bindings). Returns NULL on a malformed blob (see `ku_last_error`).
/// Free with `ku_runnable_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_graph_deserialize(bytes: *const u8, len: usize) -> *mut KuRunnable {
    match deserialize_graph(raw_slice(bytes, len)) {
        Ok(r) => Box::into_raw(Box::new(KuRunnable(r))),
        Err(e) => {
            set_err(format!("ku_graph_deserialize: {e:?}"));
            std::ptr::null_mut()
        }
    }
}

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

/// Number of output nodes in the runnable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_output_count(h: *const KuRunnable) -> usize {
    if h.is_null() {
        return 0;
    }
    let r = &(*h).0;
    r.outputs.len()
}

/// The i-th output NodeId; KU_ERR if the handle is null or `i` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_output(h: *const KuRunnable, i: usize) -> u32 {
    if h.is_null() {
        return KU_ERR;
    }
    let r = &(*h).0;
    r.outputs.get(i).map_or(KU_ERR, |n| n.0)
}

/// Number of input bindings in the runnable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_input_count(h: *const KuRunnable) -> usize {
    if h.is_null() {
        return 0;
    }
    let r = &(*h).0;
    r.inputs.len()
}

/// The i-th input's NodeId; KU_ERR if the handle is null or `i` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_input_node(h: *const KuRunnable, i: usize) -> u32 {
    if h.is_null() {
        return KU_ERR;
    }
    let r = &(*h).0;
    r.inputs.get(i).map_or(KU_ERR, |b| b.node.0)
}

/// The i-th input's role: 0 = weight (bound by name), 1 = runtime feed. KU_ERR if the
/// handle is null or `i` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_input_role(h: *const KuRunnable, i: usize) -> u32 {
    if h.is_null() {
        return KU_ERR;
    }
    let r = &(*h).0;
    match r.inputs.get(i) {
        Some(b) => match b.role {
            InputRole::Weight => 0,
            InputRole::Runtime => 1,
        },
        None => KU_ERR,
    }
}

/// Write up to `cap` bytes of the i-th input's name (UTF-8, no trailing NUL) into `out`;
/// returns the full length. Size-then-write: pass cap=0 first. 0 on a null handle/bad index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_input_name(h: *const KuRunnable, i: usize, out: *mut u8, cap: usize) -> usize {
    if h.is_null() {
        return 0;
    }
    let r = &(*h).0;
    let Some(b) = r.inputs.get(i) else {
        return 0;
    };
    let name = b.name.as_bytes();
    let n = name.len().min(cap);
    if n > 0 && !out.is_null() {
        std::ptr::copy_nonoverlapping(name.as_ptr(), out, n);
    }
    name.len()
}

/// Free a runnable handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_runnable_free(h: *mut KuRunnable) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}
