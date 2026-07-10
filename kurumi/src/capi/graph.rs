//! C ABI: graph lifecycle plus passes/inspection. Node builders (input / constant / scalar)
//! live in graph/leaves.rs; runnable-graph serialization lives in graph/serialize.rs.

pub(crate) mod leaves;
pub(crate) mod serialize;

use crate::capi::{KU_ERR, KuGraph, build, catch};
use kurumi_core::{Graph, NodeId, amp, dump, node_count, simplify};

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
