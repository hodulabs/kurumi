//! C ABI: graph lifecycle & node builders (input / constant / scalar).

use crate::capi::{KU_ERR, KuGraph, build, catch, dtype_from_u32, raw_slice, set_err, usize_slice};
use kurumi_core::{Graph, NodeId, Storage, amp, dump, node_count, simplify};

/// A baked constant of `dtype` from `nbytes` of little-endian element data,
/// row-major over `shape`. Covers every dtype (ku_constant_f32 is the f32 shortcut).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_constant(
    g: *mut KuGraph,
    dtype: u32,
    data: *const u8,
    nbytes: usize,
    shape: *const usize,
    rank: usize,
) -> u32 {
    let Some(dt) = dtype_from_u32(dtype) else {
        set_err(format!("ku_constant: bad dtype {dtype}"));
        return KU_ERR;
    };
    if data.is_null() && nbytes > 0 {
        set_err("ku_constant: null data".into());
        return KU_ERR;
    }
    let storage = Storage::from_bytes(dt, raw_slice(data, nbytes));
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| Ok(gr.const_storage(storage, shape)))
}

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

/// An `Input` node (fed at eval via `KuFeeds`). `dtype` is a `KuDType` index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_input(g: *mut KuGraph, shape: *const usize, rank: usize, dtype: u32) -> u32 {
    let Some(dt) = dtype_from_u32(dtype) else {
        set_err(format!("ku_input: bad dtype {dtype}"));
        return KU_ERR;
    };
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| Ok(gr.input(shape, dt)))
}

/// A baked f32 constant of `len` elements laid out row-major over `shape`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_constant_f32(
    g: *mut KuGraph,
    data: *const f32,
    len: usize,
    shape: *const usize,
    rank: usize,
) -> u32 {
    if data.is_null() && len > 0 {
        set_err("ku_constant_f32: null data".into());
        return KU_ERR;
    }
    let data = raw_slice(data, len).to_vec();
    let shape = usize_slice(shape, rank).to_vec();
    build(g, |gr| Ok(gr.constant(data, shape)))
}

/// A scalar with the same dtype/shape-broadcastable form as `like`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_scalar(g: *mut KuGraph, like: u32, v: f32) -> u32 {
    build(g, |gr| Ok(gr.scalar(NodeId(like), v)))
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
