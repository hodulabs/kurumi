//! C ABI: backend handle (created once, reused across evals), eval/grad, feeds.

use crate::capi::{KuBackend, KuFeeds, KuGraph, KuPlan, KuTensor, catch, raw_slice, set_err};
use kurumi_core::realize::Plan;
use kurumi_core::{Backend, CpuBackend, Feeds, NodeId};
use std::ptr;

/// `kind`: 0 = CPU, 1 = Metal (macOS; NULL if the device is unavailable).
#[unsafe(no_mangle)]
pub extern "C" fn ku_backend_new(kind: u32) -> *mut KuBackend {
    let b: Option<Box<dyn Backend>> = match kind {
        0 => Some(Box::new(CpuBackend)),
        1 => metal_backend(),
        _ => {
            set_err(format!("ku_backend_new: unknown kind {kind}"));
            return ptr::null_mut();
        }
    };
    match b {
        Some(b) => Box::into_raw(Box::new(KuBackend(b))),
        None => ptr::null_mut(),
    }
}
#[cfg(target_os = "macos")]
fn metal_backend() -> Option<Box<dyn Backend>> {
    match kurumi_metal::MetalBackend::new() {
        Some(m) => Some(Box::new(m)),
        None => {
            set_err("ku_backend_new: no Metal device".into());
            None
        }
    }
}
#[cfg(not(target_os = "macos"))]
fn metal_backend() -> Option<Box<dyn Backend>> {
    set_err("ku_backend_new: Metal backend is macOS-only".into());
    None
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_backend_free(b: *mut KuBackend) {
    if !b.is_null() {
        drop(Box::from_raw(b));
    }
}

/// Evaluate `node` (no `Input` nodes) -> a new `KuTensor` (NULL on error).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_eval(g: *mut KuGraph, node: u32, backend: *const KuBackend) -> *mut KuTensor {
    if g.is_null() || backend.is_null() {
        set_err("ku_eval: null graph/backend handle".into());
        return ptr::null_mut();
    }
    let (b, gr) = (&(*backend).0, &(*g).0);
    catch(ptr::null_mut(), || Box::into_raw(Box::new(KuTensor(b.eval(gr, NodeId(node))))))
}

/// Evaluate `node`, supplying `Input` nodes from `feeds` -> a new `KuTensor`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_eval_with(
    g: *mut KuGraph,
    node: u32,
    backend: *const KuBackend,
    feeds: *const KuFeeds,
) -> *mut KuTensor {
    if g.is_null() || backend.is_null() || feeds.is_null() {
        set_err("ku_eval_with: null graph/backend/feeds handle".into());
        return ptr::null_mut();
    }
    let (b, gr, fe) = (&(*backend).0, &(*g).0, &(*feeds).0);
    catch(ptr::null_mut(), || Box::into_raw(Box::new(KuTensor(b.eval_with(gr, NodeId(node), fe)))))
}

/// Reverse-mode gradients of `sum(out)` w.r.t. each of the `n` `wrt` nodes, written
/// into `out_grads[0..n]`. Returns 0 on success, -1 on error (see `ku_last_error`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_grad(g: *mut KuGraph, out: u32, wrt: *const u32, n: usize, out_grads: *mut u32) -> i32 {
    if g.is_null() || out_grads.is_null() || (wrt.is_null() && n > 0) {
        set_err("ku_grad: null graph/wrt/out_grads handle".into());
        return -1;
    }
    let wrt: Vec<NodeId> = raw_slice(wrt, n).iter().map(|&x| NodeId(x)).collect();
    let gr = &mut (*g).0;
    catch(-1, || match kurumi_core::grad(gr, NodeId(out), &wrt) {
        Ok(gs) => unsafe {
            for (i, ng) in gs.iter().enumerate() {
                *out_grads.add(i) = ng.0;
            }
            0
        },
        Err(e) => {
            set_err(format!("{e:?}"));
            -1
        }
    })
}

// feeds

#[unsafe(no_mangle)]
pub extern "C" fn ku_feeds_new() -> *mut KuFeeds {
    Box::into_raw(Box::new(KuFeeds(Feeds::new())))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_feeds_free(f: *mut KuFeeds) {
    if !f.is_null() {
        drop(Box::from_raw(f));
    }
}
/// Bind `node` (an `Input`) to a copy of `tensor` for the next eval.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_feeds_set(f: *mut KuFeeds, node: u32, tensor: *const KuTensor) {
    if f.is_null() || tensor.is_null() {
        return;
    }
    (*f).0.insert(NodeId(node), (*tensor).0.clone());
}

// plan-replay (compile once, run per feeds; consts are never re-copied)

/// Compile `node` into a replayable plan (CPU f32 fused path). NULL if the graph
/// leaves that path: fall back to `ku_eval_with`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_plan_compile(g: *const KuGraph, node: u32) -> *mut KuPlan {
    if g.is_null() {
        set_err("ku_plan_compile: null graph handle".into());
        return ptr::null_mut();
    }
    let gr = &(*g).0;
    catch(ptr::null_mut(), || match Plan::compile(gr, NodeId(node)) {
        Some(p) => Box::into_raw(Box::new(KuPlan(p))),
        None => {
            set_err("ku_plan_compile: graph is not on the f32 fused plan path".into());
            ptr::null_mut()
        }
    })
}
/// Replay the plan with fresh `Input` feeds -> a new `KuTensor`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_plan_run(p: *const KuPlan, g: *const KuGraph, feeds: *const KuFeeds) -> *mut KuTensor {
    if p.is_null() || g.is_null() || feeds.is_null() {
        set_err("ku_plan_run: null plan/graph/feeds handle".into());
        return ptr::null_mut();
    }
    let (pl, gr, fe) = (&(*p).0, &(*g).0, &(*feeds).0);
    catch(ptr::null_mut(), || Box::into_raw(Box::new(KuTensor(pl.run(gr, fe)))))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_plan_free(p: *mut KuPlan) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}
