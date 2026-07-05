// Multi-output linalg factorizations. Two-factor results go through the parent's
// `write2`; svd (three factors) writes `out` inline. Each null-checks g and out, then
// runs under `catch` so a bad node id returns KU_ERR instead of aborting.

use super::write2;
use crate::capi::{KU_ERR, KuGraph, catch, null_handles, set_err};
use kurumi_core::NodeId;

/// slogdet -> [sign, logabsdet] into `out[2]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_slogdet(g: *mut KuGraph, x: u32, out: *mut u32) -> u32 {
    if null_handles(g, out) {
        return KU_ERR;
    }
    let gr = &mut (*g).0;
    catch(KU_ERR, || write2(out, gr.slogdet(NodeId(x))))
}
/// eigh -> [eigenvalues, eigenvectors] into `out[2]` (ascending).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_eigh(g: *mut KuGraph, x: u32, out: *mut u32) -> u32 {
    if null_handles(g, out) {
        return KU_ERR;
    }
    let gr = &mut (*g).0;
    catch(KU_ERR, || write2(out, gr.eigh(NodeId(x))))
}
/// qr -> [Q, R] into `out[2]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_qr(g: *mut KuGraph, x: u32, out: *mut u32) -> u32 {
    if null_handles(g, out) {
        return KU_ERR;
    }
    let gr = &mut (*g).0;
    catch(KU_ERR, || write2(out, gr.qr(NodeId(x))))
}
/// svd -> [U, S, V] into `out[3]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_svd(g: *mut KuGraph, x: u32, out: *mut u32) -> u32 {
    if null_handles(g, out) {
        return KU_ERR;
    }
    let gr = &mut (*g).0;
    catch(KU_ERR, || match gr.svd(NodeId(x)) {
        Ok((u, s, v)) => unsafe {
            *out = u.0;
            *out.add(1) = s.0;
            *out.add(2) = v.0;
            0
        },
        Err(e) => {
            set_err(format!("{e:?}"));
            KU_ERR
        }
    })
}
/// topk -> [values, indices] into `out[2]`; `largest != 0` for top-k else bottom-k.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ku_topk(g: *mut KuGraph, x: u32, k: usize, axis: usize, largest: u32, out: *mut u32) -> u32 {
    if null_handles(g, out) {
        return KU_ERR;
    }
    let gr = &mut (*g).0;
    catch(KU_ERR, || write2(out, gr.topk(NodeId(x), k, axis, largest != 0)))
}
