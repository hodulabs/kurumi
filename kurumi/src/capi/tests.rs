//! C ABI logic test: calls the `extern "C"` fns directly (in-crate; same raw-pointer
//! calling convention a C consumer uses) through build -> eval -> grad -> error ->
//! feeds, plus Metal when present. Real symbol export is covered by the cdylib
//! crate-type (see kurumi.h); this validates the ABI behavior.

use crate::capi::eval::*;
use crate::capi::graph::*;
use crate::capi::ops::nn::{conv::*, pool::*};
use crate::capi::ops::*;
use crate::capi::ops::{contract::*, indexing::*, linalg::*, movement::*, nn::*, rng::*, signal::*, spatial::*};
use crate::capi::tensor::*;
use crate::capi::*;
use std::ffi::CStr;
use std::ptr;

const KU_F32: u32 = 13;
const KU_I32: u32 = 7;

unsafe fn read_f32(t: *const KuTensor) -> Vec<f32> {
    let n = ku_tensor_len(t);
    let mut buf = vec![0f32; n];
    let got = ku_tensor_data_f32(t, buf.as_mut_ptr(), n);
    assert_eq!(got, n as isize);
    buf
}

#[test]
fn capi_build_eval_grad_error_feeds() {
    unsafe {
        let g = ku_graph_new();
        let cpu = ku_backend_new(0); // KU_CPU
        assert!(!cpu.is_null());

        // z = relu(matmul(x, I) - 2.5); x = [[1,2],[3,4]] -> [0,0,0.5,1.5]
        let x = ku_constant_f32(g, [1., 2., 3., 4.].as_ptr(), 4, [2usize, 2].as_ptr(), 2);
        let eye = ku_constant_f32(g, [1., 0., 0., 1.].as_ptr(), 4, [2usize, 2].as_ptr(), 2);
        let m = ku_matmul(g, x, eye);
        assert_ne!(m, KU_ERR);
        let s = ku_scalar(g, m, 2.5);
        let d = ku_sub(g, m, s);
        assert_ne!(d, KU_ERR);
        let z = ku_relu(g, d);
        let t = ku_eval(g, z, cpu);
        assert!(!t.is_null());
        assert_eq!(read_f32(t), vec![0., 0., 0.5, 1.5]);
        ku_tensor_free(t);

        // grad: out = a*a ; grad(sum(out), [a]) = 2a ; a=[1,2,3] -> [2,4,6]
        let a = ku_constant_f32(g, [1., 2., 3.].as_ptr(), 3, [3usize].as_ptr(), 1);
        let sq = ku_mul(g, a, a);
        let mut grads = [0u32; 1];
        assert_eq!(ku_grad(g, sq, [a].as_ptr(), 1, grads.as_mut_ptr()), 0);
        let gt = ku_eval(g, grads[0], cpu);
        assert_eq!(read_f32(gt), vec![2., 4., 6.]);
        ku_tensor_free(gt);

        // error path: matmul [2,3] @ [2,2] -> contract mismatch -> KU_ERR + message
        let x2 = ku_constant_f32(g, [0.; 6].as_ptr(), 6, [2usize, 3].as_ptr(), 2);
        let bad = ku_matmul(g, x2, eye);
        assert_eq!(bad, KU_ERR);
        assert!(!ku_last_error().is_null());
        let msg = CStr::from_ptr(ku_last_error()).to_string_lossy().into_owned();
        assert!(msg.contains("dot_general"), "unexpected error: {msg}");

        // feeds: y = relu(input); feed [-1, 2] -> [0, 2]
        let inp = ku_input(g, [2usize].as_ptr(), 1, KU_F32);
        let y = ku_relu(g, inp);
        let feeds = ku_feeds_new();
        let ftv = ku_tensor_f32([-1., 2.].as_ptr(), 2, [2usize].as_ptr(), 1);
        ku_feeds_set(feeds, inp, ftv);
        let yt = ku_eval_with(g, y, cpu, feeds);
        assert_eq!(read_f32(yt), vec![0., 2.]);
        ku_tensor_free(yt);
        ku_tensor_free(ftv);
        ku_feeds_free(feeds);

        ku_backend_free(cpu);
        ku_graph_free(g);
    }
}

#[test]
fn capi_generic_dtype_op_plan_and_passes() {
    unsafe {
        let g = ku_graph_new();
        let cpu = ku_backend_new(0);

        // generic dtype: bake an i32 constant from raw bytes, read the bytes back.
        let src: [i32; 4] = [1, -2, 3, -4];
        let bytes = std::slice::from_raw_parts(src.as_ptr() as *const u8, 16);
        let c = ku_constant(g, KU_I32, bytes.as_ptr(), 16, [4usize].as_ptr(), 1);
        assert_ne!(c, KU_ERR);
        let ct = ku_eval(g, c, cpu);
        assert_eq!(ku_tensor_dtype(ct), KU_I32);
        assert_eq!(ku_tensor_nbytes(ct), 16);
        let mut out = vec![0u8; 16];
        ku_tensor_bytes(ct, out.as_mut_ptr());
        assert_eq!(std::slice::from_raw_parts(out.as_ptr() as *const i32, 4), &src);
        ku_tensor_free(ct);

        // sum(add(a, b), axis=0) over [4]: (1+10)+(2+20)+(3+30)+(4+40) = 110.
        let a = ku_constant_f32(g, [1., 2., 3., 4.].as_ptr(), 4, [4usize].as_ptr(), 1);
        let b = ku_constant_f32(g, [10., 20., 30., 40.].as_ptr(), 4, [4usize].as_ptr(), 1);
        let tot = ku_sum(g, ku_add(g, a, b), 0);
        let tt = ku_eval(g, tot, cpu);
        assert_eq!(read_f32(tt), vec![110.]);
        ku_tensor_free(tt);

        // passes / inspect: node_count > 0, simplify preserves the value, dump nonempty.
        assert!(ku_node_count(g, tot) > 0);
        let simp = ku_simplify(g, tot);
        let st = ku_eval(g, simp, cpu);
        assert_eq!(read_f32(st), vec![110.]);
        ku_tensor_free(st);
        assert!(ku_dump(g, tot, ptr::null_mut(), 0) > 0);

        // plan-replay: y = input * input, compiled once and run with fresh feeds.
        let inp = ku_input(g, [3usize].as_ptr(), 1, KU_F32);
        let y = ku_mul(g, inp, inp);
        let plan = ku_plan_compile(g, y);
        assert!(!plan.is_null());
        let feeds = ku_feeds_new();
        let f1 = ku_tensor_f32([1., 2., 3.].as_ptr(), 3, [3usize].as_ptr(), 1);
        ku_feeds_set(feeds, inp, f1);
        let r1 = ku_plan_run(plan, g, feeds);
        assert_eq!(read_f32(r1), vec![1., 4., 9.]);
        ku_tensor_free(r1);
        ku_tensor_free(f1);
        ku_plan_free(plan);
        ku_feeds_free(feeds);

        ku_backend_free(cpu);
        ku_graph_free(g);
    }
}

#[test]
fn capi_extended_ops_smoke() {
    unsafe {
        let g = ku_graph_new();
        let cpu = ku_backend_new(0);

        // tril / triu on [[1,2],[3,4]]
        let m = ku_constant_f32(g, [1., 2., 3., 4.].as_ptr(), 4, [2usize, 2].as_ptr(), 2);
        assert_eq!(read_f32(ku_eval(g, ku_tril(g, m, 0), cpu)), vec![1., 0., 3., 4.]);
        assert_eq!(read_f32(ku_eval(g, ku_triu(g, m, 0), cpu)), vec![1., 2., 0., 4.]);

        // tile [1,2] x2 along axis 0 -> [1,2,1,2]
        let v = ku_constant_f32(g, [1., 2.].as_ptr(), 2, [2usize].as_ptr(), 1);
        assert_eq!(read_f32(ku_eval(g, ku_tile(g, v, 0, 2), cpu)), vec![1., 2., 1., 2.]);

        // broadcast_to [3] -> [2,3]
        let r = ku_constant_f32(g, [1., 2., 3.].as_ptr(), 3, [3usize].as_ptr(), 1);
        let bc = ku_broadcast_to(g, r, [2usize, 3].as_ptr(), 2);
        assert_eq!(read_f32(ku_eval(g, bc, cpu)), vec![1., 2., 3., 1., 2., 3.]);

        // einsum "ij,jk->ik" against the identity reproduces m
        let eye = ku_constant_f32(g, [1., 0., 0., 1.].as_ptr(), 4, [2usize, 2].as_ptr(), 2);
        let eq = std::ffi::CString::new("ij,jk->ik").unwrap();
        let es = ku_einsum(g, eq.as_ptr(), [m, eye].as_ptr(), 2);
        assert_eq!(read_f32(ku_eval(g, es, cpu)), vec![1., 2., 3., 4.]);

        // celu is identity on positives
        assert_eq!(read_f32(ku_eval(g, ku_celu(g, v, 1.0), cpu)), vec![1., 2.]);

        // max_pool1d over [1,1,4], k=s=2 -> [2,4]
        let seq = ku_constant_f32(g, [1., 2., 3., 4.].as_ptr(), 4, [1usize, 1, 4].as_ptr(), 3);
        assert_eq!(read_f32(ku_eval(g, ku_max_pool1d(g, seq, 2, 2), cpu)), vec![2., 4.]);

        // scatter (set): write updates [5,7] at indices [0,2] of [0;4] -> [5,0,7,0]
        let operand = ku_constant_f32(g, [0.; 4].as_ptr(), 4, [4usize].as_ptr(), 1);
        let idx_src: [i32; 2] = [0, 2];
        let idx_bytes = std::slice::from_raw_parts(idx_src.as_ptr() as *const u8, 8);
        let idx = ku_constant(g, KU_I32, idx_bytes.as_ptr(), 8, [2usize].as_ptr(), 1);
        let upd = ku_constant_f32(g, [5., 7.].as_ptr(), 2, [2usize].as_ptr(), 1);
        let sc = ku_scatter(g, operand, idx, upd, 0, 0);
        assert_eq!(read_f32(ku_eval(g, sc, cpu)), vec![5., 0., 7., 0.]);

        // structural build checks (values covered by kurumi_core tests)
        let img = ku_constant_f32(g, [1., 2., 3., 4.].as_ptr(), 4, [1usize, 1, 2, 2].as_ptr(), 4);
        let ker = ku_constant_f32(g, [1.].as_ptr(), 1, [1usize, 1, 1, 1].as_ptr(), 4);
        assert_ne!(ku_conv2d(g, img, ker, 1, 1, 0, 0, 1, 1), KU_ERR);
        assert_ne!(ku_resize_bilinear(g, img, 4, 4), KU_ERR);
        assert_ne!(ku_sort(g, v, 0, 0), KU_ERR);
        assert_ne!(ku_argsort(g, v, 0, 1), KU_ERR);
        assert_ne!(ku_bitcast(g, m, KU_I32), KU_ERR);
        assert_ne!(ku_hann_window(g, 8), KU_ERR);
        assert_ne!(ku_randint(g, [4usize].as_ptr(), 1, 7, 0, 10), KU_ERR);

        // error path: bad scatter combiner -> KU_ERR + message
        assert_eq!(ku_scatter(g, operand, idx, upd, 0, 99), KU_ERR);
        let msg = CStr::from_ptr(ku_last_error()).to_string_lossy().into_owned();
        assert!(msg.contains("combine"), "unexpected error: {msg}");

        ku_backend_free(cpu);
        ku_graph_free(g);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn capi_metal_backend_matches_cpu() {
    unsafe {
        let metal = ku_backend_new(1); // KU_METAL
        if metal.is_null() {
            return; // no device (headless CI)
        }
        let cpu = ku_backend_new(0);
        let g = ku_graph_new();
        // matmul then gelu; Metal must match the CPU oracle
        let x = ku_constant_f32(g, [1., 2., 3., 4., 5., 6.].as_ptr(), 6, [2usize, 3].as_ptr(), 2);
        let wv: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let w = ku_constant_f32(g, wv.as_ptr(), 12, [3usize, 4].as_ptr(), 2);
        let m = ku_matmul(g, x, w);
        assert_ne!(m, KU_ERR);
        let z = ku_gelu(g, m);
        let cv = read_f32(ku_eval(g, z, cpu));
        let mv = read_f32(ku_eval(g, z, metal));
        assert_eq!(cv.len(), mv.len());
        for (c, d) in cv.iter().zip(&mv) {
            assert!((c - d).abs() < 1e-3, "metal {d} vs cpu {c}");
        }
        ku_graph_free(g);
        ku_backend_free(cpu);
        ku_backend_free(metal);
    }
}

// Boundary safety: invalid node ids and null handles must return the KU_ERR / null
// sentinels, not abort (extern "C" cannot unwind past a panic) or segfault. Every call
// here crashed the process before the catch_unwind / null-check guards were added.
#[test]
fn capi_boundary_is_panic_and_null_safe() {
    unsafe {
        // silence the panic backtraces from the intentionally-bad calls below.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let g = ku_graph_new();
        let cpu = ku_backend_new(0);
        let a = ku_constant_f32(g, [1., 2.].as_ptr(), 2, [2usize].as_ptr(), 1);

        // bad node id into a unary/binary builder -> KU_ERR (was: index-oob panic -> abort)
        assert_eq!(ku_relu(g, KU_ERR), KU_ERR);
        assert_eq!(ku_add(g, KU_ERR, a), KU_ERR);
        assert_eq!(ku_add(g, a, KU_ERR), KU_ERR);
        assert!(!ku_last_error().is_null());

        // null graph handle -> KU_ERR (was: null deref -> segfault)
        assert_eq!(ku_add(ptr::null_mut(), a, a), KU_ERR);
        assert_eq!(ku_relu(ptr::null_mut(), a), KU_ERR);

        // eval with a bad node id -> null; null handles -> null (was: abort/segfault)
        assert!(ku_eval(g, KU_ERR, cpu).is_null());
        assert!(ku_eval(ptr::null_mut(), a, cpu).is_null());
        assert!(ku_eval(g, a, ptr::null()).is_null());

        // grad / multi-output / tensor readers tolerate bad ids and null handles
        let mut grads = [0u32; 1];
        assert_eq!(ku_grad(g, KU_ERR, [a].as_ptr(), 1, grads.as_mut_ptr()), -1);
        assert_eq!(ku_grad(ptr::null_mut(), a, [a].as_ptr(), 1, grads.as_mut_ptr()), -1);
        let mut out2 = [0u32; 2];
        assert_eq!(ku_qr(g, KU_ERR, out2.as_mut_ptr()), KU_ERR);
        assert_eq!(ku_qr(g, a, ptr::null_mut()), KU_ERR);
        assert_eq!(ku_tensor_rank(ptr::null()), 0);
        assert_eq!(ku_tensor_data_f32(ptr::null(), ptr::null_mut(), 0), -1);

        ku_backend_free(cpu);
        ku_graph_free(g);
        std::panic::set_hook(prev);
    }
}

// Header-drift guard: every `ku_*` the cdylib exports must have a prototype in
// kurumi.h, or a cffi/cbindgen consumer (hodu-py) silently can't call it. The export
// set is scraped from the ABI source (`extern "C" fn ku_*` + the `ku_* =>` macro
// tables); each must appear as `ku_<name>(` in the header. Fails with the missing set.
#[test]
fn capi_header_declares_every_export() {
    // every file that defines exported symbols (the ops families, the handle/eval/tensor
    // surfaces, and the parent module for ku_last_error).
    const SRCS: &[&str] = &[
        include_str!("ops.rs"),
        include_str!("ops/contract.rs"),
        include_str!("ops/distance.rs"),
        include_str!("ops/indexing.rs"),
        include_str!("ops/linalg.rs"),
        include_str!("ops/movement.rs"),
        include_str!("ops/nn.rs"),
        include_str!("ops/nn/conv.rs"),
        include_str!("ops/nn/pool.rs"),
        include_str!("ops/rng.rs"),
        include_str!("ops/signal.rs"),
        include_str!("ops/spatial.rs"),
        include_str!("ops/stats.rs"),
        include_str!("graph.rs"),
        include_str!("eval.rs"),
        include_str!("tensor.rs"),
        include_str!("../capi.rs"),
    ];
    const HEADER: &str = include_str!("../../include/kurumi.h");

    fn ident_prefix(s: &str) -> String {
        s.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect()
    }
    fn trailing_ident(s: &str) -> String {
        let rev: String = s.trim_end().chars().rev().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
        rev.chars().rev().collect()
    }

    let mut exports: Vec<String> = Vec::new();
    for src in SRCS {
        // explicit wrappers: `... extern "C" fn ku_name(...`
        for seg in src.split("extern \"C\" fn ").skip(1) {
            let name = ident_prefix(seg);
            if name.starts_with("ku_") {
                exports.push(name);
            }
        }
        // macro-table rows: `ku_name => method`
        let parts: Vec<&str> = src.split("=>").collect();
        for left in &parts[..parts.len().saturating_sub(1)] {
            let name = trailing_ident(left);
            if name.starts_with("ku_") {
                exports.push(name);
            }
        }
    }
    exports.sort();
    exports.dedup();
    assert!(exports.len() > 250, "scraper found only {} exports; parsing likely broke", exports.len());

    let missing: Vec<&String> = exports.iter().filter(|s| !HEADER.contains(&format!("{s}("))).collect();
    assert!(missing.is_empty(), "kurumi.h is missing declarations for {} exports: {missing:?}", missing.len());
}
