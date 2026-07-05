// C ABI builder ops. Per-op wrappers over the Graph methods, split into per-family
// submodules (`ops/<family>.rs`). This file keeps the shared helpers, the "regular"
// binary/unary/reduce/clamp macro families (whose tables mix families, so they stay
// whole here), and re-exports the submodules so every `ku_*` lives under `capi::ops`.
// Each wrapper returns a node id, KU_ERR on error.

use crate::capi::{KU_ERR, KuGraph, build, set_err};
use kurumi_core::{NodeId, ScatterOp};
use std::ffi::{CStr, c_char};

// One submodule per op family (foo.rs + foo/*.rs, no mod.rs). The wrappers export
// C symbols directly (`#[no_mangle]`), so no re-export is needed for the ABI; the
// modules are crate-visible only for the in-crate ABI test.
pub(crate) mod contract;
pub(crate) mod distance;
pub(crate) mod indexing;
pub(crate) mod linalg;
pub(crate) mod movement;
pub(crate) mod nn;
pub(crate) mod rng;
pub(crate) mod signal;
pub(crate) mod spatial;
pub(crate) mod stats;

// map a u32 to the scatter combiner (Set=0, Add=1, Max=2, Min=3).
fn scatter_op(c: u32) -> Option<ScatterOp> {
    match c {
        0 => Some(ScatterOp::Set),
        1 => Some(ScatterOp::Add),
        2 => Some(ScatterOp::Max),
        3 => Some(ScatterOp::Min),
        _ => None,
    }
}
// borrow a C string as &str (None if null / not utf-8).
unsafe fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_str().ok()).flatten()
}
// multi-output linalg: write the factor node ids to `out`, return 0 / KU_ERR.
fn write2(out: *mut u32, r: Result<(NodeId, NodeId), kurumi_core::Error>) -> u32 {
    match r {
        Ok((a, b)) => unsafe {
            *out = a.0;
            *out.add(1) = b.0;
            0
        },
        Err(e) => {
            set_err(format!("{e:?}"));
            KU_ERR
        }
    }
}

// binary elementwise / two-operand ops (all shape/dtype checked -> Result)
macro_rules! bin {
    ($($c:ident => $m:ident),* $(,)?) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, a: u32, b: u32) -> u32 {
            build(g, |gr| gr.$m(NodeId(a), NodeId(b)))
        }
    )* };
}
bin! {
    ku_add => add, ku_sub => sub, ku_mul => mul, ku_div => div, ku_max => max, ku_min => min,
    ku_pow => pow, ku_atan2 => atan2, ku_idiv => idiv, ku_rem => rem, ku_and => and, ku_or => or,
    ku_xor => xor, ku_shl => shl, ku_shr => shr, ku_lt => cmp_lt, ku_eq => cmp_eq, ku_le => le,
    ku_gt => gt, ku_ge => ge, ku_ne => ne, ku_logaddexp => logaddexp, ku_xlogy => xlogy,
    ku_prelu => prelu, ku_complex => complex, ku_solve => solve, ku_gather_nd => gather_nd,
    ku_broadcast_like => broadcast_like, ku_beta => beta, ku_mse_loss => mse_loss,
    ku_l1_loss => l1_loss, ku_hinge_loss => hinge_loss, ku_kl_div => kl_div, ku_bce_loss => bce_loss,
    ku_bce_with_logits => bce_with_logits, ku_lstsq => lstsq,
}

// unary ops that cannot fail (single primitive or a total decomposition)
macro_rules! un {
    ($($c:ident => $m:ident),* $(,)?) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32) -> u32 {
            build(g, |gr| Ok(gr.$m(NodeId(x))))
        }
    )* };
}
un! {
    ku_neg => neg, ku_recip => recip, ku_sqrt => sqrt, ku_square => square, ku_abs => abs,
    ku_sign => sign, ku_floor => floor, ku_ceil => ceil, ku_round => round, ku_exp => exp,
    ku_exp2 => exp2, ku_exp10 => exp10, ku_ln => ln, ku_log2 => log2, ku_log10 => log10,
    ku_sin => sin, ku_cos => cos, ku_tan => tan, ku_sinh => sinh, ku_cosh => cosh,
    ku_asin => asin, ku_acos => acos, ku_atan => atan, ku_asinh => asinh, ku_acosh => acosh,
    ku_atanh => atanh, ku_relu => relu, ku_gelu => gelu, ku_gelu_erf => gelu_erf, ku_silu => silu,
    ku_sigmoid => sigmoid, ku_tanh => tanh, ku_softplus => softplus, ku_softsign => softsign,
    ku_mish => mish, ku_selu => selu, ku_hardsigmoid => hardsigmoid, ku_hardswish => hardswish,
    ku_erf => erf, ku_erfc => erfc, ku_erfinv => erfinv, ku_gamma => gamma, ku_lgamma => lgamma,
    ku_digamma => digamma, ku_i0 => i0, ku_logical_not => logical_not, ku_ones_like => ones_like,
    ku_zeros_like => zeros_like,
}

// unary ops that validate / decompose to a fallible chain -> Result
macro_rules! un_r {
    ($($c:ident => $m:ident),* $(,)?) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32) -> u32 {
            build(g, |gr| gr.$m(NodeId(x)))
        }
    )* };
}
un_r! {
    ku_expm1 => expm1, ku_log1p => log1p, ku_logit => logit, ku_log_sigmoid => log_sigmoid,
    ku_sinc => sinc, ku_isnan => isnan, ku_isinf => isinf, ku_isfinite => isfinite,
    ku_bitwise_not => bitwise_not, ku_flatten => flatten, ku_t => t, ku_trace => trace,
    ku_diagonal => diagonal, ku_diag_embed => diag_embed, ku_sum_all => sum_all,
    ku_prod_all => prod_all, ku_mean_all => mean_all, ku_cov => cov, ku_corrcoef => corrcoef,
    ku_real => real, ku_imag => imag, ku_conj => conj, ku_cabs => cabs, ku_angle => angle,
    ku_det => det, ku_cholesky => cholesky, ku_inv => inv, ku_pinv => pinv, ku_eigvals => eigvals,
    ku_matrix_exp => matrix_exp, ku_fft2 => fft2, ku_ifft2 => ifft2, ku_rope => rope,
}

// reductions / cumulative ops over one axis (keepdim = false)
macro_rules! red {
    ($($c:ident => $m:ident),* $(,)?) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32, axis: usize) -> u32 {
            build(g, |gr| gr.$m(NodeId(x), axis))
        }
    )* };
}
red! {
    ku_sum => sum, ku_prod => prod, ku_mean => mean, ku_reduce_max => reduce_max,
    ku_reduce_min => reduce_min, ku_all => all, ku_any => any, ku_argmax => argmax,
    ku_argmin => argmin, ku_median => median, ku_mode => mode, ku_std => std, ku_var => var,
    ku_softmax => softmax, ku_log_softmax => log_softmax, ku_logsumexp => logsumexp,
    ku_logsum => logsum, ku_l1_norm => l1_norm, ku_l2_norm => l2_norm, ku_cumsum => cumsum,
    ku_cumprod => cumprod, ku_cummax => cummax, ku_cummin => cummin, ku_squeeze => squeeze,
    ku_unsqueeze => unsqueeze, ku_fft => fft, ku_ifft => ifft, ku_rfft => rfft, ku_hilbert => hilbert,
}

// unary op taking one f32 attr (clamp bounds)
macro_rules! un_f {
    ($($c:ident => $m:ident),* $(,)?) => { $(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $c(g: *mut KuGraph, x: u32, v: f32) -> u32 {
            build(g, |gr| gr.$m(NodeId(x), v))
        }
    )* };
}
un_f! { ku_clamp_min => clamp_min, ku_clamp_max => clamp_max }
