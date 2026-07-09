/* kurumi C ABI: graph-builder surface.
 *
 * Build a graph of nodes (each op returns a uint32_t node id; KU_ERR on error, then
 * ku_last_error() for the message), create a backend once, and eval/grad. Handles are
 * opaque; free each with its matching ku_*_free. Not thread-safe per handle.
 *
 * Op coverage: every builder op has a named wrapper `ku_<name>` exported from the
 * cdylib. This header declares the full set explicitly (one prototype per exported
 * symbol), so other-language bindings can pull it straight from here (cffi / cbindgen);
 * a test in src/capi/tests.rs fails the build if an export ever lacks a declaration.
 *
 * Tensor exchange: raw f32 copy (ku_tensor_f32 / ku_tensor_data_f32) or generic-dtype
 * raw bytes (ku_tensor_new / ku_tensor_bytes). No DLPack zero-copy path yet. */
#ifndef KURUMI_H
#define KURUMI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error sentinel returned by builder ops (valid node ids are < KU_ERR). */
#define KU_ERR 0xFFFFFFFFu

/* dtype tag (index matches kurumi's DType declaration order). */
/* clang-format off */
typedef enum {
  KU_BOOL = 0, KU_U8, KU_U16, KU_U32, KU_U64,
  KU_I8, KU_I16, KU_I32, KU_I64,
  KU_F8E4M3, KU_F8E5M2, KU_F16, KU_BF16, KU_F32, KU_F64,
  KU_C64, KU_C128
} KuDType;
/* clang-format on */

/* backend kind for ku_backend_new. */
typedef enum { KU_CPU = 0, KU_METAL = 1 } KuBackendKind;

typedef struct KuGraph KuGraph;
typedef struct KuTensor KuTensor;
typedef struct KuFeeds KuFeeds;
typedef struct KuBackend KuBackend;
typedef struct KuPlan KuPlan;
typedef struct KuRunnable KuRunnable;

typedef uint32_t KuNode;

/* Last error on this thread (NUL-terminated), or NULL. Valid until the next failing
 * call on the same thread. */
const char *ku_last_error(void);

/* graph lifecycle */
KuGraph *ku_graph_new(void);
void ku_graph_free(KuGraph *g);
KuNode ku_input(KuGraph *g, const size_t *shape, size_t rank, uint32_t dtype);
KuNode ku_constant_f32(KuGraph *g, const float *data, size_t len, const size_t *shape, size_t rank);
/* Constant of any dtype from nbytes of little-endian element data (row-major shape). */
KuNode ku_constant(KuGraph *g, uint32_t dtype, const uint8_t *data, size_t nbytes, const size_t *shape, size_t rank);
KuNode ku_scalar(KuGraph *g, KuNode like, float v);

/* builder ops (return a node id; KU_ERR on error) */
KuNode ku_add(KuGraph *g, KuNode a, KuNode b);
KuNode ku_sub(KuGraph *g, KuNode a, KuNode b);
KuNode ku_mul(KuGraph *g, KuNode a, KuNode b);
KuNode ku_div(KuGraph *g, KuNode a, KuNode b);
KuNode ku_max(KuGraph *g, KuNode a, KuNode b);
KuNode ku_min(KuGraph *g, KuNode a, KuNode b);
KuNode ku_pow(KuGraph *g, KuNode a, KuNode b);
/* arithmetic */
KuNode ku_atan2(KuGraph *g, KuNode a, KuNode b);
KuNode ku_idiv(KuGraph *g, KuNode a, KuNode b);
KuNode ku_rem(KuGraph *g, KuNode a, KuNode b);
/* bitwise (integer) */
KuNode ku_and(KuGraph *g, KuNode a, KuNode b);
KuNode ku_or(KuGraph *g, KuNode a, KuNode b);
KuNode ku_xor(KuGraph *g, KuNode a, KuNode b);
KuNode ku_shl(KuGraph *g, KuNode a, KuNode b);
KuNode ku_shr(KuGraph *g, KuNode a, KuNode b);
/* comparison (-> BOOL) */
KuNode ku_lt(KuGraph *g, KuNode a, KuNode b);
KuNode ku_le(KuGraph *g, KuNode a, KuNode b);
KuNode ku_gt(KuGraph *g, KuNode a, KuNode b);
KuNode ku_ge(KuGraph *g, KuNode a, KuNode b);
KuNode ku_eq(KuGraph *g, KuNode a, KuNode b);
KuNode ku_ne(KuGraph *g, KuNode a, KuNode b);
/* misc two-operand */
KuNode ku_logaddexp(KuGraph *g, KuNode a, KuNode b);
KuNode ku_xlogy(KuGraph *g, KuNode a, KuNode b);
KuNode ku_prelu(KuGraph *g, KuNode a, KuNode b);
KuNode ku_complex(KuGraph *g, KuNode a, KuNode b);
KuNode ku_beta(KuGraph *g, KuNode a, KuNode b);
KuNode ku_broadcast_like(KuGraph *g, KuNode a, KuNode b);
/* linalg solves: a=A, b=rhs */
KuNode ku_solve(KuGraph *g, KuNode a, KuNode b);
KuNode ku_lstsq(KuGraph *g, KuNode a, KuNode b);
/* nd gather: a=operand, b=indices */
KuNode ku_gather_nd(KuGraph *g, KuNode a, KuNode b);
/* losses: a=input/logits, b=target */
KuNode ku_mse_loss(KuGraph *g, KuNode a, KuNode b);
KuNode ku_l1_loss(KuGraph *g, KuNode a, KuNode b);
KuNode ku_hinge_loss(KuGraph *g, KuNode a, KuNode b);
KuNode ku_kl_div(KuGraph *g, KuNode a, KuNode b);
KuNode ku_bce_loss(KuGraph *g, KuNode a, KuNode b);
KuNode ku_bce_with_logits(KuGraph *g, KuNode a, KuNode b);
KuNode ku_matmul(KuGraph *g, KuNode a, KuNode b);
/* General contraction: contract lc/rc axes, batch lb/rb axes (StableHLO dot_general). */
KuNode ku_dot_general(KuGraph *g, KuNode a, KuNode b, const size_t *lc, size_t nlc, const size_t *rc, size_t nrc,
                      const size_t *lb, size_t nlb, const size_t *rb, size_t nrb);
/* Weight-only quantized matmul: act[M,K] x dequant(qweight)[N,K]^T -> [M,N]. qweight is a
 * KU_U8-packed constant, scales/mins are KU_F16 [N, K/group_size]; mins = KU_ERR for
 * symmetric. Build the operands with ku_quantize + ku_constant. */
KuNode ku_quant_matmul(KuGraph *g, KuNode act, KuNode qweight, KuNode scales, KuNode mins, uint8_t bits,
                       size_t group_size);
/* Quantize an f32 weight [rows, cols] for ku_quant_matmul. Writes out_packed
 * (rows*cols*bits/8 bytes), out_scales (rows*cols/group_size f16 bits), and out_mins
 * (same count, or NULL when symmetric != 0). bits is 2, 4, or 8. */
void ku_quantize(const float *w, size_t rows, size_t cols, uint8_t bits, size_t group_size, uint32_t symmetric,
                 uint8_t *out_packed, uint16_t *out_scales, uint16_t *out_mins);

KuNode ku_neg(KuGraph *g, KuNode x);
KuNode ku_recip(KuGraph *g, KuNode x);
KuNode ku_sqrt(KuGraph *g, KuNode x);
KuNode ku_exp(KuGraph *g, KuNode x);
KuNode ku_relu(KuGraph *g, KuNode x);
KuNode ku_gelu(KuGraph *g, KuNode x);
KuNode ku_sigmoid(KuGraph *g, KuNode x);
KuNode ku_tanh(KuGraph *g, KuNode x);
KuNode ku_silu(KuGraph *g, KuNode x);
/* arithmetic / rounding */
KuNode ku_square(KuGraph *g, KuNode x);
KuNode ku_abs(KuGraph *g, KuNode x);
KuNode ku_sign(KuGraph *g, KuNode x);
KuNode ku_floor(KuGraph *g, KuNode x);
KuNode ku_ceil(KuGraph *g, KuNode x);
KuNode ku_round(KuGraph *g, KuNode x);
/* exp / log */
KuNode ku_exp2(KuGraph *g, KuNode x);
KuNode ku_exp10(KuGraph *g, KuNode x);
KuNode ku_ln(KuGraph *g, KuNode x);
KuNode ku_log2(KuGraph *g, KuNode x);
KuNode ku_log10(KuGraph *g, KuNode x);
KuNode ku_expm1(KuGraph *g, KuNode x);
KuNode ku_log1p(KuGraph *g, KuNode x);
/* trigonometric / hyperbolic */
KuNode ku_sin(KuGraph *g, KuNode x);
KuNode ku_cos(KuGraph *g, KuNode x);
KuNode ku_tan(KuGraph *g, KuNode x);
KuNode ku_sinh(KuGraph *g, KuNode x);
KuNode ku_cosh(KuGraph *g, KuNode x);
KuNode ku_asin(KuGraph *g, KuNode x);
KuNode ku_acos(KuGraph *g, KuNode x);
KuNode ku_atan(KuGraph *g, KuNode x);
KuNode ku_asinh(KuGraph *g, KuNode x);
KuNode ku_acosh(KuGraph *g, KuNode x);
KuNode ku_atanh(KuGraph *g, KuNode x);
KuNode ku_sinc(KuGraph *g, KuNode x);
/* activations */
KuNode ku_gelu_erf(KuGraph *g, KuNode x);
KuNode ku_softplus(KuGraph *g, KuNode x);
KuNode ku_softsign(KuGraph *g, KuNode x);
KuNode ku_mish(KuGraph *g, KuNode x);
KuNode ku_selu(KuGraph *g, KuNode x);
KuNode ku_hardsigmoid(KuGraph *g, KuNode x);
KuNode ku_hardswish(KuGraph *g, KuNode x);
KuNode ku_logit(KuGraph *g, KuNode x);
KuNode ku_log_sigmoid(KuGraph *g, KuNode x);
/* special functions */
KuNode ku_erf(KuGraph *g, KuNode x);
KuNode ku_erfc(KuGraph *g, KuNode x);
KuNode ku_erfinv(KuGraph *g, KuNode x);
KuNode ku_gamma(KuGraph *g, KuNode x);
KuNode ku_lgamma(KuGraph *g, KuNode x);
KuNode ku_digamma(KuGraph *g, KuNode x);
KuNode ku_i0(KuGraph *g, KuNode x);
/* predicates / logical (-> BOOL) */
KuNode ku_logical_not(KuGraph *g, KuNode x);
KuNode ku_bitwise_not(KuGraph *g, KuNode x);
KuNode ku_isnan(KuGraph *g, KuNode x);
KuNode ku_isinf(KuGraph *g, KuNode x);
KuNode ku_isfinite(KuGraph *g, KuNode x);
/* same-shape fills */
KuNode ku_ones_like(KuGraph *g, KuNode x);
KuNode ku_zeros_like(KuGraph *g, KuNode x);
/* complex parts */
KuNode ku_real(KuGraph *g, KuNode x);
KuNode ku_imag(KuGraph *g, KuNode x);
KuNode ku_conj(KuGraph *g, KuNode x);
KuNode ku_cabs(KuGraph *g, KuNode x);
KuNode ku_angle(KuGraph *g, KuNode x);
/* linalg (whole-tensor) */
KuNode ku_t(KuGraph *g, KuNode x);
KuNode ku_flatten(KuGraph *g, KuNode x);
KuNode ku_trace(KuGraph *g, KuNode x);
KuNode ku_diagonal(KuGraph *g, KuNode x);
KuNode ku_diag_embed(KuGraph *g, KuNode x);
KuNode ku_det(KuGraph *g, KuNode x);
KuNode ku_cholesky(KuGraph *g, KuNode x);
KuNode ku_inv(KuGraph *g, KuNode x);
KuNode ku_pinv(KuGraph *g, KuNode x);
KuNode ku_eigvals(KuGraph *g, KuNode x);
KuNode ku_matrix_exp(KuGraph *g, KuNode x);
/* reduce-all (over every axis) */
KuNode ku_sum_all(KuGraph *g, KuNode x);
KuNode ku_prod_all(KuGraph *g, KuNode x);
KuNode ku_mean_all(KuGraph *g, KuNode x);
KuNode ku_cov(KuGraph *g, KuNode x);
KuNode ku_corrcoef(KuGraph *g, KuNode x);
/* whole-tensor fft + rope */
KuNode ku_fft2(KuGraph *g, KuNode x);
KuNode ku_ifft2(KuGraph *g, KuNode x);
KuNode ku_rope(KuGraph *g, KuNode x);

KuNode ku_sum(KuGraph *g, KuNode x, size_t axis);
KuNode ku_mean(KuGraph *g, KuNode x, size_t axis);
KuNode ku_reduce_max(KuGraph *g, KuNode x, size_t axis);
KuNode ku_softmax(KuGraph *g, KuNode x, size_t axis);
/* reductions over one axis (keepdim = false) */
KuNode ku_prod(KuGraph *g, KuNode x, size_t axis);
KuNode ku_reduce_min(KuGraph *g, KuNode x, size_t axis);
KuNode ku_all(KuGraph *g, KuNode x, size_t axis);
KuNode ku_any(KuGraph *g, KuNode x, size_t axis);
KuNode ku_argmax(KuGraph *g, KuNode x, size_t axis);
KuNode ku_argmin(KuGraph *g, KuNode x, size_t axis);
KuNode ku_median(KuGraph *g, KuNode x, size_t axis);
KuNode ku_mode(KuGraph *g, KuNode x, size_t axis);
KuNode ku_std(KuGraph *g, KuNode x, size_t axis);
KuNode ku_var(KuGraph *g, KuNode x, size_t axis);
/* softmax family / norms over one axis */
KuNode ku_log_softmax(KuGraph *g, KuNode x, size_t axis);
KuNode ku_logsumexp(KuGraph *g, KuNode x, size_t axis);
KuNode ku_logsum(KuGraph *g, KuNode x, size_t axis);
KuNode ku_l1_norm(KuGraph *g, KuNode x, size_t axis);
KuNode ku_l2_norm(KuGraph *g, KuNode x, size_t axis);
/* cumulative scans over one axis */
KuNode ku_cumsum(KuGraph *g, KuNode x, size_t axis);
KuNode ku_cumprod(KuGraph *g, KuNode x, size_t axis);
KuNode ku_cummax(KuGraph *g, KuNode x, size_t axis);
KuNode ku_cummin(KuGraph *g, KuNode x, size_t axis);
/* squeeze / unsqueeze one axis */
KuNode ku_squeeze(KuGraph *g, KuNode x, size_t axis);
KuNode ku_unsqueeze(KuGraph *g, KuNode x, size_t axis);
/* 1-D fft over one axis */
KuNode ku_fft(KuGraph *g, KuNode x, size_t axis);
KuNode ku_ifft(KuGraph *g, KuNode x, size_t axis);
KuNode ku_rfft(KuGraph *g, KuNode x, size_t axis);
KuNode ku_hilbert(KuGraph *g, KuNode x, size_t axis);

/* movement (shapes given as row-major dim/axis lists) */
KuNode ku_reshape(KuGraph *g, KuNode x, const size_t *shape, size_t rank);
KuNode ku_expand(KuGraph *g, KuNode x, const size_t *shape, size_t rank);
KuNode ku_permute(KuGraph *g, KuNode x, const size_t *perm, size_t rank);
KuNode ku_transpose(KuGraph *g, KuNode x, size_t i, size_t j);
KuNode ku_flip(KuGraph *g, KuNode x, const size_t *axes, size_t n);
KuNode ku_slice(KuGraph *g, KuNode x, const size_t *ranges, size_t rank); /* 2*rank (start,end) */
KuNode ku_pad(KuGraph *g, KuNode x, const size_t *pads, size_t rank);     /* 2*rank (lo,hi), 0 */
/* non-zero padding: mode 0=reflect, 1=replicate, 2=circular */
KuNode ku_pad_mode(KuGraph *g, KuNode x, const size_t *pads, size_t rank, uint32_t mode); /* 2*rank (lo,hi) */
KuNode ku_concat(KuGraph *g, const KuNode *parts, size_t n, size_t axis);
KuNode ku_stack(KuGraph *g, const KuNode *parts, size_t n, size_t axis);
/* split x into n pieces of `sizes` along axis; piece ids -> out[0..n]. 0 ok / KU_ERR. */
KuNode ku_split(KuGraph *g, KuNode x, const size_t *sizes, size_t n, size_t axis, KuNode *out);

/* indexing */
KuNode ku_where(KuGraph *g, KuNode cond, KuNode a, KuNode b); /* cond ? a : b */
KuNode ku_gather(KuGraph *g, KuNode x, KuNode idx, size_t axis);
KuNode ku_gather_along(KuGraph *g, KuNode x, KuNode idx, size_t axis);
KuNode ku_take_along_dim(KuGraph *g, KuNode x, KuNode idx, size_t axis);
KuNode ku_onehot(KuGraph *g, KuNode idx, size_t num_classes);

/* activations with an attr / nn (norm, pool) */
KuNode ku_leaky_relu(KuGraph *g, KuNode x, float slope);
KuNode ku_elu(KuGraph *g, KuNode x, float alpha);
KuNode ku_clamp(KuGraph *g, KuNode x, float lo, float hi);
KuNode ku_clamp_min(KuGraph *g, KuNode x, float v);
KuNode ku_clamp_max(KuGraph *g, KuNode x, float v);
KuNode ku_layernorm(KuGraph *g, KuNode x, size_t axis, float eps);
KuNode ku_rmsnorm(KuGraph *g, KuNode x, size_t axis, float eps);
KuNode ku_group_norm(KuGraph *g, KuNode x, size_t groups, float eps);
KuNode ku_cross_entropy(KuGraph *g, KuNode logits, KuNode targets, size_t axis);
KuNode ku_sdpa(KuGraph *g, KuNode q, KuNode k, KuNode v, uint32_t causal);

/* generators / cast / rng (seed-based, reproducible) */
KuNode ku_iota(KuGraph *g, const size_t *shape, size_t rank, size_t axis, uint32_t dtype);
KuNode ku_cast(KuGraph *g, KuNode x, uint32_t dtype);
KuNode ku_rand_uniform(KuGraph *g, const size_t *shape, size_t rank, uint64_t seed);
KuNode ku_randn(KuGraph *g, const size_t *shape, size_t rank, uint64_t seed);
KuNode ku_dropout(KuGraph *g, KuNode x, float p, uint64_t seed);

/* multi-output factorizations: factor ids -> out[]; return 0 ok / KU_ERR.
 * slogdet->[sign,logabsdet], eigh->[vals,vecs], qr->[Q,R], svd->[U,S,V], topk->[vals,idx]. */
KuNode ku_slogdet(KuGraph *g, KuNode x, KuNode *out);
KuNode ku_eigh(KuGraph *g, KuNode x, KuNode *out);
KuNode ku_qr(KuGraph *g, KuNode x, KuNode *out);
KuNode ku_svd(KuGraph *g, KuNode x, KuNode *out);
KuNode ku_topk(KuGraph *g, KuNode x, size_t k, size_t axis, uint32_t largest, KuNode *out);

/* sort / scatter / masking / dynamic-shape indexing. sort/argsort take a descending flag;
 * masked_select/compress/nonzero/unique take an upper-bound k (the static output length).
 * scatter combine: 0=set, 1=add, 2=max, 3=min. */
KuNode ku_sort(KuGraph *g, KuNode x, size_t axis, uint32_t descending);
KuNode ku_argsort(KuGraph *g, KuNode x, size_t axis, uint32_t descending);
KuNode ku_scatter(KuGraph *g, KuNode operand, KuNode indices, KuNode updates, size_t axis, uint32_t combine);
KuNode ku_scatter_along(KuGraph *g, KuNode operand, KuNode indices, KuNode updates, size_t axis, uint32_t combine);
KuNode ku_scatter_nd(KuGraph *g, KuNode x, KuNode idx, KuNode updates, uint32_t combine);
KuNode ku_masked_fill(KuGraph *g, KuNode x, KuNode mask, float value);
KuNode ku_masked_select(KuGraph *g, KuNode x, KuNode mask, size_t k);
KuNode ku_compress(KuGraph *g, KuNode mask, KuNode x, size_t k);
KuNode ku_nonzero(KuGraph *g, KuNode x, size_t k);
KuNode ku_unique(KuGraph *g, KuNode x, size_t k);

/* movement / shape */
KuNode ku_tile(KuGraph *g, KuNode x, size_t axis, size_t n);
KuNode ku_repeat_interleave(KuGraph *g, KuNode x, size_t axis, size_t n);
KuNode ku_roll(KuGraph *g, KuNode x, size_t shift, size_t axis);
KuNode ku_broadcast_to(KuGraph *g, KuNode x, const size_t *shape, size_t rank);
KuNode ku_slice_step(KuGraph *g, KuNode x, const size_t *ranges, size_t rank); /* 3*rank (start,end,step) */
KuNode ku_tril(KuGraph *g, KuNode x, int64_t diagonal);
KuNode ku_triu(KuGraph *g, KuNode x, int64_t diagonal);
KuNode ku_detach(KuGraph *g, KuNode x);
KuNode ku_bitcast(KuGraph *g, KuNode x, uint32_t dtype);
KuNode ku_einsum(KuGraph *g, const char *equation, const KuNode *operands, size_t n);

/* convolution (stride/padding/dilation per spatial dim; transpose adds output_padding) */
KuNode ku_conv1d(KuGraph *g, KuNode input, KuNode weight, size_t stride, size_t padding, size_t dilation);
KuNode ku_conv2d(KuGraph *g, KuNode input, KuNode weight, size_t sh, size_t sw, size_t ph, size_t pw, size_t dh,
                 size_t dw);
KuNode ku_conv3d(KuGraph *g, KuNode input, KuNode weight, size_t sd, size_t sh, size_t sw, size_t pd, size_t ph,
                 size_t pw, size_t dd, size_t dh, size_t dw);
KuNode ku_conv_transpose1d(KuGraph *g, KuNode input, KuNode weight, size_t stride, size_t padding,
                           size_t output_padding, size_t dilation);
KuNode ku_conv_transpose2d(KuGraph *g, KuNode input, KuNode weight, size_t sh, size_t sw, size_t ph, size_t pw,
                           size_t oph, size_t opw, size_t dh, size_t dw);
KuNode ku_conv_transpose3d(KuGraph *g, KuNode input, KuNode weight, size_t sd, size_t sh, size_t sw, size_t pd,
                           size_t ph, size_t pw, size_t opd, size_t oph, size_t opw, size_t dd, size_t dh, size_t dw);

/* pooling (no padding); k/s per spatial dim. reduce_window mode = "max"|"sum"|"avg". */
KuNode ku_max_pool1d(KuGraph *g, KuNode x, size_t k, size_t s);
KuNode ku_avg_pool1d(KuGraph *g, KuNode x, size_t k, size_t s);
KuNode ku_min_pool1d(KuGraph *g, KuNode x, size_t k, size_t s);
KuNode ku_sum_pool1d(KuGraph *g, KuNode x, size_t k, size_t s);
KuNode ku_max_pool2d(KuGraph *g, KuNode x, size_t kh, size_t kw, size_t sh, size_t sw);
KuNode ku_avg_pool2d(KuGraph *g, KuNode x, size_t kh, size_t kw, size_t sh, size_t sw);
KuNode ku_min_pool2d(KuGraph *g, KuNode x, size_t kh, size_t kw, size_t sh, size_t sw);
KuNode ku_sum_pool2d(KuGraph *g, KuNode x, size_t kh, size_t kw, size_t sh, size_t sw);
KuNode ku_max_pool3d(KuGraph *g, KuNode x, size_t kd, size_t kh, size_t kw, size_t sd, size_t sh, size_t sw);
KuNode ku_avg_pool3d(KuGraph *g, KuNode x, size_t kd, size_t kh, size_t kw, size_t sd, size_t sh, size_t sw);
KuNode ku_min_pool3d(KuGraph *g, KuNode x, size_t kd, size_t kh, size_t kw, size_t sd, size_t sh, size_t sw);
KuNode ku_sum_pool3d(KuGraph *g, KuNode x, size_t kd, size_t kh, size_t kw, size_t sd, size_t sh, size_t sw);
KuNode ku_reduce_window(KuGraph *g, KuNode x, const size_t *window, size_t nw, const size_t *stride, size_t ns,
                        const size_t *dilation, size_t nd, const char *mode);

/* fft / signal (pass window = KU_ERR for none) */
KuNode ku_fftn(KuGraph *g, KuNode x, const size_t *axes, size_t n);
KuNode ku_ifftn(KuGraph *g, KuNode x, const size_t *axes, size_t n);
KuNode ku_irfft(KuGraph *g, KuNode x, size_t axis, size_t n);
KuNode ku_fft_conv(KuGraph *g, KuNode a, KuNode b, size_t axis);
KuNode ku_stft(KuGraph *g, KuNode x, size_t n_fft, size_t hop, KuNode window);
KuNode ku_istft(KuGraph *g, KuNode frames, size_t hop, KuNode window);
KuNode ku_hann_window(KuGraph *g, size_t n);
KuNode ku_hamming_window(KuGraph *g, size_t n);
KuNode ku_blackman_window(KuGraph *g, size_t n);
KuNode ku_bartlett_window(KuGraph *g, size_t n);

/* rng (seed-based; rand_uniform_keyed keys off a runtime scalar-int node) */
KuNode ku_randint(KuGraph *g, const size_t *shape, size_t rank, uint64_t seed, int64_t lo, int64_t hi);
KuNode ku_rand_range(KuGraph *g, const size_t *shape, size_t rank, uint64_t seed, float lo, float hi);
KuNode ku_bernoulli(KuGraph *g, const size_t *shape, size_t rank, uint64_t seed, float p);
KuNode ku_rand_uniform_keyed(KuGraph *g, const size_t *shape, size_t rank, KuNode seed);

/* extra norms / losses / stats */
KuNode ku_instance_norm(KuGraph *g, KuNode x, float eps);
KuNode ku_lrn(KuGraph *g, KuNode x, size_t size, float alpha, float beta, float k);
KuNode ku_norm_p(KuGraph *g, KuNode x, float p, size_t axis);
KuNode ku_nll_loss(KuGraph *g, KuNode log_probs, KuNode target, size_t axis);
KuNode ku_huber_loss(KuGraph *g, KuNode pred, KuNode target, float delta);
KuNode ku_celu(KuGraph *g, KuNode x, float alpha);
KuNode ku_quantile(KuGraph *g, KuNode x, size_t axis, float q);
KuNode ku_std_correction(KuGraph *g, KuNode x, size_t axis, size_t correction);
KuNode ku_var_correction(KuGraph *g, KuNode x, size_t axis, size_t correction);
KuNode ku_cdist(KuGraph *g, KuNode a, KuNode b, float p);
KuNode ku_pdist(KuGraph *g, KuNode a, float p);
KuNode ku_cosine_similarity(KuGraph *g, KuNode a, KuNode b, size_t axis);

/* spatial resize */
KuNode ku_resize(KuGraph *g, KuNode x, const size_t *axes, size_t na, const size_t *sizes, size_t ns,
                 const char *interp, const char *coord);
KuNode ku_resize_bilinear(KuGraph *g, KuNode x, size_t out_h, size_t out_w);
KuNode ku_resize_bicubic(KuGraph *g, KuNode x, size_t out_h, size_t out_w);
KuNode ku_upsample_nearest2d(KuGraph *g, KuNode x, size_t factor);
KuNode ku_space_to_depth(KuGraph *g, KuNode x, size_t r);
KuNode ku_depth_to_space(KuGraph *g, KuNode x, size_t r);

/* backend + eval */
KuBackend *ku_backend_new(uint32_t kind); /* NULL if unavailable */
void ku_backend_free(KuBackend *b);
KuTensor *ku_eval(KuGraph *g, KuNode node, const KuBackend *backend);
KuTensor *ku_eval_with(KuGraph *g, KuNode node, const KuBackend *backend, const KuFeeds *feeds);
/* Evaluate n `ids` in one shared pass (a subgraph common to them computes once) ->
 * out[0..n], each a fresh KuTensor (free via ku_tensor_free). 0 ok, -1 err. */
int32_t ku_eval_many(KuGraph *g, const KuNode *ids, size_t n, const KuBackend *backend, const KuFeeds *feeds,
                     KuTensor **out);

/* Reverse-mode grads of sum(out) wrt n `wrt` nodes -> out_grads[0..n]. 0 ok, -1 err. */
int32_t ku_grad(KuGraph *g, KuNode out, const KuNode *wrt, size_t n, KuNode *out_grads);

/* passes & inspection (return the new root, or a count/length) */
KuNode ku_simplify(KuGraph *g, KuNode root); /* algebraic simplification */
KuNode ku_amp(KuGraph *g, KuNode root);      /* mixed precision (f16 matmuls) */
size_t ku_node_count(const KuGraph *g, KuNode root);
/* Write up to `cap` bytes of the graph dump into `out` (UTF-8, no NUL); returns full
 * length. Call with cap=0 to size the buffer first. */
size_t ku_dump(const KuGraph *g, KuNode root, uint8_t *out, size_t cap);
/* Shape / dtype of a node, so a frontend can broadcast / promote before the strict
 * builder ops. ku_node_rank sizes the buffer for ku_node_shape (SIZE_MAX on error);
 * ku_node_dtype returns a KuDType index (KU_ERR on error). */
size_t ku_node_rank(const KuGraph *g, KuNode node);
void ku_node_shape(const KuGraph *g, KuNode node, size_t *out);
uint32_t ku_node_dtype(const KuGraph *g, KuNode node);

/* runnable-graph serialization (the `.hodu` graph section): serialize the graph, its output
 * nodes, and input bindings into a self-contained blob, then rebuild it elsewhere. Inputs
 * bind by (node, role, name): role 0 = weight (bound by name from the weight table), 1 =
 * runtime feed. in_names is an array of NUL-terminated UTF-8 strings. Size-then-write:
 * cap=0 with out=NULL returns the length. */
size_t ku_graph_serialize(const KuGraph *g, const KuNode *outputs, size_t n_out, const KuNode *in_nodes,
                          const uint8_t *in_roles, const char *const *in_names, size_t n_in, uint8_t *out, size_t cap);
/* Same, but writes only the nodes reachable from outputs (remapped dense) -- drops backward/
 * dead arena nodes so a training graph exports a clean inference program. */
size_t ku_graph_serialize_reachable(const KuGraph *g, const KuNode *outputs, size_t n_out, const KuNode *in_nodes,
                                    const uint8_t *in_roles, const char *const *in_names, size_t n_in, uint8_t *out,
                                    size_t cap);
/* Rebuild a blob into a runnable handle (NULL on a malformed blob); free with ku_runnable_free. */
KuRunnable *ku_graph_deserialize(const uint8_t *bytes, size_t len);
/* Move the rebuilt graph out of the runnable (call once); free it with ku_graph_free. */
KuGraph *ku_runnable_take_graph(KuRunnable *h);
size_t ku_runnable_output_count(const KuRunnable *h);
KuNode ku_runnable_output(const KuRunnable *h, size_t i);
size_t ku_runnable_input_count(const KuRunnable *h);
KuNode ku_runnable_input_node(const KuRunnable *h, size_t i);
uint32_t ku_runnable_input_role(const KuRunnable *h, size_t i);
/* Write up to cap bytes of the i-th input name (UTF-8, no NUL); returns the full length. */
size_t ku_runnable_input_name(const KuRunnable *h, size_t i, uint8_t *out, size_t cap);
void ku_runnable_free(KuRunnable *h);

/* plan-replay: compile once, run per feeds (consts never re-copied) */
KuPlan *ku_plan_compile(const KuGraph *g, KuNode node); /* NULL if off the f32 fused path */
KuTensor *ku_plan_run(const KuPlan *p, const KuGraph *g, const KuFeeds *feeds);
void ku_plan_free(KuPlan *p);

/* feeds */
KuFeeds *ku_feeds_new(void);
void ku_feeds_free(KuFeeds *f);
void ku_feeds_set(KuFeeds *f, KuNode node, const KuTensor *tensor);

/* tensor */
KuTensor *ku_tensor_f32(const float *data, size_t len, const size_t *shape, size_t rank);
/* Input tensor of any dtype from nbytes of little-endian element data (row-major). */
KuTensor *ku_tensor_new(uint32_t dtype, const uint8_t *data, size_t nbytes, const size_t *shape, size_t rank);
void ku_tensor_free(KuTensor *t);
size_t ku_tensor_rank(const KuTensor *t);
void ku_tensor_shape(const KuTensor *t, size_t *out);
size_t ku_tensor_len(const KuTensor *t);
uint32_t ku_tensor_dtype(const KuTensor *t);
/* Copy up to `cap` f32 into `out`; returns count, or -1 if not F32. */
ptrdiff_t ku_tensor_data_f32(const KuTensor *t, float *out, size_t cap);
/* Generic raw readout (any dtype): size the buffer with ku_tensor_nbytes, then copy
 * the little-endian bytes with ku_tensor_bytes; interpret via ku_tensor_dtype. */
size_t ku_tensor_nbytes(const KuTensor *t);
void ku_tensor_bytes(const KuTensor *t, uint8_t *out);

#ifdef __cplusplus
}
#endif
#endif /* KURUMI_H */
