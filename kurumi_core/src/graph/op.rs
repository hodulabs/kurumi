//! IR data types: the closed primitive op set + the arena node. The extensibility surface:
//! a new op is a variant here plus its inference rule (`graph/infer.rs`), a builder
//! (`graph.rs`), and a VJP (`grad.rs`).

use crate::{DType, Storage};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u32);

/// Closed primitive set. Grows by decomposition, never by per-backend kernels.
#[derive(Clone, Debug)]
pub enum Op {
    Const { data: Storage, shape: Vec<usize> },
    Input { shape: Vec<usize>, dtype: DType }, // a value fed at eval time (params/inputs); no baked data
    Iota { shape: Vec<usize>, axis: usize, dtype: DType }, // index along `axis`; arange/positions O(1) in IR
    // counter-based uniform [0,1) RNG (threefry2x32 over seed + flat index): pure &
    // parallel (each element independent), reproducible. src=[seed] (scalar int).
    RandUniform { shape: Vec<usize> },
    Cast { to: DType },    // dtype conversion (promotion is the frontend inserting these)
    Bitcast { to: DType }, // reinterpret bits, same width (no value change)
    Detach,                // identity forward; stops the gradient (stop_gradient). not a
    // compute kernel: an autograd-graph node, transparent to every backend.
    // elementwise; operands must match shape, no broadcast at this layer
    Add,
    Mul,
    Max,
    Neg,
    IDiv, // integer division (truncating; x/0 = 0)
    And,  // bitwise/logical and (bool + int)
    Or,
    Xor,
    Shl, // left shift by the rhs value (int)
    Shr,
    CmpLt, // a < b  -> BOOL
    CmpEq, // a == b -> BOOL
    Where, // src = [cond(BOOL), a, b]; cond ? a : b
    // unary math primitives; exp/ln/cos/tanh/sigmoid/erf decompose to these
    Recip,
    Sqrt,
    Exp2,
    Log2,
    Sin,
    Floor,
    Sum { axis: usize },                      // reduce-sum, keepdim = false
    Prod { axis: usize },                     // reduce-product, keepdim = false
    ReduceMax { axis: usize },                // reduce-max, keepdim = false
    ArgReduce { axis: usize, kind: ArgKind }, // index of max/min along axis -> I64
    // stable softmax over `axis`: exp(x - rowmax) / rowsum. Not strictly a primitive --
    // promoted so a backend runs ONE kernel instead of the reduce_max+sub+exp+sum+div chain
    // (dispatch-count win, like QuantMatmul fuses dequant). Oracle computes the decomposed math.
    Softmax { axis: usize },
    // RMSNorm over `axis`: x / sqrt(mean(x^2) + eps) (no centering, no learnable scale). Fused
    // like Softmax: one kernel vs the mul+sum+scale+add+sqrt+div chain; oracle decomposes.
    RmsNorm { axis: usize, eps: f32 },
    // Fused scaled-dot-product attention over trailing [S, dh] (leading dims batch). q,k,v same
    // shape; out = softmax((q@k^T)/sqrt(dh) [+causal -inf]) @ v. Fused like Softmax (one kernel
    // vs the two-matmul+softmax chain); the oracle computes the decomposed math. src = [q, k, v].
    Sdpa { causal: bool },
    // movement
    Reshape { shape: Vec<usize> },
    Permute { perm: Vec<usize> },
    Expand { shape: Vec<usize> },                 // broadcast size-1 dims to target
    Slice { ranges: Vec<(usize, usize, usize)> }, // per-axis [start, end) by step
    Flip { axes: Vec<usize> },
    Pad { pads: Vec<(usize, usize)> }, // per-axis (lo, hi), value 0
    // contraction (StableHLO dot_general). matmul/dot/bmm/einsum decompose to this in the frontend.
    DotGeneral { lhs_contract: Vec<usize>, rhs_contract: Vec<usize>, lhs_batch: Vec<usize>, rhs_batch: Vec<usize> },
    // weight-only quantized matmul: act[M,K] x dequant(qweight)[N,K]^T -> [M,N]. Not a
    // decomposition (fusing dequant into the contraction is the memory win); the oracle
    // dequantizes then runs DotGeneral. src = [act, qweight(U8 packed), scales(F16)],
    // plus mins(F16) when !symmetric. Weights are frozen: no VJP.
    QuantMatmul { bits: u8, group_size: usize, symmetric: bool },
    // indexed access along one axis (jnp.take semantics, a pragmatic subset of the
    // full StableHLO gather/scatter). src: [operand, indices(+updates for scatter)].
    // dense linear algebra (LU with partial pivoting), in the operand's own float dtype
    // f32 or f64 (native, no promotion; like LAPACK sgesv/dgesv). A is the trailing
    // [N, N] of each batch. Solve: src=[A, B] -> X with A*X=B.
    Solve,    // [.., N, N] x [.., N, K] -> [.., N, K]
    Det,      // [.., N, N] -> [..] (one scalar per batch)
    Cholesky, // symmetric positive-definite [.., N, N] -> lower-triangular L, A = L*L^T
    // symmetric eigendecomposition (Jacobi): [.., N, N] -> packed [.., N, N+1]
    // (columns 0..N eigenvectors, column N eigenvalues; ascending). eigh/svd slice it.
    Eigh,
    // Householder QR (reduced), K=min(M,N): r_factor picks R [.., K, N] else Q [.., M, K].
    Qr { r_factor: bool },
    // eigenvalues of a general (nonsymmetric) real matrix: [.., N, N] -> complex [.., N].
    Eigvals,
    // complex construction / part extraction. Complex: src=[re, im] real -> complex;
    // Real/Imag: src=[z] complex -> real. conj/abs/angle decompose from these.
    Complex,                                     // [re, im] -> complex (C64 from F32, C128 from F64)
    Real,                                        // complex -> real part
    Imag,                                        // complex -> imaginary part
    Gather { axis: usize },                      // OOB index clamped
    Scatter { axis: usize, combine: ScatterOp }, // OOB index dropped
    // elementwise indexing along one axis (torch take_along_dim / index_add). Unlike
    // Gather, each position carries its OWN index (idx matches the output shape).
    GatherAlong { axis: usize },                      // src=[operand, idx]; out=idx.shape
    ScatterAlong { axis: usize, combine: ScatterOp }, // src=[operand, idx, updates]
    Argsort { axis: usize, descending: bool },        // src=[x]; I64 permutation, same shape
}

/// How scatter combines an update with the existing value at a target.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ScatterOp {
    Set,
    Add,
    Max,
    Min,
}

/// Whether `ArgReduce` returns the index of the max or the min along an axis.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ArgKind {
    Max,
    Min,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub dtype: DType,
    pub src: Vec<NodeId>,
    // derived at record time and stored: derivation is O(depth) recursive and a backend
    // walk asks for it many times per node, so caching makes `shape` O(1) and turns an
    // O(N^2) eval-time walk into O(N). (DAG is append-only, so it never stales.)
    pub shape: Vec<usize>,
}
