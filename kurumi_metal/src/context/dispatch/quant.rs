//! Weight-only quant matmul launchers. `dequant_gemv` for decode/small-batch (M <= 8: one
//! simdgroup per output column, 32-bit word loads, decode-once-reuse across the row block) and
//! `dequant_gemm` for prefill (threadgroup-tiled, decoded weight tile reused across the tile's
//! rows). Both baked per (bits, sym) and templated on the activation dtype; checked against the
//! CPU quant oracle.

use crate::Buffer;
use crate::context::{MetalContext, set_u32};
use crate::dtype::msl_ty;
use crate::msl::quant::{dequant_gemm_msl, dequant_gemv_msl};
use kurumi_core::DType;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputeCommandEncoder;

// bind the shared buffers (0-4) and constants (5-8) both quant kernels take. bits/sym are
// baked into the kernel source, not passed at runtime.
#[allow(clippy::too_many_arguments)]
unsafe fn bind_quant(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    act: &Buffer,
    qw: &Buffer,
    scales: &Buffer,
    mins: &Buffer,
    out: &Buffer,
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(act), 0, 0);
        enc.setBuffer_offset_atIndex(Some(qw), 0, 1);
        enc.setBuffer_offset_atIndex(Some(scales), 0, 2);
        enc.setBuffer_offset_atIndex(Some(mins), 0, 3);
        enc.setBuffer_offset_atIndex(Some(out), 0, 4);
        set_u32(enc, m as u32, 5);
        set_u32(enc, k as u32, 6);
        set_u32(enc, n as u32, 7);
        set_u32(enc, group_size as u32, 8);
    }
}

impl MetalContext {
    /// Decode/small-batch quant GEMV (M <= 8): one thread per (n, M-row block), grid `N x 1`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dequant_gemv_dev(
        &self,
        act: &Buffer,
        qw: &Buffer,
        scales: &Buffer,
        mins: &Buffer,
        m: usize,
        k: usize,
        n: usize,
        group_size: usize,
        bits: u8,
        symmetric: bool,
        dt: DType,
    ) -> Buffer {
        let src = dequant_gemv_msl(msl_ty(dt), bits, symmetric, m == 1);
        let pso = self.cached(&src, "dequant_gemv");
        let out = self.empty(m * n, dt);
        // one simdgroup (32 lanes) per output column: N threadgroups of 32 threads.
        self.run_groups(
            &pso,
            |enc| unsafe { bind_quant(enc, act, qw, scales, mins, &out, m, k, n, group_size) },
            n,
            1,
            32,
            1,
        );
        out
    }

    /// Prefill quant GEMM (M > 8): threadgroup-tiled 16x16, decoded weight tile reused.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dequant_gemm_dev(
        &self,
        act: &Buffer,
        qw: &Buffer,
        scales: &Buffer,
        mins: &Buffer,
        m: usize,
        k: usize,
        n: usize,
        group_size: usize,
        bits: u8,
        symmetric: bool,
        dt: DType,
    ) -> Buffer {
        let src = dequant_gemm_msl(msl_ty(dt), bits, symmetric);
        let pso = self.cached(&src, "dequant_gemm");
        let out = self.empty(m * n, dt);
        // 64x64 output tile per threadgroup (16x16 threads, each a 4x4 micro-tile).
        self.run_groups(
            &pso,
            |enc| unsafe { bind_quant(enc, act, qw, scales, mins, &out, m, k, n, group_size) },
            n.div_ceil(64),
            m.div_ceil(64),
            16,
            16,
        );
        out
    }
}
