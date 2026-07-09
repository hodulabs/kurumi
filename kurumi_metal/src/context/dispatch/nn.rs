//! Fused nn-primitive launchers (softmax, ...). Kernel sources are in `msl::nn`.

use crate::Buffer;
use crate::context::{MetalContext, set_u32};
use crate::dtype::msl_ty;
use crate::msl::nn::{rmsnorm_msl, sdpa_flash_msl, softmax_msl};
use kurumi_core::DType;
use objc2_metal::MTLComputeCommandEncoder;

impl MetalContext {
    /// Device-resident softmax over an axis (layout outer x axis_len x inner, one thread per
    /// line). `out_n` = total elements (shape-preserving), `n_lines` = out_n / axis_len.
    pub(crate) fn softmax_dev(
        &self,
        input: &Buffer,
        axis_len: usize,
        inner: usize,
        n_lines: usize,
        out_n: usize,
        dt: DType,
    ) -> Buffer {
        Self::tick(2);
        let pso = self.cached(&softmax_msl(msl_ty(dt)), "softmax_k");
        let out = self.empty(out_n, dt);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(input), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
                set_u32(enc, axis_len as u32, 2);
                set_u32(enc, inner as u32, 3);
            },
            n_lines,
        );
        out
    }

    /// Device-resident RMSNorm over an axis (one thread per line). Shape-preserving.
    pub(crate) fn rmsnorm_dev(
        &self,
        input: &Buffer,
        axis_len: usize,
        inner: usize,
        n_lines: usize,
        out_n: usize,
        eps: f32,
        dt: DType,
    ) -> Buffer {
        Self::tick(2);
        let pso = self.cached(&rmsnorm_msl(msl_ty(dt)), "rmsnorm_k");
        let out = self.empty(out_n, dt);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(input), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
                set_u32(enc, axis_len as u32, 2);
                set_u32(enc, inner as u32, 3);
                let ptr = std::ptr::NonNull::new(&eps as *const f32 as *mut std::ffi::c_void).unwrap();
                enc.setBytes_length_atIndex(ptr, 4, 4);
            },
            n_lines,
        );
        out
    }

    /// Flash-attention FORWARD (online softmax): ONE thread per (batch, query-row); grid =
    /// `batch*s`. q,k,v are `[batch, s, dh]` flattened f32 buffers; out is `[batch, s, dh]`.
    /// Never materializes the SxS scores (O(dh) per-thread state). `scale` = 1/sqrt(dh) is
    /// host-computed to match the oracle exactly. Caller guarantees `dh <= SDPA_MAX_DH`.
    pub(crate) fn sdpa_dev(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        batch: usize,
        s: usize,
        dh: usize,
        causal: bool,
    ) -> Buffer {
        Self::tick(2);
        let pso = self.cached(&sdpa_flash_msl(), "sdpa_flash_k");
        let out = self.empty(batch * s * dh, DType::F32);
        let scale = 1.0f32 / (dh as f32).sqrt();
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(k), 0, 1);
                enc.setBuffer_offset_atIndex(Some(v), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 3);
                set_u32(enc, s as u32, 4);
                set_u32(enc, dh as u32, 5);
                let sp = std::ptr::NonNull::new(&scale as *const f32 as *mut std::ffi::c_void).unwrap();
                enc.setBytes_length_atIndex(sp, 4, 6);
                set_u32(enc, causal as u32, 7);
            },
            batch * s,
        );
        out
    }
}
