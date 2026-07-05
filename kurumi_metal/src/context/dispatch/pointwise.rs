//! Per-element launchers: the fused pointwise kernel (source from `fused_msl`), compare
//! (-> BOOL), select (`where`), and zero-pad. One thread per output element.

use crate::Buffer;
use crate::context::{MetalContext, set_bytes, set_u32};
use crate::dtype::msl_ty;
use crate::msl::pointwise::{cmp_msl, pad_msl, where_msl};
use kurumi_core::DType;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

impl MetalContext {
    /// Run a fused-elementwise kernel: `leaves` bound at buffer(0..N), output (of
    /// dtype `dt`) at buffer(N), one thread per element.
    pub(crate) fn fused_ew(&self, src: &str, leaves: &[&Buffer], n: usize, dt: DType) -> Buffer {
        Self::tick(1);
        let pso = self.cached(src, "fused_k");
        let out = self.empty(n, dt);
        self.run_1d(
            &pso,
            |enc| unsafe {
                for (i, l) in leaves.iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(*l), 0, i);
                }
                enc.setBuffer_offset_atIndex(Some(&out), 0, leaves.len());
            },
            n,
        );
        out
    }

    /// Device comparison `a OP b` (`op` = "<" or "==") -> BOOL (uchar 0/1) buffer.
    /// Keeps comparisons device-resident so where/select chains don't fall to host.
    pub(crate) fn cmp_dev(&self, a: &Buffer, b: &Buffer, op: &str, n: usize, in_dt: DType) -> Buffer {
        let pso = self.cached(&cmp_msl(msl_ty(in_dt), op), "cmp_k");
        let out = self.empty(n, DType::BOOL);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(b), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 2);
            },
            n,
        );
        out
    }

    /// Device select `cond ? a : b` (cond BOOL/uchar, a/b/out dtype `dt`).
    pub(crate) fn where_dev(&self, cond: &Buffer, a: &Buffer, b: &Buffer, n: usize, dt: DType) -> Buffer {
        let pso = self.cached(&where_msl(msl_ty(dt)), "where_k");
        let out = self.empty(n, dt);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(cond), 0, 0);
                enc.setBuffer_offset_atIndex(Some(a), 0, 1);
                enc.setBuffer_offset_atIndex(Some(b), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 3);
            },
            n,
        );
        out
    }

    /// Device-resident zero-pad (dtype `dt`): `out` has `out_shape`, the original
    /// `in_shape` region starts at `lo` per axis; outside it is 0.
    pub(crate) fn pad_dev(
        &self,
        input: &Buffer,
        out_shape: &[usize],
        lo: &[u32],
        in_shape: &[u32],
        in_stride: &[u32],
        dt: DType,
    ) -> Buffer {
        Self::tick(6);
        let pso = self.cached(&pad_msl(msl_ty(dt)), "pad_k");
        let n: usize = out_shape.iter().product();
        let out = self.empty(n, dt);
        let out_u: Vec<u32> = out_shape.iter().map(|&x| x as u32).collect();
        let enc = self.encoder();
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
            set_u32(&enc, out_shape.len() as u32, 2);
            set_bytes(&enc, &out_u, 3);
            set_bytes(&enc, lo, 4);
            set_bytes(&enc, in_shape, 5);
            set_bytes(&enc, in_stride, 6);
        }
        let tg = pso.maxTotalThreadsPerThreadgroup().min(n.max(1));
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: n, height: 1, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
        out
    }
}
