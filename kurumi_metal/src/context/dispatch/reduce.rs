//! Reduction launchers: axis reduce (materialized or fused-producer parallel), argmax/
//! argmin, and argsort. Fold semantics match the CPU oracle.

use crate::Buffer;
use crate::context::{MetalContext, set_u32};
use crate::dtype::msl_ty;
use crate::msl::reduce::{argreduce_msl, argsort_msl, reduce_msl};
use kurumi_core::DType;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

impl MetalContext {
    /// Run a fused parallel-reduce kernel (source from `fused_reduce_msl`): `leaves` at
    /// buffer(0..N), output (`out_n` lines, dtype `dt`) at buffer(N). One threadgroup of
    /// `tg` threads per output line (shared-memory tree reduce inside).
    pub(crate) fn reduce_fused(&self, src: &str, leaves: &[&Buffer], out_n: usize, tg: usize, dt: DType) -> Buffer {
        Self::tick(2);
        let pso = self.cached(src, "reduce_k");
        let out = self.empty(out_n, dt);
        let enc = self.encoder();
        enc.setComputePipelineState(&pso);
        unsafe {
            for (i, l) in leaves.iter().enumerate() {
                enc.setBuffer_offset_atIndex(Some(*l), 0, i);
            }
            enc.setBuffer_offset_atIndex(Some(&out), 0, leaves.len());
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: out_n, height: 1, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
        out
    }

    /// Device-resident reduce of `input` (dtype `dt`) along an axis (keepdim=false): folds
    /// `acc = op(acc, x)`. `tag` = "sum"|"max"|"prod". Floats accumulate in `float`, ints in
    /// their own type (exact). max seeds `acc` with the first element (avoids a per-dtype type-min
    /// sentinel). Layout outer x axis_len x inner; `out_n` = outer*inner. Compiled once per (tag, dtype). Serial fold.
    pub(crate) fn reduce_dev(
        &self,
        tag: &str,
        input: &Buffer,
        axis_len: usize,
        inner: usize,
        out_n: usize,
        dt: DType,
    ) -> Buffer {
        Self::tick(2);
        let pso = self.cached(&reduce_msl(tag, dt), "reduce_k");
        let out = self.empty(out_n, dt);
        let enc = self.encoder();
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
            set_u32(&enc, axis_len as u32, 2);
            set_u32(&enc, inner as u32, 3);
        }
        let tg = pso.maxTotalThreadsPerThreadgroup().min(out_n.max(1));
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: out_n, height: 1, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
        out
    }

    /// argmax/argmin along an axis -> I64 index buffer (`is_max` picks the test;
    /// ties keep the first occurrence, matching the CPU oracle).
    pub(crate) fn argreduce_dev(
        &self,
        input: &Buffer,
        axis_len: usize,
        inner: usize,
        out_n: usize,
        in_dt: DType,
        is_max: bool,
    ) -> Buffer {
        Self::tick(2);
        let pso = self.cached(&argreduce_msl(msl_ty(in_dt), is_max), "argreduce_k");
        let out = self.empty(out_n, DType::I64);
        let enc = self.encoder();
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
            set_u32(&enc, axis_len as u32, 2);
            set_u32(&enc, inner as u32, 3);
        }
        let tg = pso.maxTotalThreadsPerThreadgroup().min(out_n.max(1));
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: out_n, height: 1, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
        out
    }

    /// Device-resident argsort along an axis -> I64 permutation (same shape as input). One thread
    /// per line (outer x inner) runs a stable in-place insertion sort of the index array, comparing
    /// `in[idx]`. Strict comparison keeps ties in original order (matches the CPU oracle). `descending`
    /// flips the order. O(L^2) per line; swap for a bitonic pass if very long sort axes get hot.
    pub(crate) fn argsort_dev(
        &self,
        input: &Buffer,
        axis_len: usize,
        inner: usize,
        n_lines: usize,
        out_n: usize,
        dt: DType,
        descending: bool,
    ) -> Buffer {
        let pso = self.cached(&argsort_msl(msl_ty(dt), descending), "argsort_k");
        let out = self.empty(out_n, DType::I64);
        let enc = self.encoder();
        enc.setComputePipelineState(&pso);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
            set_u32(&enc, axis_len as u32, 2);
            set_u32(&enc, inner as u32, 3);
        }
        let tg = pso.maxTotalThreadsPerThreadgroup().min(n_lines.max(1));
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: n_lines, height: 1, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
        out
    }
}
