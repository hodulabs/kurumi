//! Gather/scatter launchers (jnp.take + inverse): gather, take_along_dim, and their
//! scatter duals with a shared combine body (Set direct, i32/u32 native atomics, f32 CAS).

use crate::Buffer;
use crate::context::{MetalContext, set_u32};
use crate::dtype::msl_ty;
use crate::msl::indexing::{copy_msl, gather_along_msl, gather_msl, scatter_along_msl, scatter_msl};
use kurumi_core::DType;
use objc2_metal::MTLComputeCommandEncoder;

impl MetalContext {
    /// Device-resident gather (jnp.take) along an axis: operand [pre, da, post],
    /// `idx` flattened (length k), output [pre, k, post]. OOB indices are clamped.
    pub(crate) fn gather_dev(
        &self,
        operand: &Buffer,
        idx: &[i32],
        k: usize,
        post: usize,
        da: usize,
        n: usize,
        dt: DType,
    ) -> Buffer {
        Self::tick(4);
        let pso = self.cached(&gather_msl(msl_ty(dt)), "gather_k");
        let ibuf = self.buffer_bytes(idx);
        let out = self.empty(n, dt);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(operand), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&ibuf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 2);
                set_u32(enc, k as u32, 3);
                set_u32(enc, post as u32, 4);
                set_u32(enc, da as u32, 5);
            },
            n,
        );
        out
    }

    /// Device-resident gather_along / take_along_dim: `out[..i..] = operand[.. idx[..i..] ..]`
    /// along `axis`. `idx` (i32) has the output's shape; `inner` = product of dims after
    /// the axis. OOB indices are clamped (matching the CPU oracle).
    pub(crate) fn gather_along_dev(
        &self,
        operand: &Buffer,
        idx: &[i32],
        op_axis: usize,
        out_axis: usize,
        inner: usize,
        n: usize,
        dt: DType,
    ) -> Buffer {
        Self::tick(4);
        let pso = self.cached(&gather_along_msl(msl_ty(dt)), "gather_along_k");
        let ibuf = self.buffer_bytes(idx);
        let out = self.empty(n, dt);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(operand), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&ibuf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 2);
                set_u32(enc, op_axis as u32, 3);
                set_u32(enc, out_axis as u32, 4);
                set_u32(enc, inner as u32, 5);
            },
            n,
        );
        out
    }

    /// Device-resident scatter_along (index_add / take_along_dim scatter). `out` = copy of
    /// `operand`, then each update j combines into `out[.. idx[j] ..]` along the axis. Set: any
    /// dtype (direct). Add/Max/Min: f32-only (float-CAS; no 16-bit atomic, so f16/bf16 stay on oracle). OOB dropped.
    pub(crate) fn scatter_along_dev(
        &self,
        operand: &Buffer,
        idx: &[i32],
        updates: &Buffer,
        op_axis: usize,
        upd_axis: usize,
        inner: usize,
        op_n: usize,
        n_upd: usize,
        combine: &str,
        dt: DType,
    ) -> Buffer {
        Self::tick(4);
        let ty = msl_ty(dt);
        // 1. out = copy(operand)
        let out = self.empty(op_n, dt);
        self.dev_copy(operand, &out, ty, op_n);
        // 2. scatter-combine over the update positions
        let spso = self.cached(&scatter_along_msl(ty, combine, dt), "scatter_along_k");
        let ibuf = self.buffer_bytes(idx);
        self.run_1d(
            &spso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(&ibuf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(updates), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 2);
                set_u32(enc, op_axis as u32, 3);
                set_u32(enc, upd_axis as u32, 4);
                set_u32(enc, inner as u32, 5);
            },
            n_upd,
        );
        out
    }

    /// Device-resident general scatter (jnp.take inverse). operand [pre, da, post],
    /// `idx` length k (one index per axis slot), updates [pre, k, post]. `out` = copy of
    /// operand, then each update combines into `out[.. idx[ki] ..]`. Set: direct write, any
    /// dtype (racy on dup indices, but Set-with-dups is ill-defined). Add/Max/Min: f32/i32/u32 atomic. OOB dropped.
    pub(crate) fn scatter_dev(
        &self,
        operand: &Buffer,
        idx: &[i32],
        updates: &Buffer,
        da: usize,
        k: usize,
        post: usize,
        op_n: usize,
        n_upd: usize,
        combine: &str,
        dt: DType,
    ) -> Buffer {
        Self::tick(4);
        let ty = msl_ty(dt);
        // 1. out = copy(operand)
        let out = self.empty(op_n, dt);
        self.dev_copy(operand, &out, ty, op_n);
        // 2. scatter-combine over the update positions
        let spso = self.cached(&scatter_msl(ty, combine, dt), "scatter_k");
        let ibuf = self.buffer_bytes(idx);
        self.run_1d(
            &spso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(&ibuf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(updates), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 2);
                set_u32(enc, da as u32, 3);
                set_u32(enc, k as u32, 4);
                set_u32(enc, post as u32, 5);
            },
            n_upd,
        );
        out
    }

    // copy `n` elements (MSL type `ty`) from `src` to `dst` (device memcpy; scatter starts from an operand copy).
    fn dev_copy(&self, src: &Buffer, dst: &Buffer, ty: &str, n: usize) {
        let cpso = self.cached(&copy_msl(ty), "copy_k");
        self.run_1d(
            &cpso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(src), 0, 0);
                enc.setBuffer_offset_atIndex(Some(dst), 0, 1);
            },
            n,
        );
    }
}
