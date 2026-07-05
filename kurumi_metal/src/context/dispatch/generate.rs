//! Value-generator launchers: iota (index along an axis) and counter-based uniform RNG.
//! Kernel sources are in `msl::generate`.

use crate::Buffer;
use crate::context::{MetalContext, set_u32};
use crate::dtype::msl_ty;
use crate::msl::generate::{RAND_MSL, iota_msl};
use kurumi_core::DType;
use objc2_metal::MTLComputeCommandEncoder;

impl MetalContext {
    /// Device-resident iota: `out[gid] = (gid / stride) % axis_len` cast to `dt`
    /// (the index along `axis`). arange/positions/eye/tril build on this.
    pub(crate) fn iota_dev(&self, stride: usize, axis_len: usize, n: usize, dt: DType) -> Buffer {
        let pso = self.cached(&iota_msl(msl_ty(dt)), "iota_k");
        let out = self.empty(n, dt);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(&out), 0, 0);
                set_u32(enc, stride as u32, 1);
                set_u32(enc, axis_len as u32, 2);
            },
            n,
        );
        out
    }

    /// Device-resident counter-based uniform `[0,1)` RNG (F32). Each element is
    /// `threefry2x32(seed, index)`: bit-for-bit identical to the CPU oracle.
    pub(crate) fn rand_dev(&self, seed: u64, n: usize) -> Buffer {
        let pso = self.cached(RAND_MSL, "rand_k");
        let out = self.empty(n, DType::F32);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(&out), 0, 0);
                let ptr = std::ptr::NonNull::new(&seed as *const u64 as *mut std::ffi::c_void).unwrap();
                enc.setBytes_length_atIndex(ptr, 8, 1);
            },
            n,
        );
        out
    }
}
