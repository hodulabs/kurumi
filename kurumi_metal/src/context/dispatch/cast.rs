//! Dtype cast launcher. Kernel source (Rust `as` semantics) is in `msl::cast`.

use crate::Buffer;
use crate::context::MetalContext;
use crate::msl::cast::cast_msl;
use kurumi_core::DType;
use objc2_metal::MTLComputeCommandEncoder;

impl MetalContext {
    pub(crate) fn cast_dev(&self, input: &Buffer, n: usize, src_dt: DType, dst_dt: DType) -> Buffer {
        Self::tick(5);
        let pso = self.cached(&cast_msl(src_dt, dst_dt), "cast_k");
        let out = self.empty(n, dst_dt);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(input), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
            },
            n,
        );
        out
    }
}
