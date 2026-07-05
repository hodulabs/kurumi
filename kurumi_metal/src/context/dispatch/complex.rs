//! Complex seam on device (C64 = float2 = (re, im)): construct from real+imag parts,
//! extract each part, and the real<->C64 cast (im=0 / take-real). C128 = double, no
//! device path.

use crate::Buffer;
use crate::context::MetalContext;
use crate::msl::complex::{COMPLEX_MSL, R2C_MSL, part_msl};
use kurumi_core::DType;
use objc2_metal::MTLComputeCommandEncoder;

impl MetalContext {
    // (re, im) two f32 buffers -> one float2 buffer
    pub(crate) fn complex_dev(&self, re: &Buffer, im: &Buffer, n: usize) -> Buffer {
        let pso = self.cached(COMPLEX_MSL, "complex_k");
        let out = self.empty(n, DType::C64);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(re), 0, 0);
                enc.setBuffer_offset_atIndex(Some(im), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 2);
            },
            n,
        );
        out
    }

    // real f32 -> float2 with imag 0 (the f32 -> C64 cast)
    pub(crate) fn r2c_dev(&self, re: &Buffer, n: usize) -> Buffer {
        let pso = self.cached(R2C_MSL, "r2c_k");
        let out = self.empty(n, DType::C64);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(re), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
            },
            n,
        );
        out
    }

    // float2 -> real part (C64 -> F32, and the C64 -> f32 cast)
    pub(crate) fn real_dev(&self, z: &Buffer, n: usize) -> Buffer {
        self.part_dev(z, n, "x")
    }
    // float2 -> imaginary part
    pub(crate) fn imag_dev(&self, z: &Buffer, n: usize) -> Buffer {
        self.part_dev(z, n, "y")
    }
    fn part_dev(&self, z: &Buffer, n: usize, comp: &str) -> Buffer {
        let pso = self.cached(&part_msl(comp), "part_k");
        let out = self.empty(n, DType::F32);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(z), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
            },
            n,
        );
        out
    }
}
