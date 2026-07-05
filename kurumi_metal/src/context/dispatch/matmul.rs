//! GEMM launchers: MPS real batched GEMM (cached kernel object) and a naive complex
//! (float2) GEMM for quantum gate application (MPS has no complex path).

use crate::Buffer;
use crate::context::{MetalContext, set_u32};
use crate::dtype::mps_ty;
use crate::msl::matmul::CMATMUL_MSL;
use kurumi_core::DType;
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2_metal::MTLComputeCommandEncoder;
use objc2_metal_performance_shaders::{MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};

impl MetalContext {
    /// Device-resident f32 batched GEMM C[batch,m,n] = op(A) @ op(B) via MPS (vendored peak GEMM),
    /// one encode for the whole batch. `a_shape`/`b_shape` are physical (row-major) per-matrix shapes;
    /// `trans_l`/`trans_r` apply the transpose, so ANY dot_general (2D or batched, canonical or
    /// transposed, incl. autograd backward + attention) runs on the GPU. batch=1 = plain 2D; inputs are device buffers.
    pub(crate) fn mps_matmul_dev(
        &self,
        a: &Buffer,
        a_shape: (usize, usize),
        trans_l: bool,
        b: &Buffer,
        b_shape: (usize, usize),
        trans_r: bool,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        dt: DType,
    ) -> Buffer {
        Self::tick(0);
        let c = self.empty(batch * m * n, dt);
        let ty = mps_ty(dt);
        let es = dt.width();
        // batched descriptor = per-matrix layout + matrixBytes stride between batches
        let desc = |rows: usize, cols: usize| unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(
                rows,
                cols,
                batch,
                cols * es,
                rows * cols * es,
                ty,
            )
        };
        // MPS encodes its own encoder into the command buffer: close ours first
        // (its work then orders after the preceding custom kernels).
        self.end_encoder();
        let cmd = self.cmd();
        // the GEMM kernel object is shape-parameterized and reusable across evals -> cache
        // it (made-once/encode-many). Only the buffer-wrapping MPSMatrix objects rebind.
        let mm = self.mm_kernel(trans_l, trans_r, m, n, k, batch);
        unsafe {
            let ma = MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), a, &desc(a_shape.0, a_shape.1));
            let mb = MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), b, &desc(b_shape.0, b_shape.1));
            let mc = MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &c, &desc(m, n));
            mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(&cmd, &ma, &mb, &mc);
        }
        c
    }

    // cached MPS GEMM kernel object for a (trans, dims, batch) shape (made-once/encode-many).
    fn mm_kernel(
        &self,
        trans_l: bool,
        trans_r: bool,
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Retained<MPSMatrixMultiplication> {
        let key = (trans_l, trans_r, m, n, k, batch);
        if let Some(mm) = self.mm_cache.borrow().get(&key) {
            return mm.clone();
        }
        let mm = unsafe {
            let mm = MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(), &self.device, trans_l, trans_r, m, n, k, 1.0, 0.0,
            );
            mm.setBatchSize(batch);
            mm
        };
        self.mm_cache.borrow_mut().insert(key, mm.clone());
        mm
    }

    /// Device-resident complex GEMM (C64/float2): MPS has no complex path, so a naive
    /// one-thread-per-output kernel accumulates `cmul` over k. Handles batch + optional transpose
    /// of either operand (per-matrix `op(A)[M,K] @ op(B)[K,N]`), so ANY complex dot_general runs on
    /// the GPU: quantum gate application (state @ gate), multi-qubit batched gates, transposed autograd backward.
    pub(crate) fn cmatmul_dev(
        &self,
        a: &Buffer,
        b: &Buffer,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        trans_l: bool,
        trans_r: bool,
    ) -> Buffer {
        Self::tick(0);
        // per-batch: A is m*k, B is k*n elements. trans_l/trans_r pick the physical layout:
        // A[i,t] = TL ? A[t,i] (kxm) : A[i,t] (mxk);  B[t,j] = TR ? B[j,t] (nxk) : B[t,j] (kxn).
        let pso = self.cached(CMATMUL_MSL, "cmatmul");
        let out = self.empty(batch * m * n, DType::C64);
        self.run_1d(
            &pso,
            |enc| unsafe {
                enc.setBuffer_offset_atIndex(Some(a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(b), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 2);
                set_u32(enc, m as u32, 3);
                set_u32(enc, n as u32, 4);
                set_u32(enc, k as u32, 5);
                set_u32(enc, trans_l as u32, 6);
                set_u32(enc, trans_r as u32, 7);
            },
            batch * m * n,
        );
        out
    }
}
