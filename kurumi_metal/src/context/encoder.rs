//! Command-buffer / compute-encoder batching and the elementwise dispatch helpers.

use crate::Pipeline;
use crate::context::{CmdBuf, Enc, MetalContext};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize,
};

impl MetalContext {
    // the pending command buffer GPU ops encode into (created lazily).
    pub(crate) fn cmd(&self) -> CmdBuf {
        let mut p = self.pending.borrow_mut();
        if p.is_none() {
            *p = Some(self.queue.commandBuffer().expect("command buffer"));
        }
        p.clone().unwrap()
    }

    // the shared open serial compute encoder for custom kernels (created lazily on
    // the pending buffer). Consecutive dispatches into it stay ordered.
    pub(crate) fn encoder(&self) -> Enc {
        let mut e = self.enc.borrow_mut();
        if e.is_none() {
            *e = Some(self.cmd().computeCommandEncoder().expect("compute encoder"));
        }
        e.clone().unwrap()
    }

    // close the open encoder (before an MPS GEMM encodes its own, or before commit).
    pub(crate) fn end_encoder(&self) {
        if let Some(e) = self.enc.borrow_mut().take() {
            e.endEncoding();
        }
    }

    /// Commit + wait the pending GPU work, if any. Call before reading any device
    /// buffer back to the host (a host boundary or the final output).
    pub(crate) fn flush(&self) {
        self.end_encoder();
        if let Some(cmd) = self.pending.borrow_mut().take() {
            cmd.commit();
            cmd.waitUntilCompleted();
            if std::env::var_os("KURUMI_GPUTIME").is_some() {
                let gpu = cmd.GPUEndTime() - cmd.GPUStartTime();
                Self::bump_flush((gpu * 1e6) as u128);
            }
        }
    }

    // encode a 1-D elementwise pso over `n` threads into the shared open encoder
    // (dispatches stay ordered; no per-op encoder, no commit: `flush` does that).
    pub(crate) fn run_1d(
        &self,
        pso: &Pipeline,
        bind: impl Fn(&ProtocolObject<dyn MTLComputeCommandEncoder>),
        n: usize,
    ) {
        let enc = self.encoder();
        enc.setComputePipelineState(pso);
        bind(&enc);
        let tg = pso.maxTotalThreadsPerThreadgroup().min(n.max(1));
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: n, height: 1, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
    }

    // encode gx x gy threadgroups of tx x ty threads (for tiled kernels with threadgroup
    // memory) into the shared open encoder.
    pub(crate) fn run_groups(
        &self,
        pso: &Pipeline,
        bind: impl Fn(&ProtocolObject<dyn MTLComputeCommandEncoder>),
        gx: usize,
        gy: usize,
        tx: usize,
        ty: usize,
    ) {
        let enc = self.encoder();
        enc.setComputePipelineState(pso);
        bind(&enc);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: gx, height: gy, depth: 1 },
            MTLSize { width: tx, height: ty, depth: 1 },
        );
    }
}
