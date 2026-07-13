//! Compute-pipeline compile + kernel cache.

use crate::Pipeline;
use crate::context::MetalContext;
use objc2_foundation::NSString;
use objc2_metal::{MTLDevice, MTLLibrary};

impl MetalContext {
    /// Compile a generated kernel once per source: all device kernels (fused / reduce /
    /// strided / pad, each parameterized by element dtype) share this cache, so dtype
    /// variants compile separately and reuse across a training loop.
    pub(crate) fn cached(&self, src: &str, func: &str) -> Pipeline {
        let hit = self.fused.borrow().get(src).cloned(); // drop the borrow before borrow_mut
        hit.unwrap_or_else(|| {
            let p = self.pipeline(src, func);
            self.fused.borrow_mut().insert(src.to_string(), p.clone());
            p
        })
    }

    // compile one MSL function into a compute pipeline state (cache this across runs)
    pub(crate) fn pipeline(&self, src: &str, func: &str) -> Pipeline {
        let lib = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(src), None)
            .unwrap_or_else(|e| panic!("MSL compile failed: {e}"));
        let f = lib.newFunctionWithName(&NSString::from_str(func)).expect("function");
        self.device.newComputePipelineStateWithFunction_error(&f).expect("pipeline")
    }
}
