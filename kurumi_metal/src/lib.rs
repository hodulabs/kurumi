//! kurumi Metal backend (Apple Silicon). objc2-metal device/buffers/dispatch +
//! MSL kernels. Modules are flat files (foo.rs), no mod.rs:
//! `msl` (MSL sources) / `context` (device layer) / `backend` (engine seam).

#![cfg(target_os = "macos")]

// MTLCreateSystemDefaultDevice needs CoreGraphics; the protocols need Metal.
#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {}
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

mod backend;
mod context;
mod dtype;
mod msl;
#[cfg(test)]
mod tests;

pub use backend::MetalBackend;
pub use context::MetalContext;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice};

// shared GPU handle types (used by `context` and `backend`)
pub(crate) type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub(crate) type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub(crate) type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub(crate) type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
