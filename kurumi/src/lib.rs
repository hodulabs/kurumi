/// C ABI graph-builder surface: exported from the cdylib/staticlib. See kurumi.h.
mod capi;

pub use kurumi_core::*;
/// Metal backend (Apple Silicon only).
#[cfg(target_os = "macos")]
pub use kurumi_metal as metal;
