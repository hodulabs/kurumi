//! Device-op launchers by category. Each submodule adds `impl MetalContext` methods that
//! encode one GPU kernel into the pending command buffer, using the context core
//! (`cmd`/`empty`/`cached`/`run_1d`/`encoder`) and the MSL sources in `crate::msl`.
//! Split by op family, mirroring graph/ops/.

mod cast;
mod complex;
mod generate;
mod hostgemm;
mod indexing;
mod linalg;
mod matmul;
mod nn;
mod pointwise;
mod quant;
mod reduce;
