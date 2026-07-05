//! Backend seam: the engine targets a `Backend` to evaluate a graph. The CPU
//! reference runs every op + dtype (it is the interpreter); device backends
//! (Metal, later CUDA/WebGPU) accelerate the ops they have kernels for and fall
//! back to the CPU oracle (`eval_op`) for the rest: every op runs on every
//! backend, acceleration grows without breaking completeness.

use crate::{DType, Error, Feeds, Graph, NodeId, Storage, TensorVal, dot_dispatch};

pub trait Backend {
    fn name(&self) -> &str;
    /// Evaluate node `id` of `g`, supplying `Input` nodes from `feeds`, to a host
    /// tensor. This is the seam: a model graph is built once, then run each step
    /// with fresh feeds. (`eval` is the no-input convenience.)
    fn eval_with(&self, g: &Graph, id: NodeId, feeds: &Feeds) -> TensorVal;

    /// Evaluate a graph with no `Input` nodes (baked constants only).
    fn eval(&self, g: &Graph, id: NodeId) -> TensorVal {
        self.eval_with(g, id, &Feeds::new())
    }
}

/// Reference CPU backend = the interpreter (every op, every dtype).
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }
    fn eval_with(&self, g: &Graph, id: NodeId, feeds: &Feeds) -> TensorVal {
        crate::interpret_with(g, id, feeds)
    }
}

// Direct GEMM/cast on the CPU: the reference device backends check against.
impl CpuBackend {
    pub fn matmul(&self, a: &Storage, m: usize, k: usize, b: &Storage, n: usize) -> Result<Storage, Error> {
        if a.dtype() != b.dtype() {
            return Err(Error::backend(format!("matmul dtype mismatch: {:?} vs {:?}", a.dtype(), b.dtype())));
        }
        if !a.dtype().is_numeric() {
            return Err(Error::backend(format!("matmul needs a numeric dtype, got {:?}", a.dtype())));
        }
        let ta = TensorVal { shape: vec![m, k], storage: a.clone() };
        let tb = TensorVal { shape: vec![k, n], storage: b.clone() };
        Ok(dot_dispatch(&ta, &tb, &[1], &[0], &[], &[]).storage)
    }
    pub fn cast(&self, src: &Storage, to: DType) -> Result<Storage, Error> {
        Ok(crate::cast(src, to)) // CPU handles every pair
    }
}

#[cfg(test)]
mod tests;
