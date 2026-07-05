//! Lazy Tensor handle: a cheap-clone handle over an IR node in a shared record
//! graph. Ops record nodes: shape/dtype checked eagerly at record time, so an
//! error points at the user's line, not an eval stack. `realize`/Display run the
//! backend. The surface the frontend and C ABI sit on; ops not wrapped here still
//! flow through the raw `Graph` builder via `Ctx::build`.

use crate::{Backend, CpuBackend, DType, Error, Graph, Key, NodeId, TensorVal};
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

/// A device context: owns the shared record graph and the backend that realizes
/// it. Cheap to clone (`Rc`); every `Tensor` born from it holds one.
/// single-threaded record (`Rc`/`RefCell`); swap to `Arc` + lock when threads
/// must share handles.
#[derive(Clone)]
pub struct Ctx(Rc<CtxInner>);

struct CtxInner {
    graph: RefCell<Graph>,
    backend: Box<dyn Backend>,
}

impl Ctx {
    /// Context over the CPU reference backend.
    pub fn cpu() -> Ctx {
        Ctx::with_backend(Box::new(CpuBackend))
    }

    /// Context over any backend (e.g. Metal, injected from the top crate).
    pub fn with_backend(backend: Box<dyn Backend>) -> Ctx {
        Ctx(Rc::new(CtxInner { graph: RefCell::new(Graph::new()), backend }))
    }

    /// The backend's name (its device label).
    pub fn device(&self) -> &str {
        self.0.backend.name()
    }

    /// An f32 constant tensor.
    pub fn constant(&self, data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let n = self.0.graph.borrow_mut().constant(data, shape);
        self.wrap(n)
    }

    /// Uniform `[0,1)` f32 tensor from `key`. The key is *consumed* (one draw); use
    /// `key.split()` for more: reusing a key is a compile error, not a silent
    /// correlated-randomness bug (the JAX footgun the borrow checker rules out).
    pub fn uniform(&self, key: Key, shape: Vec<usize>) -> Tensor {
        let n = self.0.graph.borrow_mut().rand_uniform(shape, key.raw());
        self.wrap(n)
    }

    /// Standard normal `N(0,1)` f32 tensor from `key` (Box-Muller). Key consumed.
    pub fn normal(&self, key: Key, shape: Vec<usize>) -> Tensor {
        let n = self.0.graph.borrow_mut().randn(shape, key.raw());
        self.wrap(n)
    }

    /// Escape hatch: record with the raw `Graph` builder and wrap the result:
    /// so any op not surfaced as a `Tensor` method still flows through the handle.
    /// `ctx.build(|g| g.some_op(t.node(), ..))`.
    pub fn build(&self, f: impl FnOnce(&mut Graph) -> Result<NodeId, Error>) -> Result<Tensor, Error> {
        let n = f(&mut self.0.graph.borrow_mut())?;
        Ok(self.wrap(n))
    }

    fn wrap(&self, node: NodeId) -> Tensor {
        let g = self.0.graph.borrow();
        let (shape, dtype) = (g.shape(node).to_vec(), g.dtype(node));
        Tensor(Rc::new(TensorInner { ctx: self.clone(), node, shape, dtype }))
    }
}

/// A lazy tensor: a handle to an IR node in its `Ctx`'s graph. Clone is a
/// refcount bump (`Rc`): pass and clone freely; the lazy scheduler owns buffer
/// reuse, so there is no owned-vs-borrowed dance to play.
#[derive(Clone)]
pub struct Tensor(Rc<TensorInner>);

struct TensorInner {
    ctx: Ctx,
    node: NodeId,
    shape: Vec<usize>,
    dtype: DType,
}

impl Tensor {
    pub fn node(&self) -> NodeId {
        self.0.node
    }
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }
    pub fn dtype(&self) -> DType {
        self.0.dtype
    }
    pub fn rank(&self) -> usize {
        self.0.shape.len()
    }
    pub fn ctx(&self) -> &Ctx {
        &self.0.ctx
    }

    /// `let [b, s, d] = t.dims()?;`: rank-checked shape destructure.
    pub fn dims<const N: usize>(&self) -> Result<[usize; N], Error> {
        <[usize; N]>::try_from(self.0.shape.as_slice())
            .map_err(|_| Error::Shape { op: "dims", msg: format!("rank {} != {N}", self.0.shape.len()) })
    }

    /// Realize this node to a concrete value on the ctx's backend.
    pub fn realize(&self) -> TensorVal {
        self.0.ctx.0.backend.eval(&self.0.ctx.0.graph.borrow(), self.0.node)
    }

    /// Realize and read out as f32 (casting if the dtype isn't f32).
    pub fn to_vec(&self) -> Vec<f32> {
        self.realize().storage.into_f32()
    }

    // record a binary op into the shared graph, wrap the result (shape/dtype
    // checked eagerly by the builder). The `borrow_mut` ends before `wrap` borrows.
    fn bin(
        &self,
        rhs: &Tensor,
        f: impl FnOnce(&mut Graph, NodeId, NodeId) -> Result<NodeId, Error>,
    ) -> Result<Tensor, Error> {
        let n = f(&mut self.0.ctx.0.graph.borrow_mut(), self.0.node, rhs.0.node)?;
        Ok(self.0.ctx.wrap(n))
    }

    fn un(&self, f: impl FnOnce(&mut Graph, NodeId) -> NodeId) -> Tensor {
        let n = f(&mut self.0.ctx.0.graph.borrow_mut(), self.0.node);
        self.0.ctx.wrap(n)
    }

    /// 2-D matrix multiply: contract this tensor's last axis with `rhs`'s
    /// second-to-last. 2-D only: batched matmul goes through
    /// `ctx.build(|g| g.dot_general(..))` with explicit batch dims.
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor, Error> {
        let (lc, rc) = (self.rank().saturating_sub(1), rhs.rank().saturating_sub(2));
        self.bin(rhs, |g, a, b| g.dot_general(a, b, vec![lc], vec![rc], vec![], vec![]))
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor, Error> {
        self.0.ctx.build(|g| g.reshape(self.0.node, shape))
    }
    pub fn permute(&self, perm: Vec<usize>) -> Result<Tensor, Error> {
        self.0.ctx.build(|g| g.permute(self.0.node, perm))
    }
    pub fn transpose(&self, i: usize, j: usize) -> Result<Tensor, Error> {
        self.0.ctx.build(|g| g.transpose(self.0.node, i, j))
    }
    pub fn sum(&self, axis: usize) -> Result<Tensor, Error> {
        self.0.ctx.build(|g| g.sum(self.0.node, axis))
    }
    pub fn mean(&self, axis: usize) -> Result<Tensor, Error> {
        self.0.ctx.build(|g| g.mean(self.0.node, axis))
    }
    pub fn softmax(&self, axis: usize) -> Result<Tensor, Error> {
        self.0.ctx.build(|g| g.softmax(self.0.node, axis))
    }
}

// binary ops record + wrap; unary ops are infallible builders. The rest of the
// ~100-op surface is one `ctx.build(..)` away: we surface the hot path here.
macro_rules! bin_ops {
    ($($m:ident),*) => { impl Tensor { $(
        pub fn $m(&self, rhs: &Tensor) -> Result<Tensor, Error> {
            self.bin(rhs, |g, a, b| g.$m(a, b))
        }
    )* }};
}
bin_ops!(add, sub, mul, div, max, min);

macro_rules! un_ops {
    ($($m:ident),*) => { impl Tensor { $(
        pub fn $m(&self) -> Tensor {
            self.un(|g, a| g.$m(a))
        }
    )* }};
}
un_ops!(neg, relu, gelu, sqrt, exp);

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tv = self.realize();
        let data = tv.storage.into_f32();
        let n = data.len().min(8);
        write!(
            f,
            "Tensor<{:?}, {:?}> {:?}{}",
            tv.shape,
            self.0.dtype,
            &data[..n],
            if data.len() > n { " ..." } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // handle-built graph realizes to the same values as the raw builder + oracle.
    #[test]
    fn handle_matches_oracle() {
        let ctx = Ctx::cpu();
        let a = ctx.constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = ctx.constant(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let y = a.matmul(&b).unwrap().add(&a).unwrap().relu();
        // [[19,22],[43,50]] + [[1,2],[3,4]] = [[20,24],[46,54]], relu = identity
        assert_eq!(y.to_vec(), vec![20.0, 24.0, 46.0, 54.0]);
        assert_eq!(y.shape(), &[2, 2]);
    }

    #[test]
    fn record_errors_eagerly() {
        let ctx = Ctx::cpu();
        let a = ctx.constant(vec![1.0, 2.0], vec![2]);
        let b = ctx.constant(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(a.add(&b).is_err()); // shape mismatch at record time
    }

    #[test]
    fn dims_destructure() {
        let ctx = Ctx::cpu();
        let t = ctx.constant(vec![0.0; 24], vec![2, 3, 4]);
        let [b, s, d] = t.dims().unwrap();
        assert_eq!((b, s, d), (2, 3, 4));
        assert!(t.dims::<2>().is_err());
    }
}
