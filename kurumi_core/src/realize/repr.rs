//! Fused-representation data model: the lazy elementwise `Expr`, the `Repr` (a
//! materialized leaf buffer or fused expression), and the `Realized` handle with its
//! force / materialize methods. The scheduler (`super`) builds these; the tape executor
//! runs the fused ones. Materialization goes through `eval_fused`, bit-identical to the
//! interpreter oracle.

use crate::lower::index::{self, View};
use crate::realize::tape::{eval_fused, eval_fused_into};
use crate::{Storage, TensorVal};
use std::borrow::Cow;
use std::rc::Rc;

// Op kind tags: carried instead of fn pointers so the row executor dispatches one
// monomorphic loop per op (which then auto-vectorizes), while scalar `apply` stays the
// SAME f32 op as the oracle -> bit-identical.
#[derive(Clone, Copy)]
pub(super) enum UnOp {
    Neg,
    Recip,
    Sqrt,
    Exp2,
    Log2,
    Sin,
}

#[derive(Clone, Copy)]
pub(super) enum BinOp {
    Add,
    Mul,
    Max,
}

impl UnOp {
    #[inline]
    pub(super) fn apply(self, x: f32) -> f32 {
        match self {
            UnOp::Neg => -x,
            UnOp::Recip => 1.0 / x,
            UnOp::Sqrt => x.sqrt(),
            UnOp::Exp2 => x.exp2(),
            UnOp::Log2 => x.log2(),
            UnOp::Sin => x.sin(),
        }
    }
}

// Lazy elementwise computation over a common output shape. Every leaf reads a source
// buffer through its view at the same output coordinate (the no-broadcast binary rule
// guarantees all operands share that shape). Children are `Rc` so `as_expr` is a shallow
// clone (refcount bump): a deep chain stays O(depth) allocations, not O(depth^2).
#[derive(Clone)]
pub(super) enum Expr {
    Load { buf: Rc<[f32]>, view: View },
    Unary(UnOp, Rc<Expr>),
    Add(Rc<Expr>, Rc<Expr>),
    Mul(Rc<Expr>, Rc<Expr>),
    Max(Rc<Expr>, Rc<Expr>),
}

#[derive(Clone)]
pub(super) enum Repr {
    Leaf { buf: Rc<[f32]>, view: View },
    Fused { shape: Vec<usize>, expr: Expr },
}

/// A realized node: a materialized leaf buffer (read through a view) or a fused
/// lazy elementwise expression.
#[derive(Clone)]
pub struct Realized(pub(super) Repr);

impl Realized {
    pub fn force(&self) -> TensorVal {
        match &self.0 {
            Repr::Leaf { buf, view } => {
                TensorVal { shape: view.shape.clone(), storage: Storage::F32(index::read(buf, view)) }
            }
            Repr::Fused { shape, expr } => {
                TensorVal { shape: shape.clone(), storage: Storage::F32(eval_fused(expr, shape)) }
            }
        }
    }

    /// Force into a reused buffer (no per-call output allocation). The fused expr
    /// streams straight into `out`; a leaf gathers through its view.
    pub fn force_into(&self, out: &mut Vec<f32>) {
        match &self.0 {
            Repr::Fused { shape, expr } => eval_fused_into(expr, shape, out),
            Repr::Leaf { buf, view } => {
                out.clear();
                out.extend_from_slice(&index::read(buf, view));
            }
        }
    }

    pub(super) fn shape(&self) -> &[usize] {
        match &self.0 {
            Repr::Leaf { view, .. } => &view.shape,
            Repr::Fused { shape, .. } => shape,
        }
    }

    // Borrow contiguous row-major data without copying when this is already a plain source
    // leaf (a `contiguous` view's buffer *is* its row-major data); else materialize once.
    // Lets gemm/reduce read weights and prior results in place, not re-copy at every boundary.
    pub(super) fn force_contig(&self) -> (Cow<'_, [f32]>, Vec<usize>) {
        if let Repr::Leaf { buf, view } = &self.0
            && view.contiguous
            && view.guards.is_empty()
        {
            debug_assert_eq!(buf.len(), view.shape.iter().product::<usize>());
            return (Cow::Borrowed(&**buf), view.shape.clone());
        }
        let tv = self.force();
        (Cow::Owned(tv.storage.into_f32()), tv.shape)
    }

    // Force to a materialized buffer + identity view (one pass).
    pub(super) fn into_leaf(self) -> (Rc<[f32]>, View) {
        match self.0 {
            Repr::Leaf { buf, view } => (buf, view),
            Repr::Fused { shape, expr } => {
                let data = eval_fused(&expr, &shape);
                (Rc::from(data), View::source(shape))
            }
        }
    }

    pub(super) fn as_expr(&self) -> Expr {
        match &self.0 {
            Repr::Leaf { buf, view } => Expr::Load { buf: buf.clone(), view: view.clone() },
            Repr::Fused { expr, .. } => expr.clone(),
        }
    }

    pub(super) fn is_fused(&self) -> bool {
        matches!(self.0, Repr::Fused { .. })
    }
}
