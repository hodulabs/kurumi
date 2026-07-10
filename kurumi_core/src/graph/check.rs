//! Record-time validation guards shared by the strict op builders: dtype-class checks
//! (numeric add, float sqrt, ...), reduction-axis bounds, and the same-shape/same-dtype
//! binary contract. Every `graph/ops/*` builder calls these before `push`, so a bad op is a
//! clean `Err` at record time, never a panic in inference.
use crate::Error;
use crate::graph::{Graph, NodeId, Op};

impl Graph {
    // record-time dtype-class guard for the strict ops (numeric add, float sqrt, ...)
    pub(in crate::graph) fn require(&self, op: &'static str, a: NodeId, ok: bool, want: &str) -> Result<(), Error> {
        if ok { Ok(()) } else { Err(Error::shape(op, format!("{want} dtype required, got {:?}", self.dtype(a)))) }
    }

    // axis in range (shared by every reduction).
    fn reduce_axis_ok(&self, op: &'static str, a: NodeId, axis: usize) -> Result<(), Error> {
        let rank = self.shape(a).len();
        if axis >= rank {
            return Err(Error::shape(op, format!("axis {axis} out of range for rank {rank}")));
        }
        Ok(())
    }

    // axis + numeric element type: for ORDER-based reductions (max/argmax/argmin/
    // argsort): complex has no total order, so it's rejected here.
    pub(in crate::graph) fn reduce_check(&self, op: &'static str, a: NodeId, axis: usize) -> Result<(), Error> {
        self.reduce_axis_ok(op, a, axis)?;
        self.require(op, a, self.dtype(a).is_numeric(), "numeric")
    }

    // axis + field-arith element type: for sum/prod, which are well-defined on complex
    // (complex add / complex mul). `is_arith` = numeric OR complex.
    pub(in crate::graph) fn reduce_check_arith(&self, op: &'static str, a: NodeId, axis: usize) -> Result<(), Error> {
        self.reduce_axis_ok(op, a, axis)?;
        self.require(op, a, self.dtype(a).is_arith(), "numeric or complex")
    }

    // shared check for the strict same-shape, same-dtype binary ops
    pub(in crate::graph) fn bin(&mut self, name: &'static str, op: Op, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        self.same_shape(name, a, b)?;
        self.same_dtype(name, a, b)?;
        Ok(self.push(op, vec![a, b]))
    }

    pub(in crate::graph) fn same_shape(&self, op: &'static str, a: NodeId, b: NodeId) -> Result<(), Error> {
        let (lhs, rhs) = (self.shape(a), self.shape(b));
        if lhs != rhs {
            return Err(Error::shape(op, format!("{lhs:?} vs {rhs:?}")));
        }
        Ok(())
    }

    // primitives are strict on dtype (like shape): promotion is explicit, upstream
    pub(in crate::graph) fn same_dtype(&self, op: &'static str, a: NodeId, b: NodeId) -> Result<(), Error> {
        let (lhs, rhs) = (self.dtype(a), self.dtype(b));
        if lhs != rhs {
            return Err(Error::shape(op, format!("dtype {lhs:?} vs {rhs:?}")));
        }
        Ok(())
    }
}
