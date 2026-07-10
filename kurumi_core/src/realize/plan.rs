//! plan-memoized executor: compile a graph once (schedule + const leaves materialized
//! into shared buffers), then replay with fresh `Input` feeds -- the train / inference
//! loop, weights never re-copied and the output buffer reused. Replay drives the same
//! scheduler (`go`) as one-shot `realize`.

use crate::realize::{Sched, consumer_counts, fused_supported, go};
use crate::{Feeds, Graph, NodeId, Op, TensorVal};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// A compiled, replayable schedule for the f32 fused path. `None` from `compile`
/// means the graph leaves the fused path.
pub struct Plan {
    output: NodeId,
    shape: Vec<usize>,
    counts: HashMap<NodeId, usize>,
    consts: HashMap<NodeId, Rc<[f32]>>,
}

impl Plan {
    pub fn compile(g: &Graph, id: NodeId) -> Option<Plan> {
        if !fused_supported(g, id, true) {
            return None;
        }
        // materialize every reachable const leaf exactly once (the whole point).
        let mut consts = HashMap::new();
        let mut seen = HashSet::new();
        let mut stack = vec![id];
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if let Op::Const { data, .. } = &g.node(n).op {
                consts.insert(n, Rc::from(data.as_f32()));
            }
            stack.extend_from_slice(&g.node(n).src);
        }
        Some(Plan { output: id, shape: g.shape(id).to_vec(), counts: consumer_counts(g, id), consts })
    }

    fn sched<'a>(&'a self, g: &'a Graph, feeds: &'a Feeds) -> Sched<'a> {
        Sched { g, counts: &self.counts, feeds, consts: Some(&self.consts) }
    }

    /// Replay with fresh inputs. `g` must be the graph `compile` was called on.
    pub fn run(&self, g: &Graph, feeds: &Feeds) -> TensorVal {
        go(&self.sched(g, feeds), self.output, &mut HashMap::new()).force()
    }

    /// Replay into a reused buffer (no per-run output allocation).
    pub fn run_into(&self, g: &Graph, feeds: &Feeds, out: &mut Vec<f32>) {
        go(&self.sched(g, feeds), self.output, &mut HashMap::new()).force_into(out);
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}
