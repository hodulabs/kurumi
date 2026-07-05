//! Pass inspector: reachable set / topological order of a subgraph, a human dump, and
//! node counts -- for seeing what a rewrite pass did (diff the dump or the count
//! before/after).

use crate::graph::{Graph, NodeId, Op};
use std::collections::HashSet;
use std::fmt::Write;

/// Nodes reachable from `root` in topological order (each node after its sources).
pub fn reachable(g: &Graph, root: NodeId) -> Vec<NodeId> {
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    // iterative post-order: push (id, expanded); on the second visit, emit.
    let mut stack = vec![(root, false)];
    while let Some((id, expanded)) = stack.pop() {
        if expanded {
            order.push(id);
            continue;
        }
        if !seen.insert(id) {
            continue;
        }
        stack.push((id, true));
        for &s in &g.node(id).src {
            if !seen.contains(&s) {
                stack.push((s, false));
            }
        }
    }
    order
}

/// Number of distinct nodes reachable from `root` (the size metric a rewrite pass shrinks).
pub fn node_count(g: &Graph, root: NodeId) -> usize {
    reachable(g, root).len()
}

/// One line per reachable node, topological: `%id = Op(src, ..) : shape`.
pub fn dump(g: &Graph, root: NodeId) -> String {
    let mut s = String::new();
    for id in reachable(g, root) {
        let n = g.node(id);
        let srcs: Vec<String> = n.src.iter().map(|s| format!("%{}", s.0)).collect();
        let _ = writeln!(s, "%{} = {}({}) : {:?}", id.0, label(&n.op), srcs.join(", "), n.shape);
    }
    s
}

// short op label: keeps a Const's data out of the dump.
fn label(op: &Op) -> String {
    match op {
        Op::Const { shape, .. } => format!("Const{shape:?}"),
        Op::Input { shape, .. } => format!("Input{shape:?}"),
        other => format!("{other:?}"),
    }
}
