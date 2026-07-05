//! Graph rewrite pass: local algebraic + movement simplification and CSE, applied
//! bottom-up (sources simplified before consumers, so rewrites cascade). Every rule
//! preserves interpreter semantics, so a rewritten graph evaluates identically with
//! fewer nodes (tests check that against the oracle). It mainly earns its keep on
//! autograd output (grad seeds ones/zeros and stacks transposes/reshapes). New nodes are
//! appended to the arena; the old ones stay but drop out of the new root's reachable set.

use crate::Storage;
use crate::graph::inspect::reachable;
use crate::graph::{Graph, NodeId, Op};
use std::collections::HashMap;

/// Simplify the subgraph rooted at `root` to a fixpoint, returning the new root.
pub fn simplify(g: &mut Graph, root: NodeId) -> NodeId {
    let mut root = root;
    let mut prev = usize::MAX;
    // rules only remove/merge nodes, so the count is monotone: iterate until it
    // stops dropping (composed movements can expose a fresh identity). Capped anyway.
    for _ in 0..8 {
        root = simplify_once(g, root);
        let n = reachable(g, root).len();
        if n >= prev {
            break;
        }
        prev = n;
    }
    root
}

fn simplify_once(g: &mut Graph, root: NodeId) -> NodeId {
    let mut remap: HashMap<NodeId, NodeId> = HashMap::new();
    let mut cse: HashMap<String, NodeId> = HashMap::new();
    for id in reachable(g, root) {
        let new = rewrite(g, id, &remap, &mut cse);
        remap.insert(id, new);
    }
    remap[&root]
}

fn rewrite(g: &mut Graph, id: NodeId, remap: &HashMap<NodeId, NodeId>, cse: &mut HashMap<String, NodeId>) -> NodeId {
    let old_src = g.node(id).src.clone();
    if old_src.is_empty() {
        return id; // Const / Input / Iota leaf: unchanged
    }
    let src: Vec<NodeId> = old_src.iter().map(|s| remap[s]).collect();
    let op = g.node(id).op.clone(); // leaves handled above, so no big Const data here

    if let Some(r) = rule(g, &op, &src) {
        return r;
    }
    // CSE: an identical (op, src) already built this pass? (pure ops -> safe to share)
    let key = format!("{op:?}|{src:?}");
    if let Some(&existing) = cse.get(&key) {
        return existing;
    }
    let new = if src == old_src { id } else { g.push(op, src) };
    cse.insert(key, new);
    new
}

// one local, value-preserving rewrite; `Some(r)` replaces the node with `r`.
fn rule(g: &mut Graph, op: &Op, src: &[NodeId]) -> Option<NodeId> {
    match op {
        // double inverse
        Op::Neg if matches!(g.node(src[0]).op, Op::Neg) => Some(g.node(src[0]).src[0]),
        Op::Recip if matches!(g.node(src[0]).op, Op::Recip) => Some(g.node(src[0]).src[0]),
        // multiplicative / additive identity (grad seed = ones; accumulation adds zeros)
        Op::Mul if is_all(g, src[1], 1.0) => Some(src[0]),
        Op::Mul if is_all(g, src[0], 1.0) => Some(src[1]),
        Op::Add if is_all(g, src[1], 0.0) => Some(src[0]),
        Op::Add if is_all(g, src[0], 0.0) => Some(src[1]),
        // reshape to the same shape = identity; reshape after reshape = one reshape
        Op::Reshape { shape } if g.node(src[0]).shape == *shape => Some(src[0]),
        Op::Reshape { shape } => {
            let inner = match g.node(src[0]).op {
                Op::Reshape { .. } => Some(g.node(src[0]).src[0]),
                _ => None,
            }?;
            Some(g.push(Op::Reshape { shape: shape.clone() }, vec![inner]))
        }
        // identity permute = drop; permute after permute = one composed permute
        Op::Permute { perm } if perm.iter().copied().eq(0..perm.len()) => Some(src[0]),
        Op::Permute { perm } => {
            let (inner_perm, inner) = match &g.node(src[0]).op {
                Op::Permute { perm: p } => (p.clone(), g.node(src[0]).src[0]),
                _ => return None,
            };
            // out axis i reads (permute p then perm) = inner axis inner_perm[perm[i]]
            let composed: Vec<usize> = perm.iter().map(|&i| inner_perm[i]).collect();
            Some(g.push(Op::Permute { perm: composed }, vec![inner]))
        }
        _ => None,
    }
}

fn is_all(g: &Graph, id: NodeId, v: f32) -> bool {
    matches!(&g.node(id).op, Op::Const { data: Storage::F32(d), .. } if d.iter().all(|&x| x == v))
}

#[cfg(test)]
mod tests;
