//! einsum equation parsing: `...` ellipsis expansion into explicit letters.

use crate::{Error, Graph, NodeId};

impl Graph {
    // Rewrite `...` into fresh explicit letters (batch dims, right-aligned across
    // operands) and make the output explicit, so the core einsum never sees ellipsis.
    pub(super) fn expand_ellipsis(&self, equation: &str, operands: &[NodeId]) -> Result<String, Error> {
        let (ins, out) = match equation.split_once("->") {
            Some((l, r)) => (l, Some(r)),
            None => (equation, None),
        };
        let parts: Vec<&str> = ins.split(',').map(|s| s.trim()).collect();
        if parts.len() != operands.len() {
            return Err(Error::shape("einsum", "operand count != subscript groups"));
        }
        let explicit = |p: &str| p.chars().filter(|c| c.is_ascii_alphabetic()).count();
        // widest ellipsis across operands
        let mut ell = 0usize;
        for (p, &node) in parts.iter().zip(operands) {
            if p.contains("...") {
                let rank = self.shape(node).len();
                if rank < explicit(p) {
                    return Err(Error::shape("einsum", "operand rank < its explicit subscripts"));
                }
                ell = ell.max(rank - explicit(p));
            }
        }
        // fresh letters for the batch dims
        let used: std::collections::HashSet<char> = equation.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        let batch: Vec<char> = ('A'..='Z').chain('a'..='z').filter(|c| !used.contains(c)).take(ell).collect();
        if batch.len() < ell {
            return Err(Error::shape("einsum", "ran out of letters for ellipsis"));
        }
        let batch_str: String = batch.iter().collect();
        // expand each input part (right-aligned suffix of `batch`)
        let new_ins: Vec<String> = parts
            .iter()
            .zip(operands)
            .map(|(p, &node)| {
                if p.contains("...") {
                    let er = self.shape(node).len() - explicit(p);
                    let suffix: String = batch[ell - er..].iter().collect();
                    p.replace("...", &suffix)
                } else {
                    p.to_string()
                }
            })
            .collect();
        let new_out = match out {
            Some(o) => o.trim().replace("...", &batch_str),
            None => {
                // implicit: batch dims first, then explicit letters appearing exactly once
                let all: Vec<char> = parts.iter().flat_map(|p| p.chars().filter(|c| c.is_ascii_alphabetic())).collect();
                let mut once: Vec<char> =
                    all.iter().filter(|&&c| all.iter().filter(|&&x| x == c).count() == 1).copied().collect();
                once.sort_unstable();
                once.dedup();
                format!("{batch_str}{}", once.iter().collect::<String>())
            }
        };
        Ok(format!("{}->{}", new_ins.join(","), new_out))
    }
}
