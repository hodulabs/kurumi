//! Einstein summation front-end: decomposes to diagonal + reduce + `dot_general` + permute
//! (so autodiff & every backend get it for free). The `dot_general` primitive it lowers to
//! lives in `contract.rs`. Equation parsing is in `einsum/parse.rs`, the contraction builders
//! in `einsum/contract.rs`.

use crate::{Error, Graph, NodeId};

mod contract;
mod parse;

impl Graph {
    /// Einstein summation: `einsum("ij,jk->ik", &[a, b])` etc. Decomposes to diagonal +
    /// reduce + `dot_general` + permute (autodiff & every backend for free). Any operand
    /// count, implicit/explicit output, repeated indices (diagonal/trace), `...` ellipsis (batch dims).
    pub fn einsum(&mut self, equation: &str, operands: &[NodeId]) -> Result<NodeId, Error> {
        let expanded;
        let equation = if equation.contains("...") {
            expanded = self.expand_ellipsis(equation, operands)?;
            expanded.as_str()
        } else {
            equation
        };
        let (ins, out) = match equation.split_once("->") {
            Some((l, r)) => (l, Some(r)),
            None => (equation, None),
        };
        let in_subs: Vec<Vec<char>> = ins.split(',').map(|s| s.trim().chars().collect()).collect();
        if in_subs.len() != operands.len() || operands.is_empty() {
            return Err(Error::shape("einsum", "operand count != subscript groups"));
        }
        // implicit output: indices appearing exactly once across the ORIGINAL inputs
        // (before diagonal collapse), sorted, so "ii" -> trace, "ij,jk" -> "ik".
        let out_subs: Vec<char> = match out {
            Some(s) => s.trim().chars().collect(),
            None => {
                let mut once: Vec<char> = Vec::new();
                let all: Vec<char> = in_subs.iter().flatten().copied().collect();
                for &c in &all {
                    if all.iter().filter(|&&x| x == c).count() == 1 {
                        once.push(c);
                    }
                }
                once.sort_unstable();
                once
            }
        };
        // collapse repeated indices within each operand to a single occurrence
        let mut items: Vec<(Vec<char>, NodeId)> = Vec::with_capacity(operands.len());
        for (sub, &node) in in_subs.iter().zip(operands) {
            let (n, s) = self.einsum_diag(node, sub.clone())?;
            items.push((s, n));
        }
        if items.len() == 1 {
            let (s, n) = items.pop().unwrap();
            return self.einsum_unary(&s, &out_subs, n);
        }
        self.einsum_multi(items, &out_subs)
    }
}
