//! Einstein summation front-end: decomposes to diagonal + reduce + `dot_general` + permute
//! (so autodiff & every backend get it for free). The `dot_general` primitive it lowers to
//! lives in `contract.rs`.

use crate::{Error, Graph, NodeId};

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

    // Rewrite `...` into fresh explicit letters (batch dims, right-aligned across
    // operands) and make the output explicit, so the core einsum never sees ellipsis.
    fn expand_ellipsis(&self, equation: &str, operands: &[NodeId]) -> Result<String, Error> {
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

    // Collapse every repeated index in one operand into a single axis (diagonal).
    fn einsum_diag(&mut self, mut node: NodeId, mut cs: Vec<char>) -> Result<(NodeId, Vec<char>), Error> {
        while let Some(c) = cs.iter().find(|&&c| cs.iter().filter(|&&x| x == c).count() >= 2).copied() {
            let r = cs.len();
            let p1 = cs.iter().position(|&x| x == c).unwrap();
            let p2 = cs.iter().rposition(|&x| x == c).unwrap();
            if p2 != r - 1 {
                node = self.transpose(node, p2, r - 1)?;
                cs.swap(p2, r - 1);
            }
            if p1 != r - 2 {
                node = self.transpose(node, p1, r - 2)?;
                cs.swap(p1, r - 2);
            }
            let sh = self.shape(node);
            if sh[r - 2] != sh[r - 1] {
                return Err(Error::shape("einsum", "repeated index has mismatched sizes"));
            }
            let n = sh[r - 1];
            let b: usize = sh[..r - 2].iter().product::<usize>().max(1);
            let flat = self.reshape(node, vec![b, n * n])?;
            let diag = self.slice_step(flat, vec![(0, b, 1), (0, (n - 1) * (n + 1) + 1, n + 1)])?;
            let mut ns: Vec<usize> = sh[..r - 2].to_vec();
            ns.push(n);
            node = self.reshape(diag, ns)?;
            cs.truncate(r - 2);
            cs.push(c);
        }
        Ok((node, cs))
    }

    // Fold N operands pairwise via einsum_binary. Each fold contracts only indices
    // not needed downstream (absent from later operands and the output).
    fn einsum_multi(&mut self, mut items: Vec<(Vec<char>, NodeId)>, out: &[char]) -> Result<NodeId, Error> {
        while items.len() > 1 {
            let (sa, a) = items.remove(0);
            let (sb, b) = items.remove(0);
            let target: Vec<char> = if items.is_empty() {
                out.to_vec()
            } else {
                // keep indices still live: in the output or any remaining operand
                let mut keep: Vec<char> = out.to_vec();
                for (s, _) in &items {
                    keep.extend(s);
                }
                let mut inter = Vec::new();
                for &c in sa.iter().chain(sb.iter()) {
                    if keep.contains(&c) && !inter.contains(&c) {
                        inter.push(c);
                    }
                }
                inter
            };
            let node = self.einsum_binary(&sa, &sb, a, b, &target)?;
            items.push((target, node));
        }
        Ok(items.pop().unwrap().1)
    }

    fn einsum_unary(&mut self, subs: &[char], out: &[char], x: NodeId) -> Result<NodeId, Error> {
        let mut cur = x;
        let mut cs = subs.to_vec();
        // sum out axes whose index isn't in the output (descending keeps axes valid)
        for d in (0..cs.len()).rev() {
            if !out.contains(&cs[d]) {
                cur = self.sum(cur, d)?;
                cs.remove(d);
            }
        }
        let perm: Vec<usize> = out.iter().map(|c| cs.iter().position(|x| x == c).unwrap()).collect();
        self.permute(cur, perm)
    }

    fn einsum_binary(
        &mut self,
        ls: &[char],
        rs: &[char],
        lhs: NodeId,
        rhs: NodeId,
        out: &[char],
    ) -> Result<NodeId, Error> {
        // 1. sum out indices that appear in only one operand and not the output
        let (mut lhs, mut ls) = (lhs, ls.to_vec());
        for d in (0..ls.len()).rev() {
            if !rs.contains(&ls[d]) && !out.contains(&ls[d]) {
                lhs = self.sum(lhs, d)?;
                ls.remove(d);
            }
        }
        let (mut rhs, mut rs) = (rhs, rs.to_vec());
        for d in (0..rs.len()).rev() {
            if !ls.contains(&rs[d]) && !out.contains(&rs[d]) {
                rhs = self.sum(rhs, d)?;
                rs.remove(d);
            }
        }
        // 2. shared indices -> batch (in output) or contract (not in output)
        let batch: Vec<char> = ls.iter().filter(|c| rs.contains(c) && out.contains(c)).copied().collect();
        let contract: Vec<char> = ls.iter().filter(|c| rs.contains(c) && !out.contains(c)).copied().collect();
        let pos = |subs: &[char], cs: &[char]| -> Vec<usize> {
            cs.iter().map(|c| subs.iter().position(|x| x == c).unwrap()).collect()
        };
        let y =
            self.dot_general(lhs, rhs, pos(&ls, &contract), pos(&rs, &contract), pos(&ls, &batch), pos(&rs, &batch))?;
        // 3. dot_general lays out [batch ++ lhs_free ++ rhs_free]; permute to `out`
        let mut y_subs = batch.clone();
        y_subs.extend(ls.iter().filter(|c| !batch.contains(c) && !contract.contains(c)));
        y_subs.extend(rs.iter().filter(|c| !batch.contains(c) && !contract.contains(c)));
        let perm: Vec<usize> = out.iter().map(|c| y_subs.iter().position(|x| x == c).unwrap()).collect();
        self.permute(y, perm)
    }
}
