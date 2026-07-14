//! einsum contraction builders: diagonal collapse, pairwise fold, and unary/binary
//! lowering to `dot_general` + permute.

use crate::{Error, Graph, NodeId};

impl Graph {
    // Collapse every repeated index in one operand into a single axis (diagonal).
    pub(super) fn einsum_diag(&mut self, mut node: NodeId, mut cs: Vec<char>) -> Result<(NodeId, Vec<char>), Error> {
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
    pub(super) fn einsum_multi(&mut self, mut items: Vec<(Vec<char>, NodeId)>, out: &[char]) -> Result<NodeId, Error> {
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

    pub(super) fn einsum_unary(&mut self, subs: &[char], out: &[char], x: NodeId) -> Result<NodeId, Error> {
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
