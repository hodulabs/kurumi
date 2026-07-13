//! View-like movement: reshape/permute/expand primitives and the reshape/permute-derived
//! decompositions (transpose, flatten, squeeze/unsqueeze, broadcast).

use crate::{Error, Graph, NodeId, Op};

impl Graph {
    // primitives

    pub fn reshape(&mut self, a: NodeId, shape: Vec<usize>) -> Result<NodeId, Error> {
        let (from, to): (usize, usize) = (self.shape(a).iter().product(), shape.iter().product());
        if from != to {
            return Err(Error::shape("reshape", format!("numel {from} -> {to}")));
        }
        Ok(self.push(Op::Reshape { shape }, vec![a]))
    }

    pub fn permute(&mut self, a: NodeId, perm: Vec<usize>) -> Result<NodeId, Error> {
        let rank = self.shape(a).len();
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        if sorted != (0..rank).collect::<Vec<_>>() {
            return Err(Error::shape("permute", format!("{perm:?} is not a permutation of 0..{rank}")));
        }
        Ok(self.push(Op::Permute { perm }, vec![a]))
    }

    pub fn expand(&mut self, a: NodeId, shape: Vec<usize>) -> Result<NodeId, Error> {
        let from = self.shape(a);
        let ok = from.len() == shape.len() && from.iter().zip(&shape).all(|(&f, &t)| f == t || f == 1);
        if !ok {
            return Err(Error::shape("expand", format!("{from:?} -> {shape:?}")));
        }
        Ok(self.push(Op::Expand { shape }, vec![a]))
    }

    // decompositions

    /// Swap axes `i` and `j` (generalized transpose).
    pub fn transpose(&mut self, x: NodeId, i: usize, j: usize) -> Result<NodeId, Error> {
        let r = self.shape(x).len();
        let mut perm: Vec<usize> = (0..r).collect();
        perm.swap(i, j);
        self.permute(x, perm)
    }

    /// Transpose the last two axes (2D-style `t`).
    pub fn t(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let r = self.shape(x).len();
        self.transpose(x, r - 2, r - 1)
    }

    /// Flatten to 1-D.
    pub fn flatten(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let n: usize = self.shape(x).iter().product();
        self.reshape(x, vec![n])
    }

    /// Remove `axis` if it has size 1 (else no-op).
    pub fn squeeze(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let mut s = self.shape(x);
        if s[axis] == 1 {
            s.remove(axis);
        }
        self.reshape(x, s)
    }

    /// Insert a size-1 axis at `axis`.
    pub fn unsqueeze(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let mut s = self.shape(x);
        s.insert(axis, 1);
        self.reshape(x, s)
    }

    /// Broadcast to `shape` (NumPy rules): prepend size-1 axes, then expand.
    pub fn broadcast_to(&mut self, x: NodeId, shape: Vec<usize>) -> Result<NodeId, Error> {
        let xs = self.shape(x);
        let x = if xs.len() < shape.len() {
            let mut s = vec![1usize; shape.len() - xs.len()];
            s.extend_from_slice(&xs);
            self.reshape(x, s)?
        } else {
            x
        };
        self.expand(x, shape)
    }

    /// Broadcast `x` to `other`'s shape.
    pub fn broadcast_like(&mut self, x: NodeId, other: NodeId) -> Result<NodeId, Error> {
        let shape = self.shape(other);
        self.broadcast_to(x, shape)
    }
}
