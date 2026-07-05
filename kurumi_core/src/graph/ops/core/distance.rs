//! Pairwise distances & similarity: cdist / pdist (Lp) and cosine similarity. Pure
//! decompositions over broadcast + norm_p -> autodiff & every backend.

use crate::{Error, Graph, NodeId};

impl Graph {
    /// Pairwise Lp distance between rows of `a = [m, d]` and `b = [n, d]` -> `[m, n]`.
    pub fn cdist(&mut self, a: NodeId, b: NodeId, p: f32) -> Result<NodeId, Error> {
        let (sa, sb) = (self.shape(a), self.shape(b));
        if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[1] {
            return Err(Error::shape("cdist", "expects a=[m,d], b=[n,d] with matching d"));
        }
        let (m, n, d) = (sa[0], sb[0], sa[1]);
        let ar = self.reshape(a, vec![m, 1, d])?;
        let br = self.reshape(b, vec![1, n, d])?;
        let ab = self.broadcast_to(ar, vec![m, n, d])?;
        let bb = self.broadcast_to(br, vec![m, n, d])?;
        let diff = self.sub(ab, bb)?;
        self.norm_p(diff, p, 2) // Lp over the feature axis -> [m, n]
    }

    /// Pairwise Lp self-distance matrix of `a = [m, d]` -> `[m, m]` (diagonal 0).
    pub fn pdist(&mut self, a: NodeId, p: f32) -> Result<NodeId, Error> {
        self.cdist(a, a, p)
    }

    /// Cosine similarity along `axis`: `(a*b) / (||a||*||b||)`.
    pub fn cosine_similarity(&mut self, a: NodeId, b: NodeId, axis: usize) -> Result<NodeId, Error> {
        let prod = self.mul(a, b)?;
        let dot = self.sum(prod, axis)?;
        let na = self.norm_p(a, 2.0, axis)?;
        let nb = self.norm_p(b, 2.0, axis)?;
        let denom = self.mul(na, nb)?;
        self.div(dot, denom)
    }
}
