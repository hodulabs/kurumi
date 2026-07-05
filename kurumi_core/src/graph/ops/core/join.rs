//! Join & split along an axis: concat/stack (pad + add) and split (slices).

use crate::{Error, Graph, NodeId};

impl Graph {
    /// Concatenate `parts` along `axis`. Each part is zero-padded to the full extent
    /// at its offset and summed (0 is the additive identity, so the regions don't
    /// overlap). Works for any numeric/bool dtype.
    pub fn concat(&mut self, parts: &[NodeId], axis: usize) -> Result<NodeId, Error> {
        let lens: Vec<usize> = parts.iter().map(|&p| self.shape(p)[axis]).collect();
        let total: usize = lens.iter().sum();
        let mut offset = 0usize;
        let mut acc: Option<NodeId> = None;
        for (i, &p) in parts.iter().enumerate() {
            let r = self.shape(p).len();
            let mut pads = vec![(0, 0); r];
            pads[axis] = (offset, total - offset - lens[i]);
            let padded = self.pad(p, pads)?;
            acc = Some(match acc {
                None => padded,
                Some(a) => self.add(a, padded)?,
            });
            offset += lens[i];
        }
        acc.ok_or_else(|| Error::shape("concat", "needs at least one part"))
    }

    /// Stack `parts` on a new axis `axis` (unsqueeze each, then concat).
    pub fn stack(&mut self, parts: &[NodeId], axis: usize) -> Result<NodeId, Error> {
        let un: Vec<NodeId> = parts.iter().map(|&p| self.unsqueeze(p, axis)).collect::<Result<_, _>>()?;
        self.concat(&un, axis)
    }

    /// Split `x` along `axis` into chunks of the given `sizes` (slices).
    pub fn split(&mut self, x: NodeId, sizes: &[usize], axis: usize) -> Result<Vec<NodeId>, Error> {
        let sh = self.shape(x);
        let mut out = Vec::with_capacity(sizes.len());
        let mut start = 0usize;
        for &sz in sizes {
            let mut ranges: Vec<(usize, usize)> = sh.iter().map(|&d| (0, d)).collect();
            ranges[axis] = (start, start + sz);
            out.push(self.slice(x, ranges)?);
            start += sz;
        }
        Ok(out)
    }
}
