//! Loss functions (elementwise / per-sample, unreduced: the caller reduces with
//! mean/sum). Pure decompositions, numerically-stable variants where it matters.

use crate::{Error, Graph, NodeId};

impl Graph {
    /// Squared error `(pred - target)^2` (elementwise).
    pub fn mse_loss(&mut self, pred: NodeId, target: NodeId) -> Result<NodeId, Error> {
        let d = self.sub(pred, target)?;
        Ok(self.square(d))
    }

    /// Absolute error `|pred - target|` (elementwise).
    pub fn l1_loss(&mut self, pred: NodeId, target: NodeId) -> Result<NodeId, Error> {
        let d = self.sub(pred, target)?;
        Ok(self.abs(d))
    }

    /// Huber loss (elementwise): `0.5*d^2` for `|d| < delta`, else `delta*(|d| - 0.5*delta)`.
    pub fn huber_loss(&mut self, pred: NodeId, target: NodeId, delta: f32) -> Result<NodeId, Error> {
        let d = self.sub(pred, target)?;
        let ad = self.abs(d);
        let dc = self.scalar(ad, delta);
        let small = self.cmp_lt(ad, dc)?;
        let sq = self.square(d);
        let half = self.scalar(sq, 0.5);
        let quad = self.mul(sq, half)?;
        // delta*(|d| - 0.5*delta)
        let off = self.scalar(ad, 0.5 * delta);
        let shifted = self.sub(ad, off)?;
        let dl = self.scalar(shifted, delta);
        let lin = self.mul(shifted, dl)?;
        self.select(small, quad, lin)
    }

    /// Binary cross-entropy `-(t*log(p) + (1-t)*log(1-p))` (elementwise; `p in (0,1)`).
    /// Uses `xlogy` so `t=0`/`t=1` endpoints don't produce NaN.
    pub fn bce_loss(&mut self, pred: NodeId, target: NodeId) -> Result<NodeId, Error> {
        let one = self.ones_like(pred);
        let omt = self.sub(one, target)?;
        let omp = self.sub(one, pred)?;
        let a = self.xlogy(target, pred)?;
        let b = self.xlogy(omt, omp)?;
        let s = self.add(a, b)?;
        Ok(self.neg(s))
    }

    /// Binary cross-entropy from logits (overflow-safe):
    /// `-(t*log sigma(x) + (1-t)*log sigma(-x))` via `log_sigmoid`.
    pub fn bce_with_logits(&mut self, logits: NodeId, target: NodeId) -> Result<NodeId, Error> {
        let one = self.ones_like(logits);
        let omt = self.sub(one, target)?;
        let nx = self.neg(logits);
        let lsp = self.log_sigmoid(logits)?;
        let lsn = self.log_sigmoid(nx)?;
        let a = self.mul(target, lsp)?;
        let b = self.mul(omt, lsn)?;
        let s = self.add(a, b)?;
        Ok(self.neg(s))
    }

    /// KL divergence `p*(log p - log q)` (elementwise; `xlogy` handles `p=0`).
    pub fn kl_div(&mut self, p: NodeId, q: NodeId) -> Result<NodeId, Error> {
        let pp = self.xlogy(p, p)?;
        let pq = self.xlogy(p, q)?;
        self.sub(pp, pq)
    }

    /// Negative log-likelihood `-sum target*log_probs` over `axis` (per-sample).
    /// `log_probs` are log-probabilities (e.g. from `log_softmax`), `target` one-hot.
    pub fn nll_loss(&mut self, log_probs: NodeId, target: NodeId, axis: usize) -> Result<NodeId, Error> {
        let prod = self.mul(target, log_probs)?;
        let s = self.sum(prod, axis)?;
        Ok(self.neg(s))
    }

    /// Hinge loss `max(0, 1 - pred*target)` (elementwise; `target in {-1, +1}`).
    pub fn hinge_loss(&mut self, pred: NodeId, target: NodeId) -> Result<NodeId, Error> {
        let pt = self.mul(pred, target)?;
        let one = self.ones_like(pt);
        let m = self.sub(one, pt)?;
        self.clamp_min(m, 0.0)
    }

    /// Cross-entropy over `axis` (the class dim): `-sum(targets * log_softmax(logits))`,
    /// numerically stable (log-sum-exp). `targets` are probabilities (one-hot for hard
    /// labels). Returns the per-example loss (axis reduced). All pointwise+reduce.
    pub fn cross_entropy(&mut self, logits: NodeId, targets: NodeId, axis: usize) -> Result<NodeId, Error> {
        let full = self.shape(logits);
        let m = self.reduce_max(logits, axis)?;
        let m_b = self.broadcast_back(m, &full, axis)?;
        let shifted = self.sub(logits, m_b)?; // logits - max
        let e = self.exp(shifted);
        let se = self.sum(e, axis)?;
        let lse = self.ln(se); // log-sum-exp of the shifted logits
        let lse_b = self.broadcast_back(lse, &full, axis)?;
        let logsm = self.sub(shifted, lse_b)?; // log_softmax = shifted - lse
        let prod = self.mul(targets, logsm)?;
        let s = self.sum(prod, axis)?;
        Ok(self.neg(s))
    }
}
