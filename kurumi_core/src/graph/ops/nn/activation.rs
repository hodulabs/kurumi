//! Neural-net activations and the softmax family. Activations decompose; softmax is a fused
//! primitive (`Op::Softmax`).

use crate::{Error, Graph, NodeId, Op};

impl Graph {
    /// sigmoid(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(&mut self, x: NodeId) -> NodeId {
        let nx = self.neg(x);
        let e = self.exp(nx);
        let one = self.scalar(x, 1.0);
        let d = self.add(one, e).expect("scalar shares x's shape");
        self.recip(d)
    }

    /// ReLU: `max(x, 0)`.
    pub fn relu(&mut self, x: NodeId) -> NodeId {
        let z = self.zeros_like(x);
        self.max(x, z).expect("relu: same-shape max")
    }

    /// SiLU / swish: `x * sigmoid(x)`. The SwiGLU gate activation (Llama MLP).
    pub fn silu(&mut self, x: NodeId) -> NodeId {
        let s = self.sigmoid(x);
        self.mul(x, s).expect("silu: same-shape mul is always valid")
    }

    /// GPT-2 gelu: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
    pub fn gelu(&mut self, x: NodeId) -> NodeId {
        let x2 = self.mul(x, x).expect("same shape");
        let x3 = self.mul(x2, x).expect("same shape");
        let c = self.scalar(x, 0.044715);
        let cx3 = self.mul(c, x3).expect("scalar shares x's shape");
        let inner = self.add(x, cx3).expect("same shape");
        let k = self.scalar(x, (2.0_f32 / std::f32::consts::PI).sqrt());
        let scaled = self.mul(k, inner).expect("scalar shares x's shape");
        let t = self.tanh(scaled);
        let one = self.scalar(x, 1.0);
        let onep = self.add(one, t).expect("scalar shares x's shape");
        let half = self.scalar(x, 0.5);
        let hx = self.mul(half, x).expect("scalar shares x's shape");
        self.mul(hx, onep).expect("same shape")
    }

    /// Exact GELU via the error function: `0.5*x*(1 + erf(x/sqrt(2)))`.
    pub fn gelu_erf(&mut self, x: NodeId) -> NodeId {
        let inv_sqrt2 = self.scalar(x, std::f32::consts::FRAC_1_SQRT_2);
        let xs = self.mul(x, inv_sqrt2).expect("same shape");
        let e = self.erf(xs);
        let one = self.scalar(e, 1.0);
        let onep = self.add(one, e).expect("same shape");
        let half = self.scalar(x, 0.5);
        let hx = self.mul(half, x).expect("same shape");
        self.mul(hx, onep).expect("same shape")
    }

    /// Softplus `ln(1 + e^x)`, numerically stable: `relu(x) + ln(1 + e^(-|x|))`.
    pub fn softplus(&mut self, x: NodeId) -> NodeId {
        let r = self.relu(x);
        let a = self.abs(x);
        let na = self.neg(a);
        let e = self.exp(na);
        let one = self.scalar(x, 1.0);
        let onep = self.add(one, e).expect("same shape");
        let l = self.ln(onep);
        self.add(r, l).expect("same shape")
    }

    /// Mish: `x * tanh(softplus(x))`.
    pub fn mish(&mut self, x: NodeId) -> NodeId {
        let sp = self.softplus(x);
        let t = self.tanh(sp);
        self.mul(x, t).expect("same shape")
    }

    /// ELU: `where(x>0, x, alpha(e^x-1))`.
    pub fn elu(&mut self, x: NodeId, alpha: f32) -> NodeId {
        let z = self.zeros_like(x);
        let pos = self.cmp_lt(z, x).expect("same shape"); // x > 0
        let e = self.exp(x);
        let one = self.scalar(x, 1.0);
        let em1 = self.sub(e, one).expect("same shape");
        let a = self.scalar(x, alpha);
        let neg_branch = self.mul(a, em1).expect("same shape");
        self.select(pos, x, neg_branch).expect("same shape")
    }

    /// Leaky ReLU: `where(x>0, x, slope*x)`.
    pub fn leaky_relu(&mut self, x: NodeId, slope: f32) -> NodeId {
        let z = self.zeros_like(x);
        let pos = self.cmp_lt(z, x).expect("same shape");
        let s = self.scalar(x, slope);
        let neg_branch = self.mul(s, x).expect("same shape");
        self.select(pos, x, neg_branch).expect("same shape")
    }

    /// Parametric ReLU: `where(x>0, x, slope*x)` with a tensor (broadcast) `slope`.
    pub fn prelu(&mut self, x: NodeId, slope: NodeId) -> Result<NodeId, Error> {
        let z = self.zeros_like(x);
        let pos = self.cmp_lt(z, x)?;
        let neg_branch = self.mul(slope, x)?;
        self.select(pos, x, neg_branch)
    }

    /// Hard sigmoid: `clamp((x+3)/6, 0, 1)`.
    pub fn hardsigmoid(&mut self, x: NodeId) -> NodeId {
        let three = self.scalar(x, 3.0);
        let s = self.add(x, three).expect("same shape");
        let sixth = self.scalar(x, 1.0 / 6.0);
        let y = self.mul(s, sixth).expect("same shape");
        self.clamp(y, 0.0, 1.0).expect("same shape")
    }

    /// Hard swish: `x * hardsigmoid(x)`.
    pub fn hardswish(&mut self, x: NodeId) -> NodeId {
        let h = self.hardsigmoid(x);
        self.mul(x, h).expect("same shape")
    }

    /// softmax(x, axis) = exp(x - max) / sum(exp(x - max)), numerically stable. A fused
    /// primitive: the backend runs one kernel, the interp oracle computes the decomposed math.
    pub fn softmax(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let rank = self.shape(x).len();
        if axis >= rank {
            return Err(Error::shape("softmax", format!("axis {axis} out of range for rank {rank}")));
        }
        self.require("softmax", x, self.dtype(x).is_float(), "float")?;
        Ok(self.push(Op::Softmax { axis }, vec![x]))
    }

    /// Log-softmax over `axis`: `x - logsumexp(x)`.
    pub fn log_softmax(&mut self, x: NodeId, axis: usize) -> Result<NodeId, Error> {
        let full = self.shape(x);
        let lse = self.logsumexp(x, axis)?;
        let lse_b = self.broadcast_back(lse, &full, axis)?;
        self.sub(x, lse_b)
    }

    /// SELU: `lambda*elu(x, alpha)` with the self-normalizing constants.
    pub fn selu(&mut self, x: NodeId) -> NodeId {
        let e = self.elu(x, 1.673_263_2);
        let lam = self.scalar(e, 1.050_701);
        self.mul(lam, e).expect("selu same shape")
    }

    /// CELU: `max(0,x) + min(0, alpha*(exp(x/alpha) - 1))`.
    pub fn celu(&mut self, x: NodeId, alpha: f32) -> NodeId {
        let r = self.relu(x);
        let inv = self.scalar(x, 1.0 / alpha);
        let xa = self.mul(x, inv).expect("same shape");
        let e = self.exp(xa);
        let one = self.scalar(e, 1.0);
        let em1 = self.sub(e, one).expect("same shape");
        let a = self.scalar(em1, alpha);
        let neg = self.mul(a, em1).expect("same shape");
        let zero = self.zeros_like(neg);
        let m = self.min(neg, zero).expect("same shape");
        self.add(r, m).expect("celu same shape")
    }

    /// Softsign: `x / (1 + |x|)`.
    pub fn softsign(&mut self, x: NodeId) -> NodeId {
        let a = self.abs(x);
        let one = self.scalar(a, 1.0);
        let d = self.add(one, a).expect("same shape");
        self.div(x, d).expect("softsign same shape")
    }
}
