//! VJP rules for the real<->complex seam (Complex/Real/Imag). Real-pair Cauchy-Riemann:
//! a complex node's cotangent is dL/dre + i*dL/dim, so these arms carry no conjugation
//! (the holomorphic conj lives in `cfactor`, applied by mul/recip/dot_general).

use crate::grad::acc;
use crate::{Error, Graph, NodeId};
use std::collections::HashMap;

/// Complex(re, im): grad_re = real(ct), grad_im = imag(ct).
pub(super) fn complex_vjp(
    g: &mut Graph,
    s: &[NodeId],
    ct: NodeId,
    cot: &mut HashMap<NodeId, NodeId>,
) -> Result<(), Error> {
    let gre = g.real(ct)?;
    acc(g, cot, s[0], gre)?;
    let gim = g.imag(ct)?;
    acc(g, cot, s[1], gim)
}

/// Real(z): grad_z = complex(ct, 0).
pub(super) fn real_vjp(
    g: &mut Graph,
    s: &[NodeId],
    ct: NodeId,
    cot: &mut HashMap<NodeId, NodeId>,
) -> Result<(), Error> {
    let zero = g.zeros_like(ct);
    let gz = g.complex(ct, zero)?;
    acc(g, cot, s[0], gz)
}

/// Imag(z): grad_z = complex(0, ct).
pub(super) fn imag_vjp(
    g: &mut Graph,
    s: &[NodeId],
    ct: NodeId,
    cot: &mut HashMap<NodeId, NodeId>,
) -> Result<(), Error> {
    let zero = g.zeros_like(ct);
    let gz = g.complex(zero, ct)?;
    acc(g, cot, s[0], gz)
}
