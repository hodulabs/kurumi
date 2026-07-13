//! The exhaustive per-Op decoder: read a variant tag then its attrs, rebuilding one `Op`. The tag
//! numbering mirrors the encode side (`encode/op.rs`); an unknown tag is a clean error, not a panic
//! (this is a trust boundary -- see `decode.rs`). The byte `Reader` reused below lives in the
//! parent `decode.rs`.

use super::{Reader, err};
use crate::Error;
use crate::graph::Op;

pub(crate) fn read_op(r: &mut Reader) -> Result<Op, Error> {
    let tag = r.u8()?;
    Ok(match tag {
        0 => {
            // `storage()` reads `nbytes` bytes and `from_bytes` drops a trailing partial element
            // (chunks_exact), so a corrupt Const can yield a storage shorter than its shape --
            // downstream OOB. Reject any payload whose element count != the declared shape.
            let data = r.storage()?;
            let shape = r.vec_usize()?;
            let want: usize = shape.iter().product();
            if data.len() != want {
                return Err(err(format!("Const payload has {} elements, shape {shape:?} needs {want}", data.len())));
            }
            Op::Const { data, shape }
        }
        1 => Op::Input { shape: r.vec_usize()?, dtype: r.dtype()? },
        2 => Op::Iota { shape: r.vec_usize()?, axis: r.usize()?, dtype: r.dtype()? },
        3 => Op::RandUniform { shape: r.vec_usize()? },
        4 => Op::Cast { to: r.dtype()? },
        5 => Op::Bitcast { to: r.dtype()? },
        6 => Op::Detach,
        7 => Op::Add,
        8 => Op::Mul,
        9 => Op::Max,
        10 => Op::Neg,
        11 => Op::IDiv,
        12 => Op::And,
        13 => Op::Or,
        14 => Op::Xor,
        15 => Op::Shl,
        16 => Op::Shr,
        17 => Op::CmpLt,
        18 => Op::CmpEq,
        19 => Op::Where,
        20 => Op::Recip,
        21 => Op::Sqrt,
        22 => Op::Exp2,
        23 => Op::Log2,
        24 => Op::Sin,
        25 => Op::Floor,
        26 => Op::Sum { axis: r.usize()? },
        27 => Op::Prod { axis: r.usize()? },
        28 => Op::ReduceMax { axis: r.usize()? },
        29 => Op::ArgReduce { axis: r.usize()?, kind: r.argkind()? },
        30 => Op::Softmax { axis: r.usize()? },
        31 => Op::RmsNorm { axis: r.usize()?, eps: r.f32()? },
        32 => Op::Sdpa { causal: r.bool()? },
        33 => Op::Reshape { shape: r.vec_usize()? },
        34 => Op::Permute { perm: r.vec_usize()? },
        35 => Op::Expand { shape: r.vec_usize()? },
        36 => Op::Slice { ranges: r.triples()? },
        37 => Op::Flip { axes: r.vec_usize()? },
        38 => Op::Pad { pads: r.pairs()? },
        39 => Op::DotGeneral {
            lhs_contract: r.vec_usize()?,
            rhs_contract: r.vec_usize()?,
            lhs_batch: r.vec_usize()?,
            rhs_batch: r.vec_usize()?,
        },
        40 => Op::QuantMatmul { bits: r.u8()?, group_size: r.usize()?, symmetric: r.bool()? },
        41 => Op::Solve,
        42 => Op::Det,
        43 => Op::Cholesky,
        44 => Op::Eigh,
        45 => Op::Qr { r_factor: r.bool()? },
        46 => Op::Eigvals,
        47 => Op::Complex,
        48 => Op::Real,
        49 => Op::Imag,
        50 => Op::Gather { axis: r.usize()? },
        51 => Op::Scatter { axis: r.usize()?, combine: r.scatter()? },
        52 => Op::GatherAlong { axis: r.usize()? },
        53 => Op::ScatterAlong { axis: r.usize()?, combine: r.scatter()? },
        54 => Op::Argsort { axis: r.usize()?, descending: r.bool()? },
        _ => return Err(err(format!("unknown op tag {tag}"))),
    })
}
