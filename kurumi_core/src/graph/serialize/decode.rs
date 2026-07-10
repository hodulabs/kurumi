//! Decoding half: deserialize_graph replays push to rebuild the graph (shape/dtype
//! re-inferred), plus read_op and the byte Reader.
use crate::graph::serialize::{InputBinding, InputRole, MAGIC, Runnable, VERSION};
use crate::graph::{ArgKind, Graph, NodeId, Op, ScatterOp};
use crate::{DType, Error, Storage};

/// Rebuild a graph from a blob by replaying `push` in id order (shape/dtype re-inferred).
/// This is a trust boundary: a corrupt length/tag is a clean error, never a panic. A blob
/// that is structurally valid but semantically bogus (a src referencing a later node) can
/// still panic in inference -- that is the same contract the builder has for live graphs.
pub fn deserialize_graph(bytes: &[u8]) -> Result<Runnable, Error> {
    let mut r = Reader::new(bytes);
    if r.take(4)? != MAGIC {
        return Err(err("bad magic (not a KGPH graph blob)"));
    }
    let ver = r.u8()?;
    if ver != VERSION {
        return Err(err(format!("unsupported graph blob version {ver}")));
    }
    let n = r.u32()? as usize;
    let mut g = Graph::new();
    for _ in 0..n {
        let op = read_op(&mut r)?;
        let sl = r.u32()? as usize;
        let mut src = Vec::new();
        for _ in 0..sl {
            src.push(r.node_id()?);
        }
        g.push(op, src);
    }
    let outputs = r.node_id_vec()?;
    let ni = r.u32()? as usize;
    let mut inputs = Vec::new();
    for _ in 0..ni {
        let node = r.node_id()?;
        let role = match r.u8()? {
            0 => InputRole::Weight,
            1 => InputRole::Runtime,
            x => return Err(err(format!("bad input role {x}"))),
        };
        let name = r.str()?;
        inputs.push(InputBinding { node, role, name });
    }
    Ok(Runnable { graph: g, outputs, inputs })
}

pub(super) fn read_op(r: &mut Reader) -> Result<Op, Error> {
    let tag = r.u8()?;
    Ok(match tag {
        0 => Op::Const { data: r.storage()?, shape: r.vec_usize()? },
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

// reader

pub(super) struct Reader<'a> {
    buf: &'a [u8],
    pub(super) pos: usize,
}

impl<'a> Reader<'a> {
    pub(super) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        match end {
            Some(end) => {
                let s = &self.buf[self.pos..end];
                self.pos = end;
                Ok(s)
            }
            None => Err(err("unexpected end of graph blob")),
        }
    }
    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn usize(&mut self) -> Result<usize, Error> {
        Ok(self.u64()? as usize)
    }
    fn f32(&mut self) -> Result<f32, Error> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn bool(&mut self) -> Result<bool, Error> {
        Ok(self.u8()? != 0)
    }
    fn node_id(&mut self) -> Result<NodeId, Error> {
        Ok(NodeId(self.u32()?))
    }
    fn vec_usize(&mut self) -> Result<Vec<usize>, Error> {
        let n = self.u32()? as usize;
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(self.usize()?);
        }
        Ok(v)
    }
    fn node_id_vec(&mut self) -> Result<Vec<NodeId>, Error> {
        let n = self.u32()? as usize;
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(self.node_id()?);
        }
        Ok(v)
    }
    fn triples(&mut self) -> Result<Vec<(usize, usize, usize)>, Error> {
        let n = self.u32()? as usize;
        let mut v = Vec::new();
        for _ in 0..n {
            v.push((self.usize()?, self.usize()?, self.usize()?));
        }
        Ok(v)
    }
    fn pairs(&mut self) -> Result<Vec<(usize, usize)>, Error> {
        let n = self.u32()? as usize;
        let mut v = Vec::new();
        for _ in 0..n {
            v.push((self.usize()?, self.usize()?));
        }
        Ok(v)
    }
    fn str(&mut self) -> Result<String, Error> {
        let n = self.u32()? as usize;
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| err("invalid utf8 in name"))
    }
    fn dtype(&mut self) -> Result<DType, Error> {
        Ok(match self.u8()? {
            0 => DType::BOOL,
            1 => DType::U8,
            2 => DType::U16,
            3 => DType::U32,
            4 => DType::U64,
            5 => DType::I8,
            6 => DType::I16,
            7 => DType::I32,
            8 => DType::I64,
            9 => DType::F8E4M3,
            10 => DType::F8E5M2,
            11 => DType::F16,
            12 => DType::BF16,
            13 => DType::F32,
            14 => DType::F64,
            15 => DType::C64,
            16 => DType::C128,
            x => return Err(err(format!("unknown dtype tag {x}"))),
        })
    }
    fn storage(&mut self) -> Result<Storage, Error> {
        let dt = self.dtype()?;
        let nbytes = self.u64()? as usize;
        Ok(Storage::from_bytes(dt, self.take(nbytes)?))
    }
    fn scatter(&mut self) -> Result<ScatterOp, Error> {
        Ok(match self.u8()? {
            0 => ScatterOp::Set,
            1 => ScatterOp::Add,
            2 => ScatterOp::Max,
            3 => ScatterOp::Min,
            x => return Err(err(format!("unknown scatter op {x}"))),
        })
    }
    fn argkind(&mut self) -> Result<ArgKind, Error> {
        Ok(match self.u8()? {
            0 => ArgKind::Max,
            1 => ArgKind::Min,
            x => return Err(err(format!("unknown argkind {x}"))),
        })
    }
}

fn err(msg: impl Into<String>) -> Error {
    Error::shape("deserialize_graph", msg)
}
