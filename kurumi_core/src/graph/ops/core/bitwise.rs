//! Integer/bitwise primitives (integer divide, shifts, and/or/xor).

use crate::{DType, Error, Graph, NodeId, Op, Storage};

impl Graph {
    pub fn idiv(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        if !self.dtype(a).is_int() {
            return Err(Error::shape("idiv", format!("requires integer dtype, got {:?}", self.dtype(a))));
        }
        self.bin("idiv", Op::IDiv, a, b)
    }
    pub fn shl(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        if !self.dtype(a).is_int() {
            return Err(Error::shape("shl", format!("requires integer dtype, got {:?}", self.dtype(a))));
        }
        self.bin("shl", Op::Shl, a, b)
    }
    pub fn shr(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        if !self.dtype(a).is_int() {
            return Err(Error::shape("shr", format!("requires integer dtype, got {:?}", self.dtype(a))));
        }
        self.bin("shr", Op::Shr, a, b)
    }
    pub fn and(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        self.require("and", a, !self.dtype(a).is_float(), "bool/integer")?;
        self.bin("and", Op::And, a, b)
    }
    pub fn or(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        self.require("or", a, !self.dtype(a).is_float(), "bool/integer")?;
        self.bin("or", Op::Or, a, b)
    }
    pub fn xor(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        self.require("xor", a, !self.dtype(a).is_float(), "bool/integer")?;
        self.bin("xor", Op::Xor, a, b)
    }

    /// Bitwise NOT of an integer tensor: `x ^ all_ones` (per-dtype all-ones const).
    pub fn bitwise_not(&mut self, x: NodeId) -> Result<NodeId, Error> {
        let ones = match self.dtype(x) {
            DType::I32 => Storage::I32(vec![-1]),
            DType::I64 => Storage::I64(vec![-1]),
            DType::U8 => Storage::U8(vec![u8::MAX]),
            DType::U32 => Storage::U32(vec![u32::MAX]),
            dt => return Err(Error::shape("bitwise_not", format!("requires an integer dtype, got {dt:?}"))),
        };
        let sh = self.shape(x);
        let c = self.const_storage(ones, vec![1; sh.len()]);
        let cb = self.expand(c, sh)?;
        self.xor(x, cb)
    }
}
