//! Complex construction and part extraction (C64/C128). conj/cabs decompose from
//! the Complex/Real/Imag primitives; complex arithmetic reuses add/mul/neg/recip.

use crate::{DType, Error, Graph, NodeId, Op};

impl Graph {
    // primitives

    /// Build a complex tensor from real + imaginary parts (F32 -> C64, F64 -> C128).
    pub fn complex(&mut self, re: NodeId, im: NodeId) -> Result<NodeId, Error> {
        self.same_shape("complex", re, im)?;
        self.same_dtype("complex", re, im)?;
        let dt = self.dtype(re);
        if dt != DType::F32 && dt != DType::F64 {
            return Err(Error::shape("complex", "parts must be F32 or F64"));
        }
        Ok(self.push(Op::Complex, vec![re, im]))
    }

    /// Real part of a complex tensor (C64 -> F32, C128 -> F64).
    pub fn real(&mut self, z: NodeId) -> Result<NodeId, Error> {
        self.require("real", z, self.dtype(z).is_complex(), "complex")?;
        Ok(self.push(Op::Real, vec![z]))
    }

    /// Imaginary part of a complex tensor.
    pub fn imag(&mut self, z: NodeId) -> Result<NodeId, Error> {
        self.require("imag", z, self.dtype(z).is_complex(), "complex")?;
        Ok(self.push(Op::Imag, vec![z]))
    }

    // decompositions

    /// Complex conjugate: `re - i*im`.
    pub fn conj(&mut self, z: NodeId) -> Result<NodeId, Error> {
        let re = self.real(z)?;
        let im = self.imag(z)?;
        let nim = self.neg(im);
        self.complex(re, nim)
    }

    /// Complex magnitude `|z| = sqrt(re^2 + im^2)` (returns the real dtype).
    pub fn cabs(&mut self, z: NodeId) -> Result<NodeId, Error> {
        let re = self.real(z)?;
        let im = self.imag(z)?;
        let r2 = self.square(re);
        let i2 = self.square(im);
        let s = self.add(r2, i2)?;
        Ok(self.sqrt(s))
    }

    /// Phase angle of a complex tensor: `atan2(imag, real)` (returns the real dtype).
    pub fn angle(&mut self, z: NodeId) -> Result<NodeId, Error> {
        let re = self.real(z)?;
        let im = self.imag(z)?;
        self.atan2(im, re)
    }
}
