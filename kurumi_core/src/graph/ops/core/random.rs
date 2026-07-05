//! Random sampling: the RandUniform primitive plus normal/range/int/dropout.
//! (Box-Muller / inverse-CDF; pure + reproducible given the seed.)

use crate::{DType, Error, Graph, NodeId, Op, Storage};

impl Graph {
    // primitives

    /// Uniform `[0,1)` F32 tensor from a fixed `seed` (reproducible, parallel RNG).
    /// For per-step randomness (training dropout) use [`Graph::rand_uniform_keyed`].
    pub fn rand_uniform(&mut self, shape: Vec<usize>, seed: u64) -> NodeId {
        let s = self.const_storage(Storage::I64(vec![seed as i64]), vec![1]);
        self.push(Op::RandUniform { shape }, vec![s])
    }

    /// Uniform `[0,1)` F32 tensor with the seed supplied as a runtime scalar int node
    /// (feed a step counter per eval to vary the draw in a build-once graph).
    pub fn rand_uniform_keyed(&mut self, shape: Vec<usize>, seed: NodeId) -> Result<NodeId, Error> {
        if !self.dtype(seed).is_int() {
            return Err(Error::shape("rand_uniform", format!("seed must be integer, got {:?}", self.dtype(seed))));
        }
        Ok(self.push(Op::RandUniform { shape }, vec![seed]))
    }

    // decompositions

    /// Standard normal `N(0,1)` via Box-Muller: `sqrt(-2 ln u_1)*cos(2pi u_2)`.
    pub fn randn(&mut self, shape: Vec<usize>, seed: u64) -> NodeId {
        let u1 = self.rand_uniform(shape.clone(), seed);
        let u1 = self.clamp_min(u1, 1e-7).expect("same shape"); // avoid ln(0)
        let u2 = self.rand_uniform(shape, seed ^ 0xD1B5_4A32_D192_ED03);
        let lu = self.ln(u1);
        let m2 = self.scalar(lu, -2.0);
        let r2 = self.mul(lu, m2).expect("same shape");
        let r = self.sqrt(r2);
        let tau = self.scalar(u2, std::f32::consts::TAU);
        let ang = self.mul(u2, tau).expect("same shape");
        let c = self.cos(ang);
        self.mul(r, c).expect("same shape")
    }

    /// Uniform on `[lo, hi)`: `lo + (hi-lo)*u`.
    pub fn rand_range(&mut self, shape: Vec<usize>, seed: u64, lo: f32, hi: f32) -> NodeId {
        let u = self.rand_uniform(shape, seed);
        let scale = self.scalar(u, hi - lo);
        let su = self.mul(u, scale).expect("same shape");
        let off = self.scalar(su, lo);
        self.add(su, off).expect("same shape")
    }

    /// Random integers `[lo, hi)` as I64: `floor(lo + (hi-lo)*u)`.
    pub fn randint(&mut self, shape: Vec<usize>, seed: u64, lo: i64, hi: i64) -> NodeId {
        let r = self.rand_range(shape, seed, lo as f32, hi as f32);
        let f = self.floor(r);
        self.cast(f, DType::I64)
    }

    /// Bernoulli mask (F32 0/1): 1 with probability `p`.
    pub fn bernoulli(&mut self, shape: Vec<usize>, seed: u64, p: f32) -> Result<NodeId, Error> {
        let u = self.rand_uniform(shape, seed);
        let pc = self.scalar(u, p);
        let lt = self.cmp_lt(u, pc)?; // u < p
        Ok(self.cast(lt, DType::F32))
    }

    /// Inverted dropout: zero each element with prob `p`, scale survivors by `1/(1-p)`
    /// (expected value unchanged). Reproducible given `seed`; for per-step training
    /// masks build it from [`Graph::rand_uniform_keyed`].
    pub fn dropout(&mut self, x: NodeId, p: f32, seed: u64) -> Result<NodeId, Error> {
        let shape = self.shape(x);
        let dt = self.dtype(x);
        let u = self.rand_uniform(shape, seed);
        let pc = self.scalar(u, p);
        let keep = self.ge(u, pc)?; // u >= p -> survive
        let mask = self.cast(keep, dt);
        let xm = self.mul(x, mask)?;
        let scale = self.scalar(xm, 1.0 / (1.0 - p));
        self.mul(xm, scale)
    }
}
