//! Generation ops: `iota` (index along an axis) and the counter-based RNG kernel
//! (threefry2x32 over seed + flat index -- pure, reproducible, embarrassingly parallel, backing
//! RandUniform; the algorithm + `Key` API live in `crate::rng`).

use crate::interp::indexing::indices_i64;
use crate::rng::{threefry2x32, uniform_f32};
use crate::{Op, Storage, TensorVal, iota_storage};

pub(super) fn eval(op: &Op, inputs: &[&TensorVal]) -> TensorVal {
    match op {
        Op::Iota { shape, axis, dtype } => {
            TensorVal { shape: shape.clone(), storage: iota_storage(shape, *axis, *dtype) }
        }
        Op::RandUniform { shape } => {
            let seed = indices_i64(&inputs[0].storage)[0] as u64;
            rand_uniform_gen(seed, shape)
        }
        _ => unreachable!("random::eval: non-generation op"),
    }
}

/// Counter-based uniform `[0, 1)` RNG: each element's value is `threefry(seed,
/// index)`, so it's pure, reproducible, and embarrassingly parallel (no state).
pub(crate) fn rand_uniform_gen(seed: u64, shape: &[usize]) -> TensorVal {
    let n: usize = shape.iter().product::<usize>().max(1);
    let data: Vec<f32> = (0..n as u64).map(|i| uniform_f32(threefry2x32(seed, i))).collect();
    TensorVal { shape: shape.to_vec(), storage: Storage::F32(data) }
}
