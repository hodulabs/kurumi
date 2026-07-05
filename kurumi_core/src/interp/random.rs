//! Counter-based RNG kernel (threefry2x32 over seed + flat index): pure, reproducible,
//! embarrassingly parallel. Backs RandUniform; the algorithm + `Key` API live in `crate::rng`.

use crate::rng::{threefry2x32, uniform_f32};
use crate::{Storage, TensorVal};

/// Counter-based uniform `[0, 1)` RNG: each element's value is `threefry(seed,
/// index)`, so it's pure, reproducible, and embarrassingly parallel (no state).
pub(crate) fn rand_uniform_gen(seed: u64, shape: &[usize]) -> TensorVal {
    let n: usize = shape.iter().product::<usize>().max(1);
    let data: Vec<f32> = (0..n as u64).map(|i| uniform_f32(threefry2x32(seed, i))).collect();
    TensorVal { shape: shape.to_vec(), storage: Storage::F32(data) }
}
