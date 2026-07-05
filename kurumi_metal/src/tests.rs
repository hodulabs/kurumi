use crate::*;
use half::f16;
use kurumi_core::{Graph, Storage, interpret, realize};
use std::time::Instant;

fn cpu_matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut g = Graph::new();
    let na = g.constant(a.to_vec(), vec![m, k]);
    let nb = g.constant(b.to_vec(), vec![k, n]);
    let y = g.dot_general(na, nb, vec![1], vec![0], vec![], vec![]).unwrap();
    realize::force(&g, y).f32().to_vec()
}

fn slice_storage(s: &Storage, len: usize) -> Storage {
    match s {
        Storage::U8(v) => Storage::U8(v[..len].to_vec()),
        Storage::U32(v) => Storage::U32(v[..len].to_vec()),
        Storage::I32(v) => Storage::I32(v[..len].to_vec()),
        Storage::I64(v) => Storage::I64(v[..len].to_vec()),
        Storage::F16(v) => Storage::F16(v[..len].to_vec()),
        Storage::BF16(v) => Storage::BF16(v[..len].to_vec()),
        Storage::F32(v) => Storage::F32(v[..len].to_vec()),
        _ => unreachable!(),
    }
}

mod bench;
mod device_ops;
mod matmul;
mod model;
