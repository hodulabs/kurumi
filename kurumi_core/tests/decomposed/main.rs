//! Integration tests: decomposed tensor ops vs reference math (conv/pool/special/...).

use kurumi_core::*;

mod complex;
mod conv;
mod coverage;
mod einsum;
mod elementwise;
mod indexing;
mod linalg;
mod movement;
mod rng;
mod signal;
mod special;

// decomposed tensor ops vs expected math
pub fn approx(g: &Graph, y: NodeId, want: &[f32]) {
    let o = interpret(g, y);
    let got = o.f32();
    assert_eq!(got.len(), want.len(), "len {} vs {}", got.len(), want.len());
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() < 1e-4, "[{i}] {a} vs {b}");
    }
}
