//! Model-level fwd+bwd vs the CPU oracle. This root keeps the basic device-op checks
//! (softmax/layernorm, conv2d, RoPE); attention, training, and full transformer blocks
//! live in the `attention`/`train`/`transformer` submodules.

mod attention;
mod train;
mod transformer;

use crate::tests::*;

// softmax + layernorm are now fully device-resident (reduce + broadcast +
// elementwise all on GPU); the readback must still match the CPU oracle.
#[test]
fn metal_softmax_layernorm_match_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let mut g = Graph::new();
    let x = g.constant((0..32 * 16).map(|i| ((i % 23) as f32) * 0.1 - 1.0).collect(), vec![32, 16]);
    let sm = g.softmax(x, 1).unwrap(); // max+sum reduce, broadcast, exp/div
    let ln = g.layernorm(sm, 1, 1e-5).unwrap(); // mean/var reduce, broadcast, elementwise
    let gpu = metal.eval(&g, ln);
    let cpu = interpret(&g, ln);
    assert_eq!(gpu.shape, cpu.shape);
    for (a, b) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}

// RoPE (slice + pad-concat + sin/cos + elementwise) runs device-resident and
// matches the CPU oracle.
#[test]
fn metal_rope_matches_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let (b, h, s, d) = (2usize, 2, 5, 8);
    let mut g = Graph::new();
    let x = g.constant((0..b * h * s * d).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect(), vec![b, h, s, d]);
    let y = g.rope(x).unwrap();
    let gpu = metal.eval(&g, y);
    let cpu = interpret(&g, y);
    assert_eq!(gpu.shape, cpu.shape);
    for (p, w) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((p - w).abs() < 1e-3, "{p} vs {w}");
    }
}

#[test]
fn metal_conv2d_matches_cpu() {
    use kurumi_core::Backend;
    let Some(metal) = MetalBackend::new() else { return };
    let (n, c, h, w, o, k) = (2usize, 3, 8, 8, 4, 3);
    let mut g = Graph::new();
    let gi = g.constant((0..n * c * h * w).map(|i| ((i % 17) as f32) * 0.1 - 0.5).collect(), vec![n, c, h, w]);
    let gw = g.constant((0..o * c * k * k).map(|i| ((i % 7) as f32) * 0.05).collect(), vec![o, c, k, k]);
    let y = g.conv2d(gi, gw, (2, 2), (1, 1), (1, 1)).unwrap();
    let gpu = metal.eval(&g, y);
    let cpu = interpret(&g, y);
    assert_eq!(gpu.shape, cpu.shape);
    for (a, b) in gpu.f32().iter().zip(cpu.f32()) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}
