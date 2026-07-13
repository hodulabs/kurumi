//! Random-graph differential fuzz: `MetalBackend::eval` vs the CPU interpreter oracle.
//! Per-op device tests pin one kernel at a time; this hammers random compositions so a
//! device-kernel dispatch hole (an op/dtype the builder admits but a hand-written MSL
//! kernel mishandles) can't hide between the tested ops. Float graphs compare with a
//! relative tolerance (device transcendentals differ from the CPU by ULPs); integer
//! graphs compare exactly. Skips when no Metal device is present.
use crate::tests::*;
use kurumi_core::{Backend, DType, NodeId};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n.max(1)
    }
}

fn perm(rng: &mut Rng, n: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        p.swap(i, rng.below(i + 1));
    }
    p
}

fn factor_shape(rng: &mut Rng, numel: usize) -> Vec<usize> {
    if numel <= 1 || rng.below(2) == 0 {
        return vec![numel];
    }
    let divs: Vec<usize> = (1..=numel).filter(|d| numel.is_multiple_of(*d)).collect();
    let d = divs[rng.below(divs.len())];
    vec![d, numel / d]
}

// a dtype-agnostic movement/reduce step shared by both generators. Returns None when the
// picked op does not apply to the node (caller retries a different op).
fn movement(rng: &mut Rng, g: &mut Graph, id: NodeId, shape: &[usize]) -> Option<(NodeId, Vec<usize>)> {
    match rng.below(5) {
        0 if !shape.is_empty() => {
            let axis = rng.below(shape.len());
            let r = g.reduce_max(id, axis).unwrap();
            let mut ns = shape.to_vec();
            ns.remove(axis);
            Some((r, ns))
        }
        1 => {
            let p = perm(rng, shape.len());
            let ns = p.iter().map(|&i| shape[i]).collect();
            Some((g.permute(id, p).unwrap(), ns))
        }
        2 => {
            let ns = factor_shape(rng, shape.iter().product());
            Some((g.reshape(id, ns.clone()).unwrap(), ns))
        }
        3 => {
            let ranges: Vec<(usize, usize)> = shape
                .iter()
                .map(|&d| {
                    let a = rng.below(d);
                    (a, a + 1 + rng.below(d - a))
                })
                .collect();
            let ns = ranges.iter().map(|(a, b)| b - a).collect();
            Some((g.slice(id, ranges).unwrap(), ns))
        }
        _ => {
            let pads: Vec<(usize, usize)> = shape.iter().map(|_| (rng.below(2), rng.below(2))).collect();
            let ns = shape.iter().zip(&pads).map(|(&d, &(lo, hi))| lo + d + hi).collect();
            Some((g.pad(id, pads).unwrap(), ns))
        }
    }
}

fn seed_shapes(rng: &mut Rng) -> Vec<Vec<usize>> {
    // two rank-2 matrices (so matmul has material) plus one free-rank tensor.
    vec![
        vec![1 + rng.below(4), 1 + rng.below(4)],
        vec![1 + rng.below(4), 1 + rng.below(4)],
        (0..1 + rng.below(3)).map(|_| 1 + rng.below(4)).collect(),
    ]
}

fn float_graph(rng: &mut Rng) -> (Graph, NodeId) {
    let mut g = Graph::new();
    let mut nodes: Vec<(NodeId, Vec<usize>)> = Vec::new();
    for shape in seed_shapes(rng) {
        let n: usize = shape.iter().product();
        let data = (0..n).map(|_| (rng.below(801) as f32 - 400.0) / 100.0).collect(); // [-4, 4]
        nodes.push((g.constant(data, shape.clone()), shape));
    }
    for _ in 0..8 + rng.below(6) {
        let (id, shape) = nodes[rng.below(nodes.len())].clone();
        let step = match rng.below(5) {
            0 => {
                let r = match rng.below(7) {
                    0 => g.neg(id),
                    1 => g.recip(id),
                    2 => g.sqrt(id),
                    3 => g.exp2(id),
                    4 => g.log2(id),
                    5 => g.sin(id),
                    _ => g.abs(id),
                };
                Some((r, shape))
            }
            1 => {
                let same: Vec<NodeId> = nodes.iter().filter(|t| t.1 == shape).map(|t| t.0).collect();
                let o = same[rng.below(same.len())];
                let r = match rng.below(4) {
                    0 => g.add(id, o),
                    1 => g.mul(id, o),
                    2 => g.max(id, o),
                    _ => g.min(id, o),
                }
                .unwrap();
                Some((r, shape))
            }
            2 => {
                // matmul on the first compatible [m,k] x [k,n] pair, else fall through.
                let a = nodes.iter().find(|t| t.1.len() == 2).cloned();
                a.and_then(|(aid, ash)| {
                    let b = nodes.iter().find(|t| t.1.len() == 2 && t.1[0] == ash[1]).cloned();
                    b.map(|(bid, bsh)| {
                        let r = g.dot_general(aid, bid, vec![1], vec![0], vec![], vec![]).unwrap();
                        (r, vec![ash[0], bsh[1]])
                    })
                })
            }
            _ => movement(rng, &mut g, id, &shape),
        };
        if let Some(step) = step {
            nodes.push(step);
        }
    }
    (g, nodes.last().unwrap().0)
}

fn int_graph(rng: &mut Rng, dt: DType) -> (Graph, NodeId) {
    let mut g = Graph::new();
    let mut nodes: Vec<(NodeId, Vec<usize>)> = Vec::new();
    for shape in seed_shapes(rng) {
        let n: usize = shape.iter().product();
        // values 0..8 so bitwise/max/min/reduce_max stay exact and overflow-free across dtypes.
        let vals: Vec<i64> = (0..n).map(|_| rng.below(8) as i64).collect();
        let storage = match dt {
            DType::I32 => Storage::I32(vals.iter().map(|&v| v as i32).collect()),
            DType::I64 => Storage::I64(vals),
            DType::U32 => Storage::U32(vals.iter().map(|&v| v as u32).collect()),
            DType::U8 => Storage::U8(vals.iter().map(|&v| v as u8).collect()),
            _ => unreachable!("int_graph dtype"),
        };
        nodes.push((g.const_storage(storage, shape.clone()), shape));
    }
    for _ in 0..8 + rng.below(6) {
        let (id, shape) = nodes[rng.below(nodes.len())].clone();
        let step = match rng.below(2) {
            0 => {
                let same: Vec<NodeId> = nodes.iter().filter(|t| t.1 == shape).map(|t| t.0).collect();
                let o = same[rng.below(same.len())];
                let r = match rng.below(5) {
                    0 => g.max(id, o),
                    1 => g.min(id, o),
                    2 => g.and(id, o),
                    3 => g.or(id, o),
                    _ => g.xor(id, o),
                }
                .unwrap();
                Some((r, shape))
            }
            _ => movement(rng, &mut g, id, &shape),
        };
        if let Some(step) = step {
            nodes.push(step);
        }
    }
    (g, nodes.last().unwrap().0)
}

// device float vs oracle: nan/nan and same-sign overflow compare equal, else a relative band.
fn close(a: f32, b: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.abs() > 1e30 && b.abs() > 1e30 && a.signum() == b.signum() {
        return true; // both overflowed the same way (inf vs finite-huge across a ULP threshold)
    }
    (a - b).abs() <= 2e-3 + 2e-2 * b.abs()
}

#[test]
fn metal_float_graphs_match_oracle() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    for seed in 0..40u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
        let (g, out) = float_graph(&mut rng);
        let cpu = interpret(&g, out);
        let gpu = metal.eval(&g, out);
        assert_eq!(gpu.shape, cpu.shape, "seed {seed}");
        for (a, b) in gpu.f32().iter().zip(cpu.f32()) {
            assert!(close(*a, *b), "seed {seed}: {a} vs {b}");
        }
    }
}

#[test]
fn metal_int_graphs_match_oracle() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    for dt in [DType::I32, DType::I64, DType::U32, DType::U8] {
        for seed in 0..24u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(dt as u64 + 1));
            let (g, out) = int_graph(&mut rng, dt);
            let cpu = interpret(&g, out);
            let gpu = metal.eval(&g, out);
            assert_eq!(gpu.shape, cpu.shape, "{dt:?} seed {seed}");
            assert_eq!(gpu.storage, cpu.storage, "{dt:?} seed {seed}");
        }
    }
}
