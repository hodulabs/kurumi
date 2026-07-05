use crate::realize::force;
use crate::{Graph, NodeId, interpret};

// small deterministic LCG: no external deps
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n.max(1)
    }
    fn val(&mut self) -> f32 {
        (self.below(2001) as f32 - 1000.0) / 100.0 // [-10, 10]
    }
}

fn perm(rng: &mut Rng, n: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.below(i + 1);
        p.swap(i, j);
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

fn random_graph(rng: &mut Rng) -> (Graph, NodeId) {
    let mut g = Graph::new();
    let mut nodes: Vec<(NodeId, Vec<usize>)> = Vec::new();
    for _ in 0..2 {
        let rank = 1 + rng.below(3);
        let shape: Vec<usize> = (0..rank).map(|_| 1 + rng.below(3)).collect();
        let n: usize = shape.iter().product();
        let data = (0..n).map(|_| rng.val()).collect();
        let id = g.constant(data, shape.clone());
        nodes.push((id, shape));
    }
    let steps = 4 + rng.below(10);
    for _ in 0..steps {
        let pick = rng.below(nodes.len());
        let (id, shape) = nodes[pick].clone();
        let (nid, nshape) = match rng.below(7) {
            0 => {
                let r = match rng.below(6) {
                    0 => g.neg(id),
                    1 => g.recip(id),
                    2 => g.sqrt(id),
                    3 => g.exp2(id),
                    4 => g.log2(id),
                    _ => g.sin(id),
                };
                (r, shape)
            }
            1 => {
                let same: Vec<NodeId> = nodes.iter().filter(|t| t.1 == shape).map(|t| t.0).collect();
                let o = same[rng.below(same.len())];
                let r = match rng.below(3) {
                    0 => g.add(id, o),
                    1 => g.mul(id, o),
                    _ => g.max(id, o),
                }
                .unwrap();
                (r, shape)
            }
            2 if !shape.is_empty() => {
                let axis = rng.below(shape.len());
                let r = if rng.below(2) == 0 { g.sum(id, axis) } else { g.reduce_max(id, axis) }.unwrap();
                let mut ns = shape.clone();
                ns.remove(axis);
                (r, ns)
            }
            3 => {
                let p = perm(rng, shape.len());
                let ns = p.iter().map(|&i| shape[i]).collect();
                (g.permute(id, p).unwrap(), ns)
            }
            4 => {
                let ns = factor_shape(rng, shape.iter().product());
                (g.reshape(id, ns.clone()).unwrap(), ns)
            }
            5 => {
                let ranges: Vec<(usize, usize)> = shape
                    .iter()
                    .map(|&d| {
                        let a = rng.below(d);
                        (a, a + 1 + rng.below(d - a))
                    })
                    .collect();
                let ns = ranges.iter().map(|(a, b)| b - a).collect();
                (g.slice(id, ranges).unwrap(), ns)
            }
            _ => {
                let pads: Vec<(usize, usize)> = shape.iter().map(|_| (rng.below(2), rng.below(2))).collect();
                let ns = shape.iter().zip(&pads).map(|(&d, &(lo, hi))| lo + d + hi).collect();
                (g.pad(id, pads).unwrap(), ns)
            }
        };
        nodes.push((nid, nshape));
    }
    let out = nodes.last().unwrap().0;
    (g, out)
}

// the fused engine path must agree with the materializing oracle on every
// random graph (NaN compares equal to NaN; same ops => same bits otherwise)
#[test]
fn realize_equals_oracle_over_random_graphs() {
    for seed in 0..200u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
        let (g, out) = random_graph(&mut rng);
        let oracle = interpret(&g, out).storage.into_f32();
        let fused = force(&g, out).storage.into_f32();
        assert_eq!(oracle.len(), fused.len(), "seed {seed}");
        for (a, b) in oracle.iter().zip(&fused) {
            assert!(a == b || (a.is_nan() && b.is_nan()), "seed {seed}: {a} vs {b}");
        }
    }
}
