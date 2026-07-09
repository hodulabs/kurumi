use kurumi_core::*;

#[test]
fn einsum_cases() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let b = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
    // matmul ij,jk->ik
    let mm = g.einsum("ij,jk->ik", &[a, b]).unwrap();
    let want = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
    assert_eq!(interpret(&g, mm).f32(), interpret(&g, want).f32());
    // transpose ij->ji
    let t = g.einsum("ij->ji", &[a]).unwrap();
    assert_eq!(interpret(&g, t).shape, vec![3, 2]);
    assert_eq!(interpret(&g, t).f32(), &[1., 4., 2., 5., 3., 6.]);
    // sum over rows ij->j
    let sj = g.einsum("ij->j", &[a]).unwrap();
    assert_eq!(interpret(&g, sj).f32(), &[5., 7., 9.]);
    // full reduce ij-> (implicit empty output)
    let tot = g.einsum("ij->", &[a]).unwrap();
    assert_eq!(interpret(&g, tot).f32(), &[21.]);
    // frobenius inner product ij,ij->
    let c = g.constant(vec![1.; 6], vec![2, 3]);
    let fro = g.einsum("ij,ij->", &[a, c]).unwrap();
    assert_eq!(interpret(&g, fro).f32(), &[21.]);
    // outer product i,j->ij
    let u = g.constant(vec![1., 2.], vec![2]);
    let v = g.constant(vec![3., 4., 5.], vec![3]);
    let outer = g.einsum("i,j->ij", &[u, v]).unwrap();
    assert_eq!(interpret(&g, outer).shape, vec![2, 3]);
    assert_eq!(interpret(&g, outer).f32(), &[3., 4., 5., 6., 8., 10.]);
    // batched matmul bij,bjk->bik
    let p = g.constant((0..12).map(|i| i as f32).collect(), vec![2, 2, 3]);
    let q = g.constant((0..12).map(|i| i as f32).collect(), vec![2, 3, 2]);
    let bmm = g.einsum("bij,bjk->bik", &[p, q]).unwrap();
    let bwant = g.dot_general(p, q, vec![2], vec![1], vec![0], vec![0]).unwrap();
    assert_eq!(interpret(&g, bmm).f32(), interpret(&g, bwant).f32());
    // implicit output: ik,kj (no ->) means sum over repeated k -> ij
    let imp = g.einsum("ik,kj", &[a, b]).unwrap();
    assert_eq!(interpret(&g, imp).f32(), interpret(&g, mm).f32());
}

#[test]
fn einsum_diagonal_and_multi() {
    let mut g = Graph::new();
    let m = g.constant(vec![1., 2., 3., 4.], vec![2, 2]);
    // trace: repeated index, empty output
    let tr = g.einsum("ii->", &[m]).unwrap();
    assert_eq!(interpret(&g, tr).f32(), &[5.]);
    // trace via implicit output (i appears twice -> summed away)
    let tri = g.einsum("ii", &[m]).unwrap();
    assert_eq!(interpret(&g, tri).f32(), &[5.]);
    // diagonal: keep the repeated index
    let di = g.einsum("ii->i", &[m]).unwrap();
    assert_eq!(interpret(&g, di).f32(), &[1., 4.]);
    // batched diagonal
    let x = g.constant((1..=8).map(|v| v as f32).collect(), vec![2, 2, 2]);
    let bd = g.einsum("bii->bi", &[x]).unwrap();
    assert_eq!(interpret(&g, bd).shape, vec![2, 2]);
    assert_eq!(interpret(&g, bd).f32(), &[1., 4., 5., 8.]);
    // 3-operand chain A@B@C  (B=I, C=2I -> A*2)
    let a = g.constant(vec![1., 2., 3., 4.], vec![2, 2]);
    let bb = g.constant(vec![1., 0., 0., 1.], vec![2, 2]);
    let cc = g.constant(vec![2., 0., 0., 2.], vec![2, 2]);
    let chain = g.einsum("ij,jk,kl->il", &[a, bb, cc]).unwrap();
    assert_eq!(interpret(&g, chain).f32(), &[2., 4., 6., 8.]);
    // 3-operand elementwise product of vectors
    let u = g.constant(vec![1., 2.], vec![2]);
    let v = g.constant(vec![3., 4.], vec![2]);
    let w = g.constant(vec![5., 6.], vec![2]);
    let ew = g.einsum("i,i,i->i", &[u, v, w]).unwrap();
    assert_eq!(interpret(&g, ew).f32(), &[15., 48.]);
}

// outer products (no contracted index) with reordered / interleaved output -- exercises
// dot_general with empty contract/batch, which the VQE kron path relies on.
#[test]
fn einsum_outer_products() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3., 4.], vec![2, 2]);
    let b = g.constant(vec![5., 6., 7., 8.], vec![2, 2]);
    let base = g.dot_general(a, b, vec![], vec![], vec![], vec![]).unwrap(); // [i,j,k,l]
    // ij,kl->ijkl (natural order) == the bare outer product
    let ijkl = g.einsum("ij,kl->ijkl", &[a, b]).unwrap();
    assert_eq!(interpret(&g, ijkl).shape, vec![2, 2, 2, 2]);
    assert_eq!(interpret(&g, ijkl).f32(), interpret(&g, base).f32());
    // ij,kl->ikjl (interleaved reorder) == permute(base, [0,2,1,3]) -- the bug case
    let ikjl = g.einsum("ij,kl->ikjl", &[a, b]).unwrap();
    let ref_ikjl = g.permute(base, vec![0, 2, 1, 3]).unwrap();
    assert_eq!(interpret(&g, ikjl).shape, vec![2, 2, 2, 2]);
    assert_eq!(interpret(&g, ikjl).f32(), interpret(&g, ref_ikjl).f32());
    // ij,kl->klij (swapped) == permute(base, [2,3,0,1])
    let klij = g.einsum("ij,kl->klij", &[a, b]).unwrap();
    let ref_klij = g.permute(base, vec![2, 3, 0, 1]).unwrap();
    assert_eq!(interpret(&g, klij).f32(), interpret(&g, ref_klij).f32());
    // vector outer i,j->ij
    let v = g.constant(vec![1., 2.], vec![2]);
    let w = g.constant(vec![3., 4., 5.], vec![3]);
    let vw = g.einsum("i,j->ij", &[v, w]).unwrap();
    assert_eq!(interpret(&g, vw).shape, vec![2, 3]);
    assert_eq!(interpret(&g, vw).f32(), &[3., 4., 5., 6., 8., 10.]);
    // scalar stack (the actual VQE failure path): stack of rank-0 nodes -> [n]
    let s0 = g.constant(vec![9.], vec![]);
    let s1 = g.constant(vec![10.], vec![]);
    let st = g.stack(&[s0, s1], 0).unwrap();
    assert_eq!(interpret(&g, st).shape, vec![2]);
    assert_eq!(interpret(&g, st).f32(), &[9., 10.]);
}
