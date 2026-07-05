use kurumi_core::*;

#[test]
fn scatter_min() {
    let mut g = Graph::new();
    let operand = g.constant(vec![5., 5., 5., 5.], vec![4]);
    let idx = g.const_storage(Storage::I64(vec![0, 1, 0]), vec![3]);
    let updates = g.constant(vec![3., 7., 8.], vec![3]);
    // min combine: pos0 = min(5,3,8)=3, pos1 = min(5,7)=5, pos2/3 untouched=5
    let y = g.scatter(operand, idx, updates, 0, ScatterOp::Min).unwrap();
    assert_eq!(interpret(&g, y).f32(), &[3., 5., 5., 5.]);
    let ymax = g.scatter(operand, idx, updates, 0, ScatterOp::Max).unwrap();
    assert_eq!(interpret(&g, ymax).f32(), &[8., 7., 5., 5.]);
}

#[test]
fn argmax_argmin() {
    let mut g = Graph::new();
    let asi64 = |g: &Graph, y: NodeId| match interpret(g, y).storage {
        Storage::I64(v) => v,
        s => panic!("want i64, got {:?}", s.dtype()),
    };
    // [[1,5,2],[8,3,0]] argmax over axis1 -> [1,0]; argmin -> [0,2]
    let x = g.constant(vec![1., 5., 2., 8., 3., 0.], vec![2, 3]);
    let am = g.argmax(x, 1).unwrap();
    assert_eq!(asi64(&g, am), vec![1, 0]);
    let an = g.argmin(x, 1).unwrap();
    assert_eq!(asi64(&g, an), vec![0, 2]);
    // argmax over axis0 -> [1,0,0]
    let am0 = g.argmax(x, 0).unwrap();
    assert_eq!(asi64(&g, am0), vec![1, 0, 0]);
    // ties take the first (lowest index)
    let t = g.constant(vec![3., 3., 1.], vec![3]);
    let at = g.argmax(t, 0).unwrap();
    assert_eq!(asi64(&g, at), vec![0]);
}

#[test]
fn sort_topk_takealong() {
    let mut g = Graph::new();
    let asi64 = |g: &Graph, y: NodeId| match interpret(g, y).storage {
        Storage::I64(v) => v,
        s => panic!("want i64, got {:?}", s.dtype()),
    };
    // argsort ascending / descending on [[3,1,2],[0,5,4]]
    let x = g.constant(vec![3., 1., 2., 0., 5., 4.], vec![2, 3]);
    let asc = g.argsort(x, 1, false).unwrap();
    assert_eq!(asi64(&g, asc), vec![1, 2, 0, 0, 2, 1]);
    let desc = g.argsort(x, 1, true).unwrap();
    assert_eq!(asi64(&g, desc), vec![0, 2, 1, 1, 2, 0]);
    // sort values
    let sv = g.sort(x, 1, false).unwrap();
    assert_eq!(interpret(&g, sv).f32(), &[1., 2., 3., 0., 4., 5.]);
    // topk (largest 2)
    let (tv, ti) = g.topk(x, 2, 1, true).unwrap();
    assert_eq!(interpret(&g, tv).f32(), &[3., 2., 5., 4.]);
    assert_eq!(asi64(&g, ti), vec![0, 2, 1, 2]);
    // take_along_dim: gather the argmax values
    let am = g.argmax(x, 1).unwrap(); // [0,1] -> [2,3]? argmax of [3,1,2]=0, [0,5,4]=1
    let amk = g.unsqueeze(am, 1).unwrap(); // [2,1]
    let vals = g.take_along_dim(x, amk, 1).unwrap();
    assert_eq!(interpret(&g, vals).f32(), &[3., 5.]);
}

#[test]
fn gather_along_backward() {
    // d/dx sum(take_along_dim(x, idx)) places 1 at gathered positions
    let mut g = Graph::new();
    let x = g.constant(vec![10., 20., 30., 40., 50., 60.], vec![2, 3]);
    let idx = g.const_storage(Storage::I64(vec![2, 0, 1, 1]), vec![2, 2]);
    let y = g.gather_along(x, idx, 1).unwrap(); // [[30,10],[50,50]]
    assert_eq!(interpret(&g, y).f32(), &[30., 10., 50., 50.]);
    let loss = {
        let s = g.sum(y, 1).unwrap();
        g.sum(s, 0).unwrap()
    };
    let gx = grad(&mut g, loss, &[x]).unwrap()[0];
    // row0: idx 2,0 -> grad [1,0,1]; row1: idx 1,1 -> grad [0,2,0]
    assert_eq!(interpret(&g, gx).f32(), &[1., 0., 1., 0., 2., 0.]);
}

#[test]
fn masked_dynamic_select() {
    let mut g = Graph::new();
    let x = g.constant(vec![10., 20., 30., 40., 50.], vec![5]);
    let mask = g.const_storage(Storage::BOOL(vec![true, false, true, true, false]), vec![5]);
    // compress: first 3 masked -> [10,30,40]
    let c = g.compress(mask, x, 3).unwrap();
    assert_eq!(interpret(&g, c).f32(), &[10., 30., 40.]);
    // k larger than the true count -> zero-padded tail
    let c4 = g.compress(mask, x, 4).unwrap();
    assert_eq!(interpret(&g, c4).f32(), &[10., 30., 40., 0.]);
    // masked_select flattens first
    let m2 = g.constant(vec![1., 2., 3., 4.], vec![2, 2]);
    let mk2 = g.const_storage(Storage::BOOL(vec![true, false, false, true]), vec![2, 2]);
    let ms = g.masked_select(m2, mk2, 2).unwrap();
    assert_eq!(interpret(&g, ms).f32(), &[1., 4.]);
    // nonzero: flat indices of nonzero elements
    let nz_in = g.constant(vec![0., 5., 0., 3., 0.], vec![5]);
    let nz = g.nonzero(nz_in, 2).unwrap();
    let nzv = match interpret(&g, nz).storage {
        Storage::I64(v) => v,
        s => panic!("want I64, got {s:?}"),
    };
    assert_eq!(nzv, vec![1, 3]);
    // unique: sorted distinct values
    let u_in = g.constant(vec![3., 1., 2., 3., 1.], vec![5]);
    let u = g.unique(u_in, 3).unwrap();
    assert_eq!(interpret(&g, u).f32(), &[1., 2., 3.]);
}

#[test]
fn gather_nd_cases() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
    // K=1: pick rows 0 and 2 -> shape [2,2]
    let i1 = g.const_storage(Storage::I64(vec![0, 2]), vec![2, 1]);
    let r1 = g.gather_nd(x, i1).unwrap();
    assert_eq!(interpret(&g, r1).shape, vec![2, 2]);
    assert_eq!(interpret(&g, r1).f32(), &[1., 2., 5., 6.]);
    // K=2 (full index): pick x[0,1] and x[2,0] -> shape [2]
    let i2 = g.const_storage(Storage::I64(vec![0, 1, 2, 0]), vec![2, 2]);
    let r2 = g.gather_nd(x, i2).unwrap();
    assert_eq!(interpret(&g, r2).shape, vec![2]);
    assert_eq!(interpret(&g, r2).f32(), &[2., 5.]);
    // single coord (idx rank 1): x[1,0] scalar -> shape []
    let i3 = g.const_storage(Storage::I64(vec![1, 0]), vec![2]);
    let r3 = g.gather_nd(x, i3).unwrap();
    assert_eq!(interpret(&g, r3).shape, Vec::<usize>::new());
    assert_eq!(interpret(&g, r3).f32(), &[3.]);
    // 3-d x, K=2 -> trailing dim kept
    let y = g.constant((0..12).map(|i| i as f32).collect(), vec![2, 3, 2]);
    let iy = g.const_storage(Storage::I64(vec![0, 2, 1, 0]), vec![2, 2]);
    let ry = g.gather_nd(y, iy).unwrap(); // y[0,2,:]=[4,5], y[1,0,:]=[6,7]
    assert_eq!(interpret(&g, ry).shape, vec![2, 2]);
    assert_eq!(interpret(&g, ry).f32(), &[4., 5., 6., 7.]);
    // backward: grad routes 1s to the gathered rows
    let loss = {
        let s = g.sum(r1, 1).unwrap();
        g.sum(s, 0).unwrap()
    };
    let gx = grad(&mut g, loss, &[x]).unwrap()[0];
    assert_eq!(interpret(&g, gx).f32(), &[1., 1., 0., 0., 1., 1.]);
}

#[test]
fn scatter_nd_cases() {
    let mut g = Graph::new();
    let zeros = g.constant(vec![0.; 6], vec![3, 2]);
    // K=1 Set: write rows 0 and 2
    let i1 = g.const_storage(Storage::I64(vec![0, 2]), vec![2, 1]);
    let u1 = g.constant(vec![10., 20., 50., 60.], vec![2, 2]);
    let s1 = g.scatter_nd(zeros, i1, u1, ScatterOp::Set).unwrap();
    assert_eq!(interpret(&g, s1).f32(), &[10., 20., 0., 0., 50., 60.]);
    // K=1 Add with a collision on row 0
    let idup = g.const_storage(Storage::I64(vec![0, 0]), vec![2, 1]);
    let udup = g.constant(vec![1., 1., 2., 2.], vec![2, 2]);
    let sadd = g.scatter_nd(zeros, idup, udup, ScatterOp::Add).unwrap();
    assert_eq!(interpret(&g, sadd).f32(), &[3., 3., 0., 0., 0., 0.]);
    // K=2 full index Set: x[0,1]=5, x[2,0]=7
    let i2 = g.const_storage(Storage::I64(vec![0, 1, 2, 0]), vec![2, 2]);
    let u2 = g.constant(vec![5., 7.], vec![2]);
    let s2 = g.scatter_nd(zeros, i2, u2, ScatterOp::Set).unwrap();
    assert_eq!(interpret(&g, s2).f32(), &[0., 5., 0., 0., 7., 0.]);
    // gather_nd after scatter_nd round-trips the written coords
    let back = g.gather_nd(s1, i1).unwrap();
    assert_eq!(interpret(&g, back).f32(), &[10., 20., 50., 60.]);
    // backward: with Add, d(sum(out))/d(updates) is all ones
    let upd = g.constant(vec![1., 1., 2., 2.], vec![2, 2]);
    let out = g.scatter_nd(zeros, i1, upd, ScatterOp::Add).unwrap();
    let loss = {
        let s = g.sum(out, 1).unwrap();
        g.sum(s, 0).unwrap()
    };
    let gu = grad(&mut g, loss, &[upd]).unwrap()[0];
    assert_eq!(interpret(&g, gu).f32(), &[1., 1., 1., 1.]);
}
