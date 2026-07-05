use crate::realize::*;
use crate::{DType, Feeds, Graph, Storage, TensorVal, interpret, interpret_with};

// plan-replay: compile once (const weight materialized once), replay with
// fresh Input feeds: each run matches the oracle. y = x*w + w.
#[test]
fn plan_replays_with_fresh_feeds() {
    let mut g = Graph::new();
    let w = g.constant(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);
    let x = g.input(vec![2, 2], DType::F32);
    let xw = g.mul(x, w).unwrap();
    let y = g.add(xw, w).unwrap();
    let plan = Plan::compile(&g, y).expect("fused path");
    for batch in [vec![1.0, 1.0, 1.0, 1.0], vec![0.0, 2.0, -1.0, 3.0]] {
        let feeds = Feeds::from([(x, TensorVal { shape: vec![2, 2], storage: Storage::F32(batch) })]);
        assert_eq!(plan.run(&g, &feeds).storage.into_f32(), interpret_with(&g, y, &feeds).storage.into_f32());
    }
}

// movement + elementwise + reduce: the fused path matches the oracle
#[test]
fn realize_matches_oracle() {
    let mut g = Graph::new();
    let a = g.constant((0..6).map(|x| x as f32).collect(), vec![2, 3]);
    let ap = g.permute(a, vec![1, 0]).unwrap();
    let b = g.constant(vec![10., 20., 30., 40., 50., 60.], vec![3, 2]);
    let s = g.add(ap, b).unwrap();
    let y = g.sum(s, 1).unwrap();
    assert_eq!(force(&g, y), interpret(&g, y));
}

// permute does not touch data: the backing buffer stays in source order
#[test]
fn movement_is_zero_copy() {
    let mut g = Graph::new();
    let a = g.constant((0..6).map(|x| x as f32).collect(), vec![2, 3]);
    let p = g.permute(a, vec![1, 0]).unwrap();
    let r = realize(&g, p);
    match &r.0 {
        Repr::Leaf { buf, .. } => assert_eq!(&**buf, &[0., 1., 2., 3., 4., 5.]),
        _ => panic!("movement should stay a leaf view"),
    }
    assert_eq!(r.force().f32(), &[0., 3., 1., 4., 2., 5.]);
    assert_eq!(r.force(), interpret(&g, p));
}

// a 3-deep elementwise tree stays fused (one pass, no temporaries)
#[test]
fn elementwise_chain_fuses() {
    let mut g = Graph::new();
    let x = g.constant(vec![-2., -1., 0., 1., 2., 3.], vec![2, 3]);
    let zero = g.constant(vec![0.; 6], vec![2, 3]);
    let sq = g.mul(x, x).unwrap();
    let s = g.add(sq, x).unwrap();
    let y = g.max(s, zero).unwrap(); // relu(x*x + x)
    assert!(realize(&g, y).is_fused());
    assert_eq!(force(&g, y), interpret(&g, y));
}

// the transpose happens inside the fused read: no transpose buffer
#[test]
fn movement_into_elementwise_fuses() {
    let mut g = Graph::new();
    let a = g.constant((0..6).map(|x| x as f32).collect(), vec![2, 3]);
    let ap = g.permute(a, vec![1, 0]).unwrap();
    let b = g.constant(vec![100., 200., 300., 400., 500., 600.], vec![3, 2]);
    let y = g.add(ap, b).unwrap();
    assert!(realize(&g, y).is_fused());
    assert_eq!(force(&g, y), interpret(&g, y));
}

// diamond: a fused node read by two consumers is materialized once, correct
#[test]
fn diamond_materializes_shared_node() {
    let mut g = Graph::new();
    let x = g.constant(vec![-1., 2., -3., 4.], vec![2, 2]);
    let zero = g.constant(vec![0.; 4], vec![2, 2]);
    let b = g.max(x, zero).unwrap(); // relu, fused
    let l = g.mul(b, b).unwrap(); // b consumed twice
    assert_eq!(consumer_counts(&g, l)[&b], 2);
    assert_eq!(force(&g, l), interpret(&g, l));
}

// slice then flip fuse into the read; matches oracle
#[test]
fn slice_flip_fuse() {
    let mut g = Graph::new();
    let a = g.constant((0..12).map(|x| x as f32).collect(), vec![3, 4]);
    let s = g.slice(a, vec![(0, 2), (1, 4)]).unwrap(); // [2,3]
    let f = g.flip(s, vec![1]).unwrap();
    let b = g.constant(vec![100.; 6], vec![2, 3]);
    let y = g.add(f, b).unwrap();
    assert!(realize(&g, y).is_fused());
    assert_eq!(force(&g, y), interpret(&g, y));
}

// pad's masked load happens inside the fused read: no pad buffer
#[test]
fn pad_fuses_into_elementwise() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3.], vec![3]);
    let px = g.pad(x, vec![(1, 1)]).unwrap(); // [0,1,2,3,0]
    let b = g.constant(vec![10., 20., 30., 40., 50.], vec![5]);
    let y = g.add(px, b).unwrap();
    assert!(realize(&g, y).is_fused());
    assert_eq!(force(&g, y), interpret(&g, y));
}

// movement on top of a padded view materializes the pad first; still correct
#[test]
fn movement_on_pad_materializes() {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3., 4.], vec![2, 2]);
    let px = g.pad(x, vec![(0, 1), (1, 0)]).unwrap();
    let t = g.permute(px, vec![1, 0]).unwrap();
    assert_eq!(force(&g, t), interpret(&g, t));
}

// reshape after permute can't lower -> materialize; still correct
#[test]
fn noncontig_reshape_materializes() {
    let mut g = Graph::new();
    let a = g.constant((0..6).map(|x| x as f32).collect(), vec![2, 3]);
    let p = g.permute(a, vec![1, 0]).unwrap();
    let r = g.reshape(p, vec![6]).unwrap();
    assert_eq!(force(&g, r), interpret(&g, r));
}
