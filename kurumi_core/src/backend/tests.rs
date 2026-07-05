use crate::backend::*;
use crate::{Graph, interpret};

#[test]
fn cpu_backend_eval_matches_interpret() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2., 3., 4.], vec![2, 2]);
    let b = g.constant(vec![5., 6., 7., 8.], vec![2, 2]);
    let m = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
    let r = g.add(m, a).unwrap();
    let y = g.sum(r, 1).unwrap();
    assert_eq!(CpuBackend.eval(&g, y), interpret(&g, y));
    assert_eq!(CpuBackend.name(), "cpu");
}

#[test]
fn cpu_backend_matmul_and_cast() {
    let a = Storage::I32(vec![1, 2, 3, 4]);
    let id = Storage::I32(vec![1, 0, 0, 1]);
    assert_eq!(CpuBackend.matmul(&a, 2, 2, &id, 2).unwrap(), Storage::I32(vec![1, 2, 3, 4]));
    assert_eq!(CpuBackend.cast(&a, DType::F32).unwrap(), Storage::F32(vec![1., 2., 3., 4.]));
    let b = Storage::BOOL(vec![true; 4]);
    assert!(CpuBackend.matmul(&b, 2, 2, &b, 2).is_err()); // bool has no matmul
}
