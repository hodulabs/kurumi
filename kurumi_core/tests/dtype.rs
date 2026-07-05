//! Integration tests: dtypes, cast, and cross-dtype coverage (every dtype casts/moves,
//! wrong-dtype ops rejected at record time).

use half::{bf16, f16};
use kurumi_core::*;

const ALL: [DType; 11] = [
    DType::BOOL,
    DType::U8,
    DType::U32,
    DType::I32,
    DType::I64,
    DType::F8E4M3,
    DType::F8E5M2,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

// dtype tests

#[test]
fn dtype_inferred_and_carried() {
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![1, 2, 3]), vec![3]);
    let b = g.const_storage(Storage::I32(vec![4, 5, 6]), vec![3]);
    let s = g.add(a, b).unwrap();
    assert_eq!(g.dtype(s), DType::I32); // add preserves dtype
    let f = g.cast(s, DType::F32);
    assert_eq!(g.dtype(f), DType::F32); // cast sets it
}

#[test]
fn i32_add_mul_sum() {
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![1, 2, 3, 4]), vec![2, 2]);
    let b = g.const_storage(Storage::I32(vec![10, 20, 30, 40]), vec![2, 2]);
    let s = g.mul(a, b).unwrap();
    let y = g.sum(s, 1).unwrap(); // [10+40, 90+160] = [50, 250]
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![2], storage: Storage::I32(vec![50, 250]) });
}

#[test]
fn cast_f32_to_i32_truncates() {
    let mut g = Graph::new();
    let a = g.constant(vec![1.9, -2.1, 3.5], vec![3]);
    let y = g.cast(a, DType::I32);
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![3], storage: Storage::I32(vec![1, -2, 3]) });
}

#[test]
fn cast_i32_to_f32_then_sqrt() {
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![4, 9, 16]), vec![3]);
    let f = g.cast(a, DType::F32);
    let y = g.sqrt(f);
    assert_eq!(interpret(&g, y), TensorVal { shape: vec![3], storage: Storage::F32(vec![2., 3., 4.]) });
}

#[test]
fn wide_narrow_int_dtypes() {
    // the research int set: u8/u16/u64/i8/i16 arith, wrapping, signedness, cast.
    let mut g = Graph::new();
    // u8 wraps at 256: 200 + 100 = 300 -> 44
    let a = g.const_storage(Storage::U8(vec![200, 10]), vec![2]);
    let b = g.const_storage(Storage::U8(vec![100, 5]), vec![2]);
    let s = g.add(a, b).unwrap();
    assert_eq!(interpret(&g, s).storage, Storage::U8(vec![44, 15]));

    // i8 negation + wrap: -MIN wraps to MIN
    let n = g.const_storage(Storage::I8(vec![-5, i8::MIN]), vec![2]);
    let neg = g.neg(n);
    assert_eq!(interpret(&g, neg).storage, Storage::I8(vec![5, i8::MIN]));

    // i16 sum along axis
    let m = g.const_storage(Storage::I16(vec![100, -50, 30, 40]), vec![2, 2]);
    let y = g.sum(m, 1).unwrap();
    assert_eq!(interpret(&g, y).storage, Storage::I16(vec![50, 70]));

    // u64 holds > 2^32 exactly through add
    let big = 5_000_000_000u64;
    let p = g.const_storage(Storage::U64(vec![big]), vec![1]);
    let q = g.const_storage(Storage::U64(vec![1]), vec![1]);
    let r = g.add(p, q).unwrap();
    assert_eq!(interpret(&g, r).storage, Storage::U64(vec![big + 1]));

    // cast u16 -> f32 and back
    let u = g.const_storage(Storage::U16(vec![7, 65535]), vec![2]);
    let f = g.cast(u, DType::F32);
    assert_eq!(interpret(&g, f).storage, Storage::F32(vec![7.0, 65535.0]));
    let back = g.cast(f, DType::U16);
    assert_eq!(interpret(&g, back).storage, Storage::U16(vec![7, 65535]));
}

#[test]
fn f16_add_rounds_like_f32() {
    let mut g = Graph::new();
    let a = g.const_storage(Storage::F16(vec![f16::from_f32(1.5), f16::from_f32(2.25)]), vec![2]);
    let b = g.const_storage(Storage::F16(vec![f16::from_f32(0.5), f16::from_f32(0.75)]), vec![2]);
    let y = g.add(a, b).unwrap();
    let got = interpret(&g, y);
    assert_eq!(got.dtype(), DType::F16);
    let Storage::F16(v) = got.storage else { panic!("want F16") };
    assert_eq!(v[0].to_f32(), 2.0);
    assert_eq!(v[1].to_f32(), 3.0);
}

#[test]
fn bf16_dot_general() {
    // [1,2;3,4] @ identity = itself, in bf16
    let mut g = Graph::new();
    let d = vec![bf16::from_f32(1.), bf16::from_f32(2.), bf16::from_f32(3.), bf16::from_f32(4.)];
    let a = g.const_storage(Storage::BF16(d), vec![2, 2]);
    let id = vec![bf16::from_f32(1.), bf16::ZERO, bf16::ZERO, bf16::from_f32(1.)];
    let i = g.const_storage(Storage::BF16(id), vec![2, 2]);
    let y = g.dot_general(a, i, vec![1], vec![0], vec![], vec![]).unwrap();
    let Storage::BF16(v) = interpret(&g, y).storage else { panic!("want BF16") };
    let got: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();
    assert_eq!(got, vec![1., 2., 3., 4.]);
}

#[test]
fn realize_falls_back_for_non_f32() {
    // an i32 graph: force() must route to the interpreter oracle and be correct
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I32(vec![1, 2, 3]), vec![3]);
    let b = g.const_storage(Storage::I32(vec![4, 5, 6]), vec![3]);
    let y = g.add(a, b).unwrap();
    assert_eq!(kurumi_core::realize::force(&g, y), interpret(&g, y));
    assert_eq!(kurumi_core::realize::force(&g, y).storage, Storage::I32(vec![5, 7, 9]));
}

#[test]
fn mixed_dtype_binary_is_a_record_error() {
    let mut g = Graph::new();
    let a = g.constant(vec![1., 2.], vec![2]);
    let b = g.const_storage(Storage::I32(vec![1, 2]), vec![2]);
    assert!(g.add(a, b).is_err()); // promotion is the frontend's job, not the primitive
}

#[test]
fn cast_every_pair() {
    for &from in &ALL {
        for &to in &ALL {
            let mut g = Graph::new();
            let src = g.constant(vec![1.0, 2.0, 3.0, 0.0], vec![4]);
            let a = g.cast(src, from);
            let c = g.cast(a, to);
            let out = interpret(&g, c);
            assert_eq!(out.dtype(), to, "{from:?} -> {to:?}");
            assert_eq!(out.shape, vec![4]);
        }
    }
}

#[test]
fn movement_every_dtype() {
    for &dt in &ALL {
        let mut g = Graph::new();
        let f = g.constant((0..6).map(|i| i as f32).collect(), vec![2, 3]);
        let a = g.cast(f, dt);
        let p = g.permute(a, vec![1, 0]).unwrap();
        let r = g.reshape(p, vec![6]).unwrap();
        let s = g.slice(r, vec![(1, 5)]).unwrap();
        let y = g.pad(s, vec![(1, 1)]).unwrap();
        assert_eq!(interpret(&g, y).dtype(), dt, "{dt:?}");
    }
}

// integer arithmetic / reductions round-trip through realize's interpreter fallback
#[test]
fn integer_pipeline() {
    let mut g = Graph::new();
    let a = g.const_storage(Storage::I64(vec![1, 2, 3, 4]), vec![2, 2]);
    let b = g.const_storage(Storage::I64(vec![10, 20, 30, 40]), vec![2, 2]);
    let m = g.mul(a, b).unwrap();
    let s = g.sum(m, 1).unwrap();
    assert_eq!(interpret(&g, s).storage, Storage::I64(vec![10 + 40, 90 + 160]));
}

// wrong-dtype ops must error at record time (Result), not panic at eval

fn boolean(g: &mut Graph) -> kurumi_core::NodeId {
    g.const_storage(Storage::BOOL(vec![true, false]), vec![2])
}

#[test]
fn arithmetic_on_bool_is_a_record_error() {
    let mut g = Graph::new();
    let a = boolean(&mut g);
    let b = boolean(&mut g);
    assert!(g.add(a, b).is_err());
    assert!(g.mul(a, b).is_err());
    assert!(g.max(a, b).is_err());
    assert!(g.sum(a, 0).is_err());
    assert!(g.prod(a, 0).is_err());
    assert!(g.reduce_max(a, 0).is_err());
    assert!(g.dot_general(a, b, vec![0], vec![0], vec![], vec![]).is_err());
}

#[test]
fn bitwise_on_float_is_a_record_error() {
    let mut g = Graph::new();
    let a = g.constant(vec![1.0, 2.0], vec![2]);
    let b = g.constant(vec![3.0, 4.0], vec![2]);
    assert!(g.and(a, b).is_err());
    assert!(g.or(a, b).is_err());
    assert!(g.xor(a, b).is_err());
    assert!(g.idiv(a, b).is_err()); // int-only
    assert!(g.shl(a, b).is_err());
}

#[test]
fn dot_general_dtype_mismatch_is_a_record_error() {
    let mut g = Graph::new();
    let a = g.constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = g.const_storage(Storage::I32(vec![1, 2, 3, 4]), vec![2, 2]);
    assert!(g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).is_err());
}
