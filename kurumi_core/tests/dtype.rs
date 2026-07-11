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
fn integer_dot_general_all_dtypes() {
    // [1,2;3,4] @ identity = itself, for every integer dtype. The interp dispatch used to wire
    // only I32/I64/U32/U8, so a valid I8/I16/U16/U64 matmul panicked at eval.
    macro_rules! check {
        ($variant:ident, $ty:ty) => {{
            let mut g = Graph::new();
            let a = g.const_storage(Storage::$variant(vec![1 as $ty, 2, 3, 4]), vec![2, 2]);
            let id = g.const_storage(Storage::$variant(vec![1 as $ty, 0, 0, 1]), vec![2, 2]);
            let y = g.dot_general(a, id, vec![1], vec![0], vec![], vec![]).unwrap();
            let Storage::$variant(v) = interpret(&g, y).storage else { panic!("want {}", stringify!($variant)) };
            assert_eq!(v, vec![1 as $ty, 2, 3, 4], "{} dot_general", stringify!($variant));
        }};
    }
    check!(I8, i8);
    check!(I16, i16);
    check!(I32, i32);
    check!(I64, i64);
    check!(U8, u8);
    check!(U16, u16);
    check!(U32, u32);
    check!(U64, u64);
}

#[test]
fn gather_scatter_all_integer_dtypes() {
    // the index-tensor and scatter-combiner dispatches enumerated a subset of the integer
    // dtypes the builder accepts, so gather with u16/i8/u64/... indices, or an i8/u16/...
    // scatter-add operand, panicked at eval on a valid graph.
    macro_rules! gather_with_idx {
        ($variant:ident, $ty:ty) => {{
            let mut g = Graph::new();
            let x = g.constant(vec![10.0, 20.0, 30.0], vec![3]);
            let idx = g.const_storage(Storage::$variant(vec![2 as $ty, 0]), vec![2]);
            let y = g.gather(x, idx, 0).unwrap();
            assert_eq!(interpret(&g, y).f32(), &[30.0, 10.0], "gather idx {}", stringify!($variant));
        }};
    }
    gather_with_idx!(U8, u8);
    gather_with_idx!(U16, u16);
    gather_with_idx!(U32, u32);
    gather_with_idx!(U64, u64);
    gather_with_idx!(I8, i8);
    gather_with_idx!(I16, i16);
    gather_with_idx!(I32, i32);
    gather_with_idx!(I64, i64);

    // integer scatter-add on the operand dtypes the combiner dispatch was missing.
    macro_rules! scatter_add {
        ($variant:ident, $ty:ty) => {{
            let mut g = Graph::new();
            let op = g.const_storage(Storage::$variant(vec![1 as $ty, 1, 1]), vec![3]);
            let idx = g.const_storage(Storage::I64(vec![0, 0, 2]), vec![3]);
            let up = g.const_storage(Storage::$variant(vec![5 as $ty, 3, 7]), vec![3]);
            let y = g.scatter(op, idx, up, 0, ScatterOp::Add).unwrap();
            let Storage::$variant(v) = interpret(&g, y).storage else { panic!("dtype changed") };
            assert_eq!(v, vec![9 as $ty, 1, 8], "scatter-add {}", stringify!($variant)); // op[0]+=5+3, op[2]+=7
        }};
    }
    scatter_add!(I8, i8);
    scatter_add!(I16, i16);
    scatter_add!(U16, u16);
    scatter_add!(U64, u64);
}

#[test]
fn complex_is_rejected_at_record_time() {
    // bitwise/softmax/rmsnorm/sdpa builders admitted complex, which then panicked (CPU) or was
    // silently computed on the real part (softmax cast C64->F32); they must Err at record time.
    let mut g = Graph::new();
    let f = g.constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let c = g.cast(f, DType::C64);
    assert!(g.and(c, c).is_err(), "bitwise and on complex");
    assert!(g.or(c, c).is_err());
    assert!(g.xor(c, c).is_err());
    assert!(g.softmax(c, 1).is_err(), "softmax on complex");
    assert!(g.rmsnorm(c, 1, 1e-5).is_err(), "rmsnorm on complex");
    let f3 = g.constant(vec![0.5; 8], vec![2, 2, 2]);
    let c3 = g.cast(f3, DType::C64);
    assert!(g.sdpa(c3, c3, c3, false).is_err(), "sdpa on complex");
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
