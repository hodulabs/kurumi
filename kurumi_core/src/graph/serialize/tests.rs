use crate::graph::serialize::decode::{Reader, read_op};
use crate::graph::serialize::encode::write_op;
use crate::graph::serialize::*;
use crate::graph::{ArgKind, Op, ScatterOp};
use crate::{DType, Feeds, Storage, TensorVal, interpret_with};

// every Op variant, with structurally valid attrs (encoding needs no valid graph).
fn all_ops() -> Vec<Op> {
    vec![
        Op::Const { data: Storage::F32(vec![1.5, -2.5]), shape: vec![2] },
        Op::Input { shape: vec![2, 3], dtype: DType::F32 },
        Op::Iota { shape: vec![4], axis: 0, dtype: DType::I64 },
        Op::RandUniform { shape: vec![2, 2] },
        Op::Cast { to: DType::F16 },
        Op::Bitcast { to: DType::U32 },
        Op::Detach,
        Op::Add,
        Op::Mul,
        Op::Max,
        Op::Neg,
        Op::IDiv,
        Op::And,
        Op::Or,
        Op::Xor,
        Op::Shl,
        Op::Shr,
        Op::CmpLt,
        Op::CmpEq,
        Op::Where,
        Op::Recip,
        Op::Sqrt,
        Op::Exp2,
        Op::Log2,
        Op::Sin,
        Op::Floor,
        Op::Sum { axis: 1 },
        Op::Prod { axis: 0 },
        Op::ReduceMax { axis: 2 },
        Op::ArgReduce { axis: 1, kind: ArgKind::Min },
        Op::Softmax { axis: 0 },
        Op::RmsNorm { axis: 1, eps: 1e-5 },
        Op::Sdpa { causal: true },
        Op::Reshape { shape: vec![6] },
        Op::Permute { perm: vec![1, 0] },
        Op::Expand { shape: vec![2, 3, 4] },
        Op::Slice { ranges: vec![(0, 2, 1), (1, 3, 2)] },
        Op::Flip { axes: vec![0, 2] },
        Op::Pad { pads: vec![(1, 1), (0, 2)] },
        Op::DotGeneral { lhs_contract: vec![1], rhs_contract: vec![0], lhs_batch: vec![], rhs_batch: vec![] },
        Op::QuantMatmul { bits: 4, group_size: 64, symmetric: false },
        Op::Solve,
        Op::Det,
        Op::Cholesky,
        Op::Eigh,
        Op::Qr { r_factor: true },
        Op::Eigvals,
        Op::Complex,
        Op::Real,
        Op::Imag,
        Op::Gather { axis: 0 },
        Op::Scatter { axis: 1, combine: ScatterOp::Add },
        Op::GatherAlong { axis: 2 },
        Op::ScatterAlong { axis: 0, combine: ScatterOp::Max },
        Op::Argsort { axis: 1, descending: true },
    ]
}

// exhaustive codec guard: every op tag encodes and decodes back to an identical op.
// (Op has no PartialEq, so compare its Debug form -- which includes Const bytes.)
#[test]
fn every_op_round_trips() {
    let ops = all_ops();
    assert_eq!(ops.len(), 55, "all 55 op variants must be covered");
    for op in &ops {
        let mut buf = Vec::new();
        write_op(&mut buf, op);
        let mut r = Reader::new(&buf);
        let back = read_op(&mut r).expect("decode");
        assert_eq!(r.pos, buf.len(), "trailing bytes for {op:?}");
        assert_eq!(format!("{op:?}"), format!("{back:?}"));
    }
}

// whole-blob path: a real graph serializes, replays (re-inferring shapes), and computes
// the identical value; the output/input metadata round-trips too.
#[test]
fn graph_blob_round_trips() {
    let mut g = Graph::new();
    let x = g.input(vec![2, 3], DType::F32);
    let w = g.constant(vec![2.0; 6], vec![2, 3]);
    let a = g.push(Op::Add, vec![x, w]);
    let m = g.push(Op::Mul, vec![a, w]);
    let out = g.push(Op::Sum { axis: 1 }, vec![m]);

    let outputs = vec![out];
    let inputs = vec![InputBinding { node: x, role: InputRole::Runtime, name: "x".into() }];
    let blob = serialize_graph(&g, &outputs, &inputs);

    let r = deserialize_graph(&blob).expect("deserialize");
    assert_eq!(r.outputs, outputs);
    assert_eq!(r.inputs, inputs);

    // replay is id-preserving, so the same feed (keyed by NodeId) drives both graphs.
    let mut feeds = Feeds::new();
    feeds.insert(x, TensorVal { shape: vec![2, 3], storage: Storage::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) });
    let want = interpret_with(&g, out, &feeds);
    let got = interpret_with(&r.graph, r.outputs[0], &feeds);
    assert_eq!(want, got);
}

// serialize_reachable drops arena nodes no output depends on, remaps to a dense range,
// and still computes the same value.
#[test]
fn reachable_prunes_dead_nodes() {
    let mut g = Graph::new();
    let x = g.input(vec![2, 3], DType::F32);
    let w = g.constant(vec![2.0; 6], vec![2, 3]);
    let _dead = g.push(Op::Neg, vec![w]); // reachable from w, but not an ancestor of `out`
    let a = g.push(Op::Add, vec![x, w]);
    let out = g.push(Op::Sum { axis: 1 }, vec![a]);

    let outputs = vec![out];
    let inputs = vec![InputBinding { node: x, role: InputRole::Runtime, name: "x".into() }];

    let full = serialize_graph(&g, &outputs, &inputs);
    let pruned = serialize_reachable(&g, &outputs, &inputs);
    assert!(pruned.len() < full.len(), "the dead Neg node must be dropped");

    let r = deserialize_graph(&pruned).expect("deserialize");
    assert_eq!(r.inputs.len(), 1);
    assert_eq!(r.inputs[0].name, "x");

    // ids were remapped, so feed via the returned bindings, not the original NodeIds.
    let mut feeds = Feeds::new();
    feeds.insert(
        r.inputs[0].node,
        TensorVal { shape: vec![2, 3], storage: Storage::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) },
    );
    let got = interpret_with(&r.graph, r.outputs[0], &feeds);
    assert_eq!(got.f32().to_vec(), vec![12.0, 21.0]);
}

#[test]
fn rejects_bad_magic() {
    assert!(deserialize_graph(b"XXXX\x01").is_err());
    assert!(deserialize_graph(&[]).is_err());
}
