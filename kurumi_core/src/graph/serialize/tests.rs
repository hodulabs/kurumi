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

// two named entries share one arena: their output cones overlap (both go through `a`), and
// serialize_multi prunes to the union, dense-remaps once, and writes both entries. Each
// entry's outputs/inputs round-trip and eval correctly against the shared rebuilt graph.
#[test]
fn multi_entry_round_trips() {
    let mut g = Graph::new();
    let x = g.input(vec![2, 3], DType::F32);
    let w = g.constant(vec![2.0; 6], vec![2, 3]);
    let a = g.push(Op::Add, vec![x, w]); // shared by both cones
    let fwd = g.push(Op::Sum { axis: 1 }, vec![a]);
    let scaled = g.push(Op::Mul, vec![a, w]);
    let bwd = g.push(Op::Sum { axis: 1 }, vec![scaled]);

    let fwd_in = vec![InputBinding { node: x, role: InputRole::Runtime, name: "x".into() }];
    let bwd_in = vec![
        InputBinding { node: x, role: InputRole::Runtime, name: "x".into() },
        InputBinding { node: a, role: InputRole::Weight, name: "a".into() },
    ];
    let blob = serialize_multi(&g, &[("forward", &[fwd], &fwd_in), ("forward_backward", &[bwd], &bwd_in)]);

    let mr = deserialize_multi(&blob).expect("deserialize_multi");
    assert_eq!(mr.entries.len(), 2);
    assert_eq!(mr.entries[0].name, "forward");
    assert_eq!(mr.entries[1].name, "forward_backward");
    assert_eq!(mr.entries[0].inputs.len(), 1);
    assert_eq!(mr.entries[1].inputs.len(), 2);
    assert_eq!(mr.entries[1].inputs[1].name, "a");
    assert_eq!(mr.entries[1].inputs[1].role, InputRole::Weight);

    // "x" is the same shared node in both entries (one arena) -- feed it once, eval both outputs.
    let x_node = mr.entries[0].inputs[0].node;
    assert_eq!(mr.entries[1].inputs[0].node, x_node);
    let mut feeds = Feeds::new();
    feeds.insert(x_node, TensorVal { shape: vec![2, 3], storage: Storage::F32(vec![1., 2., 3., 4., 5., 6.]) });
    let fwd_val = interpret_with(&mr.graph, mr.entries[0].outputs[0], &feeds);
    let bwd_val = interpret_with(&mr.graph, mr.entries[1].outputs[0], &feeds);
    assert_eq!(fwd_val.f32().to_vec(), vec![12.0, 21.0]);
    assert_eq!(bwd_val.f32().to_vec(), vec![24.0, 42.0]);

    // entry 0 is what the back-compat single-entry path returns.
    let r = deserialize_graph(&blob).expect("deserialize_graph");
    assert_eq!(r.outputs, mr.entries[0].outputs);
    assert_eq!(r.inputs, mr.entries[0].inputs);
}

const GOLDEN_V2: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/graph_v2.kgph");

// a fixed all-const graph whose serialized bytes are frozen as the v2 wire-format baseline.
fn golden_graph() -> (Graph, Vec<NodeId>, Vec<InputBinding>) {
    let mut g = Graph::new();
    let x = g.constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let w = g.constant(vec![2.0; 6], vec![2, 3]);
    let a = g.push(Op::Add, vec![x, w]);
    let out = g.push(Op::Sum { axis: 1 }, vec![a]);
    (g, vec![out], vec![])
}

// dev tool (run with --ignored): (re)generate the committed v2 golden blob after an
// intentional wire-format change.
#[test]
#[ignore = "regenerates the committed v2 golden blob"]
fn gen_golden_blob() {
    let (g, outs, ins) = golden_graph();
    let blob = serialize_graph(&g, &outs, &ins);
    std::fs::create_dir_all(std::path::Path::new(GOLDEN_V2).parent().unwrap()).unwrap();
    std::fs::write(GOLDEN_V2, blob).unwrap();
}

// the v2 wire format is frozen: re-serializing the fixed graph must reproduce the committed
// bytes (catches an accidental encoder change that a round-trip alone would miss), and the
// committed bytes still decode and compute the known value.
#[test]
fn golden_blob_is_stable() {
    let golden = std::fs::read(GOLDEN_V2).expect("run `cargo test gen_golden_blob -- --ignored` first");
    let (g, outs, ins) = golden_graph();
    assert_eq!(serialize_graph(&g, &outs, &ins), golden, "v2 graph wire format changed; regenerate if intentional");
    let r = deserialize_graph(&golden).unwrap();
    assert_eq!(interpret_with(&r.graph, r.outputs[0], &Feeds::new()).f32(), &[12.0, 21.0]);
}

#[test]
fn rejects_bad_magic() {
    assert!(deserialize_graph(b"XXXX\x01").is_err());
    assert!(deserialize_graph(&[]).is_err());
}

// A structurally valid blob whose only node's src references itself (id 0, not yet built) must
// be a clean error, not an index-OOB panic in inference. (Hand-forged bytes: MAGIC + VERSION,
// then one Add node with src [0].)
#[test]
fn rejects_forward_src_ref() {
    let mut b = b"KGPH\x02".to_vec();
    b.extend_from_slice(&1u32.to_le_bytes()); // n = 1 node
    b.push(7); // Op::Add (no attrs)
    b.extend_from_slice(&1u32.to_le_bytes()); // 1 src
    b.extend_from_slice(&0u32.to_le_bytes()); // src = node 0 (itself / forward)
    assert!(deserialize_graph(&b).is_err(), "a self/forward src ref must be a clean error");
}

// A Const whose payload holds fewer elements than its declared shape (here 1 f32 for shape [2])
// must be a clean error, not a short storage that OOBs downstream.
#[test]
fn rejects_const_payload_underfill() {
    let mut b = b"KGPH\x02".to_vec();
    b.extend_from_slice(&1u32.to_le_bytes()); // n = 1 node
    b.push(0); // Op::Const
    b.push(13); // storage dtype = F32
    b.extend_from_slice(&4u64.to_le_bytes()); // nbytes = 4 (one f32)
    b.extend_from_slice(&1.0f32.to_le_bytes()); // payload: one f32
    b.extend_from_slice(&1u32.to_le_bytes()); // shape rank = 1
    b.extend_from_slice(&2u64.to_le_bytes()); // shape = [2] -> needs 2 elems, got 1
    assert!(deserialize_graph(&b).is_err(), "a Const underfilling its shape must be a clean error");
}
