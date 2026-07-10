//! Engine-facing backend: implements `kurumi_core::Backend::eval` (whole-graph
//! execution: GPU matmul, CPU fallback for the rest) plus the GPU GEMM/cast
//! dispatch by dtype.

mod eval;
mod fuse;
mod hostgemm;

use crate::dtype::{dev_dtype, msl_ty};
use crate::{Buffer, MetalContext, Pipeline};
use fuse::{FExpr, Leaf, Val, View, fused_msl};
use kurumi_core::{Backend, DType, Feeds, Graph, NodeId, Op, Storage, TensorVal, eval_op};
use std::cell::RefCell;
use std::collections::HashMap;

/// Engine backend backed by the Metal device. Implements `kurumi_core::Backend`.
/// f32 pointwise ops fuse into one kernel; reduce/broadcast/matmul are device
/// primitives; the rest fall back to the CPU oracle.
pub struct MetalBackend {
    ctx: MetalContext,
    gemm_f32: Pipeline, // host-path simdgroup GEMM (f16/int via host_op)
    gemm_f16: Pipeline,
    // uploaded constant (weight) buffers, keyed by (graph id, node id): a re-evaluated
    // fixed graph (training loop) uploads its weights ONCE, not every eval. Graph-id key
    // => ABA-safe across graphs sharing NodeIds. Read-only, so sharing across evals is safe.
    const_cache: RefCell<HashMap<(u64, u32), Buffer>>,
}

impl MetalBackend {
    pub fn new() -> Option<Self> {
        let ctx = MetalContext::new()?;
        let gemm_f32 = ctx.gemm_pipeline();
        let gemm_f16 = ctx.gemm_f16_pipeline();
        Some(Self { ctx, gemm_f32, gemm_f16, const_cache: RefCell::new(HashMap::new()) })
    }
}

impl MetalBackend {
    /// Cast a storage to any dtype. Metal-native pairs (f32/f16/bf16 + bool + 8 ints)
    /// run the GPU cast kernel (`!=0` for bool targets, saturating truncation for
    /// float->int); f64/fp8/complex fall back to the CPU oracle, so every pair works.
    pub fn cast(&self, src: &Storage, to: DType) -> Result<Storage, kurumi_core::Error> {
        let from = src.dtype();
        // complex casts (re/im preservation, part extraction) aren't the scalar GPU cast
        // -> CPU oracle. Other Metal-native pairs use the GPU kernel.
        if dev_dtype(from) && dev_dtype(to) && !from.is_complex() && !to.is_complex() {
            let n = src.len();
            let buf = self.ctx.upload(src);
            let out = self.ctx.cast_dev(&buf, n, from, to);
            self.ctx.flush();
            return Ok(self.ctx.download(&out, n, to));
        }
        // f64/fp8/complex have no device path -> the CPU cast (exact, all pairs).
        let tv = TensorVal { shape: vec![src.len()], storage: src.clone() };
        Ok(eval_op(&Op::Cast { to }, &[&tv]).storage)
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }
    /// Evaluate the whole graph (Input nodes from `feeds`). f32 pointwise ops fuse
    /// and run device-resident, matmul/reduce/broadcast/permute are device
    /// primitives, the rest fall back to the CPU oracle. Any graph runs on Metal.
    fn eval_with(&self, g: &Graph, id: NodeId, feeds: &Feeds) -> TensorVal {
        self.ctx.recycle(); // reclaim the prior eval's intermediate buffers
        if std::env::var_os("KURUMI_PHASE").is_some() {
            let _ = MetalContext::take_flush_stats();
            let t0 = std::time::Instant::now();
            let v = self.eval_memo(g, id, feeds, &mut HashMap::new());
            let enc = t0.elapsed();
            let t1 = std::time::Instant::now();
            let out = self.to_host(&v);
            let (nf, gpu) = MetalContext::take_flush_stats();
            let d = MetalContext::take_dispatch();
            eprintln!(
                "  encode {:.2} ms | flush+dl {:.2} ms | flushes {nf}, GPU {gpu:.2} ms | mm {} fused {} reduce {} strided {} gather {} cast {} pad {}",
                enc.as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3,
                d[0],
                d[1],
                d[2],
                d[3],
                d[4],
                d[5],
                d[6]
            );
            return out;
        }
        let v = self.eval_memo(g, id, feeds, &mut HashMap::new());
        self.to_host(&v)
    }

    /// Evaluate several nodes in ONE shared pass: `recycle` the prior eval's buffers
    /// once up front, then walk every id through a single `memo`, so a subgraph common
    /// to the outputs (the forward trunk under many grads) computes once. Memoized
    /// intermediate buffers stay in `inuse` (never recycled mid-pass), so reusing them
    /// across outputs is safe. (No profiling branch here -- keep the shared path clean.)
    fn eval_many_with(&self, g: &Graph, ids: &[NodeId], feeds: &Feeds) -> Vec<TensorVal> {
        self.ctx.recycle(); // reclaim the prior eval's intermediate buffers, once
        let mut memo = HashMap::new();
        ids.iter()
            .map(|&id| {
                let v = self.eval_memo(g, id, feeds, &mut memo);
                self.to_host(&v)
            })
            .collect()
    }
}

impl MetalBackend {
    fn to_host(&self, v: &Val) -> TensorVal {
        match v {
            Val::Host(t) => t.clone(),
            Val::Dev { buf, shape, dt } => {
                self.ctx.flush(); // finish pending GPU work before reading back
                let n = shape.iter().product();
                TensorVal { shape: shape.clone(), storage: self.ctx.download(buf, n, *dt) }
            }
            Val::Fused { shape, leaves, expr, dt } => {
                let buf = self.materialize(shape, leaves, expr, *dt);
                self.ctx.flush();
                let n = shape.iter().product();
                TensorVal { shape: shape.clone(), storage: self.ctx.download(&buf, n, *dt) }
            }
        }
    }
    // get an input as a device buffer; uploads a host value, materializes a fused
    // chain into one kernel. (Only called on f32/f16/bf16 values.)
    fn to_dev(&self, v: &Val) -> Buffer {
        match v {
            Val::Dev { buf, .. } => buf.clone(),
            Val::Host(t) => self.ctx.upload(&t.storage),
            Val::Fused { shape, leaves, expr, dt } => self.materialize(shape, leaves, expr, *dt),
        }
    }

    // emit ONE kernel for the whole fused pointwise chain (output dtype `dt`); each
    // leaf's view (if any) is baked into the kernel's per-leaf index math.
    fn materialize(&self, shape: &[usize], leaves: &[Leaf], expr: &FExpr, dt: DType) -> Buffer {
        let n: usize = shape.iter().product();
        let src = fused_msl(expr, leaves, msl_ty(dt));
        let refs: Vec<&Buffer> = leaves.iter().map(|l| &l.buf).collect();
        self.ctx.fused_ew(&src, &refs, n, dt)
    }

    // fold a movement (broadcast/permute/slice) into a strided fused leaf: the input is
    // materialized once, then read through `view` by a pointwise consumer (no strided_dev
    // dispatch, no enlarged intermediate), or by `materialize` for a non-fusable consumer
    // (reduce/matmul/output).
    fn strided_view(&self, a: Val, base: i64, strides: Vec<i64>, shape: Vec<usize>, dt: DType) -> Val {
        let buf = self.to_dev(&a);
        let view = View { base, strides, out_shape: shape.clone() };
        Val::Fused { shape, leaves: vec![Leaf { buf, view: Some(view) }], expr: FExpr::Leaf(0), dt }
    }

    // view any device-dtype Val as a fused chain (leaves + expr) for combining.
    fn as_fused(&self, v: Val) -> (Vec<Leaf>, FExpr) {
        match v {
            Val::Fused { leaves, expr, .. } => (leaves, expr),
            Val::Dev { buf, .. } => (vec![Leaf::plain(buf)], FExpr::Leaf(0)),
            Val::Host(t) => (vec![Leaf::plain(self.ctx.upload(&t.storage))], FExpr::Leaf(0)),
        }
    }

    fn eval_memo(&self, g: &Graph, id: NodeId, feeds: &Feeds, memo: &mut HashMap<NodeId, Val>) -> Val {
        if let Some(v) = memo.get(&id) {
            return v.clone();
        }
        let node = g.node(id);
        let shape = g.shape(id);
        if matches!(node.op, Op::Input { .. }) {
            let v = Val::Host(feeds.get(&id).expect("metal: missing feed for an Input node").clone());
            memo.insert(id, v.clone());
            return v;
        }
        if matches!(node.op, Op::Detach) {
            // identity: pass the input Val straight through (no kernel, no host round-
            // trip). detach only affects autograd, never the forward value.
            let v = self.eval_memo(g, node.src[0], feeds, memo);
            memo.insert(id, v.clone());
            return v;
        }
        let dt = g.dtype(id);
        let dev = dev_dtype(dt);
        if let Op::Const { data, .. } = &node.op
            && dev
        {
            // weight/constant: upload once, keep device-resident across evals (keyed by
            // graph id -> ABA-safe; read-only, so one buffer is shared across consumers/evals).
            let key = (g.id(), id.0);
            let hit = self.const_cache.borrow().get(&key).cloned(); // drop the borrow before borrow_mut
            let buf = hit.unwrap_or_else(|| {
                let b = self.ctx.upload(data);
                self.const_cache.borrow_mut().insert(key, b.clone());
                b
            });
            let v = Val::Dev { buf, shape, dt };
            memo.insert(id, v.clone());
            return v;
        }
        let v = if let Some(v) = self.eval_matmul(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_quant(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_index(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_pointwise(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_complex(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_reduce_arg(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_linalg(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_generate(g, node, &shape, dt, feeds, memo) {
            v
        } else if let Some(v) = self.eval_nn(g, node, &shape, dt, feeds, memo) {
            v
        } else {
            self.eval_fused(g, node, &shape, dt, feeds, memo)
        };
        memo.insert(id, v.clone());
        v
    }

    fn host_op(&self, op: &Op, refs: &[&TensorVal]) -> TensorVal {
        match op {
            // canonical row-major 2D matmul -> GPU (fall back to CPU on a dtype the
            // device can't run, e.g. f64); batched/transposed dots stay on CPU.
            Op::DotGeneral { lhs_contract, rhs_contract, lhs_batch, rhs_batch }
                if lhs_contract.as_slice() == [1]
                    && rhs_contract.as_slice() == [0]
                    && lhs_batch.is_empty()
                    && rhs_batch.is_empty()
                    && refs[0].shape.len() == 2
                    && refs[1].shape.len() == 2 =>
            {
                let (a, b) = (refs[0], refs[1]);
                let (m, k, nn) = (a.shape[0], a.shape[1], b.shape[1]);
                match self.matmul(&a.storage, m, k, &b.storage, nn) {
                    Ok(storage) => TensorVal { shape: vec![m, nn], storage },
                    Err(_) => eval_op(op, refs),
                }
            }
            op => eval_op(op, refs),
        }
    }
}
