//! Engine-facing backend: implements `kurumi_core::Backend::eval` (whole-graph
//! execution: GPU matmul, CPU fallback for the rest) plus the GPU GEMM/cast
//! dispatch by dtype.

mod eval;
mod eval_walk;
mod fuse;
mod hostgemm;

use crate::dtype::dev_dtype;
use crate::{Buffer, MetalContext, Pipeline};
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
