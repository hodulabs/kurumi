//! Microbench harness (zero-dep std): Instant for time, a counting global allocator
//! for the memory fusion (realize) cuts vs the materializing oracle (interpret).
//! Four metrics: record ns/op (gate <100), fusion kernel count (realize passes),
//! GEMM GFLOPS, and memory bandwidth vs a streaming-copy ceiling.
//! Run: cargo bench -p kurumi_core

use kurumi_core::{Graph, NodeId, interpret, realize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

// counting allocator: total alloc calls + peak live bytes, per measured iteration
struct Counting;
static N_ALLOC: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        N_ALLOC.fetch_add(1, Relaxed);
        let live = LIVE.fetch_add(l.size(), Relaxed) + l.size();
        PEAK.fetch_max(live, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

// timing core: median wall time over many iterations (warm up first)
fn time_median<T>(mut f: impl FnMut() -> T) -> Duration {
    for _ in 0..10 {
        black_box(f());
    }
    let iters = 200;
    let mut samples: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        black_box(f());
        samples.push(t.elapsed());
    }
    samples.sort_unstable();
    samples[iters / 2]
}

// full line with the allocation snapshot (isolates fusion's memory win)
fn bench<T>(name: &str, mut f: impl FnMut() -> T) {
    black_box(f());
    N_ALLOC.store(0, Relaxed);
    LIVE.store(0, Relaxed);
    PEAK.store(0, Relaxed);
    black_box(f());
    let (allocs, peak) = (N_ALLOC.load(Relaxed), PEAK.load(Relaxed));
    let median = time_median(&mut f);
    println!("   {name:20} median {median:>10.2?}  allocs {allocs:>6}  peak {:>7} KB", peak / 1024);
}

fn weight(g: &mut Graph, rows: usize, cols: usize, seed: f32) -> NodeId {
    let data = (0..rows * cols).map(|i| ((i as f32 + 1.0) * seed).sin() * 0.1).collect();
    g.constant(data, vec![rows, cols])
}

fn iota(g: &mut Graph, n: usize, side: usize, scale: f32) -> NodeId {
    g.constant((0..n).map(|i| i as f32 * scale).collect(), vec![side, n / side])
}

// a GPT-2-style block (single head): attention + MLP + 2 residuals + 2 layernorms
fn block(s: usize, d: usize, h: usize) -> (Graph, NodeId) {
    let mut g = Graph::new();
    let x = g.constant((0..s * d).map(|i| i as f32 * 0.01).collect(), vec![s, d]);
    let (wq, wk, wv, wo) =
        (weight(&mut g, d, d, 0.3), weight(&mut g, d, d, 0.5), weight(&mut g, d, d, 0.7), weight(&mut g, d, d, 0.9));
    let (w1, w2) = (weight(&mut g, d, h, 0.2), weight(&mut g, h, d, 0.4));
    let q = g.dot_general(x, wq, vec![1], vec![0], vec![], vec![]).unwrap();
    let k = g.dot_general(x, wk, vec![1], vec![0], vec![], vec![]).unwrap();
    let v = g.dot_general(x, wv, vec![1], vec![0], vec![], vec![]).unwrap();
    let scores = g.dot_general(q, k, vec![1], vec![1], vec![], vec![]).unwrap();
    let scale = g.scalar(scores, 1.0 / (d as f32).sqrt());
    let scaled = g.mul(scores, scale).unwrap();
    let attn = g.softmax(scaled, 1).unwrap();
    let ctx = g.dot_general(attn, v, vec![1], vec![0], vec![], vec![]).unwrap();
    let proj = g.dot_general(ctx, wo, vec![1], vec![0], vec![], vec![]).unwrap();
    let r1 = g.add(x, proj).unwrap();
    let ln1 = g.layernorm(r1, 1, 1e-5).unwrap();
    let hpre = g.dot_general(ln1, w1, vec![1], vec![0], vec![], vec![]).unwrap();
    let hact = g.gelu(hpre);
    let mlp = g.dot_general(hact, w2, vec![1], vec![0], vec![], vec![]).unwrap();
    let r2 = g.add(ln1, mlp).unwrap();
    let out = g.layernorm(r2, 1, 1e-5).unwrap();
    (g, out)
}

// a deep single-consumer elementwise chain: one giant fused group for realize
// (one pass, no intermediates) vs N materialized buffers for interpret.
fn ew_chain(n: usize, depth: usize) -> (Graph, NodeId) {
    let mut g = Graph::new();
    let c = iota(&mut g, n * n, n, 1e-3);
    let k = iota(&mut g, n * n, n, 2e-3);
    let mut a = c;
    for _ in 0..depth {
        a = g.mul(a, k).unwrap();
        a = g.add(a, k).unwrap();
        a = g.max(a, c).unwrap();
    }
    (g, a)
}

fn softmax_graph(n: usize) -> (Graph, NodeId) {
    let mut g = Graph::new();
    let x = iota(&mut g, n * n, n, 1e-3);
    let y = g.softmax(x, 1).unwrap();
    (g, y)
}

// permute -> slice -> reshape: all view algebra, no compute kernel (the reshape of
// a non-contiguous view is the only forced gather).
fn movement_graph(n: usize) -> (Graph, NodeId) {
    let mut g = Graph::new();
    let x = iota(&mut g, n * n, n, 1.0);
    let p = g.permute(x, vec![1, 0]).unwrap();
    let s = g.slice(p, vec![(0, n / 2), (0, n)]).unwrap();
    let r = g.reshape(s, vec![n / 2 * n]).unwrap();
    (g, r)
}

// metric reporters

fn kernels(name: &str, (g, out): (Graph, NodeId), note: &str) {
    match realize::force_counted(&g, out) {
        (_, Some(k)) => println!("   {name:30} {k:>2} kernels   ({note})"),
        (_, None) => println!("   {name:30}  oracle    (leaves fused path: {note})"),
    }
}

fn gemm_gflops(n: usize) {
    let mut g = Graph::new();
    let a = weight(&mut g, n, n, 0.3);
    let b = weight(&mut g, n, n, 0.7);
    let c = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
    let dur = time_median(|| realize::force(&g, c));
    let gflops = 2.0 * (n as f64).powi(3) / dur.as_secs_f64() / 1e9;
    println!("   gemm {n:>4}^3        median {dur:>10.2?}  =>  {gflops:>7.1} GFLOPS");
}

// streaming-copy ceiling: the achievable DRAM bandwidth on this machine (read +
// write, buffer >> LLC). Every memory-bound op is reported as a % of this.
fn copy_ceiling(n: usize) -> f64 {
    let src = vec![1.0f32; n];
    let mut dst = vec![0.0f32; n];
    let dur = time_median(|| {
        dst.copy_from_slice(black_box(&src));
        black_box(dst[0])
    });
    let gbps = 2.0 * n as f64 * 4.0 / dur.as_secs_f64() / 1e9;
    println!("   copy   {:>4} MB    median {dur:>10.2?}  =>  {gbps:>6.1} GB/s  (100% ceiling)", (n * 4) >> 20);
    gbps
}

fn ew_bandwidth(n: usize, ceiling: f64) {
    let side = (n as f64).sqrt() as usize;
    let mut g = Graph::new();
    let a = g.constant(vec![1.5f32; side * side], vec![side, side]);
    let y = g.neg(a); // read a + write y = 2 arrays, same traffic as copy
    let bytes = 2.0 * (side * side) as f64 * 4.0;
    let mb = (side * side * 4) >> 20;

    // fresh vs reused output buffer come out ~equal -- the finding: the shallow-op gap
    // to the copy ceiling is NOT output allocation but per-call const re-materialization
    // (realize rebuilds leaves each call; `Rc::from(data.as_f32())` copies the whole
    // const buffer). The executor streams in one pass (win shows on cache-resident
    // ew_chain); plan-replay kills the re-copy.
    let fresh = time_median(|| realize::force(&g, y));
    let fresh_gbps = bytes / fresh.as_secs_f64() / 1e9;
    let mut out = Vec::new();
    let reuse = time_median(|| realize::force_into(&g, y, &mut out));
    let reuse_gbps = bytes / reuse.as_secs_f64() / 1e9;
    // plan-replay: const materialized once at compile, output reused -> the const
    // re-copy is gone, so this is the executor's true streaming bandwidth.
    let plan = realize::Plan::compile(&g, y).unwrap();
    let feeds = Default::default();
    let mut pout = Vec::new();
    plan.run_into(&g, &feeds, &mut pout); // warm
    let planned = time_median(|| plan.run_into(&g, &feeds, &mut pout));
    let plan_gbps = bytes / planned.as_secs_f64() / 1e9;
    println!("   neg {mb:>4} MB fresh  {fresh:>9.2?}  {fresh_gbps:>6.1} GB/s ({:>3.0}%)", fresh_gbps / ceiling * 100.0);
    println!("   neg {mb:>4} MB reuse  {reuse:>9.2?}  {reuse_gbps:>6.1} GB/s ({:>3.0}%)", reuse_gbps / ceiling * 100.0);
    println!("   neg {mb:>4} MB plan   {planned:>9.2?}  {plan_gbps:>6.1} GB/s ({:>3.0}%)", plan_gbps / ceiling * 100.0);
}

fn main() {
    println!("=== microbench -- CPU f32, zero-dep std harness ===\n");

    // 1. record overhead (bump-arena append; gate < 100 ns/op)
    println!("-- record overhead (gate < 100 ns/op) --");
    let ops = 1000usize;
    let dur = time_median(|| {
        let mut g = Graph::new();
        let mut a = g.constant(vec![0.0; 64], vec![64]);
        for _ in 0..ops {
            a = g.neg(a);
        }
        a
    });
    let per = dur.as_nanos() as f64 / ops as f64;
    println!("   record {ops}xneg     {per:>6.1} ns/op   [{}]\n", if per < 100.0 { "PASS" } else { "FAIL" });

    // 2. fusion kernel count (realize compute passes)
    println!("-- fusion kernel count (compute passes over data) --");
    kernels("ew_chain 256^2 depth 16", ew_chain(256, 16), "elementwise chain -> 1");
    kernels("softmax 256^2", softmax_graph(256), "target <= 3");
    kernels("movement perm after slice after reshape", movement_graph(256), "view algebra -> ~0");
    kernels("gpt2_block 16x64x256", block(16, 64, 256), "attention + MLP");
    println!();

    // 3. realize vs interpret on a genuinely-fused graph (ew_chain fuses to 1
    // kernel; softmax/block leave the fused path so they'd just be oracle==oracle).
    println!("-- realize vs interpret (fusion cuts intermediates) --");
    let (g, out) = ew_chain(256, 16);
    bench("ew_interpret", || interpret(&g, out));
    bench("ew_realize", || realize::force(&g, out));
    let (g, out) = block(16, 64, 256);
    bench("block (oracle)", || interpret(&g, out));
    println!();

    // 4. GEMM GFLOPS (f32; Accelerate/AMX on macOS, gemm crate elsewhere)
    println!("-- GEMM (f32, Accelerate cblas_sgemm on macOS) --");
    for n in [256, 512, 1024] {
        gemm_gflops(n);
    }
    println!();

    // 5. memory bandwidth (self-calibrated vs streaming-copy ceiling)
    println!("-- memory bandwidth (vs streaming-copy ceiling) --");
    let ceiling = copy_ceiling(16 << 20); // 64 MB f32
    ew_bandwidth(16 << 20, ceiling);
}
