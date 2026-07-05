//! Debug-only profiling counters (env-gated, off the hot path unless requested).
//! `KURUMI_GPUTIME` accumulates flush count + summed GPU time; the dispatch
//! counter tallies kernels by kind. Surfaced to tests/benches via `take_*`.

use crate::context::MetalContext;

thread_local! {
    static FLUSH_N: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static FLUSH_GPU_US: std::cell::Cell<u128> = const { std::cell::Cell::new(0) };
    // dispatch counts by kind: [matmul, fused, reduce, strided, gather, cast, pad]
    static DISPATCH: std::cell::Cell<[u32; 7]> = const { std::cell::Cell::new([0; 7]) };
}

impl MetalContext {
    // accumulate one flush + its GPU time (called from `flush` when GPUTIME is on).
    pub(crate) fn bump_flush(gpu_us: u128) {
        FLUSH_N.with(|c| c.set(c.get() + 1));
        FLUSH_GPU_US.with(|c| c.set(c.get() + gpu_us));
    }
    /// (#flushes, summed GPU ms) since last reset: host-bound vs GPU-bound probe.
    pub fn take_flush_stats() -> (u32, f64) {
        (FLUSH_N.with(|c| c.replace(0)), FLUSH_GPU_US.with(|c| c.replace(0)) as f64 / 1e3)
    }
    // tally one dispatch of the given kind (called at each device-op launch).
    pub(crate) fn tick(kind: usize) {
        DISPATCH.with(|c| {
            let mut a = c.get();
            a[kind] += 1;
            c.set(a);
        });
    }
    /// dispatch counts [matmul, fused, reduce, strided, gather, cast, pad] since reset.
    pub fn take_dispatch() -> [u32; 7] {
        DISPATCH.with(|c| c.replace([0; 7]))
    }
}
