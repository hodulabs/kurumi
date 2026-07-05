# kurumi_metal

The Apple Silicon backend for [kurumi](https://github.com/hodu-labs/kurumi): a Metal implementation of `kurumi_core::Backend`. macOS only.

## Design

- **MPS for GEMM, fused MSL for the rest.** Matmul runs on MetalPerformanceShaders; elementwise chains compile to one fused MSL kernel; reduce, broadcast, movement, gather/scatter, and RoPE/RMSNorm/SiLU/SDPA run device-resident.
- **Batched, device-resident execution.** GPU ops encode into one command buffer through a shared compute encoder. Intermediate buffers are pooled and MPS kernel objects cached, so a re-evaluated graph (a training loop) uploads weights and allocates buffers once.
- **Checked against the oracle.** Every kernel is validated against the `kurumi_core` interpreter; an op with no device kernel falls back to the CPU reference.
- **objc2-metal bindings.** No hand-rolled Objective-C FFI beyond the thin device layer.

## Example

```rust
use kurumi_core::{Backend, Graph};
use kurumi_metal::MetalBackend;

let backend = MetalBackend::new().expect("no Metal device");
let out = backend.eval(&graph, node);   // GPU where accelerated, CPU fallback otherwise
```

## Layout

- `context.rs`, `context/dispatch/` -- the device layer: buffers, command batching, kernel launchers
- `backend.rs`, `backend/eval/` -- the engine seam: which ops run device-resident
- `msl.rs` -- MSL kernel sources and per-dtype generators
- `dtype.rs` -- dtype-to-MSL type mapping
