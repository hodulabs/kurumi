# kurumi_core

The engine core of [kurumi](https://github.com/hodulabs/kurumi): a closed-primitive tensor IR, a reference interpreter, reverse-mode autograd, and a view-fused CPU evaluator. Pure Rust; f32 matmul uses the system BLAS (Accelerate on macOS, the `gemm` crate elsewhere).

## Design

- **Closed primitive IR.** Every operation is built from a small fixed set of `Op` primitives (`graph/op.rs`). High-level ops (softmax, attention, conv, FFTs, linear algebra) decompose down to these, so autograd and every backend cover only the primitives -- never the hundreds of surface ops.
- **The interpreter is the oracle.** `interpret` runs each primitive per dtype on the CPU. Every backend is checked against it, and device backends reuse it for ops they do not accelerate, so every op runs on every backend.
- **Static shapes, dtype-native compute.** Shape and dtype are inferred once at record time and stored on the node. 17 dtypes (bool, integers, f8/f16/bf16/f32/f64, complex); low-precision floats accumulate in f32.
- **View-fused realize.** A faster CPU path: movement only rewrites a read view over a shared buffer (0 copies) and elementwise chains fuse into one pass, materializing only at a boundary (reduce, contraction, output, multi-consumer node).
- **Counter-based RNG.** threefry2x32 (`Key`) is stateless, reproducible, and bit-identical across CPU and GPU.

## Example

```rust
use kurumi_core::{Backend, CpuBackend, Graph};

let mut g = Graph::new();
let x = g.constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let sq = g.mul(x, x).unwrap();      // elementwise; shape and dtype are checked
let y = g.relu(sq);                 // decomposes to primitives, autograd for free
let out = CpuBackend.eval(&g, y);   // TensorVal, checked against the interpreter oracle
```

## Layout

- `graph/` -- the IR: `op.rs` (primitives), `ops/` (builders by domain), inference, passes
- `interp/` -- the reference interpreter (oracle)
- `grad/` -- reverse-mode VJP rules
- `realize/`, `lower/` -- the view-fused evaluator and its index-expression lowering
- `dtype/` -- storage, dtype traits, conversions
- `tensor.rs`, `rng.rs`, `backend.rs`, `layout.rs`
