# kurumi

The top-level crate of the [kurumi](https://github.com/hodulabs/kurumi) engine: it re-exports the core, selects the Metal backend on Apple Silicon, and exposes the C ABI.

## What it provides

- **Rust facade.** Re-exports all of [`kurumi_core`](../kurumi_core) and exposes [`kurumi_metal`](../kurumi_metal) as `kurumi::metal` on macOS.
- **C ABI.** A stable C interface over the graph builder, backends, autograd, and graph passes -- every builder op, every dtype, plus plan-replay, graph inspection, and graph (de)serialization (`ku_graph_serialize` / `ku_graph_deserialize`). Built as `cdylib` and `staticlib`; the header is [`include/kurumi.h`](include/kurumi.h).

## Example (Rust)

```rust
use kurumi::{Backend, CpuBackend, Graph};

let mut g = Graph::new();
let x = g.constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let y = g.relu(g.mul(x, x).unwrap());
let out = CpuBackend.eval(&g, y);
```

## Example (C)

```c
#include "kurumi.h"

KuGraph *g = ku_graph_new();
KuBackend *cpu = ku_backend_new(KU_CPU);
KuNode x = ku_constant_f32(g, (float[]){1, 2, 3, 4}, 4, (size_t[]){2, 2}, 2);
KuNode y = ku_relu(g, ku_mul(g, x, x));
KuTensor *out = ku_eval(g, y, cpu);
```

## Build

- Rust: `cargo build -p kurumi`
- C library: `cargo build -p kurumi --release` produces `target/release/libkurumi.{dylib,a}`; include [`include/kurumi.h`](include/kurumi.h).
