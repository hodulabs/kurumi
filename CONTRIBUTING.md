# Contributing

Thanks for your interest in kurumi.

## Development

A Cargo workspace driven by [`just`](https://github.com/casey/just):

```
just check     # format check, lint (Rust + C header), and all tests -- the CI gate
just format    # format Rust, the C header, and the justfile
just --list    # all recipes
```

Requirements: a recent stable Rust (edition 2024). The Metal backend (`kurumi_metal`)
is macOS-only, on Apple Silicon; elsewhere it is skipped and the CPU path still builds
and tests.

## The interpreter is the oracle

`kurumi_core`'s per-dtype CPU interpreter defines correctness. Every backend, kernel, and
IR pass is checked against it, and reuses it for anything it does not accelerate. When you
add or change behaviour, the interpreter is the reference: a device kernel or a fused path
is correct iff it matches the interpreter on the same graph.

## Adding an op

The IR is a small closed set of primitives. High-level ops are decompositions -- they build
from existing primitives, so autograd and every backend cover them for free.

- Add the builder to the matching `kurumi_core/src/graph/ops/<family>.rs`, expressed through
  existing `Graph` methods.
- Add a primitive (an `Op` variant) only when it genuinely cannot decompose. A new primitive
  needs shape/dtype inference, an interpreter kernel, a VJP rule for autograd, and -- if it
  should run device-resident -- a Metal path.
- Cover it with a test against the interpreter; for autograd, a finite-difference grad check.

## Backends

The Metal backend mirrors the op families in three layers: `msl/` (kernel source),
`context/dispatch/` (launch), `backend/eval/` (the eval seam). A device op returns
`Some(Val)` if it ran on-device, else `None` to fall through to the CPU interpreter -- so
partial device coverage is always correct, only slower. A new kernel must match the
interpreter (the `*_device_match_oracle` tests).

## Conventions

- No `mod.rs`. Modules are `foo.rs` + `foo/*.rs` (a `foo.rs` may hold code and declare its
  submodules).
- Comments explain the logic and the reason, kernel-style: terse, ASCII, no ornament.
- Keep `just check` clean -- `clippy --all-targets -D warnings` is part of the gate.

## Pull requests

- `just check` passes (format, lint, tests).
- One focused change per pull request.

## License

By contributing, you agree that your contributions are dual licensed under
[MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE), matching the project.
