// Neural-net layers, split by concern to mirror the engine's `graph/ops/nn/*`:
// activations, normalizations, losses, attention, convolution, and pooling. Each wrapper
// exports its C symbol directly (`#[no_mangle]`); the submodules are crate-visible only
// for the in-crate ABI test.

pub(crate) mod activation;
pub(crate) mod attention;
pub(crate) mod conv;
pub(crate) mod loss;
pub(crate) mod norm;
pub(crate) mod pool;
