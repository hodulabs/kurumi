//! Eigendecomposition & orthogonal factorizations (interp kernels): symmetric `eigh` (cyclic
//! Jacobi), reduced `qr` (Householder), general `eigvals` (QR algorithm), one solver per
//! submodule. Direct LU/Cholesky solvers stay in the parent `linalg`.

mod eigh;
mod eigvals;
mod qr;

pub(crate) use eigh::eigh;
pub(crate) use eigvals::eigvals;
pub(crate) use qr::qr;
