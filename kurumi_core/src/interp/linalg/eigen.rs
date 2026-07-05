//! Eigendecomposition & orthogonal factorizations (interp kernels): symmetric `eigh` (cyclic
//! Jacobi), reduced `qr` (Householder), general `eigvals` (QR algorithm), one solver per
//! submodule. Direct LU/Cholesky solvers stay in the parent `linalg`.

mod eigvals;
mod jacobi;
mod qr;

pub(crate) use eigvals::eigvals;
pub(crate) use jacobi::eigh;
pub(crate) use qr::qr;
