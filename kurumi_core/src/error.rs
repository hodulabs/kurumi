//! Engine error type. `Shape` is a record-time op error (shape/dtype mismatch);
//! `Backend` is a runtime backend error (a device can't run a dtype, an alloc
//! failed). Shared across the IR builder, interpreter, and out-of-crate backends.

use std::fmt;

#[derive(Clone, PartialEq, Debug)]
pub enum Error {
    Shape { op: &'static str, msg: String },
    Backend { msg: String },
}

impl Error {
    pub(crate) fn shape(op: &'static str, msg: impl Into<String>) -> Self {
        Error::Shape { op, msg: msg.into() }
    }
    /// Construct a backend error (public so out-of-crate backends like
    /// `kurumi_metal` can report "this device can't do X").
    pub fn backend(msg: impl Into<String>) -> Self {
        Error::Backend { msg: msg.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Shape { op, msg } => write!(f, "{op}: {msg}"),
            Error::Backend { msg } => write!(f, "backend: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
