//! Shape/stride/axis utilities shared across the interpreter, realize path, and
//! out-of-crate backends. Pure index arithmetic, no allocation on the hot paths
//! beyond the returned stride vector.

use crate::Error;

/// Row-major (C-contiguous) strides for `shape`.
pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut st = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        st[i] = st[i + 1] * shape[i + 1];
    }
    st
}

/// Advance a coordinate odometer-style (mixed radix by shape); reused across a
/// loop so hot paths never allocate a coordinate Vec per element.
pub fn inc(coord: &mut [usize], shape: &[usize]) {
    for i in (0..coord.len()).rev() {
        coord[i] += 1;
        if coord[i] < shape[i] {
            return;
        }
        coord[i] = 0;
    }
}

/// Axes not in `batch` or `contract`, in original order (dot_general free axes).
pub fn free_axes(rank: usize, batch: &[usize], contract: &[usize]) -> Vec<usize> {
    (0..rank).filter(|i| !batch.contains(i) && !contract.contains(i)).collect()
}

/// Each axis in range and distinct (no axis used as both batch and contract).
pub(crate) fn check_axes(axes: &[usize], rank: usize) -> Result<(), Error> {
    for (k, &ax) in axes.iter().enumerate() {
        if ax >= rank {
            return Err(Error::shape("dot_general", format!("axis {ax} out of range for rank {rank}")));
        }
        if axes[..k].contains(&ax) {
            return Err(Error::shape("dot_general", format!("axis {ax} used twice")));
        }
    }
    Ok(())
}
