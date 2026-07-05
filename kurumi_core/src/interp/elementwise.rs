//! Generic elementwise interp kernels (zip / map / compare / select), written once over a
//! `Copy` type and monomorphized by the dispatch macros. Reductions are in `reduce.rs`.

pub(crate) fn zip_map<T: Copy>(x: &[T], y: &[T], f: impl Fn(T, T) -> T) -> Vec<T> {
    x.iter().zip(y).map(|(&a, &b)| f(a, b)).collect()
}

pub(crate) fn map1<T: Copy>(v: &[T], f: impl Fn(T) -> T) -> Vec<T> {
    v.iter().map(|&x| f(x)).collect()
}

// comparison -> bool (f takes refs: PartialOrd::lt / PartialEq::eq)
pub(crate) fn cmp_map<T>(x: &[T], y: &[T], f: impl Fn(&T, &T) -> bool) -> Vec<bool> {
    x.iter().zip(y).map(|(a, b)| f(a, b)).collect()
}

// where: cond ? a : b, elementwise
pub(crate) fn select_k<T: Copy>(cond: &[bool], a: &[T], b: &[T]) -> Vec<T> {
    (0..a.len()).map(|i| if cond[i] { a[i] } else { b[i] }).collect()
}
