//! Shape/view movement, grouped by sub-family: reshape/permute/expand + transpose/flatten/
//! squeeze/broadcast (view), slice/flip/pad + non-constant pad modes (slicing), tile/
//! repeat_interleave/roll (repeat). (join/split -> join.rs, tril/diagonal -> triangular.rs.)

mod repeat;
mod slicing;
mod view;
