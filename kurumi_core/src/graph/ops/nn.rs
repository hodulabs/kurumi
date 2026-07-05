//! ML fast-op layer: activation/norm/attention/conv builders decomposed onto `core`
//! (autodiff and shape rules fall out).

mod activation;
mod attention;
mod conv;
mod loss;
mod norm;
mod pool;
