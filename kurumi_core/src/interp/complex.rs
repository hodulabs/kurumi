//! Complex construction / part-extraction kernels for the interpreter.
//! (C64 = Complex<f32>, C128 = Complex<f64>.)

use crate::{Op, Storage, TensorVal};
use num_complex::Complex;

pub(super) fn eval(op: &Op, inputs: &[&TensorVal]) -> TensorVal {
    match op {
        Op::Complex => {
            let storage = complex_k(&inputs[0].storage, &inputs[1].storage);
            TensorVal { shape: inputs[0].shape.clone(), storage }
        }
        Op::Real => TensorVal { shape: inputs[0].shape.clone(), storage: real_k(&inputs[0].storage) },
        Op::Imag => TensorVal { shape: inputs[0].shape.clone(), storage: imag_k(&inputs[0].storage) },
        _ => unreachable!("complex::eval: non-complex op"),
    }
}

// combine real + imaginary parts into a complex storage
pub(crate) fn complex_k(re: &Storage, im: &Storage) -> Storage {
    match (re, im) {
        (Storage::F32(r), Storage::F32(i)) => {
            Storage::C64(r.iter().zip(i).map(|(&a, &b)| Complex::new(a, b)).collect())
        }
        (Storage::F64(r), Storage::F64(i)) => {
            Storage::C128(r.iter().zip(i).map(|(&a, &b)| Complex::new(a, b)).collect())
        }
        _ => unreachable!("complex: builder requires matching real-float inputs"),
    }
}

// real part (C64 -> F32, C128 -> F64)
pub(crate) fn real_k(z: &Storage) -> Storage {
    match z {
        Storage::C64(v) => Storage::F32(v.iter().map(|c| c.re).collect()),
        Storage::C128(v) => Storage::F64(v.iter().map(|c| c.re).collect()),
        _ => unreachable!("real: builder requires complex"),
    }
}

// imaginary part
pub(crate) fn imag_k(z: &Storage) -> Storage {
    match z {
        Storage::C64(v) => Storage::F32(v.iter().map(|c| c.im).collect()),
        Storage::C128(v) => Storage::F64(v.iter().map(|c| c.im).collect()),
        _ => unreachable!("imag: builder requires complex"),
    }
}
