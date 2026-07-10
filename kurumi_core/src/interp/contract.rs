//! Contraction (StableHLO dot_general): a general batched/multi-axis loop for every shape and
//! dtype, plus dtype dispatch and the quant-matmul oracle. The f32 2D fast path (Accelerate /
//! the `gemm` crate) is in `gemm.rs`. matmul/bmm/einsum all decompose to this one op.

mod gemm;

use crate::{DType, Num, Op, Storage, TensorVal, cast, free_axes, inc, row_major_strides};

pub(crate) use gemm::dot_general;

pub(super) fn eval(op: &Op, inputs: &[&TensorVal]) -> TensorVal {
    match op {
        Op::DotGeneral { lhs_contract, rhs_contract, lhs_batch, rhs_batch } => {
            dot_dispatch(inputs[0], inputs[1], lhs_contract, rhs_contract, lhs_batch, rhs_batch)
        }
        Op::QuantMatmul { bits, group_size, symmetric } => {
            let mins = (!*symmetric).then(|| inputs[3]);
            quant_matmul(inputs[0], inputs[1], inputs[2], mins, *bits, *group_size)
        }
        _ => unreachable!("contract::eval: non-contract op"),
    }
}

// dispatch a contraction by dtype: f32 takes the gemm fast path, every other
// numeric dtype uses the generic loop. (Operands share dtype, validated upstream.)
pub(crate) fn dot_dispatch(
    a: &TensorVal,
    b: &TensorVal,
    lc: &[usize],
    rc: &[usize],
    lb: &[usize],
    rb: &[usize],
) -> TensorVal {
    match (&a.storage, &b.storage) {
        (Storage::F32(x), Storage::F32(y)) => dot_general(x, &a.shape, y, &b.shape, lc, rc, lb, rb),
        (Storage::F64(x), Storage::F64(y)) => dot_num(x, &a.shape, y, &b.shape, lc, rc, lb, rb),
        // low-precision floats accumulate in f32, then round back: matching the reductions and
        // Metal's MPS f32 accumulator (f16 K-sum accumulation bleeds precision). Reuses the f32 GEMM.
        (Storage::F16(_), Storage::F16(_))
        | (Storage::BF16(_), Storage::BF16(_))
        | (Storage::F8E4M3(_), Storage::F8E4M3(_))
        | (Storage::F8E5M2(_), Storage::F8E5M2(_)) => dot_promoted(a, b, lc, rc, lb, rb),
        (Storage::I32(x), Storage::I32(y)) => dot_num(x, &a.shape, y, &b.shape, lc, rc, lb, rb),
        (Storage::I64(x), Storage::I64(y)) => dot_num(x, &a.shape, y, &b.shape, lc, rc, lb, rb),
        (Storage::U32(x), Storage::U32(y)) => dot_num(x, &a.shape, y, &b.shape, lc, rc, lb, rb),
        (Storage::U8(x), Storage::U8(y)) => dot_num(x, &a.shape, y, &b.shape, lc, rc, lb, rb),
        (Storage::C64(x), Storage::C64(y)) => dot_num(x, &a.shape, y, &b.shape, lc, rc, lb, rb),
        (Storage::C128(x), Storage::C128(y)) => dot_num(x, &a.shape, y, &b.shape, lc, rc, lb, rb),
        _ => unreachable!("dot_general: non-numeric or mismatched dtype"),
    }
}

// promote low-precision-float operands to f32, contract, round the result back to input dtype.
fn dot_promoted(a: &TensorVal, b: &TensorVal, lc: &[usize], rc: &[usize], lb: &[usize], rb: &[usize]) -> TensorVal {
    let dt = a.storage.dtype();
    let af = cast(&a.storage, DType::F32).into_f32();
    let bf = cast(&b.storage, DType::F32).into_f32();
    let r = dot_general(&af, &a.shape, &bf, &b.shape, lc, rc, lb, rb);
    TensorVal { shape: r.shape, storage: cast(&r.storage, dt) }
}

// weight-only quantized matmul oracle: dequantize the packed weight [N,K], then
// `act[M,K] x W[N,K]^T -> [M,N]`. src = [act, qweight(U8), scales(F16)] (+ mins(F16)
// when asymmetric). The fused kernels are checked against this. Output takes act's dtype.
pub(crate) fn quant_matmul(
    act: &TensorVal,
    qw: &TensorVal,
    scales: &TensorVal,
    mins: Option<&TensorVal>,
    bits: u8,
    group_size: usize,
) -> TensorVal {
    let (m, k, n) = (act.shape[0], act.shape[1], qw.shape[0]);
    let take_u8 = |t: &TensorVal| match &t.storage {
        Storage::U8(v) => v.clone(),
        _ => unreachable!("quant_matmul: qweight must be U8"),
    };
    let take_f16 = |t: &TensorVal| match &t.storage {
        Storage::F16(v) => v.clone(),
        _ => unreachable!("quant_matmul: scales/mins must be F16"),
    };
    let q = crate::quant::Quantized {
        packed: take_u8(qw),
        scales: take_f16(scales),
        mins: mins.map(take_f16),
        rows: n,
        cols: k,
        bits,
        group_size,
    };
    let w = crate::quant::dequantize(&q);
    let af = cast(&act.storage, DType::F32).into_f32();
    let r = dot_general(&af, &[m, k], &w, &[n, k], &[1], &[1], &[], &[]);
    TensorVal { shape: r.shape, storage: cast(&r.storage, act.storage.dtype()) }
}

fn dot_num<T: Num>(
    a_data: &[T],
    a_shape: &[usize],
    b_data: &[T],
    b_shape: &[usize],
    lhs_contract: &[usize],
    rhs_contract: &[usize],
    lhs_batch: &[usize],
    rhs_batch: &[usize],
) -> TensorVal {
    let a_strides = row_major_strides(a_shape);
    let b_strides = row_major_strides(b_shape);
    let a_free = free_axes(a_shape.len(), lhs_batch, lhs_contract);
    let b_free = free_axes(b_shape.len(), rhs_batch, rhs_contract);

    let batch_sizes: Vec<usize> = lhs_batch.iter().map(|&i| a_shape[i]).collect();
    let lf_sizes: Vec<usize> = a_free.iter().map(|&i| a_shape[i]).collect();
    let rf_sizes: Vec<usize> = b_free.iter().map(|&i| b_shape[i]).collect();
    let c_sizes: Vec<usize> = lhs_contract.iter().map(|&i| a_shape[i]).collect();

    let out_shape: Vec<usize> = batch_sizes.iter().chain(&lf_sizes).chain(&rf_sizes).copied().collect();
    let (nb, nlf) = (batch_sizes.len(), lf_sizes.len());
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let c_len: usize = c_sizes.iter().product::<usize>().max(1);

    let mut out = Vec::with_capacity(out_len);
    let mut oc = vec![0usize; out_shape.len()];
    let mut ccoord = vec![0usize; c_sizes.len()];
    for _ in 0..out_len {
        let (bcoord, rest) = oc.split_at(nb);
        let (lfcoord, rfcoord) = rest.split_at(nlf);
        let mut acc = T::zero();
        ccoord.fill(0);
        for _ in 0..c_len {
            let ai = operand_flat(&a_strides, lhs_batch, lhs_contract, &a_free, bcoord, lfcoord, &ccoord);
            let bi = operand_flat(&b_strides, rhs_batch, rhs_contract, &b_free, bcoord, rfcoord, &ccoord);
            acc = acc.add(a_data[ai].mul(b_data[bi]));
            inc(&mut ccoord, &c_sizes);
        }
        out.push(acc);
        inc(&mut oc, &out_shape);
    }
    TensorVal { shape: out_shape, storage: T::store(out) }
}

fn operand_flat(
    strides: &[usize],
    batch_axes: &[usize],
    contract_axes: &[usize],
    free_axes: &[usize],
    batch_coord: &[usize],
    free_coord: &[usize],
    contract_coord: &[usize],
) -> usize {
    let mut flat = 0;
    for (i, &ax) in batch_axes.iter().enumerate() {
        flat += batch_coord[i] * strides[ax];
    }
    for (j, &ax) in contract_axes.iter().enumerate() {
        flat += contract_coord[j] * strides[ax];
    }
    for (k, &ax) in free_axes.iter().enumerate() {
        flat += free_coord[k] * strides[ax];
    }
    flat
}

#[cfg(test)]
mod tests {
    use crate::{Graph, Storage, interpret};
    use half::f16;

    // f16 matmul must accumulate the K sum in f32: 4096 ones sum to 4096 exactly.
    // f16-native accumulation would stall near 2048 (f16 spacing is 2 above 2048).
    #[test]
    fn f16_matmul_accumulates_in_f32() {
        let k = 4096usize;
        let mut g = Graph::new();
        let a = g.const_storage(Storage::F16(vec![f16::ONE; k]), vec![1, k]);
        let b = g.const_storage(Storage::F16(vec![f16::ONE; k]), vec![k, 1]);
        let c = g.dot_general(a, b, vec![1], vec![0], vec![], vec![]).unwrap();
        let v = match interpret(&g, c).storage {
            Storage::F16(v) => v[0].to_f32(),
            _ => panic!("expected f16"),
        };
        assert_eq!(v, k as f32, "f16 matmul must accumulate in f32, got {v}");
    }
}
