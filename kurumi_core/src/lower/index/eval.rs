//! Read side: interpret a finished `View` against a source buffer. No intermediate buffers --
//! each output element is a masked load at the view's computed offset. The movement algebra that
//! builds the view lives in `movement`.

use super::View;
use crate::lower::sym::Sym;

/// Read the source through a view: out[o] = load_at(coord(o)). No intermediate
/// buffers: this is what "movement adds 0 copies" means.
pub fn read(src: &[f32], v: &View) -> Vec<f32> {
    let out_len = v.shape.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(out_len);
    let mut coord = vec![0usize; v.shape.len()];
    for _ in 0..out_len {
        out.push(load_at(src, v, &coord));
        crate::inc(&mut coord, &v.shape);
    }
    out
}

/// One element at an output coordinate: pad value (0) if any guard fails, else
/// the source element at the computed offset.
pub(crate) fn load_at(src: &[f32], v: &View, coord: &[usize]) -> f32 {
    let valid = v.guards.iter().all(|g| {
        let c = coord[g.var as usize] as i64;
        g.lo <= c && c < g.hi
    });
    if valid { src[eval_sym(&v.offset, coord) as usize] } else { 0.0 }
}

// evaluate an index expression at concrete loop-var values (closed enum walk)
fn eval_sym(e: &Sym, vals: &[usize]) -> i64 {
    match e {
        Sym::Const(c) => *c,
        Sym::Var(v) => vals[*v as usize] as i64,
        Sym::Add(ts) => ts.iter().map(|t| eval_sym(t, vals)).sum(),
        Sym::Mul(k, e) => k * eval_sym(e, vals),
        Sym::Div(e, k) => eval_sym(e, vals).div_euclid(*k),
        Sym::Mod(e, k) => eval_sym(e, vals).rem_euclid(*k),
    }
}
