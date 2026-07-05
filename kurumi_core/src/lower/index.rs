//! Movement lowering: a chain of movement ops becomes ONE index expression over the
//! source buffer (no intermediate copies). reshape on a non-contiguous view emits div/mod;
//! if the simplifier can't collapse them we signal a contiguous-copy fallback (never wrong).
//! pad adds validity guards (masked load: out-of-range coords read the pad value, not the
//! source). The RANGEIFY model that replaces a separate view algebra.

use crate::lower::sym::{self, Ranges, Sym, VarId};
use std::collections::HashMap;

/// Validity guard on an output loop var: the read is valid iff `lo <= var < hi`,
/// otherwise it yields the pad value (0). Produced by `pad`.
#[derive(Clone, Debug)]
pub struct Guard {
    pub var: VarId,
    pub lo: i64,
    pub hi: i64,
}

/// A read pattern into a source buffer: for output coordinate (var(0)..var(n-1),
/// each over [0, shape[i])), `offset` gives the flat source index; `guards` mask
/// out-of-range coords to the pad value.
#[derive(Clone, Debug)]
pub struct View {
    pub shape: Vec<usize>,
    pub offset: Sym,
    pub contiguous: bool,
    pub guards: Vec<Guard>,
}

impl View {
    pub fn source(shape: Vec<usize>) -> View {
        let offset = affine(&shape);
        View { shape, offset, contiguous: true, guards: vec![] }
    }

    pub fn permute(&self, perm: &[usize]) -> View {
        debug_assert!(self.guards.is_empty(), "movement on a guarded view must materialize first");
        // new var(i) takes the role of old var(perm[i])
        let m: HashMap<VarId, Sym> =
            perm.iter().enumerate().map(|(i, &p)| (p as VarId, sym::var(i as VarId))).collect();
        View {
            shape: perm.iter().map(|&p| self.shape[p]).collect(),
            offset: self.offset.subst(&m),
            contiguous: self.contiguous && perm.iter().enumerate().all(|(i, &p)| i == p),
            guards: vec![],
        }
    }

    pub fn expand(&self, new_shape: &[usize]) -> View {
        debug_assert!(self.guards.is_empty(), "movement on a guarded view must materialize first");
        // broadcast dim (old size 1): its loop var contributes nothing
        let m: HashMap<VarId, Sym> = (0..self.shape.len())
            .filter(|&d| self.shape[d] == 1 && new_shape[d] != 1)
            .map(|d| (d as VarId, sym::c(0)))
            .collect();
        View {
            shape: new_shape.to_vec(),
            offset: self.offset.subst(&m),
            contiguous: self.contiguous && new_shape == self.shape.as_slice(),
            guards: vec![],
        }
    }

    pub fn slice(&self, ranges: &[(usize, usize, usize)]) -> View {
        debug_assert!(self.guards.is_empty(), "movement on a guarded view must materialize first");
        // strided slice: output var(d) reads source at var(d)*step_d + start_d
        let m: HashMap<VarId, Sym> = (0..self.shape.len())
            .filter(|&d| ranges[d].0 != 0 || ranges[d].2 != 1)
            .map(|d| {
                let (start, _, step) = ranges[d];
                let mut e = sym::var(d as VarId);
                if step != 1 {
                    e = e * (step as i64);
                }
                if start != 0 {
                    e = e + sym::c(start as i64);
                }
                (d as VarId, e)
            })
            .collect();
        View {
            shape: ranges.iter().map(|&(s, e, st)| (e - s).div_ceil(st)).collect(),
            offset: self.offset.subst(&m),
            contiguous: false,
            guards: vec![],
        }
    }

    pub fn flip(&self, axes: &[usize]) -> View {
        debug_assert!(self.guards.is_empty(), "movement on a guarded view must materialize first");
        // reverse each flipped axis: output var(d) reads source at (size_d-1) - var(d)
        let m: HashMap<VarId, Sym> =
            axes.iter().map(|&d| (d as VarId, sym::c(self.shape[d] as i64 - 1) + sym::var(d as VarId) * -1)).collect();
        View { shape: self.shape.clone(), offset: self.offset.subst(&m), contiguous: false, guards: vec![] }
    }

    pub fn pad(&self, pads: &[(usize, usize)]) -> View {
        debug_assert!(self.guards.is_empty(), "movement on a guarded view must materialize first");
        // source index along axis d = out_var(d) - lo_d; valid iff lo_d <= out_var < lo_d + size_d
        let mut m: HashMap<VarId, Sym> = HashMap::new();
        let mut guards = Vec::new();
        for (d, &(lo, hi)) in pads.iter().enumerate() {
            if lo != 0 {
                m.insert(d as VarId, sym::var(d as VarId) + sym::c(-(lo as i64)));
            }
            if lo != 0 || hi != 0 {
                guards.push(Guard { var: d as VarId, lo: lo as i64, hi: (lo + self.shape[d]) as i64 });
            }
        }
        View {
            shape: pads.iter().zip(&self.shape).map(|(&(lo, hi), &s)| lo + s + hi).collect(),
            offset: self.offset.subst(&m),
            contiguous: false,
            guards,
        }
    }

    /// `Some(view)` if it lowers to a pure index expression; `None` if the
    /// simplifier can't remove the div/mod -> caller must materialize a copy.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Option<View> {
        debug_assert!(self.guards.is_empty(), "movement on a guarded view must materialize first");
        if self.contiguous {
            return Some(View { offset: affine(&new_shape), shape: new_shape, contiguous: true, guards: vec![] });
        }
        // de-linearize the new flat index over the old shape, substitute, simplify
        let k = affine(&new_shape);
        let old_st = crate::row_major_strides(&self.shape);
        let m: HashMap<VarId, Sym> = (0..self.shape.len())
            .map(|j| (j as VarId, (k.clone() / old_st[j] as i64) % self.shape[j] as i64))
            .collect();
        let offset = self.offset.subst(&m).simplify(&ranges_of(&new_shape));
        if contains_divmod(&offset) {
            None
        } else {
            Some(View { shape: new_shape, offset, contiguous: false, guards: vec![] })
        }
    }
}

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

fn affine(shape: &[usize]) -> Sym {
    let st = crate::row_major_strides(shape);
    (0..shape.len()).map(|i| sym::var(i as VarId) * st[i] as i64).reduce(|a, b| a + b).unwrap_or(sym::c(0))
}

fn ranges_of(shape: &[usize]) -> Ranges {
    (0..shape.len()).map(|i| (i as VarId, (0, shape[i] as i64 - 1))).collect()
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

fn contains_divmod(e: &Sym) -> bool {
    match e {
        Sym::Div(..) | Sym::Mod(..) => true,
        Sym::Add(ts) => ts.iter().any(contains_divmod),
        Sym::Mul(_, e) => contains_divmod(e),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // src [2,3] = 0..6 ; transpose -> [3,2] = [0,3,1,4,2,5], read straight from src
    #[test]
    fn permute_reads_transpose() {
        let src: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let v = View::source(vec![2, 3]).permute(&[1, 0]);
        assert_eq!(v.shape, vec![3, 2]);
        assert_eq!(read(&src, &v), vec![0., 3., 1., 4., 2., 5.]);
    }

    #[test]
    fn expand_reads_broadcast() {
        let src = vec![10., 20.]; // shape [1,2]
        let v = View::source(vec![1, 2]).expand(&[3, 2]);
        assert_eq!(read(&src, &v), vec![10., 20., 10., 20., 10., 20.]);
    }

    // contiguous reshape is a pure relabel: same buffer, affine offset, 0 copies
    #[test]
    fn contiguous_reshape_is_free() {
        let src: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let v = View::source(vec![2, 3]).reshape(vec![3, 2]).unwrap();
        assert!(v.contiguous);
        assert!(!contains_divmod(&v.offset));
        assert_eq!(read(&src, &v), src);
    }

    #[test]
    fn permute_then_expand_composes() {
        // src [2,1] -> permute [1,0] -> [1,2] -> expand [3,2], one fused index expr
        let src = vec![7., 9.];
        let v = View::source(vec![2, 1]).permute(&[1, 0]).expand(&[3, 2]);
        assert_eq!(v.shape, vec![3, 2]);
        assert_eq!(read(&src, &v), vec![7., 9., 7., 9., 7., 9.]);
    }

    // reshape after permute: simplifier can't collapse -> None (materialize)
    #[test]
    fn noncontiguous_reshape_falls_back() {
        let v = View::source(vec![2, 3]).permute(&[1, 0]).reshape(vec![6]);
        assert!(v.is_none());
    }

    #[test]
    fn slice_reads_subregion() {
        let src: Vec<f32> = (0..12).map(|x| x as f32).collect(); // [3,4]
        let v = View::source(vec![3, 4]).slice(&[(1, 3, 1), (1, 3, 1)]);
        assert_eq!(v.shape, vec![2, 2]);
        assert_eq!(read(&src, &v), vec![5., 6., 9., 10.]);
    }

    #[test]
    fn flip_reads_reversed() {
        let src = vec![1., 2., 3., 4., 5., 6.]; // [2,3]
        let v = View::source(vec![2, 3]).flip(&[1]);
        assert_eq!(read(&src, &v), vec![3., 2., 1., 6., 5., 4.]);
    }

    // pad reads 0 outside the source region, the value inside (masked load)
    #[test]
    fn pad_reads_zeros_outside() {
        let src = vec![1., 2., 3.]; // [3]
        let v = View::source(vec![3]).pad(&[(1, 2)]); // -> [6]
        assert_eq!(v.shape, vec![6]);
        assert_eq!(read(&src, &v), vec![0., 1., 2., 3., 0., 0.]);
    }

    #[test]
    fn pad_2d() {
        let src = vec![1., 2., 3., 4.]; // [2,2]
        let v = View::source(vec![2, 2]).pad(&[(1, 0), (0, 1)]); // -> [3,3]
        assert_eq!(v.shape, vec![3, 3]);
        assert_eq!(read(&src, &v), vec![0., 0., 0., 1., 2., 0., 3., 4., 0.]);
    }
}
