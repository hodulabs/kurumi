//! Special functions (polynomial approximations). Pure primitive compositions.

use crate::{Error, Graph, NodeId};

impl Graph {
    /// Error function `erf(x)`: Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7).
    pub fn erf(&mut self, x: NodeId) -> NodeId {
        let a = self.abs(x);
        let p = self.scalar(a, 0.327_591_1);
        let pa = self.mul(p, a).expect("same shape");
        let one = self.scalar(a, 1.0);
        let den = self.add(one, pa).expect("same shape");
        let t = self.recip(den); // t = 1/(1+p|x|)
        // Horner: ((((a5 t + a4) t + a3) t + a2) t + a1) t
        let mut poly = self.scalar(t, 1.061_405_43);
        for c in [-1.453_152, 1.421_413_7, -0.284_496_75, 0.254_829_6] {
            poly = self.mul(poly, t).expect("same shape");
            let cc = self.scalar(t, c);
            poly = self.add(poly, cc).expect("same shape");
        }
        poly = self.mul(poly, t).expect("same shape");
        let x2 = self.square(x);
        let nx2 = self.neg(x2);
        let e = self.exp(nx2);
        let pe = self.mul(poly, e).expect("same shape");
        let y = self.sub(one, pe).expect("same shape"); // 1 - poly*exp(-x^2)
        let s = self.sign(x);
        self.mul(s, y).expect("same shape")
    }

    /// Complementary error function `erfc(x) = 1 - erf(x)`.
    pub fn erfc(&mut self, x: NodeId) -> NodeId {
        let e = self.erf(x);
        let one = self.scalar(e, 1.0);
        self.sub(one, e).expect("same shape")
    }

    // Horner evaluation of a polynomial (coeffs highest-degree first) at `t`.
    fn horner(&mut self, t: NodeId, coeffs: &[f32]) -> NodeId {
        let mut p = self.scalar(t, coeffs[0]);
        for &c in &coeffs[1..] {
            p = self.mul(p, t).expect("same shape");
            let cc = self.scalar(t, c);
            p = self.add(p, cc).expect("same shape");
        }
        p
    }
    // c * x (scalar times tensor)
    fn scale(&mut self, x: NodeId, c: f32) -> NodeId {
        let cc = self.scalar(x, c);
        self.mul(x, cc).expect("same shape")
    }

    /// Log-gamma `ln Gamma(x)` for `x > 0` (Lanczos g=7; ~1e-6 rel). Differentiable ->
    /// its gradient is `digamma`. (No reflection: `x <= 0` is out of range.)
    #[allow(clippy::inconsistent_digit_grouping)]
    pub fn lgamma(&mut self, x: NodeId) -> NodeId {
        const C: [f32; 9] = [
            0.999_999_99,
            676.520_36,
            -1259.139_2,
            771.323_42,
            -176.615_02,
            12.507_343,
            -0.138_571_1,
            9.984_369_5e-6,
            1.505_632_7e-7,
        ];
        const G: f32 = 7.0;
        let one = self.scalar(x, 1.0);
        let z = self.sub(x, one).expect("same shape"); // z = x - 1
        let mut a = self.scalar(z, C[0]);
        for (i, &ci) in C.iter().enumerate().skip(1) {
            let off = self.scalar(z, i as f32);
            let d = self.add(z, off).expect("same shape"); // z + i
            let r = self.recip(d);
            let term = self.scale(r, ci);
            a = self.add(a, term).expect("same shape");
        }
        let t = self.scalar(z, G + 0.5);
        let t = self.add(z, t).expect("same shape"); // z + g + 0.5
        let lt = self.ln(t);
        let zh = self.scalar(z, 0.5);
        let zph = self.add(z, zh).expect("same shape"); // z + 0.5
        let term1 = self.mul(zph, lt).expect("same shape"); // (z+0.5)*ln(t)
        let la = self.ln(a);
        // 0.5*ln(2pi) + (z+0.5)ln(t) - t + ln(a)
        let c0 = self.scalar(z, 0.5 * (std::f32::consts::TAU).ln());
        let s1 = self.add(c0, term1).expect("same shape");
        let s2 = self.sub(s1, t).expect("same shape");
        self.add(s2, la).expect("same shape")
    }

    /// Gamma `Gamma(x) = exp(lgamma(x))` for `x > 0`.
    pub fn gamma(&mut self, x: NodeId) -> NodeId {
        let lg = self.lgamma(x);
        self.exp(lg)
    }

    /// Beta function `B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b)` for positive args.
    pub fn beta(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, Error> {
        let la = self.lgamma(a);
        let lb = self.lgamma(b);
        let ab = self.add(a, b)?;
        let lab = self.lgamma(ab);
        let s = self.add(la, lb)?;
        let d = self.sub(s, lab)?;
        Ok(self.exp(d))
    }

    /// Digamma `psi(x) = d/dx lnGamma(x)` for `x > 0`. Recurrence up 6 steps then the
    /// asymptotic series: accurate for positive reals.
    pub fn digamma(&mut self, x: NodeId) -> NodeId {
        // correction = -sum_{k=0}^{5} 1/(x+k); y = x + 6
        let mut corr = self.recip(x);
        for k in 1..6 {
            let off = self.scalar(x, k as f32);
            let xk = self.add(x, off).expect("same shape");
            let r = self.recip(xk);
            corr = self.add(corr, r).expect("same shape");
        }
        let corr = self.neg(corr);
        let six = self.scalar(x, 6.0);
        let y = self.add(x, six).expect("same shape");
        // ln(y) - 1/(2y) - 1/(12y^2) + 1/(120y^4) - 1/(252y^6)
        let ly = self.ln(y);
        let iy = self.recip(y);
        let iy2 = self.mul(iy, iy).expect("same shape");
        let iy4 = self.mul(iy2, iy2).expect("same shape");
        let iy6 = self.mul(iy4, iy2).expect("same shape");
        let t1 = self.scale(iy, -0.5);
        let t2 = self.scale(iy2, -1.0 / 12.0);
        let t3 = self.scale(iy4, 1.0 / 120.0);
        let t4 = self.scale(iy6, -1.0 / 252.0);
        let mut r = self.add(ly, t1).expect("same shape");
        for t in [t2, t3, t4, corr] {
            r = self.add(r, t).expect("same shape");
        }
        r
    }

    /// Inverse error function `erfinv(x)`, `x in (-1, 1)`: Giles (2010) single-precision.
    pub fn erfinv(&mut self, x: NodeId) -> NodeId {
        let x2 = self.square(x);
        let one = self.scalar(x, 1.0);
        let om = self.sub(one, x2).expect("same shape"); // 1 - x^2
        let lm = self.ln(om);
        let w = self.neg(lm); // w = -ln(1-x^2)
        // small branch (w < 5): w -= 2.5
        let w_s = {
            let c = self.scalar(w, 2.5);
            self.sub(w, c).expect("same shape")
        };
        let p_s = self.horner(
            w_s,
            &[
                2.810_226_36e-08,
                3.432_739_39e-07,
                -3.523_387_7e-06,
                -4.391_506_54e-06,
                0.000_218_580_87,
                -0.001_253_725_03,
                -0.004_177_681_64,
                0.246_640_727,
                1.501_409_41,
            ],
        );
        // large branch (w >= 5): w = sqrt(w) - 3
        let w_l = {
            let s = self.sqrt(w);
            let c = self.scalar(s, 3.0);
            self.sub(s, c).expect("same shape")
        };
        let p_l = self.horner(
            w_l,
            &[
                -0.000_200_214_257,
                0.000_100_950_558,
                0.001_349_343_22,
                -0.003_673_428_44,
                0.005_739_507_73,
                -0.007_622_461_3,
                0.009_438_870_47,
                1.001_674_06,
                2.832_976_82,
            ],
        );
        let five = self.scalar(w, 5.0);
        let small = self.cmp_lt(w, five).expect("same shape");
        let p = self.select(small, p_s, p_l).expect("same shape");
        self.mul(p, x).expect("same shape")
    }

    /// Modified Bessel function of the first kind, order 0: `I_0(x)` (Abramowitz &
    /// Stegun 9.8.1/9.8.2, ~1e-7).
    pub fn i0(&mut self, x: NodeId) -> NodeId {
        let ax = self.abs(x);
        // small |x| < 3.75: t = (x/3.75)^2, polynomial in t
        let ts = {
            let c = self.scalar(ax, 1.0 / 3.75);
            let s = self.mul(ax, c).expect("same shape");
            self.mul(s, s).expect("same shape")
        };
        let small =
            self.horner(ts, &[0.004_581_3, 0.036_076_8, 0.265_973_2, 1.206_749_2, 3.089_942_4, 3.515_622_9, 1.0]);
        // large |x| >= 3.75: t = 3.75/|x|; exp(|x|)/sqrt(|x|) * polynomial in t
        let tl = {
            let c = self.scalar(ax, 3.75);
            self.div(c, ax).expect("same shape")
        };
        let poly = self.horner(
            tl,
            &[
                0.003_923_77,
                -0.016_476_33,
                0.026_355_37,
                -0.020_577_06,
                0.009_162_81,
                -0.001_575_65,
                0.002_253_19,
                0.013_285_92,
                0.398_942_28,
            ],
        );
        let ex = self.exp(ax);
        let sq = self.sqrt(ax);
        let pref = self.div(ex, sq).expect("same shape");
        let large = self.mul(pref, poly).expect("same shape");
        let c = self.scalar(ax, 3.75);
        let lt = self.cmp_lt(ax, c).expect("same shape");
        self.select(lt, small, large).expect("same shape")
    }
}
