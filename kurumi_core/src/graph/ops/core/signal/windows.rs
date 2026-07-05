//! Window functions (Hann/Hamming/Blackman/Bartlett) via a generalized-cosine builder. Pure
//! constants; the STFT ops in the sibling `stft` apply them.

use crate::{Graph, NodeId};

impl Graph {
    /// Hann window `0.5 - 0.5*cos(2pi k/(n-1))` (symmetric), length `n`, F32.
    pub fn hann_window(&mut self, n: usize) -> NodeId {
        self.cosine_window(n, &[0.5, 0.5])
    }
    /// Hamming window `0.54 - 0.46*cos(2pi k/(n-1))`.
    pub fn hamming_window(&mut self, n: usize) -> NodeId {
        self.cosine_window(n, &[0.54, 0.46])
    }
    /// Blackman window `0.42 - 0.5*cos(2pi k/(n-1)) + 0.08*cos(4pi k/(n-1))`.
    pub fn blackman_window(&mut self, n: usize) -> NodeId {
        self.cosine_window(n, &[0.42, 0.5, 0.08])
    }
    /// Bartlett (triangular) window, length `n`, F32.
    pub fn bartlett_window(&mut self, n: usize) -> NodeId {
        if n <= 1 {
            return self.constant(vec![1.0; n.max(1)], vec![n.max(1)]);
        }
        let half = (n - 1) as f32 / 2.0;
        let w: Vec<f32> = (0..n).map(|k| 1.0 - ((k as f32 - half) / half).abs()).collect();
        self.constant(w, vec![n])
    }
    // generalized-cosine window: w[k] = a0 - a1*cos(2pi k/(n-1)) + a2*cos(4pi k/(n-1)) - ...
    fn cosine_window(&mut self, n: usize, a: &[f32]) -> NodeId {
        if n <= 1 {
            return self.constant(vec![1.0; n.max(1)], vec![n.max(1)]);
        }
        let d = (n - 1) as f32;
        let w: Vec<f32> = (0..n)
            .map(|k| {
                let t = std::f32::consts::TAU * k as f32 / d;
                a.iter().enumerate().map(|(i, &ai)| if i % 2 == 0 { ai } else { -ai } * (i as f32 * t).cos()).sum()
            })
            .collect();
        self.constant(w, vec![n])
    }
}
