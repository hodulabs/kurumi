//! Counter-based RNG: threefry2x32-20 (Random123) as the reproducibility primitive,
//! plus the `Key` type whose move-consuming `split` makes key reuse a *compile* error
//! (JAX's runtime split convention, enforced by the borrow checker). A random tensor is
//! a pure function of (key, index): parallel, backend-agnostic, bit-reproducible. The
//! algorithm is pinned as a stability contract (the KAT tests below lock the exact bits).

/// threefry2x32-20: (key, counter) -> 64 pseudo-random bits. Stateless and pure:
/// element i of a random tensor is `threefry2x32(seed, i)`, independent of every
/// other element (so it's parallel and order/backend-invariant).
pub(crate) fn threefry2x32(key: u64, counter: u64) -> u64 {
    const ROT: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];
    let k = [key as u32, (key >> 32) as u32];
    let ks = [k[0], k[1], 0x1BD1_1BDA ^ k[0] ^ k[1]];
    let mut x = [(counter as u32).wrapping_add(ks[0]), ((counter >> 32) as u32).wrapping_add(ks[1])];
    for r in 0..20u32 {
        x[0] = x[0].wrapping_add(x[1]);
        x[1] = x[1].rotate_left(ROT[(r % 8) as usize]);
        x[1] ^= x[0];
        if r % 4 == 3 {
            let inj = r / 4 + 1; // key injection every 4 rounds
            x[0] = x[0].wrapping_add(ks[(inj % 3) as usize]);
            x[1] = x[1].wrapping_add(ks[((inj + 1) % 3) as usize]).wrapping_add(inj);
        }
    }
    (x[0] as u64) | ((x[1] as u64) << 32)
}

/// Map 64 random bits to a uniform f32 in `[0, 1)` (top 24 bits -> full mantissa).
pub(crate) fn uniform_f32(bits: u64) -> f32 {
    ((bits as u32 >> 8) as f32) / ((1u32 << 24) as f32)
}

/// A threefry key. `split` *consumes* it, so the borrow checker rejects reusing a
/// key for two draws: which would give silently-correlated randomness (the classic
/// JAX footgun, here a compile error). Not `Copy`/`Clone` on purpose.
#[derive(Debug)]
pub struct Key(u64);

impl Key {
    /// A root key from a seed.
    pub fn new(seed: u64) -> Key {
        Key(seed)
    }

    /// Two independent subkeys; the parent is consumed.
    pub fn split(self) -> (Key, Key) {
        (Key(threefry2x32(self.0, 0)), Key(threefry2x32(self.0, 1)))
    }

    /// `n` independent subkeys; the parent is consumed.
    pub fn split_n(self, n: usize) -> Vec<Key> {
        (0..n as u64).map(|i| Key(threefry2x32(self.0, i))).collect()
    }

    // the raw seed (self.0) for the graph builder to embed; borrows, does not consume.
    pub(crate) fn raw(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Random123 known-answer tests: pins the algorithm to the standard bits
    // (interop + a stability contract: changing threefry breaks this).
    #[test]
    fn threefry_kat() {
        // Random123 KAT: ctr=0 key=0 -> {v0=0x6b200159, v1=0x99ba4efe} (packed v0|v1<<32).
        // This one matches the reference exactly -> we're the standard algorithm; the
        // all-ones value pins our own bits as the forward stability contract.
        assert_eq!(threefry2x32(0, 0), 0x99ba_4efe_6b20_0159);
        assert_eq!(threefry2x32(u64::MAX, u64::MAX), 0xbb00_2be7_1cb9_96fc);
    }

    #[test]
    fn uniform_in_range_and_reproducible() {
        let draw = |seed| (0..10_000u64).map(|i| uniform_f32(threefry2x32(seed, i))).collect::<Vec<_>>();
        let a = draw(42);
        assert_eq!(a, draw(42)); // reproducible
        assert!(a.iter().all(|&x| (0.0..1.0).contains(&x)));
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
        assert_ne!(a, draw(43)); // different seed -> different stream
    }

    #[test]
    fn split_gives_independent_deterministic_subkeys() {
        let (a, b) = Key::new(7).split();
        assert_ne!(a.raw(), b.raw()); // the two halves differ
        let (a2, b2) = Key::new(7).split();
        assert_eq!((a.raw(), b.raw()), (a2.raw(), b2.raw())); // deterministic
    }
}
