use std::f32::consts::PI;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rng {
    state: [u32; 4],
}

impl Rng {
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let mut state = [0_u32; 4];
        for chunk in state.chunks_mut(2) {
            let bits = sm.next_u64();
            chunk[0] = bits as u32;
            if chunk.len() > 1 {
                chunk[1] = (bits >> 32) as u32;
            }
        }
        if state.iter().all(|value| *value == 0) {
            state[0] = 1;
        }
        Self { state }
    }

    pub fn next_u32(&mut self) -> u32 {
        let result = self.state[0].wrapping_add(self.state[3]);
        let t = self.state[1] << 9;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(11);

        result
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 + 0.5) / (u32::MAX as f32 + 1.0)
    }

    pub fn gen_range_usize(&mut self, upper_exclusive: usize) -> usize {
        assert!(upper_exclusive > 0, "upper_exclusive must be > 0");
        let bound = upper_exclusive as u32;
        let zone = u32::MAX - (u32::MAX % bound);
        loop {
            let value = self.next_u32();
            if value < zone {
                return (value % bound) as usize;
            }
        }
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        if items.len() < 2 {
            return;
        }
        for idx in (1..items.len()).rev() {
            let swap_idx = self.gen_range_usize(idx + 1);
            items.swap(idx, swap_idx);
        }
    }

    pub fn gauss(&mut self, std: f32) -> f32 {
        let u1 = self.next_f32().clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
        let u2 = self.next_f32();
        std * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    pub fn sample_weighted(&mut self, weights: &[f32]) -> Option<usize> {
        let total: f32 = weights.iter().copied().filter(|w| *w > 0.0).sum();
        if total <= 0.0 {
            return None;
        }

        let mut threshold = self.next_f32() * total;
        for (idx, weight) in weights.iter().copied().enumerate() {
            if weight <= 0.0 {
                continue;
            }
            threshold -= weight;
            if threshold <= 0.0 {
                return Some(idx);
            }
        }
        Some(weights.len().saturating_sub(1))
    }
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn seeded_rng_is_deterministic() {
        let mut left = Rng::from_seed(42);
        let mut right = Rng::from_seed(42);
        let left_values: Vec<u32> = (0..8).map(|_| left.next_u32()).collect();
        let right_values: Vec<u32> = (0..8).map(|_| right.next_u32()).collect();
        assert_eq!(left_values, right_values);
    }

    #[test]
    fn shuffle_preserves_all_items() {
        let mut rng = Rng::from_seed(42);
        let mut values = vec![1, 2, 3, 4, 5];
        rng.shuffle(&mut values);
        values.sort_unstable();
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn weighted_sampling_prefers_nonzero_weights() {
        let mut rng = Rng::from_seed(7);
        let pick = rng.sample_weighted(&[0.0, 0.0, 3.0, 0.0]);
        assert_eq!(pick, Some(2));
    }
}
