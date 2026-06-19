//! NumPy legacy `RandomState` (MT19937) — just enough to reproduce
//! `np.random.RandomState(seed).permutation(n)`, which is exactly what
//! `pandas.DataFrame.sample(frac=1, random_state=seed)` does (verified).
//!
//! The Reptile finetune (`reptile_trainer{,_gru}.py::finetune`) shuffles the per-user
//! training rows once via `df.sample(frac=1, random_state=2025)` before batching. For users
//! whose per-split train set exceeds one batch (8192 rows) the batch *composition* depends on
//! this permutation, so we must reproduce NumPy's RNG bit-for-bit.
//!
//! This is a DIFFERENT generator from torch's MT19937 randperm (`train.rs`): NumPy seeds via
//! `init_by_array` and shuffles with descending Fisher–Yates + `random_interval`
//! (mask-rejection over 32-bit words), whereas torch uses a 32-bit Fisher–Yates with its own
//! seeding. Unit-tested against `numpy 2.4.2` reference permutations below.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct MT19937 {
    mt: [u32; N],
    mti: usize,
}

impl MT19937 {
    /// `init_genrand(s)` — the scalar seeding primitive.
    fn init_genrand(s: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = s;
        for i in 1..N {
            // 1812433253 * (mt[i-1] ^ (mt[i-1] >> 30)) + i  (mod 2^32)
            let prev = mt[i - 1] ^ (mt[i - 1] >> 30);
            mt[i] = 1812433253u32
                .wrapping_mul(prev)
                .wrapping_add(i as u32);
        }
        MT19937 { mt, mti: N }
    }

    /// NumPy's `RandomState(int_seed)` seeds a scalar int that fits in `uint32` directly via
    /// `mt19937_seed`, which is exactly the standard `init_genrand(seed)` (verified term-by-term
    /// against NumPy's C source — it is NOT `init_by_array`, which is only used for array seeds).
    pub fn new_from_seed(seed: u32) -> Self {
        Self::init_genrand(seed)
    }

    /// `genrand_int32` — one 32-bit MT output.
    pub fn next_u32(&mut self) -> u32 {
        if self.mti >= N {
            // Regenerate the whole state array.
            let mag01 = [0u32, MATRIX_A];
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] =
                    self.mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        // Tempering.
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random_interval(max)` — uniform integer in `[0, max]` via mask-rejection over 32-bit
    /// draws (NumPy's `distributions.c::random_interval` for `max <= 0xffffffff`).
    fn random_interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        // Smallest bit mask >= max.
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let value = self.next_u32() & mask;
            if value <= max {
                return value;
            }
        }
    }

    /// `RandomState.permutation(n)` = `arange(n)` then legacy in-place Fisher–Yates shuffle
    /// (descending `i`, `j = random_interval(i)`).
    pub fn permutation(seed: u32, n: usize) -> Vec<usize> {
        let mut arr: Vec<usize> = (0..n).collect();
        if n <= 1 {
            return arr;
        }
        let mut rng = Self::new_from_seed(seed);
        let mut i = n - 1;
        while i >= 1 {
            let j = rng.random_interval(i as u32) as usize;
            arr.swap(i, j);
            i -= 1;
        }
        arr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from `numpy 2.4.2`:
    //   np.random.RandomState(2025).permutation(n)
    #[test]
    fn raw_stream_matches_numpy() {
        let mut rng = MT19937::new_from_seed(2025);
        let first: Vec<u32> = (0..6).map(|_| rng.next_u32()).collect();
        assert_eq!(
            first,
            vec![581917246, 2146402014, 3813294034, 1466313566, 4005510732, 453344760]
        );
    }

    #[test]
    fn permutation_matches_numpy() {
        let p5 = MT19937::permutation(2025, 5);
        assert_eq!(p5, vec![1, 3, 0, 4, 2]);

        let p10 = MT19937::permutation(2025, 10);
        assert_eq!(p10, vec![6, 1, 5, 9, 0, 4, 7, 3, 8, 2]);

        let p17 = MT19937::permutation(2025, 17);
        assert_eq!(p17[..8], [4, 7, 13, 9, 11, 1, 10, 2]);
        assert_eq!(p17[14..], [3, 8, 12]);

        let p100 = MT19937::permutation(2025, 100);
        assert_eq!(p100[..8], [49, 77, 61, 69, 6, 95, 68, 19]);
        assert_eq!(p100[97..], [82, 94, 62]);
        assert_eq!(p100.iter().sum::<usize>(), 4950);

        let p8193 = MT19937::permutation(2025, 8193);
        assert_eq!(p8193[..8], [976, 7872, 3558, 5270, 855, 1917, 4017, 4088]);
        assert_eq!(p8193[8190..], [8146, 7902, 6718]);
        assert_eq!(p8193.iter().sum::<usize>(), 33558528);

        let p20000 = MT19937::permutation(2025, 20000);
        assert_eq!(p20000[..8], [17280, 7276, 9392, 6463, 14869, 4907, 7152, 11793]);
        assert_eq!(p20000[19997..], [15948, 11102, 16338]);
        assert_eq!(p20000.iter().sum::<usize>(), 199990000);
    }
}
