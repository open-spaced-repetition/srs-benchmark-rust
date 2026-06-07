//! Port of sklearn `TimeSeriesSplit` (default args: gap=0, no max_train_size, test_size
//! derived). Older reviews train, newer reviews test; the first train-only block is
//! excluded from evaluation.

/// One split: train indices are `0..test_start`, test indices are `test_start..test_end`.
#[derive(Debug, Clone, Copy)]
pub struct Split {
    pub test_start: usize,
    pub test_end: usize,
}

/// `TimeSeriesSplit(n_splits).split(range(n))`. Requires `n >= n_splits + 1`.
pub fn time_series_split(n: usize, n_splits: usize) -> Vec<Split> {
    let n_folds = n_splits + 1;
    let test_size = n / n_folds; // >= 1 because callers ensure n >= 6 (>= n_folds)
    let first_start = n - n_splits * test_size;
    let mut splits = Vec::with_capacity(n_splits);
    let mut s = first_start;
    for _ in 0..n_splits {
        splits.push(Split {
            test_start: s,
            test_end: s + test_size,
        });
        s += test_size;
    }
    splits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_sklearn_shape() {
        // n=12744, n_splits=5 -> test_size=2124, first_start=1770+... check union size.
        let n = 12744;
        let sp = time_series_split(n, 5);
        assert_eq!(sp.len(), 5);
        let test_size = n / 6;
        assert_eq!(test_size, 2124);
        assert_eq!(sp[0].test_start, n - 5 * test_size);
        assert_eq!(sp[4].test_end, n);
        // union of test folds = 5 * test_size
        let total: usize = sp.iter().map(|s| s.test_end - s.test_start).sum();
        assert_eq!(total, 5 * test_size);
    }
}
