//! Agglomerative hierarchical clustering for smart-preset assignment (step 5 of the pipeline).
//!
//! Linkage is computed by the `kodama` crate (a pure-Rust port of SciPy/fastcluster's
//! hierarchical clustering — O(n²) nn-chain for the reducible methods, so it scales to the
//! ~1000-deck users). The flat-cluster cut `fcluster(criterion="distance")` is reimplemented
//! here (SciPy's monotone max-distance criterion + union-find) since kodama only builds the tree.
//!
//! All methods operate on the (already-whitened) input points using Euclidean distance, where
//! Euclidean distance == Mahalanobis distance in log-param space. This matches the Python
//! prototype `FSRS smart preset assignment (simple).py`: single/complete/average there use the
//! pairwise Mahalanobis distance matrix and centroid/ward use the whitened vectors — both are
//! whitened-Euclidean, so a single Euclidean path reproduces all five.

use kodama::{linkage, Method as KMethod};

/// Hierarchical linkage method (the five SciPy methods used by the experiment matrix).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Single,
    Complete,
    Average,
    Centroid,
    Ward,
}

impl Method {
    pub fn parse(s: &str) -> Option<Method> {
        Some(match s {
            "single" => Method::Single,
            "complete" => Method::Complete,
            "average" => Method::Average,
            "centroid" => Method::Centroid,
            "ward" => Method::Ward,
            _ => return None,
        })
    }

    fn kodama(self) -> KMethod {
        match self {
            Method::Single => KMethod::Single,
            Method::Complete => KMethod::Complete,
            Method::Average => KMethod::Average,
            Method::Centroid => KMethod::Centroid,
            Method::Ward => KMethod::Ward,
        }
    }
}

/// Flat clustering equivalent to SciPy
/// `fcluster(linkage(points, method), t=threshold, criterion="distance")`.
///
/// Returns one 0-based cluster label per point, contiguous and in first-occurrence order
/// (so the label values are a canonical representation of the partition). Euclidean distance
/// is used between rows of `points`.
pub fn fcluster_distance(points: &[Vec<f64>], method: Method, threshold: f64) -> Vec<usize> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Condensed upper-triangular Euclidean distance matrix (kodama/SciPy layout: i<j).
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (&points[i], &points[j]);
            let mut s = 0.0f64;
            for k in 0..a.len() {
                let d = a[k] - b[k];
                s += d * d;
            }
            condensed.push(s.sqrt());
        }
    }

    fcluster_from_condensed(condensed, n, method, threshold)
}

/// Same flat-clustering cut as [`fcluster_distance`], but from a **precomputed** full `n×n` symmetric
/// distance matrix (e.g. KL divergence between deck predictions) instead of Euclidean over points.
/// single/complete/average are valid for any distance matrix; centroid/ward apply the Lance-Williams
/// update as SciPy does (they implicitly assume Euclidean, so on a non-Euclidean matrix they are a
/// heuristic — kept so the KL sweep can reuse the full 5-linkage matrix).
pub fn fcluster_distance_matrix(dist: &[Vec<f64>], method: Method, threshold: f64) -> Vec<usize> {
    let n = dist.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            condensed.push(dist[i][j]);
        }
    }
    fcluster_from_condensed(condensed, n, method, threshold)
}

/// Shared core: build the linkage tree from a condensed distance matrix and apply SciPy's
/// `fcluster(criterion="distance")` monotone max-distance cut. `n` ≥ 2.
fn fcluster_from_condensed(
    mut condensed: Vec<f64>,
    n: usize,
    method: Method,
    threshold: f64,
) -> Vec<usize> {
    let dend = linkage(&mut condensed, n, method.kodama());
    let steps = dend.steps();

    // SciPy fcluster(criterion="distance"): cut at `threshold` using the *monotone* height
    // MD[node] = max linkage distance anywhere in node's subtree (handles centroid inversions).
    // A merge whose MD <= threshold keeps its two children in the same flat cluster.
    // Node ids: 0..n-1 = original points; merge step s creates node n+s (children always have
    // smaller ids, so a single forward pass suffices).
    let mut md = vec![0.0f64; steps.len()]; // md[s] = MD for node n+s
    let mut rep = vec![0usize; steps.len()]; // a representative leaf for node n+s
    let mut uf = UnionFind::new(n);
    for (s, st) in steps.iter().enumerate() {
        let (a, b) = (st.cluster1, st.cluster2);
        let md_a = if a < n { 0.0 } else { md[a - n] };
        let md_b = if b < n { 0.0 } else { md[b - n] };
        md[s] = st.dissimilarity.max(md_a).max(md_b);
        let rep_a = if a < n { a } else { rep[a - n] };
        let rep_b = if b < n { b } else { rep[b - n] };
        rep[s] = rep_a;
        if md[s] <= threshold {
            uf.union(rep_a, rep_b);
        }
    }

    // Canonicalize union-find roots to first-occurrence 0-based labels.
    let mut map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let root = uf.find(i);
        let next = map.len();
        out.push(*map.entry(root).or_insert(next));
    }
    out
}

/// Agglomerative merge to **exactly `k` clusters** from points (Euclidean): cut the dendrogram after
/// its `n−k` closest merges. Used to pre-aggregate >12 decks into 12 pseudo-decks before the
/// optimal-partition search. `k ≥ n` ⇒ every point its own cluster.
pub fn fcluster_k_points(points: &[Vec<f64>], method: Method, k: usize) -> Vec<usize> {
    let n = points.len();
    if n <= k {
        return (0..n).collect();
    }
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let s: f64 = points[i].iter().zip(&points[j]).map(|(a, b)| (a - b) * (a - b)).sum();
            condensed.push(s.sqrt());
        }
    }
    fcluster_k_from_condensed(condensed, n, method, k)
}

/// Same as [`fcluster_k_points`] but from a precomputed full `n×n` distance matrix (the KL path).
pub fn fcluster_k_matrix(dist: &[Vec<f64>], method: Method, k: usize) -> Vec<usize> {
    let n = dist.len();
    if n <= k {
        return (0..n).collect();
    }
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            condensed.push(dist[i][j]);
        }
    }
    fcluster_k_from_condensed(condensed, n, method, k)
}

/// Union the first `n−k` merges of the linkage tree (dendrogram steps are in non-decreasing
/// dissimilarity) to leave exactly `k` connected components; canonicalize to 0-based labels.
fn fcluster_k_from_condensed(mut condensed: Vec<f64>, n: usize, method: Method, k: usize) -> Vec<usize> {
    let dend = linkage(&mut condensed, n, method.kodama());
    let steps = dend.steps();
    let n_merges = n - k;
    let mut rep = vec![0usize; steps.len()];
    let mut uf = UnionFind::new(n);
    for (s, st) in steps.iter().enumerate() {
        let (a, b) = (st.cluster1, st.cluster2);
        let rep_a = if a < n { a } else { rep[a - n] };
        let rep_b = if b < n { b } else { rep[b - n] };
        rep[s] = rep_a;
        if s < n_merges {
            uf.union(rep_a, rep_b);
        }
    }
    let mut map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let root = uf.find(i);
        let next = map.len();
        out.push(*map.entry(root).or_insert(next));
    }
    out
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind { parent: (0..n).collect(), rank: vec![0; n] }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two well-separated 2D blobs (4 + 3 points).
    fn two_blobs() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.0, 0.2],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.2, 9.9],
            vec![9.8, 10.1],
        ]
    }

    // The 10 prototype decks from "FSRS smart preset assignment (simple).py" (raw, unwhitened —
    // the clustering algorithm is independent of whitening).
    #[rustfmt::skip]
    fn decks10() -> Vec<Vec<f64>> {
        vec![
            vec![0.041,2.417,4.128,11.97,5.638,0.446,3.262,2.305,0.168,1.332,0.352,0.004,0.75,0.089,0.662,1.3,0.882,0.307,3.587,0.303,0.01,0.227,2.641,0.559,1.3,2.5,1.0,0.072,0.163,0.5,0.955,0.224,0.623,0.136,0.386],
            vec![0.099,0.457,1.36,25.017,5.648,0.409,3.449,2.106,0.284,1.138,0.418,0.006,0.731,0.174,0.392,1.551,1.037,0.479,3.701,0.335,0.003,0.267,2.66,0.583,1.329,2.506,1.0,0.028,0.142,0.759,0.957,0.178,0.648,0.249,0.5],
            vec![0.042,0.083,0.865,1.87,5.688,0.333,3.306,2.112,0.2,1.147,0.464,0.001,0.778,0.164,0.363,1.699,1.107,0.434,3.675,0.377,0.001,0.318,2.66,0.569,1.383,2.5,0.955,0.064,0.376,0.608,0.981,0.208,0.67,0.0,0.1],
            vec![0.07,0.135,2.239,7.486,5.104,0.501,2.939,1.952,0.018,1.108,0.595,0.007,0.687,0.605,0.263,3.717,1.572,0.287,4.162,0.46,0.019,0.539,2.644,0.728,2.51,2.5,0.902,0.059,0.317,0.539,0.966,0.232,0.738,0.014,0.117],
            vec![0.039,0.041,1.549,7.846,5.843,0.472,3.012,1.997,0.003,1.065,0.545,0.005,0.566,0.344,0.375,3.307,1.673,0.296,4.053,0.394,0.005,0.346,2.644,0.782,2.253,2.5,0.996,0.032,0.176,0.73,0.954,0.436,0.461,0.009,0.115],
            vec![0.007,0.096,2.056,5.456,5.993,0.397,3.382,1.969,0.378,1.011,0.33,0.003,0.626,0.157,0.085,1.541,1.184,0.338,3.729,0.367,0.003,0.303,2.648,0.255,1.29,2.5,1.0,0.033,0.034,0.727,0.972,0.311,0.554,0.083,0.318],
            vec![0.015,0.093,1.758,27.1,5.867,0.646,3.369,2.074,0.0,1.111,0.532,0.013,0.668,0.632,0.47,3.844,1.654,0.316,4.107,0.466,0.023,0.409,2.709,0.986,2.042,2.5,0.984,0.049,0.294,0.571,0.97,0.161,0.742,0.02,0.215],
            vec![0.217,0.217,1.609,1.715,5.219,0.463,3.16,1.799,0.094,0.828,0.524,0.006,0.725,0.269,0.144,2.484,1.851,0.146,4.174,0.51,0.001,0.459,2.733,0.679,1.702,2.5,0.999,0.038,0.32,0.663,0.92,0.416,0.5,0.072,0.304],
            vec![0.175,0.175,2.254,82.964,5.95,0.478,3.681,2.108,0.0,1.137,0.516,0.005,0.673,0.273,0.27,2.34,1.111,0.321,3.68,0.444,0.003,0.383,2.676,0.536,1.557,2.55,0.947,0.044,0.148,0.657,0.989,0.173,0.649,0.041,0.28],
            vec![1.598,1.598,6.67,47.39,5.559,0.459,2.962,2.446,0.019,1.443,0.577,0.003,0.774,0.246,0.287,2.197,1.096,0.311,3.644,0.388,0.001,0.328,2.665,0.671,1.307,2.564,0.904,0.042,0.06,0.646,0.989,0.3,0.605,0.271,0.523],
        ]
    }

    #[test]
    fn two_blobs_split() {
        // Every method/threshold should recover the two blobs.
        for m in [Method::Single, Method::Complete, Method::Average, Method::Centroid, Method::Ward] {
            assert_eq!(
                fcluster_distance(&two_blobs(), m, 0.5),
                vec![0, 0, 0, 0, 1, 1, 1],
                "method {m:?}"
            );
        }
    }

    #[test]
    fn decks10_matches_scipy() {
        // Ground truth from scipy (see _smart/scipy_cluster_ref.py), canonicalized 0-based.
        let cases: &[(Method, f64, &[usize])] = &[
            (Method::Single, 5.0, &[0, 1, 2, 2, 2, 2, 1, 2, 3, 4]),
            (Method::Complete, 5.0, &[0, 1, 2, 3, 3, 3, 1, 2, 4, 5]),
            (Method::Average, 5.0, &[0, 1, 2, 3, 3, 3, 1, 2, 4, 5]),
            (Method::Centroid, 5.0, &[0, 1, 2, 3, 3, 3, 1, 2, 4, 5]),
            (Method::Ward, 5.0, &[0, 1, 2, 3, 3, 3, 1, 2, 4, 5]),
            (Method::Single, 30.0, &[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            (Method::Average, 30.0, &[0, 0, 0, 0, 0, 0, 0, 0, 1, 2]),
            (Method::Ward, 30.0, &[0, 1, 0, 0, 0, 0, 1, 0, 2, 1]),
            (Method::Complete, 30.0, &[0, 1, 0, 0, 0, 0, 1, 0, 2, 1]),
        ];
        let pts = decks10();
        for (m, t, expected) in cases {
            assert_eq!(
                &fcluster_distance(&pts, *m, *t),
                expected,
                "method {m:?} threshold {t}"
            );
        }
    }

    #[test]
    fn singletons_when_threshold_zero() {
        // Threshold 0: nothing merges (all distinct points) -> n singletons.
        let labels = fcluster_distance(&two_blobs(), Method::Average, 0.0);
        assert_eq!(labels, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn trivial_sizes() {
        assert_eq!(fcluster_distance(&[], Method::Ward, 1.0), Vec::<usize>::new());
        assert_eq!(fcluster_distance(&[vec![1.0, 2.0]], Method::Ward, 1.0), vec![0]);
    }

    // Build a full Euclidean distance matrix from points (so the precomputed-matrix path can be
    // checked against the points path).
    fn euclid_matrix(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = points.len();
        let mut d = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let s: f64 = points[i].iter().zip(&points[j]).map(|(a, b)| (a - b) * (a - b)).sum();
                d[i][j] = s.sqrt();
                d[j][i] = s.sqrt();
            }
        }
        d
    }

    #[test]
    fn matrix_path_matches_points_path() {
        // fcluster_distance_matrix on the Euclidean matrix == fcluster_distance on the points.
        let pts = decks10();
        let mat = euclid_matrix(&pts);
        for m in [Method::Single, Method::Complete, Method::Average, Method::Centroid, Method::Ward] {
            for t in [5.0, 30.0, 0.0] {
                assert_eq!(
                    fcluster_distance_matrix(&mat, m, t),
                    fcluster_distance(&pts, m, t),
                    "method {m:?} threshold {t}"
                );
            }
        }
        assert_eq!(fcluster_distance_matrix(&[], Method::Ward, 1.0), Vec::<usize>::new());
        assert_eq!(fcluster_distance_matrix(&[vec![0.0]], Method::Ward, 1.0), vec![0]);
    }

    #[test]
    fn merge_to_exactly_k() {
        // Two well-separated blobs (4+3): cutting to k=2 must recover them; k≥n ⇒ singletons.
        let pts = two_blobs();
        let lab = fcluster_k_points(&pts, Method::Average, 2);
        assert_eq!(lab.iter().copied().max().unwrap() + 1, 2, "should be exactly 2 clusters");
        assert_eq!(lab, vec![0, 0, 0, 0, 1, 1, 1]);
        // Exactly k components for a range of k.
        for k in 1..=7 {
            let l = fcluster_k_points(&pts, Method::Average, k);
            assert_eq!(l.iter().copied().max().unwrap() + 1, k.min(7), "k={k}");
        }
        assert_eq!(fcluster_k_points(&pts, Method::Average, 10), vec![0, 1, 2, 3, 4, 5, 6]);
    }
}
