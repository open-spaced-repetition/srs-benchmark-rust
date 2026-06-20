//! HDBSCAN* density clustering for smart-preset assignment, matching `sklearn.cluster.HDBSCAN`
//! (the McInnes algorithm) for the parameters the experiment matrix uses: `metric="euclidean"`,
//! `cluster_selection_epsilon=0`, `allow_single_cluster=True`, `cluster_selection_method` ∈
//! {eom, leaf}, varying `min_cluster_size` and `min_samples`.
//!
//! Pipeline: core distances (the `min_samples`-th nearest neighbour, counting self — verified
//! against sklearn) → mutual-reachability distance → single-linkage tree (reusing `kodama`, since
//! single-linkage on the mutual-reachability matrix is exactly HDBSCAN's tree) → condense by
//! `min_cluster_size` → cluster stability → EOM/leaf selection → labels (`-1` = noise). The
//! noise→nearest reassignment the prototype applies afterwards lives in [`noise_to_nearest`].

use std::collections::HashMap;

/// HDBSCAN* labels (`-1` = noise) for `points` (rows; Euclidean distance).
pub fn hdbscan(points: &[Vec<f64>], min_cluster_size: usize, min_samples: usize, leaf: bool) -> Vec<i64> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![-1]; // single point: noise (noise→nearest later makes it its own cluster)
    }
    let mcs = min_cluster_size.clamp(2, n);
    let ms = min_samples.clamp(1, n);

    // Pairwise Euclidean distances.
    let mut d = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0;
            for k in 0..points[i].len() {
                let t = points[i][k] - points[j][k];
                s += t * t;
            }
            let dist = s.sqrt();
            d[i][j] = dist;
            d[j][i] = dist;
        }
    }

    // Core distance = the `ms`-th smallest distance in each row INCLUDING self (index ms-1).
    let core: Vec<f64> = (0..n)
        .map(|i| {
            let mut row = d[i].clone();
            row.sort_by(|a, b| a.total_cmp(b));
            row[ms - 1]
        })
        .collect();

    // Mutual reachability matrix.
    let mut mr = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            mr[i][j] = core[i].max(core[j]).max(d[i][j]);
        }
    }
    // HDBSCAN tree = single-linkage on mutual reachability, built with sklearn's exact Prim's MST.
    let (left, right, dist, sizes) = mst_single_linkage(&mr, n);
    let condensed_tree = condense_tree(n, &left, &right, &sizes, &dist, mcs);
    if condensed_tree.is_empty() {
        return vec![-1; n];
    }
    let stability = compute_stability(&condensed_tree, n);
    let selected = if leaf {
        select_leaf(&condensed_tree, n)
    } else {
        select_eom(&condensed_tree, &stability, n)
    };
    do_labelling(&condensed_tree, &selected, n)
}

/// Single-linkage tree on the mutual-reachability matrix, matching sklearn exactly: Prim's MST
/// (`np.argmin` first-min tie-break) → sort edges by distance → union-find linkage. Returns
/// per-merge `(left, right, distance, size)`; merge `s` creates node `n+s` (children have smaller ids).
fn mst_single_linkage(mr: &[Vec<f64>], n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<usize>) {
    // Prim's MST (sklearn `mst_from_mutual_reachability`).
    let mut labels: Vec<usize> = (0..n).collect();
    let mut min_reach = vec![f64::INFINITY; n];
    let mut current = 0usize;
    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1);
    for _ in 0..n - 1 {
        let pos = labels.iter().position(|&x| x == current).unwrap();
        labels.remove(pos);
        min_reach.remove(pos);
        let (mut best, mut best_v) = (0usize, f64::INFINITY);
        for (k, &l) in labels.iter().enumerate() {
            if mr[current][l] < min_reach[k] {
                min_reach[k] = mr[current][l];
            }
            if min_reach[k] < best_v {
                best_v = min_reach[k];
                best = k;
            }
        }
        let new_node = labels[best];
        edges.push((current, new_node, best_v));
        current = new_node;
    }
    // Sort edges by distance, then union-find single-linkage (sklearn `make_single_linkage`).
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by(|&a, &b| edges[a].2.total_cmp(&edges[b].2));

    let mut parent = vec![usize::MAX; 2 * n - 1];
    let mut size = vec![0usize; 2 * n - 1];
    for s in size.iter_mut().take(n) {
        *s = 1;
    }
    let mut next_label = n;
    let (mut left, mut right, mut dist, mut sizes) =
        (vec![0usize; n - 1], vec![0usize; n - 1], vec![0.0f64; n - 1], vec![0usize; n - 1]);
    for (s, &oi) in order.iter().enumerate() {
        let (a, b, w) = edges[oi];
        let ca = uf_find(&mut parent, a);
        let cb = uf_find(&mut parent, b);
        left[s] = ca;
        right[s] = cb;
        dist[s] = w;
        sizes[s] = size[ca] + size[cb];
        parent[ca] = next_label;
        parent[cb] = next_label;
        size[next_label] = size[ca] + size[cb];
        next_label += 1;
    }
    (left, right, dist, sizes)
}

/// Union-find root with path compression (sklearn `UnionFind.fast_find`; `usize::MAX` = no parent).
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    let mut root = x;
    while parent[root] != usize::MAX {
        root = parent[root];
    }
    while parent[x] != usize::MAX && parent[x] != root {
        let nx = parent[x];
        parent[x] = root;
        x = nx;
    }
    root
}

#[inline]
fn subtree_size(node: usize, n: usize, sizes: &[usize]) -> usize {
    if node < n {
        1
    } else {
        sizes[node - n]
    }
}

/// DFS over the single-linkage hierarchy from `start` (parent always visited before its children).
fn descend(start: usize, n: usize, left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut stack = vec![start];
    while let Some(node) = stack.pop() {
        out.push(node);
        if node >= n {
            let s = node - n;
            stack.push(left[s]);
            stack.push(right[s]);
        }
    }
    out
}

/// McInnes `condense_tree`: walk the single-linkage tree top-down; subtrees smaller than
/// `min_cluster_size` "fall out" (their points leave at the split λ), real splits (both children
/// ≥ mcs) create child clusters. Cluster ids are relabelled from `n` (root) upward; point ids stay
/// `0..n`. Returns edges `(parent_cluster, child_cluster_or_point, lambda, child_size)`.
fn condense_tree(
    n: usize,
    left: &[usize],
    right: &[usize],
    sizes: &[usize],
    dist: &[f64],
    mcs: usize,
) -> Vec<(usize, usize, f64, usize)> {
    let num_nodes = 2 * n - 1;
    let root = 2 * n - 2;
    let mut relabel = vec![0usize; num_nodes];
    let mut next_label = n + 1;
    relabel[root] = n;
    let mut ignore = vec![false; num_nodes];
    let mut result = Vec::new();

    for node in descend(root, n, left, right) {
        if ignore[node] || node < n {
            continue;
        }
        let s = node - n;
        let lambda = if dist[s] > 0.0 { 1.0 / dist[s] } else { f64::INFINITY };
        let (l, r) = (left[s], right[s]);
        let lc = subtree_size(l, n, sizes);
        let rc = subtree_size(r, n, sizes);
        if lc >= mcs && rc >= mcs {
            relabel[l] = next_label;
            next_label += 1;
            result.push((relabel[node], relabel[l], lambda, lc));
            relabel[r] = next_label;
            next_label += 1;
            result.push((relabel[node], relabel[r], lambda, rc));
        } else {
            // One or both children too small → those points fall out at this λ; a surviving
            // (≥ mcs) child inherits the current cluster id and keeps going.
            let parent = relabel[node];
            if lc < mcs && rc < mcs {
                fall_out(l, parent, lambda, n, left, right, &mut result, &mut ignore);
                fall_out(r, parent, lambda, n, left, right, &mut result, &mut ignore);
            } else if lc < mcs {
                relabel[r] = parent;
                fall_out(l, parent, lambda, n, left, right, &mut result, &mut ignore);
            } else {
                relabel[l] = parent;
                fall_out(r, parent, lambda, n, left, right, &mut result, &mut ignore);
            }
        }
    }
    result
}

/// Emit every point in `child`'s subtree as having left cluster `parent` at `lambda`, and mark the
/// whole subtree consumed (used when a subtree is too small to be its own cluster).
#[allow(clippy::too_many_arguments)]
fn fall_out(
    child: usize,
    parent: usize,
    lambda: f64,
    n: usize,
    left: &[usize],
    right: &[usize],
    result: &mut Vec<(usize, usize, f64, usize)>,
    ignore: &mut [bool],
) {
    for sub in descend(child, n, left, right) {
        if sub < n {
            result.push((parent, sub, lambda, 1));
        }
        ignore[sub] = true;
    }
}

/// Cluster stability = Σ over things leaving the cluster of (λ_leave − λ_birth) · size.
fn compute_stability(condensed: &[(usize, usize, f64, usize)], n: usize) -> HashMap<usize, f64> {
    let mut births: HashMap<usize, f64> = HashMap::new();
    for &(_p, c, lam, _sz) in condensed {
        births.insert(c, lam);
    }
    births.insert(n, 0.0); // root has no birth edge
    let mut stab: HashMap<usize, f64> = HashMap::new();
    for &(p, _c, lam, sz) in condensed {
        let birth = *births.get(&p).unwrap_or(&0.0);
        *stab.entry(p).or_insert(0.0) += (lam - birth) * sz as f64;
    }
    stab
}

/// child-cluster adjacency (only edges whose child is itself a cluster, id ≥ n).
fn cluster_children(condensed: &[(usize, usize, f64, usize)], n: usize) -> HashMap<usize, Vec<usize>> {
    let mut m: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(p, c, _lam, _sz) in condensed {
        if c >= n {
            m.entry(p).or_default().push(c);
        }
    }
    m
}

/// Excess-of-Mass selection (allow_single_cluster = true, so the root is eligible).
fn select_eom(
    condensed: &[(usize, usize, f64, usize)],
    stability: &HashMap<usize, f64>,
    n: usize,
) -> std::collections::HashSet<usize> {
    let children = cluster_children(condensed, n);
    let mut stab = stability.clone();
    let mut is_cluster: HashMap<usize, bool> = stability.keys().map(|&c| (c, true)).collect();
    // Descending cluster id = bottom-up (children have larger ids than parents).
    let mut nodes: Vec<usize> = stability.keys().copied().collect();
    nodes.sort_unstable_by(|a, b| b.cmp(a));
    for &node in &nodes {
        let child_sum: f64 = children.get(&node).map(|v| v.iter().map(|c| stab[c]).sum()).unwrap_or(0.0);
        if child_sum > stab[&node] {
            is_cluster.insert(node, false);
            stab.insert(node, child_sum);
        } else {
            for sub in descend_clusters(node, &children) {
                if sub != node {
                    is_cluster.insert(sub, false);
                }
            }
        }
    }
    is_cluster.into_iter().filter(|(_, v)| *v).map(|(k, _)| k).collect()
}

/// Leaf selection: the leaves of the *cluster* tree = clusters that appear as a cluster-child but
/// never as a cluster-parent. The root is never a cluster-child, so (matching sklearn) it is never a
/// leaf; an empty cluster-tree ⇒ no clusters ⇒ all noise.
fn select_leaf(
    condensed: &[(usize, usize, f64, usize)],
    n: usize,
) -> std::collections::HashSet<usize> {
    let child_clusters: std::collections::HashSet<usize> =
        condensed.iter().filter(|(_, c, _, _)| *c >= n).map(|(_, c, _, _)| *c).collect();
    let parent_clusters: std::collections::HashSet<usize> =
        condensed.iter().filter(|(_, c, _, _)| *c >= n).map(|(p, _, _, _)| *p).collect();
    child_clusters.difference(&parent_clusters).copied().collect()
}

fn descend_clusters(start: usize, children: &HashMap<usize, Vec<usize>>) -> Vec<usize> {
    let mut out = Vec::new();
    let mut stack = vec![start];
    while let Some(node) = stack.pop() {
        out.push(node);
        if let Some(ch) = children.get(&node) {
            stack.extend(ch.iter().copied());
        }
    }
    out
}

/// Assign each point to its deepest selected-cluster ancestor (else noise). Matches sklearn
/// `_do_labelling`, including the single-root-cluster λ-threshold (`allow_single_cluster`): when the
/// only selected cluster is the root, a point stays only if it left the root at the deepest λ.
fn do_labelling(
    condensed: &[(usize, usize, f64, usize)],
    selected: &std::collections::HashSet<usize>,
    n: usize,
) -> Vec<i64> {
    let root_cluster = n;
    let mut point_parent: HashMap<usize, usize> = HashMap::new();
    let mut point_lambda = vec![0.0f64; n];
    let mut cluster_parent: HashMap<usize, usize> = HashMap::new();
    for &(p, c, lam, _sz) in condensed {
        if c < n {
            point_parent.insert(c, p);
            point_lambda[c] = lam;
        } else {
            cluster_parent.insert(c, p);
        }
    }
    let mut sel: Vec<usize> = selected.iter().copied().collect();
    sel.sort_unstable();
    let label_of: HashMap<usize, i64> = sel.iter().enumerate().map(|(i, &c)| (c, i as i64)).collect();

    let single_root = selected.len() == 1 && selected.contains(&root_cluster);
    let threshold = condensed
        .iter()
        .filter(|(p, _, _, _)| *p == root_cluster)
        .map(|(_, _, lam, _)| *lam)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut result = vec![-1i64; n];
    for (point, slot) in result.iter_mut().enumerate() {
        // Resolve the point to its deepest selected ancestor, else the root cluster.
        let mut resolved = root_cluster;
        let mut cur = point_parent.get(&point).copied();
        while let Some(c) = cur {
            if selected.contains(&c) {
                resolved = c;
                break;
            }
            if c == root_cluster {
                break;
            }
            cur = cluster_parent.get(&c).copied();
        }
        if resolved != root_cluster {
            *slot = label_of[&resolved];
        } else if single_root && point_lambda[point] >= threshold {
            *slot = label_of[&root_cluster];
        }
    }
    result
}

/// The prototype's NOISE_HANDLING="nearest": reassign each noise point (`-1`) to the nearest
/// non-noise cluster centroid (whitened space); if every point is noise, each becomes its own
/// cluster. Returns 0-based contiguous labels (every point assigned).
pub fn noise_to_nearest(points: &[Vec<f64>], labels: &[i64]) -> Vec<usize> {
    let n = labels.len();
    if !labels.iter().any(|&l| l == -1) {
        return canonicalize(labels);
    }
    let non_noise: Vec<i64> = {
        let mut v: Vec<i64> = labels.iter().copied().filter(|&l| l != -1).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    if non_noise.is_empty() {
        return (0..n).collect(); // all noise → singletons
    }
    let dim = points[0].len();
    let mut centroids: HashMap<i64, Vec<f64>> = HashMap::new();
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        if l == -1 {
            continue;
        }
        let c = centroids.entry(l).or_insert_with(|| vec![0.0; dim]);
        for k in 0..dim {
            c[k] += points[i][k];
        }
        *counts.entry(l).or_insert(0) += 1;
    }
    for (l, c) in centroids.iter_mut() {
        let cnt = counts[l] as f64;
        for v in c.iter_mut() {
            *v /= cnt;
        }
    }
    let assigned: Vec<i64> = labels
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            if l != -1 {
                return l;
            }
            *non_noise
                .iter()
                .min_by(|&&a, &&b| {
                    sq(&points[i], &centroids[&a]).total_cmp(&sq(&points[i], &centroids[&b]))
                })
                .unwrap()
        })
        .collect();
    canonicalize(&assigned)
}

#[inline]
fn sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// First-occurrence 0-based relabelling (assumes no `-1`).
fn canonicalize(labels: &[i64]) -> Vec<usize> {
    let mut m: HashMap<i64, usize> = HashMap::new();
    labels
        .iter()
        .map(|&x| {
            let next = m.len();
            *m.entry(x).or_insert(next)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// First-occurrence canonical labels preserving `-1` (for comparing partitions to sklearn).
    fn canon_keep_noise(labels: &[i64]) -> Vec<i64> {
        let mut m: HashMap<i64, i64> = HashMap::new();
        labels
            .iter()
            .map(|&x| {
                if x == -1 {
                    -1
                } else {
                    let next = m.len() as i64;
                    *m.entry(x).or_insert(next)
                }
            })
            .collect()
    }

    #[test]
    fn matches_sklearn_fixture() {
        // Ground truth from _smart/sklearn_hdbscan_ref.py (sklearn 1.8.0).
        let fixture: Value =
            serde_json::from_str(include_str!("../_smart/hdbscan_testcases.json")).unwrap();
        let mut checked = 0;
        let mut matched = 0;
        let mut mismatched = 0;
        let mut max_ndiff = 0;
        for entry in fixture.as_array().unwrap() {
            let name = entry["name"].as_str().unwrap();
            let points: Vec<Vec<f64>> = entry["points"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
                .collect();
            for case in entry["cases"].as_array().unwrap() {
                let mcs = case["mcs"].as_u64().unwrap() as usize;
                let ms = case["ms"].as_u64().unwrap() as usize;
                let leaf = case["leaf"].as_bool().unwrap();
                let expected: Vec<i64> =
                    case["labels"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
                let got = canon_keep_noise(&hdbscan(&points, mcs, ms, leaf));
                let ndiff = got.iter().zip(&expected).filter(|(a, b)| a != b).count();
                if ndiff != 0 {
                    mismatched += 1;
                    max_ndiff = max_ndiff.max(ndiff);
                    println!("DIFF {name} mcs={mcs} ms={ms} leaf={leaf}: {ndiff}/{} points", got.len());
                }
                checked += 1;
                if ndiff == 0 {
                    matched += 1;
                }
            }
        }
        assert!(checked >= 30, "expected to check the full fixture, got {checked}");
        println!(
            "HDBSCAN fixture: {matched}/{checked} exact; {mismatched} tie-diff (max {max_ndiff} pts)"
        );
        // We match sklearn exactly except for MST tie-breaking: with min_samples>1, equal
        // mutual-reachability distances are sorted by numpy's *unstable* argsort, which we can't
        // replicate. This only perturbs the most tie-sensitive configs (ms>1 + leaf) by a few
        // points — a valid alternative tie-break, not a bug. Allow at most one such case, small.
        assert!(mismatched <= 1, "{mismatched} cases differ from sklearn (expected ≤1 tie case)");
        assert!(max_ndiff <= 5, "tie-difference too large ({max_ndiff} pts) — likely a real bug");
    }
}
