"""Reimplement the Rust HDBSCAN algorithm in Python and compare to sklearn, to localize the
ms>1 divergence (Rust translation bug vs MST/tie/algorithm difference)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import HDBSCAN

DECKS = np.array([
[0.041,2.417,4.128,11.97,5.638,0.446,3.262,2.305,0.168,1.332,0.352,0.004,0.75,0.089,0.662,1.3,0.882,0.307,3.587,0.303,0.01,0.227,2.641,0.559,1.3,2.5,1,0.072,0.163,0.5,0.955,0.224,0.623,0.136,0.386],
[0.099,0.457,1.36,25.017,5.648,0.409,3.449,2.106,0.284,1.138,0.418,0.006,0.731,0.174,0.392,1.551,1.037,0.479,3.701,0.335,0.003,0.267,2.66,0.583,1.329,2.506,1,0.028,0.142,0.759,0.957,0.178,0.648,0.249,0.5],
[0.042,0.083,0.865,1.87,5.688,0.333,3.306,2.112,0.2,1.147,0.464,0.001,0.778,0.164,0.363,1.699,1.107,0.434,3.675,0.377,0.001,0.318,2.66,0.569,1.383,2.5,0.955,0.064,0.376,0.608,0.981,0.208,0.67,0,0.1],
[0.07,0.135,2.239,7.486,5.104,0.501,2.939,1.952,0.018,1.108,0.595,0.007,0.687,0.605,0.263,3.717,1.572,0.287,4.162,0.46,0.019,0.539,2.644,0.728,2.51,2.5,0.902,0.059,0.317,0.539,0.966,0.232,0.738,0.014,0.117],
[0.039,0.041,1.549,7.846,5.843,0.472,3.012,1.997,0.003,1.065,0.545,0.005,0.566,0.344,0.375,3.307,1.673,0.296,4.053,0.394,0.005,0.346,2.644,0.782,2.253,2.5,0.996,0.032,0.176,0.73,0.954,0.436,0.461,0.009,0.115],
[0.007,0.096,2.056,5.456,5.993,0.397,3.382,1.969,0.378,1.011,0.33,0.003,0.626,0.157,0.085,1.541,1.184,0.338,3.729,0.367,0.003,0.303,2.648,0.255,1.29,2.5,1,0.033,0.034,0.727,0.972,0.311,0.554,0.083,0.318],
[0.015,0.093,1.758,27.1,5.867,0.646,3.369,2.074,0,1.111,0.532,0.013,0.668,0.632,0.47,3.844,1.654,0.316,4.107,0.466,0.023,0.409,2.709,0.986,2.042,2.5,0.984,0.049,0.294,0.571,0.97,0.161,0.742,0.02,0.215],
[0.217,0.217,1.609,1.715,5.219,0.463,3.16,1.799,0.094,0.828,0.524,0.006,0.725,0.269,0.144,2.484,1.851,0.146,4.174,0.51,0.001,0.459,2.733,0.679,1.702,2.5,0.999,0.038,0.32,0.663,0.92,0.416,0.5,0.072,0.304],
[0.175,0.175,2.254,82.964,5.95,0.478,3.681,2.108,0,1.137,0.516,0.005,0.673,0.273,0.27,2.34,1.111,0.321,3.68,0.444,0.003,0.383,2.676,0.536,1.557,2.55,0.947,0.044,0.148,0.657,0.989,0.173,0.649,0.041,0.28],
[1.598,1.598,6.67,47.39,5.559,0.459,2.962,2.446,0.019,1.443,0.577,0.003,0.774,0.246,0.287,2.197,1.096,0.311,3.644,0.388,0.001,0.328,2.665,0.671,1.307,2.564,0.904,0.042,0.06,0.646,0.989,0.3,0.605,0.271,0.523],
])

def my_hdbscan(P, mcs, ms, leaf):
    n = len(P)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    core = np.sort(D, axis=1)[:, ms - 1]
    mr = np.maximum(np.maximum(core[:, None], core[None, :]), D)
    Z = linkage(squareform(mr, checks=False), method="single")  # (n-1) x 4
    # build hierarchy arrays
    left = [int(Z[s, 0]) for s in range(n - 1)]
    right = [int(Z[s, 1]) for s in range(n - 1)]
    dist = [Z[s, 2] for s in range(n - 1)]
    sizes = [int(Z[s, 3]) for s in range(n - 1)]
    def sub_size(node): return 1 if node < n else sizes[node - n]
    def descend(start):
        out, st = [], [start]
        while st:
            nd = st.pop(); out.append(nd)
            if nd >= n: st += [left[nd - n], right[nd - n]]
        return out
    # condense
    root = 2 * n - 2
    relabel = {root: n}; nxt = n + 1
    ignore = [False] * (2 * n - 1); res = []
    for node in descend(root):
        if ignore[node] or node < n: continue
        s = node - n; lam = 1.0 / dist[s] if dist[s] > 0 else np.inf
        l, r = left[s], right[s]; lc, rc = sub_size(l), sub_size(r)
        if lc >= mcs and rc >= mcs:
            relabel[l] = nxt; nxt += 1; res.append((relabel[node], relabel[l], lam, lc))
            relabel[r] = nxt; nxt += 1; res.append((relabel[node], relabel[r], lam, rc))
        else:
            par = relabel[node]
            def fo(ch):
                for sub in descend(ch):
                    if sub < n: res.append((par, sub, lam, 1))
                    ignore[sub] = True
            if lc < mcs and rc < mcs: fo(l); fo(r)
            elif lc < mcs: relabel[r] = par; fo(l)
            else: relabel[l] = par; fo(r)
    # stability
    births = {c: lam for (_p, c, lam, _s) in res}; births[n] = 0.0
    stab = {}
    for (p, _c, lam, sz) in res: stab[p] = stab.get(p, 0.0) + (lam - births[p]) * sz
    # select
    children = {}
    for (p, c, _l, _s) in res:
        if c >= n: children.setdefault(p, []).append(c)
    if leaf:
        parents = set(p for (p, c, _l, _s) in res if c >= n)
        selected = set(c for c in stab if c not in parents)
    else:
        st2 = dict(stab); is_c = {c: True for c in stab}
        for node in sorted(stab, reverse=True):
            cs = sum(st2[c] for c in children.get(node, []))
            if cs > st2[node]:
                is_c[node] = False; st2[node] = cs
            else:
                stk = [node]
                while stk:
                    x = stk.pop()
                    if x != node: is_c[x] = False
                    stk += children.get(x, [])
        selected = set(c for c in is_c if is_c[c])
    # label
    pp = {c: p for (p, c, _l, _s) in res if c < n}
    cp = {c: p for (p, c, _l, _s) in res if c >= n}
    sel = sorted(selected); lof = {c: i for i, c in enumerate(sel)}
    out = [-1] * n
    for pt in range(n):
        cur = pp.get(pt)
        while cur is not None:
            if cur in lof: out[pt] = lof[cur]; break
            cur = cp.get(cur)
    return out

def canon(lab):
    m = {}; o = []
    for x in lab:
        x = int(x)
        if x == -1: o.append(-1); continue
        if x not in m: m[x] = len(m)
        o.append(m[x])
    return o

for mcs in [2, 5, 10, 20]:
    for ms in [1, 5]:
        for leaf in [False, True]:
            sk = canon(HDBSCAN(min_cluster_size=min(mcs, 10), min_samples=min(ms, 10),
                               metric="euclidean", cluster_selection_method="leaf" if leaf else "eom",
                               allow_single_cluster=True).fit_predict(DECKS))
            mine = canon(my_hdbscan(DECKS, min(mcs, 10), min(ms, 10), leaf))
            tag = "OK " if sk == mine else "DIFF"
            print(f"{tag} mcs={mcs} ms={ms} {'leaf' if leaf else 'eom'}: sklearn={sk} mine={mine}")
