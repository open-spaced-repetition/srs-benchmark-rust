"""Ground truth for the smart-preset HDBSCAN experiments.

Emits RAW sklearn HDBSCAN labels (canonicalized: non-noise -> first-occurrence 0-based,
noise stays -1) for the 16-config grid min_cluster_size{2,5,10,20} x min_samples{1,5} x
method{eom,leaf} on several point sets. Used to validate the Rust `src/hdbscan.rs` port
(the noise->nearest reassignment is applied later, in the smart model, and tested separately).

Also prints, for a tiny crafted set, sklearn's implied CORE DISTANCES so the Rust uses the
same convention (which neighbor index `min_samples` maps to).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cluster import HDBSCAN
import sklearn

MCS = [2, 5, 10, 20]
MS = [1, 5]
METHODS = ["eom", "leaf"]

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

rng = np.random.default_rng(1)
centers = rng.normal(0, 4, (5, 8))
BLOBS = np.array([centers[rng.integers(5)] + rng.normal(0, 0.6, 8) for _ in range(40)])

def canon(labels):
    labels = list(labels)
    m = {}
    out = []
    for x in labels:
        x = int(x)
        if x == -1:
            out.append(-1)
            continue
        if x not in m:
            m[x] = len(m)
        out.append(m[x])
    return out

print(f"sklearn {sklearn.__version__}")

# --- core-distance convention probe ---
crafted = np.array([[0.0], [1.0], [2.0], [4.0], [8.0]])
D = np.abs(crafted[:, 0:1] - crafted[:, 0].reshape(1, -1))
for ms in (1, 2, 3):
    # candidate A: sorted row including self (index ms-1); candidate B: kth OTHER (index ms)
    a = np.sort(D, axis=1)[:, ms - 1]
    b = np.sort(D, axis=1)[:, ms]
    print(f"core probe ms={ms}: include-self[ms-1]={list(a)}  exclude-self[ms]={list(b)}")

import json
fixture = []
for name, P in [("decks10", DECKS), ("blobs40_8d", BLOBS)]:
    n = len(P)
    print(f"\n=== pointset {name}: n={n}, d={P.shape[1]} ===")
    entry = {"name": name, "points": P.tolist(), "cases": []}
    for mcs in MCS:
        for ms in MS:
            for method in METHODS:
                m = HDBSCAN(min_cluster_size=min(mcs, n), min_samples=min(ms, n),
                            metric="euclidean", cluster_selection_method=method,
                            allow_single_cluster=True)
                lab = canon(m.fit_predict(P))
                k = len(set(x for x in lab if x != -1))
                noise = sum(1 for x in lab if x == -1)
                print(f"CASE {name} mcs={mcs} ms={ms} {method:>4} k={k} noise={noise} : {lab}")
                entry["cases"].append({"mcs": mcs, "ms": ms, "leaf": method == "leaf", "labels": lab})
    fixture.append(entry)
json.dump(fixture, open("_smart/hdbscan_testcases.json", "w"))
print("\nwrote _smart/hdbscan_testcases.json")
