"""
Compare HNSW against the exact brute-force baseline, and show the recall/latency
tradeoff as ef_search varies.

Run:  python -m benchmarks.bench_hnsw
"""
import time
import numpy as np
from vsearch.dataset import make_dataset, exact_neighbors
from vsearch.brute_force import brute_force_search
from vsearch.hnsw import HNSW


def recall_at_k(predicted, truth):
    hits = sum(len(set(p) & set(t.tolist())) for p, t in zip(predicted, truth))
    return hits / (len(truth) * len(truth[0]))


def measure(search_fn, queries, k):
    latencies, preds = [], []
    for q in queries:
        t0 = time.perf_counter()
        preds.append(search_fn(q, k))
        latencies.append((time.perf_counter() - t0) * 1000)
    return preds, np.array(latencies)


def row(label, preds, lat, truth):
    print(f"{label:<22}{recall_at_k(preds, truth):>11.3f}"
          f"{np.percentile(lat, 50):>11.2f}{1000 / np.mean(lat):>8.0f}")


vectors, queries = make_dataset()          # 10k vectors, dim 128, 100 queries
k = 10
ground_truth = exact_neighbors(vectors, queries, k)

print("Building HNSW index...")
t0 = time.perf_counter()
index = HNSW(M=16, ef_construction=200, ef_search=200, seed=0).build(vectors)
print(f"build time: {time.perf_counter() - t0:.1f}s  ({len(vectors)} vectors)\n")

print(f"{'method':<22}{'recall@10':>11}{'p50 (ms)':>11}{'QPS':>8}")
print("-" * 52)

preds, lat = measure(lambda q, k: brute_force_search(vectors, q, k), queries, k)
row("brute force", preds, lat, ground_truth)

for ef in [50, 100, 200, 400]:
    index.ef_search = ef
    preds, lat = measure(lambda q, k: index.search(q, k), queries, k)
    row(f"HNSW ef_search={ef}", preds, lat, ground_truth)
