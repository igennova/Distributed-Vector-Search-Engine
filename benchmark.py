"""
Benchmark harness.  (PROVIDED — your ruler for every phase.)

Runs any search function over the query set and reports:
  - recall@k  (vs the exact oracle)   -> quality
  - p50 / p99 latency in ms           -> speed
  - QPS                               -> throughput

Usage:
    python benchmark.py
"""
import time
import numpy as np
from dataset import make_dataset, exact_neighbors


def recall_at_k(predicted, truth):
    hits, total = 0, 0
    for p, t in zip(predicted, truth):
        hits += len(set(np.asarray(p).tolist()) & set(np.asarray(t).tolist()))
        total += len(t)
    return hits / total


def benchmark(search_fn, vectors, queries, k=10, ground_truth=None, label=""):
    if ground_truth is None:
        ground_truth = exact_neighbors(vectors, queries, k)

    latencies, preds = [], []
    for q in queries:
        t0 = time.perf_counter()
        idx = search_fn(vectors, q, k)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms
        preds.append(np.asarray(idx))

    latencies = np.array(latencies)
    recall = recall_at_k(preds, ground_truth)

    print(f"--- {label or search_fn.__name__} | n={len(vectors)} dim={vectors.shape[1]} k={k} ---")
    print(f"recall@{k} : {recall:.3f}")
    print(f"latency   : p50={np.percentile(latencies,50):.2f}ms  p99={np.percentile(latencies,99):.2f}ms")
    print(f"QPS       : {1000.0/np.mean(latencies):.0f}")
    return recall


if __name__ == "__main__":
    from search import brute_force_search

    vectors, queries = make_dataset()
    # Brute force IS the oracle, so its recall must be 1.000. If it isn't, your
    # implementation is wrong — fix it before moving to Phase 1.
    benchmark(brute_force_search, vectors, queries, k=10, label="brute force")
