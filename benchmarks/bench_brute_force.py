"""
Benchmark exact brute-force search: recall@k (vs. the exact ground truth), p50/p99
latency, and QPS.

Run:  python -m benchmarks.bench_brute_force
"""
import time
import numpy as np
from vsearch.dataset import make_dataset, exact_neighbors
from vsearch.brute_force import brute_force_search


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
    vectors, queries = make_dataset()
    # Brute force is exact, so recall here should always be 1.000.
    benchmark(brute_force_search, vectors, queries, k=10, label="brute force")
