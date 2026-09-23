"""
Sharded vs single-index search on the same data: recall, per-query latency, build time.

Run:  python benchmark_cluster.py
"""
import time
import numpy as np
from dataset import make_dataset, exact_neighbors
from cluster import Coordinator

vectors, queries = make_dataset()          # 10k vectors, dim 128, 100 queries
k = 10
truth = exact_neighbors(vectors, queries, k)

print(f"{'shards':>6}{'recall@10':>11}{'p50 (ms)':>10}{'build (s)':>11}")
print("-" * 38)
for num_shards in [1, 2, 4]:
    t0 = time.perf_counter()
    coord = Coordinator(num_shards, M=16, ef_construction=100, ef_search=50, seed=0).add(vectors)
    build = time.perf_counter() - t0

    latencies, hits = [], 0
    for q, t in zip(queries, truth):
        start = time.perf_counter()
        got = coord.search(q, k)
        latencies.append((time.perf_counter() - start) * 1000)
        hits += len(set(got) & set(t.tolist()))

    print(f"{num_shards:>6}{hits / (k * len(queries)):>11.3f}"
          f"{np.percentile(latencies, 50):>10.2f}{build:>11.1f}")
