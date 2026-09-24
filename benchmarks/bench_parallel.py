"""
Sequential vs threaded vs multi-process fan-out across 4 shards: recall, p50 latency,
and build time. Runs once on 10k vectors and once on tiny shards (200 vectors total).

Run:  python -m benchmarks.bench_parallel
"""
import time
import numpy as np
from vsearch.dataset import make_dataset, exact_neighbors
from vsearch.cluster import Coordinator, MODES

NUM_SHARDS = 4
K = 10


def run(mode, vectors, queries, truth):
    t0 = time.perf_counter()
    with Coordinator(NUM_SHARDS, mode=mode, M=16, ef_construction=100, ef_search=50,
                     seed=0) as coord:
        coord.add(vectors)
        build = time.perf_counter() - t0

        for q in queries[:5]:                 # warm-up: first calls pay one-time costs
            coord.search(q, K)

        latencies, hits = [], 0
        for q, t in zip(queries, truth):
            start = time.perf_counter()
            got = coord.search(q, K)
            latencies.append((time.perf_counter() - start) * 1000)
            hits += len(set(got) & set(t.tolist()))

    return hits / (K * len(queries)), np.percentile(latencies, 50), build


def main():
    for label, n in [("10,000 vectors", 10_000), ("200 vectors (tiny shards)", 200)]:
        vectors, queries = make_dataset(n_vectors=n)
        truth = exact_neighbors(vectors, queries, K)

        print(f"\n{label}, {NUM_SHARDS} shards")
        print(f"{'mode':<12}{'recall@10':>11}{'p50 (ms)':>10}{'build (s)':>11}")
        print("-" * 44)
        for mode in MODES:
            recall, p50, build = run(mode, vectors, queries, truth)
            print(f"{mode:<12}{recall:>11.3f}{p50:>10.3f}{build:>11.1f}")


if __name__ == "__main__":    # required: "spawn" workers re-import this module
    main()
