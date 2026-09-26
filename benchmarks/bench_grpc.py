"""
Shards over gRPC vs shards over local pipes: recall, p50 latency, and build time on
4 shards. Runs once on 10k vectors and once on tiny shards (200 vectors total).

Run:  python -m benchmarks.bench_grpc
"""
import time
from contextlib import ExitStack

import numpy as np
from vsearch.client import GrpcCoordinator, local_grpc_cluster
from vsearch.cluster import Coordinator
from vsearch.dataset import make_dataset, exact_neighbors

NUM_SHARDS = 4
K = 10
PARAMS = dict(M=16, ef_construction=100, ef_search=50)


def open_coordinator(kind, stack):
    if kind == "grpc":
        addresses = stack.enter_context(local_grpc_cluster(NUM_SHARDS, seed=0, **PARAMS))
        return stack.enter_context(GrpcCoordinator(addresses))
    mode = "sequential" if kind == "sequential" else "processes"
    return stack.enter_context(Coordinator(NUM_SHARDS, mode=mode, seed=0, **PARAMS))


def run(kind, vectors, queries, truth):
    with ExitStack() as stack:
        coord = open_coordinator(kind, stack)
        t0 = time.perf_counter()
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
        print(f"{'transport':<22}{'recall@10':>11}{'p50 (ms)':>10}{'build (s)':>11}")
        print("-" * 54)
        for kind, name in [("sequential", "in-process sequential"),
                           ("processes", "processes over pipes"),
                           ("grpc", "gRPC servers")]:
            recall, p50, build = run(kind, vectors, queries, truth)
            print(f"{name:<22}{recall:>11.3f}{p50:>10.3f}{build:>11.1f}")


if __name__ == "__main__":
    main()
