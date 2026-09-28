"""
Replicated shards over gRPC: what a second copy costs, and what happens when a server
crashes in the middle of a run.

Run:  python -m benchmarks.bench_replication
"""
import time

import numpy as np
from vsearch.client import GrpcCoordinator, ShardUnavailableError, local_grpc_cluster
from vsearch.dataset import make_dataset, exact_neighbors

NUM_SHARDS = 4
K = 10
PARAMS = dict(M=16, ef_construction=100, ef_search=50)


def timed_search(coord, query):
    start = time.perf_counter()
    ids = coord.search(query, K)
    return ids, (time.perf_counter() - start) * 1000


def cost_of_replicas(vectors, queries, truth):
    print(f"{NUM_SHARDS} shards, {len(vectors):,} vectors")
    print(f"{'replicas':>8}{'servers':>9}{'recall@10':>11}{'p50 (ms)':>10}{'build (s)':>11}")
    print("-" * 49)
    for replicas in [1, 2]:
        with local_grpc_cluster(NUM_SHARDS, replicas=replicas, seed=0, **PARAMS) as cluster, \
                GrpcCoordinator(cluster.addresses) as coord:
            t0 = time.perf_counter()
            coord.add(vectors)
            build = time.perf_counter() - t0
            for q in queries[:5]:                  # warm-up
                coord.search(q, K)
            results = [timed_search(coord, q) for q in queries]

        hits = sum(len(set(ids) & set(t.tolist())) for (ids, _), t in zip(results, truth))
        p50 = np.percentile([ms for _, ms in results], 50)
        print(f"{replicas:>8}{NUM_SHARDS * replicas:>9}{hits / (K * len(queries)):>11.3f}"
              f"{p50:>10.3f}{build:>11.1f}")


def crash_mid_run(vectors, queries, truth):
    print(f"\nCrash test: {NUM_SHARDS} shards x 2 replicas, one server killed (SIGKILL) "
          f"halfway through {len(queries)} queries")
    with local_grpc_cluster(NUM_SHARDS, replicas=2, seed=0, **PARAMS) as cluster, \
            GrpcCoordinator(cluster.addresses) as coord:
        coord.add(vectors)
        for q in queries[:5]:                      # warm-up
            coord.search(q, K)

        half = len(queries) // 2
        before, after, failed, hits = [], [], 0, 0
        for i, (q, t) in enumerate(zip(queries, truth)):
            if i == half:
                cluster.kill(shard=1, replica=0)
            try:
                ids, ms = timed_search(coord, q)
            except ShardUnavailableError:
                failed += 1
                continue
            (before if i < half else after).append(ms)
            hits += len(set(ids) & set(t.tolist()))

        answered = len(queries) - failed
        print(f"queries answered     : {answered}/{len(queries)}")
        print(f"recall@10            : {hits / (K * answered):.3f}")
        print(f"failovers            : {coord.failovers}")
        print(f"p50 before the crash : {np.percentile(before, 50):.3f} ms")
        print(f"p50 after the crash  : {np.percentile(after, 50):.3f} ms")
        print(f"slowest after crash  : {max(after):.3f} ms")


def main():
    vectors, queries = make_dataset()
    truth = exact_neighbors(vectors, queries, K)
    cost_of_replicas(vectors, queries, truth)
    crash_mid_run(vectors, queries, truth)


if __name__ == "__main__":
    main()
