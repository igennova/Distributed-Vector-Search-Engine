"""
Namespaces: what it buys to keep each tenant's vectors on one shard.

100,000 GloVe vectors belong to 200 tenants (500 each), on 4 shard servers over gRPC,
stored in two ways:

  spread      vectors go round-robin over the shards with a "tenant" payload field;
              a tenant's search is a filtered search on every shard
  namespace   each tenant is a namespace, stored on the one shard its name hashes to;
              a tenant's search asks that shard only

Then one shard is killed, to see how many tenants can still search.

Needs the GloVe cache (see bench_glove).
Run:  python -m benchmarks.bench_namespace
"""
import time

import numpy as np
from vsearch.client import GrpcCoordinator, ShardUnavailableError, local_grpc_cluster
from vsearch.dataset import glove_benchmark_split, normalize

N, TENANTS, NUM_SHARDS, K, QUERIES = 100_000, 200, 4, 10, 400
PARAMS = dict(M=16, ef_construction=100, ef_search=100)


def run(layout, base, queries, tenant_of, members):
    names = [f"tenant-{t}" for t in range(TENANTS)]
    rng = np.random.default_rng(1)
    asked = rng.integers(0, TENANTS, size=len(queries))

    def search(coord, q, tenant):
        if layout == "namespace":
            return coord.search(q, K, namespace=names[tenant])
        return coord.search(q, K, where={"tenant": names[tenant]})

    with local_grpc_cluster(NUM_SHARDS, seed=0, **PARAMS) as cluster, \
            GrpcCoordinator(cluster.addresses, timeout=2.0) as coord:
        start = time.perf_counter()
        if layout == "namespace":
            to_global = np.empty(len(base), dtype=np.int64)     # global id -> row in `base`
            for tenant, rows in enumerate(members):
                to_global[coord.size:coord.size + len(rows)] = rows
                coord.add(base[rows], namespace=names[tenant])
        else:
            to_global = np.arange(len(base))
            for begin in range(0, len(base), 10_000):
                rows = range(begin, min(begin + 10_000, len(base)))
                coord.add(base[rows], [{"tenant": names[tenant_of[i]]} for i in rows])
        build = time.perf_counter() - start

        for q, tenant in zip(queries[:20], asked):              # warm-up
            search(coord, q, tenant)
        latencies, hits = [], 0
        unit = normalize(base)
        for q, tenant in zip(queries, asked):
            begin = time.perf_counter()
            got = search(coord, q, tenant)
            latencies.append((time.perf_counter() - begin) * 1000)
            rows = members[tenant]
            truth = rows[np.argsort(-(unit[rows] @ (q / np.linalg.norm(q))))[:K]]
            hits += len(set(to_global[got].tolist()) & set(truth.tolist()))

        sizes = list(coord.shard_sizes)
        cluster.kill(0)
        alive = 0
        for tenant in range(TENANTS):
            try:
                search(coord, queries[0], tenant)
                alive += 1
            except ShardUnavailableError:
                pass
    shards_asked = 1 if layout == "namespace" else NUM_SHARDS
    print(f"{layout:<10}{shards_asked:>8}{hits / (K * len(queries)):>10.3f}"
          f"{np.percentile(latencies, 50):>10.2f}{np.percentile(latencies, 99):>10.2f}"
          f"{build:>10.0f}{alive / TENANTS:>12.0%}   {sizes}")


def main():
    _, base, queries = glove_benchmark_split(N)
    queries = queries[:QUERIES]
    tenant_of = np.random.default_rng(0).permutation(len(base)) % TENANTS
    members = [np.flatnonzero(tenant_of == t) for t in range(TENANTS)]
    print(f"{len(base):,} vectors, {TENANTS} tenants, {NUM_SHARDS} shards, {len(queries)} searches, "
          f"each inside one tenant\n")
    print(f"{'layout':<10}{'shards':>8}{'recall':>10}{'p50 ms':>10}{'p99 ms':>10}{'build s':>10}"
          f"{'tenants ok':>12}   vectors per shard")
    print(f"{'':<10}{'asked':>8}{'':>40}{'1 shard down':>12}")
    print("-" * 100)
    for layout in ("spread", "namespace"):
        run(layout, base, queries, tenant_of, members)


if __name__ == "__main__":
    main()
