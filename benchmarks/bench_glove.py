"""
Real data: Stanford GloVe word vectors (100 dims) at 10k and 100k vectors.

Compares HNSW (recall, latency, build time, memory) against exact search done the fast way:
one numpy matrix-vector product per query, not a Python loop. Queries are 1,000 words held
out of the index.

Needs data/glove.6B.zip from https://nlp.stanford.edu/projects/glove/ (862 MB).
Run:  python -m benchmarks.bench_glove            # 10k and 100k
      python -m benchmarks.bench_glove 400000     # any sizes, up to the 400k words available
"""
import sys
import time

import numpy as np
from vsearch.cluster import Coordinator
from vsearch.dataset import exact_neighbors, glove_benchmark_split, normalize
from vsearch.hnsw import HNSW

K = 10
SIZES = [10_000, 100_000]
EF_SEARCH = [50, 100, 200]
PARAMS = dict(M=16, ef_construction=100)


def timed(search, queries):
    """Run search on every query; returns (results, latencies in ms)."""
    latencies, results = [], []
    for q in queries:
        start = time.perf_counter()
        results.append(search(q))
        latencies.append((time.perf_counter() - start) * 1000)
    return results, np.array(latencies)


def report(n, label, search, queries, truth):
    results, latencies = timed(search, queries)
    hits = np.mean([len(set(r) & set(t.tolist())) / K for r, t in zip(results, truth)])
    print(f"{n:>8,}  {label:<26}{hits:>10.3f}"
          f"{np.percentile(latencies, 50):>10.2f}{1000 / latencies.mean():>9,.0f}")


def index_megabytes(index):
    """Rough memory of an index: its vector matrix plus the Python dicts and lists of the graph."""
    total = index._data.nbytes
    for layer in index.graph:
        total += sys.getsizeof(layer) + sum(sys.getsizeof(nbrs) for nbrs in layer.values())
    total += 28 * len(index.vectors)          # one int object per node id, shared by the lists
    return total / 1e6


def main(sizes=SIZES):
    print(f"{'vectors':>8}  {'method':<26}{'recall@10':>10}{'p50 (ms)':>10}{'QPS':>9}")
    print("-" * 65)
    notes = []
    for n in sizes:
        _, base, queries = glove_benchmark_split(n)
        truth = exact_neighbors(base, queries, K)
        unit = normalize(base)

        def exact(q):
            sims = unit @ (q / np.linalg.norm(q))
            top = np.argpartition(-sims, K)[:K]
            return top[np.argsort(-sims[top])].tolist()

        report(n, "exact (numpy)", exact, queries, truth)

        t0 = time.perf_counter()
        index = HNSW(seed=0, **PARAMS).build(base)
        build = time.perf_counter() - t0
        for ef in EF_SEARCH:
            index.ef_search = ef
            report(n, f"HNSW ef_search={ef}", lambda q: index.search(q, K), queries, truth)
        notes.append(f"{n:>8,} vectors: HNSW build {build:.0f}s, index ~{index_megabytes(index):.0f} MB")
        del index

        if n == sizes[-1]:
            t0 = time.perf_counter()
            with Coordinator(4, mode="processes", seed=0, ef_search=100, **PARAMS) as coord:
                coord.add(base)
                build = time.perf_counter() - t0
                report(n, "4 shards, ef_search=100", lambda q: coord.search(q, K), queries, truth)
            notes.append(f"{n:>8,} vectors: 4 shard processes build {build:.0f}s")

    print()
    for note in notes:
        print(note)


if __name__ == "__main__":    # required: "spawn" workers re-import this module
    main([int(arg) for arg in sys.argv[1:]] or SIZES)   # e.g. python -m benchmarks.bench_glove 400000
