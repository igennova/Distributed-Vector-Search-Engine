"""
Filtered search: three ways to answer "the 10 nearest vectors among those that match a
filter", on GloVe word vectors, as the filter gets more selective.

  filter after   search the unfiltered top 10, then drop what does not match
  graph          walk the HNSW graph, keeping only matching nodes as results
  exact          compare the query with every matching vector

Each vector is given a random group; a filter matching 10% of the data selects one group
out of ten. Recall is measured against the exact filtered answer.

Needs the GloVe cache (see bench_glove).
Run:  python -m benchmarks.bench_filter            # 100k vectors
      python -m benchmarks.bench_filter 400000
"""
import sys
import time

import numpy as np
from vsearch.cluster import filter_strategy
from vsearch.dataset import glove_benchmark_split, normalize
from vsearch.hnsw import HNSW

K = 10
PARAMS = dict(M=16, ef_construction=100, ef_search=100)
FRACTIONS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001]
QUERIES = 200


def measure(search, queries, truth):
    """Mean recall@K, mean number of results, and median latency in ms."""
    recalls, counts, latencies = [], [], []
    for q, expected in zip(queries, truth):
        start = time.perf_counter()
        got = search(q)
        latencies.append((time.perf_counter() - start) * 1000)
        counts.append(len(got))
        recalls.append(len(set(got) & expected) / len(expected))
    return np.mean(recalls), np.mean(counts), np.percentile(latencies, 50)


def main(n=100_000):
    _, base, queries = glove_benchmark_split(n)
    n = len(base)                      # the queries are held out of the words available
    queries = queries[:QUERIES]
    unit = normalize(base)
    start = time.perf_counter()
    index = HNSW(seed=0, **PARAMS).build(base)
    print(f"{n:,} GloVe vectors, HNSW built in {time.perf_counter() - start:.0f}s, "
          f"ef_search={PARAMS['ef_search']}, {len(queries)} queries\n")

    rng = np.random.default_rng(0)
    print(f"{'':>19} | {'filter after':^17} | {'graph':^17} | {'exact':^8} | engine")
    print(f"{'matching':>8} {'vectors':>10} | {'recall':>7} {'results':>9} | {'recall':>7} {'ms':>9} |"
          f" {'ms':>8} | picks")
    print("-" * 80)
    for fraction in FRACTIONS:
        allowed = rng.random(n) < fraction
        ids = np.flatnonzero(allowed)
        truth = []
        for q in queries:
            sims = unit[ids] @ (q / np.linalg.norm(q))
            truth.append(set(ids[np.argsort(-sims)[:K]].tolist()))

        after = measure(lambda q: [i for i in index.search(q, K) if allowed[i]], queries, truth)
        graph = measure(lambda q: [i for _, i in index.search_with_distances(q, K, allowed)],
                        queries, truth)
        exact = measure(lambda q: [i for _, i in index.exact_among(q, ids, K)], queries, truth)
        assert exact[0] == 1.0
        print(f"{fraction:>8.1%} {len(ids):>10,} | {after[0]:>7.3f} {after[1]:>9.1f} |"
              f" {graph[0]:>7.3f} {graph[2]:>9.2f} | {exact[2]:>8.2f} |"
              f" {filter_strategy(len(ids), n, PARAMS['ef_search'])}")


if __name__ == "__main__":
    main(*(int(arg) for arg in sys.argv[1:2]))
