# Distributed Vector Search Engine

A distributed approximate-nearest-neighbor (ANN) search engine built from scratch in Python —
covering indexing, sharding, query routing, replication, and distributed query execution. The
focus is on understanding the internals of systems like Qdrant, Milvus, and Pinecone rather than
feature parity.

## Overview

Vector search powers semantic search, retrieval-augmented generation, and recommendation systems:
data is embedded into high-dimensional vectors, and a query returns the vectors closest to it by
cosine similarity. Doing this exactly is linear in the dataset size; doing it fast at scale
requires an ANN index and, beyond one machine, sharding and a coordinator that fans queries out
and merges the results.

<img width="665" height="500" alt="Distributed Vector Search Architecture" src="https://github.com/user-attachments/assets/7a9013d4-02c2-4fb0-a609-221fb36434ce" />


## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```
vsearch/            the engine
  dataset.py          synthetic data + exact ground truth
  brute_force.py      exact search (baseline)
  hnsw.py             HNSW index
  cluster.py          shards + scatter-gather coordinator
tests/              pytest suite
benchmarks/         recall / latency / QPS benchmarks
DECISIONS.md        design decisions and trade-offs
```

## Usage

```bash
pytest                                   # run all tests
python -m benchmarks.bench_brute_force   # exact baseline: recall, p50/p99, QPS
python -m benchmarks.bench_hnsw          # HNSW vs brute force across ef_search
python -m benchmarks.bench_cluster       # 1 vs 2 vs 4 shards
```

## Benchmarks

Measured on a synthetic dataset (10,000 vectors, dim 128, k=10):

| Method               | recall@10 | p50 latency | QPS  |
|----------------------|-----------|-------------|------|
| Brute force (exact)  | 1.000     | ~12.9 ms    | ~77  |
| HNSW, ef_search=50   | 0.564     | ~1.8 ms     | ~541 |
| HNSW, ef_search=200  | 0.930     | ~5.2 ms     | ~192 |
| HNSW, ef_search=400  | 0.987     | ~8.3 ms     | ~120 |

`ef_search` trades recall for latency at query time with no rebuild. Exact search scales linearly
with dataset size, which is the motivation for the ANN index. (The dataset is random Gaussian, a
worst case for ANN; real clustered embeddings reach high recall at lower `ef`.) Run it with
`python -m benchmarks.bench_hnsw`.

Sharded search (same data, ef_search=50, shards queried sequentially):

| Shards | recall@10 | p50 latency | build time |
|--------|-----------|-------------|------------|
| 1      | 0.554     | ~1.8 ms     | ~29 s      |
| 2      | 0.701     | ~3.7 ms     | ~26 s      |
| 4      | 0.876     | ~5.5 ms     | ~20 s      |

More shards raise recall and cut build time, but add total query work; latency only drops once
shards are queried in parallel. Run it with `python -m benchmarks.bench_cluster`.

## Roadmap

- [x] Exact brute-force cosine baseline + benchmark harness
- [x] HNSW index (graph-based ANN), single node
- [x] Sharded search: coordinator with scatter-gather and top-K merge (in-process)
- [ ] Parallel fan-out across shards
- [ ] Index persistence (serialization, mmap)
- [ ] gRPC service around a single shard
- [ ] Shards as separate processes behind the coordinator
- [ ] Replication
- [ ] Node-failure handling
- [ ] Write-ahead log / snapshots for durability
- [ ] Benchmarks across growing dataset sizes
- [ ] Docker / Kubernetes deployment

## Design notes

Key design decisions and trade-offs for each stage are documented in
[`DECISIONS.md`](DECISIONS.md).
