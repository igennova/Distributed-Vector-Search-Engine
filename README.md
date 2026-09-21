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

```
          Client
            │
     Query Coordinator
            │  (fan-out)
   ┌────────┼────────┐
   ▼        ▼        ▼
 Shard 1  Shard 2  Shard 3
 (ANN)    (ANN)    (ANN)
   └────────┼────────┘
            ▼
       Top-K Merge
            │
         Results
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python test_search.py     # correctness tests
python benchmark.py       # recall@k, latency (p50/p99), and QPS vs. an exact oracle
```

## Benchmarks

Measured on a synthetic dataset (10,000 vectors, dim 128, k=10):

| Index        | recall@10 | p50 latency | QPS |
|--------------|-----------|-------------|-----|
| Brute force  | 1.000     | ~12 ms      | ~80 |

Exact brute-force search scales linearly with dataset size, which is the motivation for the ANN
index. Results across larger datasets are added as each stage lands.

## Roadmap

- [x] Exact brute-force cosine baseline + benchmark harness
- [ ] HNSW index (graph-based ANN), single node
- [ ] Index persistence (serialization, mmap)
- [ ] gRPC service around a single shard
- [ ] Sharding: N shard processes + a query coordinator (scatter-gather)
- [ ] Parallel fan-out + top-K merge across shards
- [ ] Replication
- [ ] Node-failure handling
- [ ] Write-ahead log / snapshots for durability
- [ ] Benchmarks across growing dataset sizes
- [ ] Docker / Kubernetes deployment

## Design notes

Key design decisions and trade-offs for each stage are documented in
[`DECISIONS.md`](DECISIONS.md).
