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
  cluster.py          shards, per-shard worker processes, scatter-gather coordinator
  server.py           one shard served over gRPC
  storage.py          snapshots + write-ahead log, so a shard survives restarts
  client.py           gRPC coordinator (replication, failover) + local cluster launcher
  protos/shard.proto  the gRPC contract (Add, Search)
scripts/            gen_protos.sh regenerates the gRPC code from the .proto
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
python -m benchmarks.bench_parallel      # sequential vs threads vs processes fan-out
python -m benchmarks.bench_grpc          # shards over local pipes vs over gRPC
python -m benchmarks.bench_replication   # 1 vs 2 replicas, and a server crash mid-run
python -m benchmarks.bench_persistence   # fsync cost, and restart from log vs snapshot
```

### Running shards as network services

Start one server per shard (each in its own terminal, or on its own machine):

```bash
python -m vsearch.server --port 50051 --seed 0
python -m vsearch.server --port 50052 --seed 1
python -m vsearch.server --port 50053 --seed 2
python -m vsearch.server --port 50054 --seed 3
```

Then point a coordinator at them:

```python
from vsearch.client import GrpcCoordinator
from vsearch.dataset import make_dataset

vectors, queries = make_dataset()
with GrpcCoordinator([f"127.0.0.1:{port}" for port in range(50051, 50055)]) as coord:
    coord.add(vectors)
    print(coord.search(queries[0], k=10))
```

Servers listen on `127.0.0.1` by default. The service has no authentication, so only pass
`--host 0.0.0.0` on a trusted network.

By default a shard lives in memory only, so a restarted server starts empty. Give each server
its own `--data-dir` and it keeps a snapshot plus a write-ahead log there, and recovers both
on startup:

```bash
python -m vsearch.server --port 50051 --seed 0 --data-dir data/shard-0
# recovered 2500 vectors from data/shard-0 (snapshot at seq 1, replayed 0 log records)
```

`--fsync always` (the default) forces every log write to disk before the server replies;
`--fsync off` is faster but can lose the last writes if the machine loses power.
`--snapshot-every N` takes a snapshot after every N written vectors.

For replicas, start more than one server per shard with that shard's seed (for example a second
set on ports 50061–50064 with `--seed 0` to `--seed 3`) and pass one list of addresses per shard.
Writes go to every copy, and a search keeps working if a copy dies:

```python
shards = [[f"127.0.0.1:{50051 + i}", f"127.0.0.1:{50061 + i}"] for i in range(4)]
with GrpcCoordinator(shards) as coord:
    coord.add(vectors)
    ids = coord.search(queries[0])                   # raises ShardUnavailableError if a shard has no live copy
    ids, missing = coord.search_partial(queries[0])  # or: results from the live shards + the missing ones
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

Parallel fan-out (4 shards, same data and parameters):

| Fan-out                        | recall@10 | p50 latency | build time |
|--------------------------------|-----------|-------------|------------|
| Sequential                     | 0.876     | ~5.6 ms     | ~18 s      |
| Threads                        | 0.876     | ~30.5 ms    | ~18 s      |
| Processes (one worker/shard)   | 0.876     | ~1.6 ms     | ~4.9 s     |

Each worker process owns its shard's index, so only the query and its k results cross process
boundaries. Threads are *slower* than sequential: the search makes thousands of small numpy calls
that release and reacquire the GIL, and with several threads waiting every release becomes a
context switch (~12,800 per query with 4 threads). Run it with `python -m benchmarks.bench_parallel`.

Shards as gRPC services (4 shard servers as separate processes, same data and parameters):

| Transport              | 10k vectors: p50 | tiny shards (200 vectors): p50 |
|------------------------|------------------|--------------------------------|
| In-process, sequential | ~5.3 ms          | ~0.60 ms                       |
| Processes over pipes   | ~1.5 ms          | ~0.23 ms                       |
| gRPC servers           | ~1.9 ms          | ~0.60 ms                       |

Recall is identical across transports (0.876 on 10k). gRPC adds roughly 0.35 ms per query over
local pipes (protobuf encoding, HTTP/2, and the Python gRPC stack). On 10k vectors that is small
next to the search itself; on tiny shards it cancels the whole benefit of searching in parallel.
Run it with `python -m benchmarks.bench_grpc`.

Replication (4 shards over gRPC; laptop numbers vary from run to run, so these are ranges over
three runs):

| Replicas | Servers | recall@10 | p50 latency | build time |
|----------|---------|-----------|-------------|------------|
| 1        | 4       | 0.876     | ~1.9–3.5 ms | ~5.8 s     |
| 2        | 8       | 0.876     | ~2.1–2.7 ms | ~8–10 s    |

A search still reads one copy per shard, so a second replica adds no measurable latency. Writes go
to both copies, so building takes about 1.5x longer with twice the servers.

Crash test: with 2 replicas, one server is killed (SIGKILL) halfway through 100 queries. All 100
queries were answered with unchanged recall (0.876). One query failed over to the other copy
(~3.4 ms instead of ~2 ms); after that the dead server was tried last, so later queries went
straight to its replica. Run it with `python -m benchmarks.bench_replication`.

Persistence (one shard of 2,500 vectors, dim 128, on a MacBook SSD):

| Log append, no index work      | per append |
|--------------------------------|------------|
| No fsync                       | ~1.9 µs    |
| `fsync`                        | ~22 µs     |
| `F_FULLFSYNC` (macOS)          | ~3.4 ms    |

| Restart of the shard           | time       |
|--------------------------------|------------|
| Replay the whole log (1.3 MB)  | ~4.4 s     |
| Load a snapshot (1.9 MB)       | ~3.5 ms    |

Writing each vector through the log barely changes write throughput (~850 writes/s with or
without it), because the HNSW insert (~1.1 ms) dwarfs a 22 µs fsync. Replaying the log means
re-inserting every vector, while a snapshot loads the finished graph, which is why periodic
snapshots matter. On macOS, plain `fsync` does not flush the drive's own cache; only
`F_FULLFSYNC` does, at about 150x the cost. Run it with `python -m benchmarks.bench_persistence`.

## Roadmap

- [x] Exact brute-force cosine baseline + benchmark harness
- [x] HNSW index (graph-based ANN), single node
- [x] Sharded search: coordinator with scatter-gather and top-K merge (in-process)
- [x] Parallel fan-out: one long-lived worker process per shard
- [x] Shards as network services (gRPC) behind the coordinator
- [x] Replication: write to every replica, read from one, fail over when a server dies
- [x] Durability: snapshots + write-ahead log; a restarted server recovers its data
- [ ] Replica resync: copy data from a healthy twin when a replica missed writes
- [ ] Benchmarks across growing dataset sizes
- [ ] Docker / Kubernetes deployment

## Design notes

Key design decisions and trade-offs for each stage are documented in
[`DECISIONS.md`](DECISIONS.md).
