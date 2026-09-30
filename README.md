# Distributed Vector Search Engine

[![CI](https://github.com/igennova/Distributed-Vector-Search-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/igennova/Distributed-Vector-Search-Engine/actions/workflows/ci.yml)

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
  server.py           one shard served over gRPC, plus replica catch-up
  storage.py          snapshots + write-ahead log, so a shard survives restarts
  client.py           gRPC coordinator (replication, failover, repair) + local cluster launcher
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
python -m benchmarks.bench_resync        # catching up a replica: from the log vs a snapshot
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

If a write reaches some replicas of a shard but not others, or a replica loses its disk, the
coordinator refuses new writes to that shard until the copies agree again. `repair()` brings
every replica level with the most up-to-date copy of its shard (this needs `--data-dir`):

```python
synced = coord.repair()   # one ReplicaSync per replica that caught up, via "log" or "snapshot"
```

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

Measured on a MacBook with a synthetic dataset (10,000 vectors, dim 128, k=10). Laptop timings
vary from run to run, so they are rounded or given as ranges over repeated runs.

| Method               | recall@10 | p50 latency | QPS    |
|----------------------|-----------|-------------|--------|
| Brute force (exact)  | 1.000     | ~13 ms      | ~70    |
| HNSW, ef_search=50   | 0.564     | ~0.4 ms     | ~2,300 |
| HNSW, ef_search=200  | 0.930     | ~1.2 ms     | ~800   |
| HNSW, ef_search=400  | 0.987     | ~2.0 ms     | ~450   |

`ef_search` trades recall for latency at query time with no rebuild. At 0.93 recall HNSW is about
10x faster than exact search on 10k vectors, and the gap grows with the dataset because exact
search is linear. (The dataset is random Gaussian, a worst case for ANN; real clustered
embeddings reach high recall at lower `ef`.) Run it with `python -m benchmarks.bench_hnsw`.

### Making the index fast

Profiling the build showed 84% of the time inside the distance function: 7 million calls to
`np.linalg.norm` (57% of the total) and 3.5 million single-pair distance calls, each paying
Python and numpy call overhead. Vectors are now normalized once when inserted, so cosine distance
is `1 - dot`; they live in one contiguous float32 matrix, and the distances to a whole neighbor
list come from a single matrix-vector product.

| 10k vectors, M=16, ef=200 | build     | search p50  | recall@10 |
|---------------------------|-----------|-------------|-----------|
| Before                    | ~42 s     | ~5.2 ms     | 0.930     |
| After                     | ~9–14 s   | ~1.0–1.3 ms | 0.930     |

Same algorithm, same recall. What's left is the Python beam-search loop itself (heap operations
and bookkeeping); going much further would take compiled code.

### Sharding and fan-out

Sharded search (same data, ef_search=50, shards queried sequentially):

| Shards | recall@10 | p50 latency | build time |
|--------|-----------|-------------|------------|
| 1      | 0.554     | ~0.3 ms     | ~5.4 s     |
| 2      | 0.701     | ~0.6 ms     | ~4.9 s     |
| 4      | 0.876     | ~1.1 ms     | ~4.2 s     |

More shards raise recall and cut build time, but add total query work; latency only drops once
shards are queried in parallel. Run it with `python -m benchmarks.bench_cluster`.

Parallel fan-out (4 shards, same data and parameters):

| Fan-out                        | recall@10 | p50 latency  | build time |
|--------------------------------|-----------|--------------|------------|
| Sequential                     | 0.876     | ~1.2–1.3 ms  | ~4 s       |
| Threads                        | 0.876     | ~1.8 ms      | ~4 s       |
| Processes (one worker/shard)   | 0.876     | ~0.4–0.55 ms | ~1.4 s     |

Each worker process owns its shard's index, so only the query and its k results cross process
boundaries. Threads are slower than sequential because search is CPU-bound Python and the GIL
lets one thread run at a time. Before distances were batched, the effect was dramatic: every
search made thousands of tiny numpy calls that release and reacquire the GIL, 4 threads caused
~12,800 context switches per query, and threads were 5x slower than sequential. With batched
distances that fell to ~270 context switches per query, and threads are ~1.5x slower: still no
gain. On tiny shards (200 vectors) processes and sequential now come out about even, since there
is little work per shard left to spread out. Run it with `python -m benchmarks.bench_parallel`.

### Over the network

Shards as gRPC services (4 shard servers as separate processes, same data and parameters):

| Transport              | 10k vectors: p50 | tiny shards (200 vectors): p50 |
|------------------------|------------------|--------------------------------|
| In-process, sequential | ~1.0 ms          | ~0.19 ms                       |
| Processes over pipes   | ~0.34 ms         | ~0.12 ms                       |
| gRPC servers           | ~0.66 ms         | ~0.54 ms                       |

Recall is identical across transports (0.876 on 10k). gRPC adds roughly 0.3 ms per query over
local pipes (protobuf encoding, HTTP/2, and the Python gRPC stack). Now that a shard search takes
a few tenths of a millisecond, that hop is about half of each query's latency, and on tiny shards
it makes gRPC about 3x slower than searching in-process. Run it with
`python -m benchmarks.bench_grpc`.

Replication (4 shards over gRPC):

| Replicas | Servers | recall@10 | p50 latency | build time |
|----------|---------|-----------|-------------|------------|
| 1        | 4       | 0.876     | ~0.7 ms     | ~1.1 s     |
| 2        | 8       | 0.876     | ~0.7 ms     | ~1.7 s     |

A search still reads one copy per shard, so a second replica adds no measurable latency. Writes go
to both copies, so building takes about 1.5x longer with twice the servers.

Crash test: with 2 replicas, one server is killed (SIGKILL) halfway through 100 queries. All 100
queries were answered with unchanged recall (0.876), with one failover to the other copy; the
slowest query after the crash took ~1.3 ms. After that first failure the dead server was tried
last, so later queries went straight to its replica. Run it with
`python -m benchmarks.bench_replication`.

### Durability and resync

Persistence (one shard of 2,500 vectors, dim 128, on a MacBook SSD):

| Log append, no index work      | per append   |
|--------------------------------|--------------|
| No fsync                       | ~2 µs        |
| `fsync`                        | ~22–25 µs    |
| `F_FULLFSYNC` (macOS)          | ~3.4–3.6 ms  |

| Restart of the shard           | time       |
|--------------------------------|------------|
| Replay the whole log (1.3 MB)  | ~1.1 s     |
| Load a snapshot (1.9 MB)       | ~4 ms      |

With inserts now around 0.3 ms, an fsync on every write shows up in write throughput: ~3,100
writes/s in memory against ~2,600/s with fsync, roughly 20% fewer. (Before the index got faster,
a ~1.1 ms insert hid the fsync completely.) Replaying the log means re-inserting every vector,
while a snapshot loads the finished graph, which is why periodic snapshots matter. On macOS, plain
`fsync` does not flush the drive's own cache; only `F_FULLFSYNC` does, at about 150x the cost.
Run it with `python -m benchmarks.bench_persistence`.

Replica resync (a replica of a 2,500-vector shard catching up from its twin over gRPC, localhost):

| Replica missed          | Method   | Time        |
|-------------------------|----------|-------------|
| 50 vectors              | log      | ~0.04 s     |
| 500 vectors             | log      | ~0.3–0.6 s  |
| 2,000 vectors           | log      | ~1.0 s      |
| everything (empty disk) | snapshot | ~0.01 s     |

Catching up from the log means re-inserting every missed vector into the HNSW graph (~0.5 ms
each), so its cost grows with the gap. Copying a snapshot ships the finished graph (1.9 MB
here), so at this size it still wins even for a 50-vector gap. The balance shifts as shards
grow: a snapshot of a large shard is gigabytes to send, while a small gap stays cheap from the
log. Run it with `python -m benchmarks.bench_resync`.

## Roadmap

- [x] Exact brute-force cosine baseline + benchmark harness
- [x] HNSW index (graph-based ANN), single node
- [x] Sharded search: coordinator with scatter-gather and top-K merge (in-process)
- [x] Parallel fan-out: one long-lived worker process per shard
- [x] Shards as network services (gRPC) behind the coordinator
- [x] Replication: write to every replica, read from one, fail over when a server dies
- [x] Durability: snapshots + write-ahead log; a restarted server recovers its data
- [x] Replica resync: a replica that missed writes or lost its disk catches up from its twin
- [ ] Benchmarks across growing dataset sizes
- [ ] Docker / Kubernetes deployment

## Design notes

Key design decisions and trade-offs for each stage are documented in
[`DECISIONS.md`](DECISIONS.md).
