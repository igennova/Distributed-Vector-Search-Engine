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
pip install -e .          # adds the `vsearch` command
pip install fastembed     # optional: the embedding model used to search documents
```

## Quick start

Start a local cluster, index real word vectors, and search them. This needs Stanford's GloVe 6B
vectors: download `glove.6B.zip` (862 MB) from https://nlp.stanford.edu/projects/glove/ into
`data/`. The first load caches the 100-dimension vectors as a `.npy` file, after which the zip
can be deleted.

```console
$ vsearch cluster up --shards 4 --replicas 2      # 8 shard servers, running in the background
$ vsearch load glove --limit 100000
loaded 100,000 words in 20.2s
$ vsearch similar king
words closest to 'king':
  1. prince               0.768
  2. queen                0.751
  3. son                  0.702
  4. brother              0.699
  5. monarch              0.698
(8.0 ms across 4 shards)
```

Crash a server and keep searching; restart it and it recovers from its own disk:

```console
$ vsearch cluster kill 2 0                        # SIGKILL shard 2's replica 0
$ vsearch similar paris
words closest to 'paris':
  1. france               0.748
  2. london               0.734
  ...
(16.5 ms across 4 shards, 1 failover(s))
$ vsearch status
shard replica  address          state   vectors  last seq  snapshot seq
    2       0  127.0.0.1:50518  down
    2       1  127.0.0.1:50519  up       25,000        10            10
  ...
$ vsearch cluster restart 2 0
$ vsearch repair
all reachable replicas are in sync
$ vsearch cluster down                            # data stays on disk; `cluster up` brings it back
```

The cluster's state, data, and logs live in `data/cluster` (change it with `--home`).
`vsearch cluster down --wipe` deletes them.

### Search your own documents

`vsearch ingest` splits documents into passages, embeds each one, and stores the vector with the
passage as its payload. `vsearch search` embeds the question the same way and returns the
closest passages. Here the engine indexes its own documentation:

```console
$ vsearch --home data/docs cluster up --shards 2 --replicas 2
$ vsearch --home data/docs ingest README.md DECISIONS.md
  DECISIONS.md: 34 passages
  README.md: 28 passages
ingested 62 passages in 3.1s
$ vsearch --home data/docs search "what happens if the computer loses power"
passages closest to 'what happens if the computer loses power':
  1. 0.633  DECISIONS.md > Durability: snapshots + write-ahead log
     F_FULLFSYNC does, and costs ~3.4 ms per append, about 150x more. So `--fsync always` here
     survives a process or OS crash, but not necessarily a power cut. ...
(embed 380.4 ms, search 3.1 ms across 2 shards)
```

The embedding model places texts with similar meaning close together, so a question does not
have to use the document's own words. The model is [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
(384 dimensions), run locally with ONNX Runtime: no API key, and nothing leaves the machine. It
is downloaded once (about 67 MB) to `~/.cache/vsearch/models`. Without fastembed installed,
`--embedder hash` uses a built-in embedder that matches on shared words instead.

Ingesting the same files again adds nothing: each file's content hash is recorded, and unchanged
files are skipped.

A search can be limited to passages whose fields match, and documents can be kept apart in
namespaces (one per user or project). A namespace is stored on a single shard, so a search
inside it asks that shard only (shown here on a cluster started with `--shards 4`):

```console
$ vsearch --home data/docs search "how are replicas repaired" --where source=DECISIONS.md
passages closest to 'how are replicas repaired':
  1. 0.747  DECISIONS.md > Replica resync
     - `repair()` finds the replica with the highest seq for each shard and tells each replica that
     is behind to `SyncFrom` it. ...
(embed 382.8 ms, search 5.6 ms across 4 shards)
$ vsearch --home data/docs ingest DECISIONS.md --namespace alice
ingested 47 passages into namespace 'alice' (shard 3 of 4) in 1.8s
$ vsearch --home data/docs search "why is exact search sometimes faster than the graph" --namespace alice
passages closest to 'why is exact search sometimes faster than the graph':
  1. 0.799  DECISIONS.md > Metadata filtering
     - Surprise in the numbers: exact search wins far more often than expected, for any filter under
     about a third of a 100k shard. ...
(embed 386.6 ms, search 3.6 ms on shard 3 of 4)
```

### Use it from an AI assistant (MCP)

`vsearch mcp` serves the cluster over the [Model Context Protocol](https://modelcontextprotocol.io),
so an assistant or coding agent that supports MCP can search the indexed documents by itself.
It offers three tools:

| Tool | What it does |
|------|--------------|
| `search` | The passages most relevant to a question, each with its file and section; optionally inside one namespace or one file |
| `ingest` | Add files or directories to the index (left out with `--read-only`) |
| `status` | Which servers are up, how many vectors each holds, and what is indexed |

Start the cluster as above, then register the command in the client's MCP configuration. Most
clients use this form; the paths must be absolute, because the client starts the command from
its own working directory:

```json
{
  "mcpServers": {
    "vsearch": {
      "command": "/path/to/vector-search-engine/.venv/bin/vsearch",
      "args": ["--home", "/path/to/vector-search-engine/data/docs", "mcp"]
    }
  }
}
```

The client and the server exchange JSON-RPC messages over stdin and stdout, one per line, so a
session can be tried by hand:

```console
$ echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"status","arguments":{}}}' \
    | vsearch --home data/docs mcp
{"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "2 of 2 servers up
(2 shards).\nIngested: 85 passages from 2 file(s), embedder: hash.\nFiles: DECISIONS.md,
README.md\nshard 0: replica 0 up (43 vectors, seq 2)\nshard 1: replica 0 up (42 vectors, seq
2)"}], "isError": false}}
```

(The reply is one line; it is wrapped here.) The server lives as long as the client's session,
so the embedding model is loaded once: the first search of a session took 407 ms and later ones
a median of 4.5 ms, where every `vsearch search` pays about 390 ms.

### Give an assistant a long-term memory

A model forgets everything when a conversation ends. `vsearch memory` keeps facts in a cluster
of their own and recalls them by meaning, so "how should I explain things to him?" finds
"likes his explanations in simple language with analogies":

```console
$ vsearch --home data/memory cluster up --shards 2 --replicas 2
$ vsearch --home data/memory memory remember "Rahul likes his explanations in simple language with analogies." --owner rahul
remembered as m6d6a64525c
$ vsearch --home data/memory memory recall "how should I explain things to him?" --owner rahul
  m6d6a64525c  0.574  just now        Rahul likes his explanations in simple language with analogies.
  ...
```

Each owner's memories are a namespace, so they sit on one shard and stay separate from
everyone else's. When a fact changes, `remember ... --replaces ID` stores the new one and the
old one is no longer recalled; `forget ID` stops recalling a fact. Saving a fact that is
already stored returns the existing id. Among facts that match about equally well, newer ones
rank first. Over MCP, `vsearch --home data/memory mcp --memory --owner rahul` gives an assistant
`remember`, `recall`, and `forget` as tools.

The engine cannot delete yet, so a forgotten or replaced fact is hidden, not erased: its text
stays on disk in the write-ahead log and the snapshots.

### Let an agent keep the cluster healthy

`vsearch ops check` looks at the cluster, repairs what is wrong, and then checks for itself
whether that worked. What decides the steps is the `--brain`:

- `runbook` (the default) is a fixed procedure: restart whatever is down, repair, test a search.
- `openai` or `claude` hands the decisions to a model. It gets five tools and chooses each step
  from what the last one returned.

| Tool | What it does |
|------|--------------|
| `cluster_status` | Every server (up or down, vectors, last write) and a list of what is wrong |
| `restart_server` | Start a server that is down; it recovers its data from its own disk |
| `repair` | Bring replicas that missed writes up to date from their twin |
| `read_server_log` | The last lines of a server's log, to find out why it will not start |
| `probe_search` | One real search across the cluster, to confirm every shard answers |

A model brain reads its API key from the environment or from a `.env` file in the current
directory (`OPENAI_API_KEY=...`; the file is git-ignored). Here a replica has been killed:

```console
$ vsearch cluster kill 1 0
$ vsearch ops check --brain openai
found 1 problem(s):
  - shard 1 replica 0 is down
  -> cluster_status()  [ok]
       shard 0: replica 0 up (46 vectors, seq 2), replica 1 up (46 vectors, seq 2)
       shard 1: replica 0 DOWN (127.0.0.1:64155), replica 1 up (46 vectors, seq 2)
       Problems:
       - shard 1 replica 0 is down
  -> restart_server(shard=1, replica=0)  [ok]
       shard 1 replica 0 restarted and answering: it recovered 46 vectors from its disk and is at seq 2.
  -> cluster_status()  [ok]
       ...
       No problems found: every server answers and replicas agree.
  -> probe_search()  [ok]
       A search across all 2 shards answered in 3 ms.
  [gpt-5.4-mini: 5 turns, 4 tool calls, 4,137 input + 151 output tokens]

report:
  The issue was that shard 1 replica 0 was down.
  ...
checked: the cluster is healthy
```

The last line is not the model's opinion: after the model stops, the command inspects the
cluster again and sets its exit code from what it finds. When a replica's snapshot file was
corrupted so that it could not start, the model tried one restart, read the server's log,
named the cause, and stopped instead of retrying:

```
report:
  Shard 1 replica 1 is down because it fails to start: its stored snapshot is unreadable ...
  This cannot be fixed with the available tools ... A person will need to inspect or replace
  the corrupted/incompatible snapshot for shard 1 replica 1 ...

checked: still wrong:
  - shard 1 replica 1 is down
```

The agent is limited by what its tools can do, not by what it is asked to do: no tool deletes
data or stops a server, a server is restarted at most twice per run, the model gets at most 12
turns, and `--dry-run` describes changes instead of making them. Every model turn and tool call
is appended to `ops-journal.jsonl` in the cluster's folder. `vsearch ops watch --interval 30`
repeats the check, and does not try again on a problem it already failed to fix. With a model
brain, the cluster's status and log excerpts are sent to that model's provider.

## Project layout

```
vsearch/            the engine
  dataset.py          synthetic data, GloVe loader, exact ground truth
  brute_force.py      exact search (baseline)
  hnsw.py             HNSW index
  cluster.py          shards, per-shard worker processes, scatter-gather coordinator
  server.py           one shard served over gRPC, plus replica catch-up
  storage.py          snapshots + write-ahead log, so a shard survives restarts
  client.py           gRPC coordinator (replication, failover, repair) + local cluster launcher
  filtering.py        filters and the per-shard payload index that answers them
  chunking.py         split a document into passages along headings and sentences
  embedding.py        text -> vector: a local embedding model, or word hashing
  ingest.py           files -> passages -> vectors with payloads
  state.py            a local cluster's recorded state (cluster.json) and server status
  documents.py        a document collection on a cluster: ingest files, search passages
  mcp.py              Model Context Protocol server over stdio (JSON-RPC), tool-agnostic
  mcp_server.py       the engine's MCP tools: search, ingest, status
  processes.py        start, identify, and stop the shard server processes
  health.py           one definition of a healthy cluster: facts per server, and problems
  ops.py              the operator's tools, their journal, and the fixed runbook
  memory.py           an agent's long-term memory: remember, recall, forget
  agent.py            the agent loop, and the models that can drive it
  cli.py              the `vsearch` command
evals/              ops_eval.py (fault scenarios, grading, results), retrieval_eval.py
  protos/shard.proto  the gRPC contract (Add, Search, Status, and the resync calls)
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
python -m benchmarks.bench_glove         # real word vectors (needs the GloVe download, below)
```

The GloVe benchmark uses the same download as the quick start.

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

### Payloads

A vector can carry a payload: any JSON-serializable dict, such as the text it was made from and
where that text came from. Payloads are stored with the vector (in the write-ahead log and the
snapshots, so they survive restarts and replica resync) and come back with search results:

```python
coord.add(vectors, payloads=[{"text": "Writes go to every replica...", "source": "DECISIONS.md"},
                             None,          # a vector without a payload
                             ...])
for hit in coord.search_hits(query, k=5):
    print(hit.id, hit.distance, hit.payload)
```

### Filters and namespaces

A search can be restricted to the vectors whose payload matches a filter. Every field has to
match; a list accepts any of its values:

```python
coord.search_hits(query, k=5, where={"source": "DECISIONS.md"})
coord.search_hits(query, k=5, where={"lang": ["en", "de"], "year": 2024})
```

Each shard answers a filtered search in whichever of two ways is cheaper: if few vectors
match, it compares the query with every one of them (exact); if many match, it walks the HNSW
graph and keeps only matching nodes. Filtering the unfiltered top-k afterwards is not used: it
returns fewer than k results, or none.

A namespace is a separate set of vectors inside the same cluster, for one user, project, or
agent:

```python
coord.add(vectors, payloads, namespace="alice")
coord.search_hits(query, k=5, namespace="alice")     # only alice's vectors, from one shard
```

All of a namespace's vectors go to the one shard its name hashes to, so searching it costs one
request instead of one per shard, and it keeps working when other shards are down. A search
without a namespace still covers every shard.

## Benchmarks

Measured on a MacBook. Laptop timings vary from run to run, so they are rounded.

### Real data: GloVe word vectors

Stanford GloVe 6B (400,000 words, 100 dims), indexed most-frequent-first and queried with 1,000
words held out of the index. The exact baseline is one numpy matrix-vector product per query.

| Vectors | Method                  | recall@10 | p50 latency | QPS     |
|---------|-------------------------|-----------|-------------|---------|
| 10,000  | exact (numpy)           | 1.000     | ~0.09 ms    | ~12,000 |
| 10,000  | HNSW, ef_search=100     | 0.883     | ~0.34 ms    | ~2,900  |
| 100,000 | exact (numpy)           | 1.000     | ~1.05 ms    | ~900    |
| 100,000 | HNSW, ef_search=100     | 0.868     | ~0.53 ms    | ~2,000  |
| 100,000 | 4 shards, ef_search=100 | 0.964     | ~0.70 ms    | ~1,200  |
| 400,000 | exact (numpy)           | 1.000     | ~3.96 ms    | ~250    |
| 400,000 | HNSW, ef_search=50      | 0.848     | ~0.35 ms    | ~2,700  |
| 400,000 | HNSW, ef_search=100     | 0.912     | ~0.65 ms    | ~1,500  |
| 400,000 | HNSW, ef_search=200     | 0.949     | ~1.14 ms    | ~890    |
| 400,000 | 4 shards, ef_search=100 | 0.960     | ~0.73 ms    | ~1,260  |

- Exact search is linear in the data: 0.09 → 1.05 → 3.96 ms as the index grows 40x. HNSW at
  `ef_search=100` goes 0.34 → 0.53 → 0.65 ms, less than 2x.
- At 10k vectors exact search wins outright: one BLAS call is cheaper than walking a graph in
  Python. The crossover is between 10k and 100k; at 400k HNSW is 3.5–11x faster at 0.95–0.85
  recall.
- This index is pure Python and the baseline's inner loop is compiled BLAS. A compiled HNSW such
  as hnswlib is far faster than this one; what carries over is how the two scale.
- Recall is not comparable across sizes: the index holds the n most frequent words, so the mix
  of words changes with n.
- Building 400k vectors takes ~264 s as one index and ~75 s as 4 shard processes. The index uses
  about 88 MB per 100k vectors (351 MB at 400k).

Run it with `python -m benchmarks.bench_glove` (10k and 100k) or
`python -m benchmarks.bench_glove 400000`.

Choosing each node's links (100k GloVe vectors, same parameters):

| Neighbor selection                  | recall@10 at ef 50 / 100 / 200 | build | links per node |
|-------------------------------------|--------------------------------|-------|----------------|
| M closest candidates                | 0.727 / 0.813 / 0.879          | ~56 s | 25.8           |
| Diversity heuristic + fill          | 0.797 / 0.870 / 0.921          | ~69 s | 27.2           |
| Diversity heuristic (the default)   | 0.791 / 0.868 / 0.919          | ~57 s | 21.6           |

Real data is clumpy, so a node's M closest candidates tend to sit in one cluster and its links
all point the same way. The heuristic from the HNSW paper keeps a candidate only if it is closer
to the node than to any neighbor already kept, which spends links on different directions: 4–6
points more recall for the same build and search cost, with fewer links. Filling the unused
slots with rejected candidates cost 20% more build time for almost no recall, so it is off. On
the random synthetic data below, which has no clusters, the heuristic changes recall by about a
point.

### Synthetic data

10,000 random Gaussian vectors, dim 128, k=10. Random vectors are a worst case for ANN.

| Method                    | recall@10 | p50 latency | QPS    |
|---------------------------|-----------|-------------|--------|
| Brute force (Python loop) | 1.000     | ~13 ms      | ~75    |
| HNSW, ef_search=50        | 0.555     | ~0.3 ms     | ~3,000 |
| HNSW, ef_search=200       | 0.926     | ~1.0 ms     | ~1,000 |
| HNSW, ef_search=400       | 0.980     | ~1.8 ms     | ~550   |

`ef_search` trades recall for latency at query time with no rebuild. The brute-force row is a
plain Python loop over every vector, which is a weak baseline; see the GloVe table above for a
comparison against vectorized exact search. Run it with `python -m benchmarks.bench_hnsw`.

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

Same algorithm, same recall (measured before the neighbor-selection change above). What's left
is the Python beam-search loop itself (heap operations and bookkeeping); going much further would
take compiled code.

### Sharding and fan-out

Sharded search (synthetic data, ef_search=50, shards queried sequentially):

| Shards | recall@10 | p50 latency | build time |
|--------|-----------|-------------|------------|
| 1      | 0.561     | ~0.3 ms     | ~5.8 s     |
| 2      | 0.713     | ~0.6 ms     | ~4.9 s     |
| 4      | 0.883     | ~1.0 ms     | ~4.0 s     |

More shards raise recall and cut build time, but add total query work; latency only drops once
shards are queried in parallel. Run it with `python -m benchmarks.bench_cluster`.

Parallel fan-out (4 shards, same data and parameters):

| Fan-out                        | recall@10 | p50 latency | build time |
|--------------------------------|-----------|-------------|------------|
| Sequential                     | 0.883     | ~1.0 ms     | ~4 s       |
| Threads                        | 0.883     | ~1.8 ms     | ~4 s       |
| Processes (one worker/shard)   | 0.883     | ~0.34 ms    | ~1.1 s     |

Each worker process owns its shard's index, so only the query and its k results cross process
boundaries. Threads are slower than sequential because search is CPU-bound Python and the GIL
lets one thread run at a time. Before distances were batched, the effect was dramatic: every
search made thousands of tiny numpy calls that release and reacquire the GIL, 4 threads caused
~12,800 context switches per query, and threads were 5x slower than sequential. With batched
distances that fell to ~270 context switches per query, and threads are ~1.5–1.7x slower: still
no gain. On tiny shards (200 vectors) the gain from processes shrinks, since there is little
work per shard left to spread out. Run it with `python -m benchmarks.bench_parallel`.

### Over the network

Shards as gRPC services (4 shard servers as separate processes, same data and parameters):

| Transport              | 10k vectors: p50 | tiny shards (200 vectors): p50 |
|------------------------|------------------|--------------------------------|
| In-process, sequential | ~1.0 ms          | ~0.20 ms                       |
| Processes over pipes   | ~0.33 ms         | ~0.11 ms                       |
| gRPC servers           | ~0.65 ms         | ~0.54 ms                       |

Recall is identical across transports (0.883 on 10k). gRPC adds roughly 0.3 ms per query over
local pipes (protobuf encoding, HTTP/2, and the Python gRPC stack). Now that a shard search takes
a few tenths of a millisecond, that hop is about half of each query's latency, and on tiny shards
it makes gRPC about 3x slower than searching in-process. Run it with
`python -m benchmarks.bench_grpc`.

Replication (4 shards over gRPC):

| Replicas | Servers | recall@10 | p50 latency | build time |
|----------|---------|-----------|-------------|------------|
| 1        | 4       | 0.883     | ~0.7 ms     | ~1.1 s     |
| 2        | 8       | 0.883     | ~0.7 ms     | ~1.6 s     |

A search still reads one copy per shard, so a second replica adds no measurable latency. Writes go
to both copies, so building takes about 1.5x longer with twice the servers.

Crash test: with 2 replicas, one server is killed (SIGKILL) halfway through 100 queries. All 100
queries were answered with unchanged recall (0.883), with one failover to the other copy; the
slowest query after the crash took ~1.2 ms. After that first failure the dead server was tried
last, so later queries went straight to its replica. Run it with
`python -m benchmarks.bench_replication`.

### Durability and resync

Persistence (one shard of 2,500 vectors, dim 128, on a MacBook SSD):

| Log append, no index work      | per append   |
|--------------------------------|--------------|
| No fsync                       | ~2 µs        |
| `fsync`                        | ~22 µs       |
| `F_FULLFSYNC` (macOS)          | ~3.4 ms      |

| Restart of the shard           | time       |
|--------------------------------|------------|
| Replay the whole log (1.3 MB)  | ~1.0 s     |
| Load a snapshot (1.9 MB)       | ~3 ms      |

An fsync per write (~22 µs) is small next to an insert (~0.33 ms), about 7% on paper. Over seven
interleaved runs, writing through the log with or without fsync could not be told apart from
in-memory writes (medians within a few percent, while each mode alone varied by more than that).
Replaying the log means re-inserting every vector, while a snapshot loads the finished graph,
which is why periodic snapshots matter. On macOS, plain `fsync` does not flush the drive's own
cache; only `F_FULLFSYNC` does, at about 150x the cost. Run it with
`python -m benchmarks.bench_persistence`.

Replica resync (a replica of a 2,500-vector shard catching up from its twin over gRPC, localhost):

| Replica missed          | Method   | Time      |
|-------------------------|----------|-----------|
| 50 vectors              | log      | ~0.03 s   |
| 500 vectors             | log      | ~0.3 s    |
| 2,000 vectors           | log      | ~0.9 s    |
| everything (empty disk) | snapshot | ~0.01 s   |

Catching up from the log means re-inserting every missed vector into the HNSW graph (~0.5 ms
each), so its cost grows with the gap. Copying a snapshot ships the finished graph (1.9 MB
here), so at this size it still wins even for a 50-vector gap. The balance shifts as shards
grow: a snapshot of a large shard is gigabytes to send, while a small gap stays cheap from the
log. Run it with `python -m benchmarks.bench_resync`.

### Filtered search and namespaces

Filtered search on GloVe (`ef_search=100`), each vector given a random group and the filter
selecting one share of the data. Recall is against the exact filtered answer. "Filter after"
searches the unfiltered top 10 and drops what does not match.

100,000 vectors:

| Matching | Filter after: recall (results) | Graph: recall | Graph | Exact | Engine picks |
|----------|--------------------------------|---------------|-------|-------|--------------|
| 50%  | 0.470 (4.8 of 10) | 0.906 | 1.15 ms  | 1.27 ms | graph |
| 20%  | 0.214 (2.2)       | 0.970 | 2.40 ms  | 0.53 ms | exact |
| 10%  | 0.109 (1.1)       | 0.983 | 4.61 ms  | 0.27 ms | exact |
| 1%   | 0.013 (0.1)       | 0.998 | 35.4 ms  | 0.03 ms | exact |
| 0.1% | 0.002 (0.0)       | 1.000 | 143 ms   | 0.01 ms | exact |

399,000 vectors:

| Matching | Graph: recall | Graph | Exact | Engine picks |
|----------|---------------|-------|-------|--------------|
| 50%  | 0.941 | 1.18 ms | 5.11 ms | graph |
| 20%  | 0.970 | 2.72 ms | 5.09 ms | graph |
| 10%  | 0.976 | 4.93 ms | 2.53 ms | exact |
| 1%   | 0.996 | 43.7 ms | 0.11 ms | exact |
| 0.1% | 1.000 | 311 ms  | 0.02 ms | exact |

- Filtering after the search returns about `10 x share` results: useless for selective filters.
- The graph walk slows down as the filter gets more selective (it has to pass about
  `ef / share` nodes), at the same rate whatever the size of the index. Exact search over the
  matching vectors does the opposite: it gets faster.
- So each shard estimates both costs and picks one. The crossover moves with size: about a
  third of the vectors at 100k, a sixth at 400k.

Namespaces (100,000 vectors owned by 200 tenants, 4 shard servers over gRPC, 400 searches each
inside one tenant). "Spread" stores tenants round-robin with a `tenant` payload field and
filters on it; "namespace" stores each tenant on the shard its name hashes to:

| Layout | Shards asked | Recall@10 | p50 | p99 | Tenants still searchable with 1 shard down |
|--------|--------------|-----------|-----|-----|---------------------------------------------|
| Spread    | 4 | 1.000 | 0.44 ms | 0.57 ms | 0%  |
| Namespace | 1 | 1.000 | 0.25 ms | 0.35 ms | 76% |

One request instead of four makes a tenant's search about 1.8x faster here, and a dead shard
takes down only the tenants stored on it. The shards stayed balanced (24,500 to 25,500 vectors
each) because the tenants are the same size; one very large tenant would make its shard large.

### Evaluations

`evals/` measures the two things a single demo cannot show: how reliably the operations agent
repairs a cluster, and how often search finds the passage that answers a question.

**Operations agent.** Each run starts a fresh 2-shard, 2-replica cluster, breaks it in one of
seven ways, lets a brain work, and grades the result in code: the cluster ends in the right
state, and the report's `Outcome:` line matches what happened. 10 runs per fault for each
prompt with `gpt-5.4-mini`, 3 for the runbook:

| Brain | Success | Claims correct | Read the right log (corrupt file) | Median turns | Tokens per run |
|-------|---------|----------------|-----------------------------------|--------------|----------------|
| Model, careful prompt | 70/70 | 70/70 | 10/10 | 5 | 5,120 |
| Model, lean prompt    | 70/70 | 70/70 | 10/10 | 5 | 4,721 |
| Runbook               | 21/21 | 21/21 | 0/3   | - | 0     |

The faults: a killed replica, one killed during a write (needs repair after the restart), one
that lost its disk, one down in each shard, a whole shard down, a corrupted snapshot (cannot
be fixed with the tools; the right answer is to stop and hand over), and a false alarm. With
no failures in 70 runs, the 95% interval for the model's success rate is 95–100%. All 140
model runs used 690,000 tokens, about $0.60, and every transcript is kept in `evals/results/`.

```bash
python -m evals.ops_eval --brain runbook --runs 3
python -m evals.ops_eval --brain openai --prompt lean --runs 10      # uses OPENAI_API_KEY
python -m evals.ops_eval --summary evals/results/*.jsonl
```

**Retrieval.** 40 questions about the design notes, each labelled with the section that
answers it. Half reuse that section's words, half are paraphrased:

| Embedder | Keyword: right section first | Paraphrased: right section first | In the top 5 (all) |
|----------|------------------------------|----------------------------------|--------------------|
| `hash` (shared words) | 90% | 15% | 92% |
| `bge-small`           | 85% | 40% | 92% |

The embedding model matters exactly where the question uses different words from the text.

```bash
python -m evals.retrieval_eval
```

## Roadmap

- [x] Exact brute-force cosine baseline + benchmark harness
- [x] HNSW index (graph-based ANN), single node
- [x] Sharded search: coordinator with scatter-gather and top-K merge (in-process)
- [x] Parallel fan-out: one long-lived worker process per shard
- [x] Shards as network services (gRPC) behind the coordinator
- [x] Replication: write to every replica, read from one, fail over when a server dies
- [x] Durability: snapshots + write-ahead log; a restarted server recovers its data
- [x] Replica resync: a replica that missed writes or lost its disk catches up from its twin
- [x] Real-data benchmarks: GloVe word vectors up to 400k, diversity heuristic for links
- [x] Command-line interface (`vsearch`)
- [x] Payloads: data stored with each vector and returned with search results
- [x] Text ingestion: chunk documents, embed them locally, search by question
- [x] Metadata filtering, and namespaces that each live on one shard
- [x] MCP server: search, ingest, and status as tools for an AI assistant
- [x] Operations agent: a model that diagnoses and repairs the cluster with tools
- [x] Agent memory: remember, recall by meaning, replace and forget
- [x] Evaluations: agent reliability over 140 runs, and retrieval on 40 labelled questions
- [ ] Docker / Kubernetes deployment

## Design notes

Key design decisions and trade-offs for each stage are documented in
[`DECISIONS.md`](DECISIONS.md).
