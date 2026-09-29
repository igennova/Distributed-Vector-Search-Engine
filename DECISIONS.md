## Phase 0 — brute-force baseline
- Metric: cosine similarity (direction, magnitude-independent) over raw dot product.
- Bug caught: unit tests passed but benchmark recall was 0.734 — I'd implemented
  dot product, not cosine. Tiny tests used unit vectors (where they're identical);
  only the benchmark's varied magnitudes exposed it.
- Baseline @ 10k vectors: recall 1.000, p50 12ms, 80 QPS. Scales linearly →
  the motivation for an ANN index in Phase 1.

## HNSW index (graph-based ANN)
- Structure: hierarchical navigable small-world graph. Search greedily descends the
  sparse upper layers, then runs a beam search (`ef_search`) on the dense layer 0.
- Neighbor selection on insert: simple M-nearest, with bounded degree (2M at layer 0,
  M above) to keep search cost bounded. The paper's diversity heuristic would give
  better recall per `ef` and is a candidate future improvement.
- Recall is tunable at query time via `ef_search` with no rebuild — the recall/latency knob.
  @ 10k vectors, dim 128: ef=50 → 0.56 recall @ 1.8ms; ef=200 → 0.93 @ 5.2ms;
  ef=400 → 0.99 @ 8.3ms (vs brute force 1.00 @ 12.9ms).
- Caveat: the dataset is 128-dim random Gaussian, a near-worst case for ANN (points are
  nearly equidistant), so it needs a higher `ef` than real, clustered embeddings would.

## Sharding (scatter-gather)
- Partitioning: round-robin by global id. Similarity search has no routing key, so every
  query fans out to every shard; round-robin keeps shards balanced. Cluster-based routing
  (query only nearby shards) is cheaper per query but can miss neighbors near shard
  boundaries and creates hot shards.
- Each shard returns its full local top-k with distances, and maps local ids to global ids.
  Returning k / num_shards would drop results whenever the true top-k sits in one shard
  (covered by a test).
- @ 10k vectors, ef_search=50, shards queried sequentially:
  1 shard 0.55 recall / 1.8ms, 2 shards 0.70 / 3.7ms, 4 shards 0.88 / 5.5ms.
  Build time 29s → 20s.
- Recall rises because each shard searches a smaller graph and the merge sees K×k
  candidates. Latency rises because sharding adds total work; it only lowers wall-clock
  latency when shards run in parallel.
- For comparison, one index at ef_search=200 gives 0.93 / 5.2ms, so sequential in-process
  sharding is not a free win. Its payoff is parallelism and holding more data than one
  machine can.

## Parallel fan-out
- Three coordinator modes: sequential, threads (a thread pool), and processes (one
  long-lived worker process per shard). Each worker builds and owns its shard's index, so
  per query only the query vector (~0.5 KB) and k results cross the pipe instead of
  megabytes of index data — move the computation to the data.
- @ 10k vectors, 4 shards, ef_search=50: sequential 5.6ms, threads 30.5ms, processes 1.6ms
  (3.5x). Build 18s → 4.9s with processes, since shards build their graphs in parallel.
  Recall is identical in every mode (0.876); a test checks results match exactly.
- Threads were 5x slower, not just no faster. Changing sys.setswitchinterval (5ms → 0.1ms)
  had no effect, which ruled out forced GIL switching. Counting context switches did explain
  it: ~1 per query sequential, 262 with one worker thread, 3.1k with two, 12.8k with four.
  The search makes thousands of small numpy calls that release and reacquire the GIL; with
  other threads waiting, every release becomes an OS-level handoff.
- Tiny shards (200 vectors) still favored processes (0.23ms vs 0.62ms). I had expected the
  overhead to dominate, but a local pipe round trip costs tens of microseconds, less than the
  ~0.15ms of work per shard. Over a real network the overhead is larger, which should move
  that crossover.
- Takeaway: CPU-bound Python needs processes for parallelism. Threads suit I/O-bound work,
  like a coordinator waiting on remote shards.

## Shards as gRPC services
- Each shard is its own server process (`vsearch/server.py`) exposing `Add` and `Search`
  from a small `.proto` contract; the coordinator is a gRPC client that only needs a list of
  addresses. Changing an address to another machine changes nothing else.
- gRPC over REST/JSON: binary protobuf is smaller and cheaper to parse than JSON text for
  vectors of floats, the contract is enforced on both sides, and clients can be generated in
  any language. Qdrant and Milvus expose gRPC for the same reasons. Cost: a codegen step and
  traffic that isn't human-readable.
- Vectors are `repeated float` (like Qdrant's API) rather than raw bytes. Raw float32 bytes
  would be faster to encode, but tie both sides to one dtype and byte order.
- Fan-out uses gRPC futures: every Search request is sent before waiting on any. The
  coordinator now only waits on the network, which releases the GIL, so it needs no
  processes of its own; the CPU work happens inside the shard servers.
- Uploads are chunked to at most 2 MB per request. gRPC rejects messages over 4 MB by
  default; a test uploads ~4.5 MB to one shard and fails with RESOURCE_EXHAUSTED if chunking
  is turned off.
- Every Search carries a deadline (5s default) so one stuck shard cannot hang a query.
- Each shard server guards its index with one lock, because an insert mutates the graph while
  a search could be walking it. Requests to one shard therefore run one at a time; real
  systems use read-write locks or immutable index segments to serve concurrent reads.
- Servers bind to 127.0.0.1 by default since the service has no authentication.
- @ 10k vectors, 4 shards: pipes 1.5ms, gRPC 1.9ms, sequential 5.3ms; recall identical
  (0.876). gRPC costs ~0.35ms per query more than local pipes. On tiny shards (200 vectors)
  that overhead cancels the parallel speedup entirely (gRPC 0.60ms vs sequential 0.60ms,
  while pipes still manage 0.23ms). The overhead-versus-work crossover I expected at the
  parallel step shows up here, once a real transport sits in between.
- Current gap: if any shard is unreachable, the whole query fails with UNAVAILABLE (covered by
  a test). Handling that is the failure-handling step.

## Replication and failover
- More shards made the system more fragile: a query needs every shard, so with servers up 99%
  of the time only ~96% of 4-shard queries would succeed. Each shard now has several replicas on
  different servers (2 in the benchmarks).
- Writes go to every replica (write-all) and fail if any replica fails, so copies cannot
  silently diverge; the coordinator also checks that replicas report the same size. Quorum
  writes (W of N replicas) would keep writes available while a replica is down, but the missed
  replica then needs a log to catch up, which comes with persistence.
- Reads need one replica per shard. The starting replica rotates per query (round-robin), which
  spreads read load across copies. Hedged requests (ask two copies, keep the first answer) would
  cut tail latency at double the work.
- A read that fails with UNAVAILABLE or DEADLINE_EXCEEDED is retried on the shard's next
  replica; retries for different shards go out in parallel. A failed replica is tried last for
  5 s (a simple circuit breaker) but never skipped entirely, so a server that comes back is
  used again.
- If every replica of a shard is down, search() raises ShardUnavailableError, and
  search_partial() returns results from the remaining shards plus the missing shard ids. That
  is the availability-vs-completeness trade-off, left to the caller; Elasticsearch similarly
  returns partial results and reports failed shards.
- Replicas stay identical here because they get the same writes in the same order with the same
  seed. Real systems replicate the data itself or an operation log instead of relying on a
  deterministic rebuild.
- Results @ 10k vectors, 4 shards: search p50 was about 2 ms with 1 or 2 replicas, and the
  difference was within noise across three runs (1.9–3.5 ms vs 2.1–2.7 ms), since a search still
  reads one copy per shard. Build time went from ~5.8 s to ~8–10 s: twice the servers doing twice
  the writes on 4 performance cores.
- The first benchmark run showed 1 replica slower than 2 (3.5 vs 2.4 ms). Two re-runs showed it
  was noise rather than an effect, so these notes report ranges instead of a single run.
- Crash test: one of 8 servers killed with SIGKILL halfway through 100 queries. 100/100 queries
  answered, recall unchanged (0.876), one failover. That query took ~3.4 ms against a ~2 ms
  p50; afterwards the circuit breaker kept queries off the dead server.
- Gap: a replica that restarts comes back empty, and nothing resyncs it yet. That needs
  persistence and a write-ahead log.

## Durability: snapshots + write-ahead log
- Options considered: snapshots only (loses writes since the last one), log only (restart
  replays everything), snapshot + log (loses nothing, restarts fast), immutable segments
  (Lucene, Milvus; a bigger redesign of the index), keeping the index on disk (mmap), and
  rebuilding from an external source of truth. Snapshot + log gives zero loss of acknowledged
  writes (RPO) and a short replay (RTO), and it is what Postgres, SQLite, and Qdrant do.
- Write path: append the batch to the log, fsync, apply it to the in-memory index, then reply.
  Anything the server has acknowledged is on disk first.
- Log record: [length][crc32][seq | count | dim | ids | vectors]. Recovery stops at the first
  record that is cut short or fails its checksum and truncates the file there, which is what a
  crash mid-append leaves. Like Postgres and SQLite, a bad record is treated as the end of the
  log; records after it are not trusted.
- Every write gets a sequence number, and the snapshot records the last one it contains.
  Recovery skips log records at or below it, because a crash can land between writing a
  snapshot and emptying the log. That makes replay safe to repeat.
- Snapshots are written to a temp file, fsynced, renamed over the old one, and the directory is
  fsynced. A rename is atomic, so a crash mid-save keeps the previous snapshot intact.
- Snapshots are plain numpy arrays (adjacency lists flattened into offsets + neighbors) rather
  than pickle: faster, and loading a pickle can execute code.
- The snapshot also stores the random generator's state. HNSW draws each node's level from it,
  and replicas only stay identical if they keep drawing the same sequence. A test shows that
  reseeding instead makes the graph diverge.
- Each safety check was verified by removing it in a copy of the code: dropping the seq skip,
  the checksum, the torn-tail truncation, or the generator restore each makes its test fail.
- Results (one 2,500-vector shard, MacBook SSD): a log append costs ~1.9 µs without fsync and
  ~22 µs with it. Written through the log, a shard still takes ~850 writes/s either way, since
  the HNSW insert (~1.1 ms) dwarfs the fsync. Restarting from the log (1.3 MB) takes ~4.4 s
  because every vector is re-inserted; loading a snapshot (1.9 MB) takes ~3.5 ms. The snapshot
  is bigger than the log because it stores the graph too.
- macOS caveat: plain fsync hands data to the drive but does not flush the drive's own cache.
  F_FULLFSYNC does, and costs ~3.4 ms per append, about 150x more. So `--fsync always` here
  survives a process or OS crash, but not necessarily a power cut. (SQLite exposes the same
  choice as `PRAGMA fullfsync`.)
- Process kills can't show the difference between fsync modes: data already handed to the OS
  survives the process dying. Only a power or kernel failure loses unsynced writes, which the
  tests can't simulate.
- Because writes fail whenever a replica is down, a server that was down never missed an
  acknowledged write, so reloading its own disk is enough to rejoin. The remaining gap is a
  write that reached one replica but failed on the other; fixing that needs a resync from the
  healthy copy.