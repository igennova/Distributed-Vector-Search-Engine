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

## Replica resync
- The gap: a write can land on one replica and fail on the other. The shard's copies then
  differ, and a replica whose disk is lost comes back empty.
- Sequence numbers now come from the coordinator. Each write carries the seq it must get, and a
  durable replica rejects it unless it is exactly its last seq + 1. Without this, a replica that
  missed seq 5 would take the next write as its own seq 5, so "seq 5" would mean different
  writes on different copies and copying a log by seq would copy the wrong data. It is the same
  idea as Raft checking the previous log index before appending.
- Before writing, the coordinator asks each replica for its last seq (`Status`). If replicas
  disagree it refuses the write (`ReplicasOutOfSyncError`) instead of piling new writes on top of
  different histories. After a failed write it forgets what it knew, so the next write re-checks.
  This assumes one coordinator writing at a time.
- `repair()` finds the replica with the highest seq for each shard and tells each replica that is
  behind to `SyncFrom` it. The data goes directly between the two replicas; the coordinator only
  decides who copies from whom, so it is not in the data path.
- Catch-up tries the peer's log first (`FetchLog` after my last seq). If the peer has already
  folded those writes into its snapshot and emptied its log, it answers FAILED_PRECONDITION, and
  the replica copies the snapshot (`FetchSnapshot`, streamed in 1 MB chunks under gRPC's 4 MB
  limit), then fetches any newer log records. Postgres standbys (WAL streaming, or a base backup
  when the WAL is gone) and Raft (AppendEntries, or InstallSnapshot) work the same way.
- Repair rolls forward: a write that failed on some replicas but landed on one ends up applied
  on all of them. The caller saw an error for a write that did happen. Making retries safe would
  take idempotent writes (a client-chosen write id the servers deduplicate on); not done yet.
- A replica that is catching up holds its own lock for the whole sync, so searches sent to it
  wait until the sync finishes, or hit their deadline and fail over to its twin.
- Results, one 2,500-vector shard on localhost, two runs: catching up 50 / 500 / 2,000 missed
  vectors from the log took ~0.14 / ~1.3 / ~4.2 s. Rebuilding an empty replica from the
  snapshot took ~0.01 s for 1.9 MB.
- That surprised me: at this size the snapshot wins even for a 50-vector gap. Applying a log
  record means re-inserting its vectors into the HNSW graph (~2 ms each), while a snapshot ships
  the finished graph. In a database where applying a record is cheap the log wins; in a vector
  index, inserts are the expensive part. The balance still flips for big shards, where a
  snapshot is gigabytes to send. A better policy would pick the method from the gap size versus
  the snapshot size and network bandwidth, instead of always trying the log first.
- Each safety check was verified by removing it in a copy of the code: the "log no longer
  covers it" check, the fall-back to the snapshot, the server's exact-next-seq check, refusing
  writes while replicas disagree, and forgetting seqs after a failed write. Each removal makes
  a test fail.

## Faster distance computation
- Numbers in the sections above were measured before this change; the README has the current
  ones.
- Profiled first (cProfile, building 2,500 vectors): 84% of the time was in the distance
  function. `np.linalg.norm` alone was 57% of the total, 7 million calls, because every distance
  recomputed both vectors' lengths. The 3.5 million distance calls each did a few hundred
  multiply-adds but paid full Python + numpy call overhead.
- Fix: normalize each vector once on insert, so cosine distance is `1 - dot`; keep all vectors
  in one contiguous float32 matrix that doubles its capacity when full (amortized O(1) append,
  like a Python list); compute a whole neighbor list's distances with one matrix-vector product;
  and reuse the distances the beam search already computed when picking neighbors instead of
  computing them again.
- Same algorithm, same recall. 10k vectors, M=16, ef=200: build ~42 s → ~9–14 s, search p50
  ~5.2 ms → ~1.0–1.3 ms, recall 0.930 both times. All tests pass unchanged.
- After the change the vector math no longer appears near the top of the profile. What's left is
  the Python beam-search loop: heap pushes and pops, visited-set updates, and 3 million `len()`
  calls. Going much further would need compiled code (Numba, Cython, or C++ as in hnswlib).
- Costs, as expected: the index only does cosine now, the original vector lengths are gone, zero
  vectors need a special case, and appends need capacity management. The snapshot format moved to
  version 2, which stores unit vectors; they are loaded back bit for bit, because renormalizing
  could change the last bits and make a restored replica's graph drift from its twin's.
  Version-1 snapshots still load (their vectors get normalized on load).
- Knock-on effects on the other benchmarks:
  - Threads went from 5x slower than sequential to ~1.5x slower. Context switches per query with
    4 threads dropped from ~12,800 to ~270, since there are far fewer tiny numpy calls handing
    the GIL around. That independently confirms the earlier GIL diagnosis.
  - gRPC's ~0.3 ms per query is now about half of total latency, and on tiny shards gRPC is ~3x
    slower than in-process. Making the compute faster made the transport matter more.
  - fsync on every write now costs roughly 20% of write throughput (~3,100 → ~2,600 writes/s);
    a 1.1 ms insert used to hide it. Speeding up one part exposed the next one.
  - Log replay on restart and log-based resync got about 4x faster; snapshots still win for
    resync at this shard size.