## Brute-force baseline
- Metric: cosine similarity (direction, magnitude-independent) over raw dot product.
- Bug caught: unit tests passed but benchmark recall was 0.734 — I'd implemented
  dot product, not cosine. Tiny tests used unit vectors (where they're identical);
  only the benchmark's varied magnitudes exposed it.
- Baseline @ 10k vectors: recall 1.000, p50 12ms, 80 QPS. Scales linearly →
  the motivation for an ANN index.

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
  - Correction: I first wrote here that fsync on every write now cost roughly 20% of write
    throughput, from a single run (~3,100 vs ~2,600 writes/s). The next run showed no gap, and
    seven interleaved runs put the three modes within a few percent of each other while each
    mode alone ranged far wider (in-memory: 1,363 to 3,252 writes/s). A 22 µs fsync against a
    ~0.33 ms insert should cost about 7%, which this machine's noise hides. The 20% was noise.
  - Log replay on restart and log-based resync got about 4x faster; snapshots still win for
    resync at this shard size.
## Real data: GloVe word vectors
- Everything so far ran on 10k random vectors: small, a worst case for ANN, and comparable to
  nothing. Switched to Stanford GloVe 6B (400,000 words, 100 dims). It ships with the words,
  unlike the ann-benchmarks copy of GloVe, so results can be checked by eye ("king" → prince,
  queen, monarch) and it can back a word-similarity demo.
- Queries are 1,000 random words held out of the index, the same ones at every size. The index
  holds the n most frequent of the remaining words, so recall is not comparable across sizes.
  Ground truth for each size is computed by exact search over that subset.
- The baseline is exact search as one numpy matrix-vector product per query. The earlier
  brute-force baseline was a Python loop, and against it I had written that HNSW was "about 10x
  faster than exact search" at 10k vectors. Against the vectorized baseline that is false: at
  10k, exact search takes 0.09 ms and beats HNSW (0.19–0.64 ms). The README was corrected.
- What does hold is the scaling. From 10k to 400k vectors exact search goes 0.09 → 1.05 →
  3.96 ms (linear), HNSW at ef=100 goes 0.34 → 0.53 → 0.65 ms. At 400k: 0.848 recall at 0.35 ms
  (11x faster than exact), 0.912 at 0.65 ms (6x), 0.949 at 1.14 ms (3.5x).
- It is an uneven comparison in the baseline's favor: its inner loop is compiled BLAS, this
  HNSW is interpreted Python. A compiled HNSW would widen the gap a lot.
- Sharding on real data: 4 shard processes at ef=100 reach 0.960 recall at 0.73 ms on 400k
  (one index: 0.912 at 0.65 ms), and build in 75 s instead of 264 s.
- Memory: ~88 MB per 100k vectors (40 MB of vectors, the rest Python dicts and lists for the
  graph), 351 MB at 400k. Storing the graph in numpy arrays would cut that; not needed yet.
- These are single-run numbers on a laptop.

## Neighbor selection: diversity heuristic
- With "link to the M closest candidates", recall on GloVe was 0.727 / 0.813 / 0.879 at
  ef 50 / 100 / 200 (100k vectors). Real data is clustered, so the M closest candidates of a
  node are usually near each other and its links all lead into the same cluster.
- The heuristic from the HNSW paper (Algorithm 4) walks the candidates from nearest to farthest
  and keeps one only if it is closer to the node than to every neighbor already kept. A rejected
  candidate is reachable through a kept one, so the slot goes to a different direction.
- Measured three variants on the same 100k vectors and queries:
  closest 0.727 / 0.813 / 0.879, build 56 s, 25.8 links per node;
  heuristic + filling unused slots with rejected candidates 0.797 / 0.870 / 0.921, 69 s, 27.2;
  heuristic without filling 0.791 / 0.868 / 0.919, 57 s, 21.6.
- Chose the heuristic without filling. I had assumed filling would help by keeping every node
  at full degree; it added 20% build time and slower searches for at most 0.6 points of recall.
  Not filling is also what hnswlib does.
- At 400k vectors the heuristic gives 0.848 / 0.912 / 0.949 against 0.817 / 0.877 / 0.924 for
  closest.
- On the random synthetic data the heuristic moves recall by about a point (0.926 vs 0.930 at
  ef=200 on one index): with no clusters there is nothing to diversify, and fewer links cost a
  little.
- The rule is a setting (`neighbor_selection`) stored in snapshots, so a restored replica keeps
  building its graph the same way as its twin. Snapshots written before the setting existed
  load as "closest".

## Command-line interface
- `vsearch cluster up` starts the shard servers as detached background processes (their own
  session, logs to a file) and records their ports and process ids in `cluster.json`, written
  with the same write-temp-then-rename used for snapshots. Later commands read that file to
  know where to connect.
- `cluster down` stops the servers but keeps their data directories, so `cluster up` again
  restarts the same layout and every server recovers from its snapshot and log. Deleting the
  data takes an explicit `--wipe`.
- Process ids are reused by the OS. Before signalling a recorded pid, the CLI checks with `ps`
  that it is still a `vsearch.server` on the recorded port, so a stale state file can never
  kill an unrelated process.
- `similar` maps results back to words through the GloVe word list: global ids are row numbers
  in that list. That only works because the dataset is fixed; storing a payload next to each
  vector is the general answer and comes with the retrieval layer.
- Each command is a new process with a new coordinator, so the "this replica just failed, try
  it last" memory does not carry over between commands: with a server down, each `similar`
  that picks it first pays one failover again (a few ms). A long-running coordinator service
  would keep that state.
- Built with argparse; no new dependency. A console script makes it `vsearch ...`, and
  `python -m vsearch ...` works without installing.

## Payloads
- A search returned ids and distances only; the CLI's `similar` could show words only because
  it looked ids up in the GloVe word list on the side. Anything built on top (retrieval for an
  LLM, agent memory) needs the stored data itself back, with where it came from.
- A payload is any JSON-serializable dict, or nothing. The coordinator encodes it to JSON text
  once; shards store and return that text and never parse it, so they stay independent of what
  is inside. "" means no payload.
- Payloads travel with their vector everywhere: in `Add`, in the write-ahead log record, in the
  snapshot, and in the log records and snapshots that resync ships. A payload that was only in
  memory would be lost exactly when its vector survived.
- Formats were extended without breaking old data. A log record gains a payload section only
  when some vector in it has a payload, so a record without payloads is byte for byte what it
  was before (a test pins the size). Snapshots moved to format 3, which adds a byte blob plus
  offsets; formats 1 and 2 still load, with no payloads. New proto fields got new field numbers.
- Payload text is stored as one blob with offsets rather than pickled, for the same reason as
  the vectors: loading a pickle can execute code.
- Uploads still split into requests of at most 2 MB, now sized by the largest payload in the
  batch as well as the vector size.
- Plain `search` is unchanged and returns ids. `search_hits` returns Hit(id, distance, payload);
  the server only sends payloads when asked, so searches that don't need them don't pay for them.
- Not done: payloads live in memory with the index. Large texts would be better kept on disk and
  read only for the final top-k.

## Floating-point details found while testing payloads
- A new test compared full results (ids, distances, payloads) between a local index and a
  replica reached over gRPC, and one distance differed in the 8th decimal place.
- First suspicion: the replicas' graphs had diverged. Checked directly in one process: vectors,
  graphs, and results are identical between copies, including after a restart from the log. So
  that was wrong.
- Actual cause: the distance to a search's starting node was computed as
  `1.0 - float(dot)`, which subtracts in float64, while every other distance was computed as
  `1.0 - dots` in float32. Distances travel as float32 over gRPC, so that one value was rounded
  on the wire and no longer equalled the local one. Earlier tests compared ids only, or
  happened not to have the starting node among the results. Fixed by computing every distance
  in float32; a test now checks every returned distance is a float32 value, and both it and
  the cross-process test fail on the old code.
- Also measured along the way: numpy's matrix-vector product can give the same row a result
  that differs in the last bit depending on its position in the batch (2 distinct values for
  one row across batch sizes 1 to 40), while memory alignment made no difference. So the same
  (node, query) distance can differ by one ulp between two searches that reach the node in
  different batches. Identical replicas walk identical paths and still agree exactly.
- Benchmarks were unaffected by the fix (same recall at every ef_search).

## Text ingestion
- The pipeline is file → passages → vectors → shards, and each vector's payload is its passage
  (`text`, `source`, `heading`, `chunk`). Nothing below the coordinator changed: a shard still
  stores vectors and opaque payload text.
- Why split at all: an embedding model reads a fixed number of tokens (512 for this one) and
  silently ignores the rest, and one vector for a whole document blurs every topic in it
  together. Passages are at most 1000 characters, roughly 250 tokens, well inside the limit.
- Where to cut: first at Markdown headings, so a passage never mixes two sections, then
  between sentences, list items, table rows, or lines of code, never inside one. A list item
  wrapped over several lines is put back together before packing; the first version cut at
  line breaks and produced passages starting mid-sentence, visible as soon as real results
  were printed.
- Neighboring passages of a section share up to 150 characters, so a fact that sits on a
  boundary is whole in one of them. Overlap never crosses a heading.
- The headings above a passage are embedded with it ("Storage > Snapshots" + text). A passage
  often never names the thing it is about; its headings do.
- Embedders share a small interface (`embed_documents`, `embed_query`, `dim`, `name`), with
  two implementations. `bge-small` is a neural model run locally through fastembed and ONNX
  Runtime: no API key, no network after the first download, no PyTorch. `hash` is feature
  hashing of words (crc32, because Python's `hash()` of a string changes per process and a
  stored vector has to match its text after a restart; a test runs a second interpreter to
  check). It needs nothing installed, so the whole pipeline is tested in CI without a model.
- A cluster records which embedder built it, and search always uses that one. Vectors from
  two embedders live in unrelated spaces, so ingesting with a different one is refused, as is
  mixing documents into a cluster of GloVe words.
- A small check, not an evaluation: 14 hand-written questions about this repository's two
  documents, each phrased differently from the text, scored by whether the right section came
  back.

  | Embedder | right section first | in the top 3 |
  |----------|---------------------|--------------|
  | `hash` (shared words) | 6 / 14 | 10 / 14 |
  | `bge-small`           | 8 / 14 | 10 / 14 |

  The model is ahead at rank 1 and level in the top 3. With 14 questions written by the
  author, two questions of difference is not evidence of much; it does show a neural model is
  not automatically far ahead of word matching on technical text full of rare terms. Embedding
  the headings helped the model at every passage size tried (right section first: 7–9 with
  headings, 4–6 without, at 300, 500, and 1000 characters). The query instruction the model's
  authors suggest for short questions made no measurable difference (8 vs 8 first, 11 vs 10 in
  the top 3), so it is not used. A proper evaluation set comes later.
- Scores are not comparable between embedders: `bge-small` gives unrelated texts 0.4 to 0.5
  and related ones 0.65 and up, while `hash` gives unrelated texts about 0. Only the order
  matters.
- Where the time goes for one `vsearch search`: about 380 ms to import the library and load
  the model, about 3 ms for the search across the shards. Each command is a new process, so
  it pays the load every time; a long-running service would pay it once (embedding a query
  with the model already loaded takes about 2 ms). Ingestion embeds about 20 passages a
  second on a laptop CPU.
- Ingestion records each file's SHA-256 in `cluster.json` after the file is stored. Unchanged
  files are skipped, so `ingest` can be run again safely.
- Each check was verified by removing it in a copy of the code and confirming a test fails:
  the embedder match, the words-versus-documents check, the unchanged-file skip, the size
  limit, the overlap, headings inside code blocks, and crc32 in place of `hash()`. The test
  for the unchanged-file skip passed without the skip at first (the changed-file branch
  produced the same "0 passages" line) and was tightened.
- Not done: the index has no delete, so a changed file is reported and skipped instead of
  replaced. A crash in the middle of a file can leave some of its passages stored without the
  file being recorded, and a re-run would add them again; fixing both needs deletes or
  idempotent writes. Only Markdown and plain text are read (no PDF or HTML).
