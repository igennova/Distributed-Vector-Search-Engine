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

## Metadata filtering
- A filter restricts a search to vectors whose payload matches: every listed field must match
  (AND), and a list of values matches any of them (OR). Equality only: no ranges, negation,
  or nested fields yet.
- Filtering the unfiltered top-k afterwards is the obvious way and the wrong one: with a filter
  matching 10% of the data it returned 1.1 results out of 10, and the right ones only by
  accident (recall 0.109). The filter has to be part of the search.
- Two ways that work, with opposite costs. Exact: compare the query with every matching
  vector; time grows with the number of matches. Graph: walk the HNSW graph, keep only matching
  nodes as results; it must pass about `ef / share` nodes to collect `ef` matches, so it gets
  slower as the filter gets more selective (1.2 ms at 50%, 35 ms at 1%, 143 ms at 0.1% on 100k
  vectors) and measured the same at 400k as at 100k for a given share.
- The graph walk still travels through nodes that do not match. They are the roads to the ones
  that do; skipping them would cut the graph into islands. It also keeps going until it holds
  `ef` matches. The first test of that passed with the unfiltered stopping rule left in (the
  allowed nodes were all far from the query, which hid it) and was rewritten.
- Each shard picks per query: exact while `matches^2 <= EXACT_ADVANTAGE * ef * size`, which is
  where the two cost estimates cross. The constant is 100; any value from 40 to 158 picks the
  faster way at every point measured on 100k and 400k vectors. Near the crossover the two are
  close, so a wrong pick there is cheap. Measured at 100 dimensions with `ef=100` on one
  machine; the dependence on `ef` is assumed, not measured.
- Surprise in the numbers: exact search wins far more often than expected, for any filter
  under about a third of a 100k shard. A matrix product over 20,000 gathered rows takes
  0.5 ms. This is the same lesson as the 10k benchmark, where numpy exact search beat HNSW.
- Exact search per vector got more expensive on the larger index in the middle of the range
  (40k to 80k scattered rows out of 399k took 2.5 to 5 ms, about 2.5x the per-vector cost at
  100k), most likely because gathering scattered rows from a 160 MB matrix misses the cache.
  Not investigated further; it is why the constant was set from both sizes.
- The benchmark's groups are random, so matching vectors are spread evenly through the graph.
  A real filter often correlates with the vectors (one source is one topic), which this does
  not measure.
- To know what matches without reading every payload, a shard keeps a payload index: (field,
  value) to the ids that have it. Only top-level strings (up to 128 characters), integers,
  booleans, and lists of those are indexed; a filter on anything else is refused with an
  error instead of quietly matching nothing. The value's type is part of the key, because in
  Python `True == 1`.
- The index is not stored and not maintained on write. It is built on the first filtered
  search and extended on later ones (0.17 s for 100,000 passage payloads; a lookup takes
  0.02 to 0.12 ms). So snapshots, the log, and resync did not change, and a shard that is never
  filtered never parses its payloads. The cost: the first filtered search after a restart
  pays for the build while holding the shard's lock.
- The filter travels as JSON text in `SearchRequest.filter`, like payloads. The coordinator
  validates it before sending anything, and the server validates it again and answers
  INVALID_ARGUMENT.

## Namespaces
- A namespace is a named, separate set of vectors in the same cluster: one per user, project,
  or agent. The idea comes from how Discord stores messages for search: keep everything that
  is searched together on one shard, so the search does not fan out.
- All vectors of a namespace go to shard `crc32(name) % shards`, and each records its
  namespace in a reserved payload field. A search inside a namespace asks that one shard, with
  the namespace added to the filter, because a shard holds many namespaces in one graph.
- Measured against spreading tenants over all shards and filtering: 1 request instead of 4,
  p50 0.25 ms instead of 0.44 ms, and with one shard down 76% of tenants could still search
  instead of none. Recall was 1.000 both ways, since a small tenant is searched exactly.
- crc32 and not `hash()`, for the same reason as in the hashing embedder: the mapping has to
  be the same in every process. A test pins three names to their shards.
- Vectors without a namespace still go round-robin, and a search without one still covers
  every shard, including namespaced vectors.
- Costs and limits. A very large namespace makes its shard large; Discord gives such guilds a
  dedicated index with several shards, and nothing like that exists here. Changing the number
  of shards would map names to different shards, with no way to move the data (consistent
  hashing plus rebalancing is the usual answer). Loading one tenant at a time builds on one
  shard at a time: 52 s against 14 s for the same 100,000 vectors spread round-robin. And a
  namespace separates data, it does not protect it: there is no authentication, and a search
  without a namespace sees everything.
- Found while building this: every new coordinator numbered its vectors from 0, so a second
  `vsearch ingest` reused the ids of the first. A coordinator now asks the shards for the
  largest id in use before its first write. The largest id rather than a count, because after
  a write that reached only some shards a count would hand out ids already taken.
- Each check was verified by removing it in a copy of the code and confirming a test fails:
  keeping only allowed nodes, the stopping rule, the type in the index key, bringing the index
  up to date, sending the filter over gRPC, rejecting a bad filter, adding the namespace to
  the filter, asking one shard, routing writes, the reserved field, crc32, and the id
  continuation.

## MCP server
- Until now only a person at a terminal could use the engine. The Model Context Protocol is
  the standard way for an AI application to call tools another program offers, so `vsearch mcp`
  makes the engine usable by any client that speaks it, with no code specific to one client.
- The protocol layer is written here (`mcp.py`, about 190 lines) instead of using the official
  SDK. What a tools-only server needs is small: JSON-RPC 2.0 over stdin and stdout, one message
  per line, and four methods (`initialize`, `ping`, `tools/list`, `tools/call`). Writing it
  keeps the project free of another dependency and lets CI test the whole session. The cost:
  only the stdio transport, no resources or prompts, a hand-written check of tool arguments
  that covers part of JSON Schema, and new protocol revisions have to be followed by hand.
- Version negotiation: the server answers with the client's protocol revision if it speaks it
  (2024-11-05, 2025-03-26, 2025-06-18), otherwise with its newest, and the client decides.
- Two kinds of failure are kept apart, as the protocol asks. A tool that ran and failed
  returns a normal result with `isError`, written for the model: what went wrong and what to
  do instead ("Nothing is indexed yet. Add documents with the ingest tool first."). A request
  the server cannot serve (unknown method, unknown tool) is a JSON-RPC error for the client
  program. Bad arguments are the first kind, so a model that sends `k: 100` reads "must be at
  most 20" and can correct itself. A bug inside a tool is caught and reported the same way
  instead of ending the session.
- stdout is the protocol channel, so one stray `print` from any library would corrupt the
  stream. While serving, everything written to `sys.stdout` is redirected to stderr; a test
  has a tool print and checks the output stays clean.
- Tools are few and their arguments flat: `search(question, k, namespace, source)` instead
  of exposing the general filter. The description of each tool and argument is what the model
  reads to decide when and how to call it, so they say when to use the tool, not how it works.
  Results are whole passages with their file and section, so an answer can cite them; `k` is
  capped at 20 to bound the size of a reply.
- The server lives as long as the client's session, unlike the command line where every call
  is a new process. The embedding model is loaded once: the first search of a session took
  407 ms, later ones a median of 4.5 ms, against about 390 ms for every `vsearch search`. The
  coordinator is kept too, so its memory of which replica failed last now carries over.
- A long-lived coordinator met a new problem: the command line may write between two of its
  writes, leaving its idea of the shards' write positions stale. It now re-reads them before
  each ingest (`refresh()`); a test ingests through the server, then the command line, then
  the server again. Two writers at the same moment are still not supported.
- `ingest` reads any path the server process can read and adds it to the index, at the
  request of the model. The client asks its user before running a tool, and `--read-only`
  leaves the tool out entirely. There is no authentication; the server is meant to run
  locally, started by the client.
- Tool calls are handled one at a time. Ingesting a large folder blocks the session until it
  finishes, with no progress reports and no cancellation.
- To share the logic between the command line and the tools, the cluster-state helpers and
  the document logic moved out of `cli.py` into `state.py` and `documents.py`.
- Tested by a client written for the tests, which starts `vsearch mcp` as a separate process
  and speaks the protocol over real pipes. Not yet tried with a third-party MCP client.
- Each check was verified by removing it in a copy of the code and confirming a test fails:
  the stdout redirection, not answering notifications, catching a crashing tool, tool errors
  as results, checking arguments, booleans not passing as integers, version negotiation,
  `--read-only`, the refresh before a write, and turning user errors into tool errors. The
  last one survived at first, because the catch-all for crashing tools hid it behind a worse
  message; the test now checks the wording.

## Operations agent
- Keeping the cluster healthy was a person's job: read `status`, restart what is down, run
  `repair`. `vsearch ops check` does it, with either a fixed procedure or a model deciding.
- An agent here is a loop of about 30 lines (`run_agent`): ask the model, run the tools it
  asks for, hand back the results, until it answers without asking for a tool or runs out of
  steps. The loop does not know which model it talks to: a model is anything with
  `begin(system, task, tools)` and `step(outcomes) -> Turn`. There are three: OpenAI, Claude,
  and a scripted one that replays fixed turns, which is how the loop and tools are tested
  without a model, a network, or a key.
- The model gets five dedicated tools with typed arguments, not a shell. A shell command is an
  opaque string; a `restart_server(shard, replica)` call is something the code can validate,
  refuse, count, and write down.
- The limits are enforced in the tools and the loop, not requested in the prompt. No tool
  deletes data or stops a server. `restart_server` refuses a server that is up, refuses one
  whose process is still running, and refuses a third attempt on the same server in one run.
  The loop stops after 12 model turns. `--dry-run` makes the changing tools describe what
  they would do. A test feeds the loop a model that calls a tool that does not exist, passes
  a wrong type, restarts a healthy server, and asks for 5,000 log lines: every call comes
  back as an error it can read, and the cluster is untouched.
- The model's report is a claim, not a result. After it stops, the command inspects the
  cluster itself and takes its exit code from that. A test has the model answer "Everything
  is fixed" without doing anything; the command prints the report, then "still wrong", and
  exits 1.
- "Healthy" is defined once (`health.py`): every server answers, and the replicas of each
  shard have applied the same writes. The status tool, the trigger for the agent, and the
  final check all use that one definition.
- Real runs with `gpt-5.4-mini`, one of each, on a 2-shard, 2-replica cluster:

  | Fault | What the model did | Turns | Tokens (in + out) | Result |
  |-------|--------------------|-------|-------------------|--------|
  | A replica killed | status, restart, status, probe | 5 | 4,137 + 151 | healthy |
  | A replica killed while a write was in flight | status, restart, status, repair, status | 6 | 5,298 + 129 | healthy |
  | A replica with a corrupt snapshot | status, restart (failed), read log, status | 5 | 5,296 + 280 | still down, cause named, handed to a person |

  The first took 11 s end to end. One run each shows the loop works; it says nothing about
  how often it works. That is what the evaluation step is for.
- The honest comparison is with the runbook, the fixed procedure a script would follow
  (restart everything that is down, repair, probe). On the first two faults it reaches the
  same healthy cluster instantly and for free. The model was only better on the third: the
  runbook reports "could not restart", while the model read the log, said the snapshot was
  unreadable, and did not retry. So a script is enough for the faults someone anticipated,
  and the model earns its cost on the ones nobody wrote a branch for.
- The third fault also shows the price of the limits. The real fix is to delete the corrupt
  snapshot and let `repair` rebuild the replica from its twin, exactly as in the lost-disk
  case, which both brains handle. No tool deletes anything, so the agent stops and says so.
  A tool for it would have to be guarded (only when the twin is healthy and in sync); not
  built.
- Every model turn and every tool call, with arguments and results, is appended to
  `ops-journal.jsonl`. `ops watch` repeats the check on an interval; if a run leaves problems
  behind, it does not start another on the same problems, since a second attempt would cost
  time and tokens and end the same way.
- OpenAI is called over HTTPS with the standard library: it is one endpoint, so there is
  nothing to install, and the tests run the client against a local HTTP server that plays
  the API, exercising the real request and response path. Claude goes through its SDK, which
  is optional and loaded on use. The Claude client is tested against a stand-in for the SDK
  and has not been run against the real API.
- Keys are read from the environment or a git-ignored `.env` file, never from the command
  line. Error messages from the API have the key removed before they are shown. The test
  suite cannot reach a real model: for every test the key file is not read, keys are removed
  from the environment, and the API address points at a closed port.
- With a model brain, the cluster's status and log excerpts leave the machine.
- Found while writing the fault tests: a coordinator that starts while a server is down
  refuses to write at all, because it asks every replica for its position first. A replica
  only falls behind when it dies during a coordinator's life, so the tests kill it between
  two writes of the same coordinator.
- To share code, process management moved out of `cli.py` into `processes.py`, and the MCP
  server and the agent now run tool calls through the same function.
- Not done: a server that hangs without exiting is reported, not fixed (nothing here stops a
  process). There is no approval step before a change other than `--dry-run`. Tools run one
  at a time.
- Each check was verified by removing it in a copy of the code and confirming a test fails:
  the restart limit, dry run, refusing to restart a live server, checking after the report,
  not retrying in `watch`, unknown tools, the step limit, journaling, removing the key from
  errors, marking failed tools for the model, and the key file not overriding the
  environment.

## Evaluations
- One run per fault showed the agent can work, not how often it does. A model does not take
  the same steps every time, so reliability has to be measured over many runs.
- Operations eval: each run starts a fresh 2-shard, 2-replica cluster, breaks it in a known
  way, lets a brain work, and grades with code. Success needs both the right end state and an
  honest claim: the report ends with `Outcome: fixed / needs a person / nothing wrong`, and it
  has to match what the final check finds. A report without that line counts as a wrong
  claim. For the false alarm, success means changing nothing; for the corrupted snapshot, it
  means stopping within the restart limit and saying a person is needed. API errors are
  recorded separately and never count as successes.
- Seven faults: a killed replica, one killed during a write (it needs a repair after its
  restart), one that lost its disk, one replica down in each shard, both replicas of a shard
  down, a corrupted snapshot, and a false alarm (the report says a replica is down; nothing is).
- Results, 10 runs per fault for each of two prompts with `gpt-5.4-mini`, 3 per fault for the
  runbook:

  | Brain | Success | Claims correct | Read the broken server's log | Median turns | Tool calls | Tokens |
  |-------|---------|----------------|------------------------------|--------------|------------|--------|
  | careful prompt | 70/70 | 70/70 | 10/10 | 5 | 4.5 | 5,120 |
  | lean prompt    | 70/70 | 70/70 | 10/10 | 5 | 4.0 | 4,721 |
  | runbook        | 21/21 | 21/21 | 0/3   | - | 4.1 | 0     |

  With 70 of 70, the 95% interval (Wilson) is 95–100% for each prompt overall, and 72–100%
  for any single fault. All 140 model runs together used 668,397 input and 20,488 output
  tokens, about $0.60 at the listed prices; the median run took 7 s of agent time, against
  1.5 s for the runbook.
- The eval is too easy to separate the brains on success: the runbook also passes every fault.
  What separates them is what happens on the fault nobody wrote a branch for. On the corrupted
  snapshot the model read that server's log in 10 of 10 runs and named the cause; the runbook
  reports that the restart failed and nothing more. Harder faults are the next step: a server
  that hangs without exiting, a port taken by another process, two faults where order matters,
  and a report that names the wrong server.
- The lean prompt ("act on the listed problems at once, ask for several tools in one turn,
  verify once") was meant to cut turns. It did not: the median stayed at 5. The model made
  about one tool call per turn under both prompts (1.0 to 1.3 calls per acting turn), and
  under the lean prompt it still began with `cluster_status`. What the prompt changed was
  where the turns went: it called `repair` right after a restart, which saved a turn when a
  replica had missed writes (6–7 turns down to 5) and added an unneeded one after a plain
  restart (9 of 10 dead-replica runs). Net, 8% fewer tokens. The habit of one call per turn
  came from the model, not the prompt.
- In 2 of the 10 corrupted-snapshot runs under each prompt, the model restarted the server a
  second time after reading its log, although the prompt says not to retry blindly. The limit
  of 2 restarts in the code is what bounded it. That is the measured argument for enforcing
  limits in code rather than asking for them in the prompt.
- Retrieval eval: 40 questions about the design notes (a frozen copy in `evals/corpus`, so
  later edits to the notes do not move the scores), each labelled with its section. Half
  reuse that section's words; half were written to avoid them. Passages are ranked by exact
  cosine similarity, so it measures the embedder and the chunking, not the index.

  | Embedder, passage size, headings | Keyword hit@1 | Paraphrase hit@1 | Paraphrase hit@5 | MRR (all) |
  |----------------------------------|---------------|------------------|------------------|-----------|
  | hash, 1000, yes      | 90% | 15% | 85% | 0.68 |
  | bge-small, 1000, yes | 85% | 40% | 85% | 0.73 |
  | bge-small, 500, yes  | 90% | 45% | 85% | 0.78 |
  | bge-small, 1000, no  | 85% | 40% | 90% | 0.74 |

  Word matching is as good as the model when the question uses the document's words, and
  nearly useless at rank 1 when it does not; the model is better there by almost 3x. Both
  find the right section in the top five about as often. Each question is 5 points of a
  kind's score, so the passage-size and heading differences (one or two questions) are noise.
- Correction: the text-ingestion notes say embedding the headings helped the model at every
  passage size, from 14 questions. On these 40 it made no difference (hit@1 62% either way).
  That earlier result does not hold up.
- The questions were written by the author of the notes, who knows their wording, so these
  scores are likely higher than other people's questions would get.
- The evals have tests of their own, because a grading bug reports a wrong number with
  confidence. Each grading rule was verified by removing it in a copy of the code and
  confirming a test fails: requiring an honest claim, the restart limit on the unfixable
  fault, changing nothing on a false alarm, reading the right server's log, excluding API
  errors, the interval formula, and the rank calculation. The false-alarm rule survived at
  first, because no test had an agent restart a healthy server; one does now.
- The tests can never reach a real model or use a real key, so the paid runs are only ever
  started by hand.

## Agent memory
- A model starts every conversation knowing nothing about earlier ones. Memory gives it three
  operations: `remember` a fact in one sentence, `recall` the facts relevant to a query, and
  `forget` one. It is built entirely from parts that already existed: each fact is embedded
  and stored with its text, tags, and time as the payload, each owner's facts are a
  namespace (one shard, separate from other owners), and the bookkeeping below is done with
  filters.
- Facts change, and the engine has no delete. So a new fact can name the facts it replaces,
  and `forget` writes a record that names its target. Recall fetches its candidates, then
  makes one filtered lookup for records that replace any of them, and drops those. Nothing
  is ever removed: a forgotten fact is no longer recalled, but its text is still in the
  write-ahead log and the snapshots. For a memory of personal facts that is a real gap;
  erasing needs a delete in the engine and a compaction that rewrites the log and snapshot.
- Measured with the embedding model, to set the thresholds:

  | Pair of facts | Similarity |
  |---------------|------------|
  | the same sentence, punctuation aside | 0.984 |
  | a paraphrase ("short answers with examples" / "brief replies that include examples") | 0.908 |
  | an update ("lives in Delhi" / "moved to Bangalore") | 0.812 |
  | a change ("favourite colour is blue" / "is green") | 0.814 |
  | unrelated ("prefers short answers" / "lives in Delhi") | 0.666 |

  Embeddings measure what a fact is about, not whether it is still true: an update is as
  close to the old fact as two different facts about the same topic. So similarity cannot
  tell that "moved to Bangalore" replaces "lives in Delhi"; the agent has to say so, with
  `replaces`. A new fact at 0.97 or more from a stored one is treated as the same fact and
  not stored again. That catches repeats, not paraphrases, which sit at 0.91, too close to
  genuinely different facts to merge automatically.
- Recall ranks by similarity plus a recency bonus of up to 0.03, halving every 30 days. In a
  small trial the gap between a relevant fact and an unrelated one was often only 0.05 to
  0.1, so a bigger bonus (0.1 was the first choice) would let any fresh fact outrank an older,
  more relevant one. At 0.03 recency only reorders near-ties, such as two versions of a fact
  stored without `replaces`. A tuned value would need an eval of its own.
- Recall looks at 4 x k candidates before dropping replaced ones and re-ranking. If more
  than that are replaced, it returns fewer than k.
- A cluster holds documents, GloVe words, or memories, never a mix, as before: they are made
  by different embedders, and a passage search should not return someone's personal facts.
- Over MCP, the owner is fixed when the server starts (`--owner`), not chosen by the model, so
  a model using one owner's memory cannot read or change another's.
- Found while testing: after `close()`, the memory store and the MCP server kept the address
  list of their old connection, so the next call found "already connected" and used a closed
  coordinator. Both now forget the addresses on close; a test makes a call after closing.
- Each check was verified by removing it in a copy of the code and confirming a test fails:
  dropping replaced facts, dropping forgotten ones, not storing a repeat, the recency bonus,
  separate owners, refusing to replace an unknown id, reconnecting after close, and keeping
  memories out of a documents cluster.

## Docker
- Running the cluster needed Python 3.13, a virtual environment, and servers started by
  hand. Now each shard server is an image, and `docker-compose.yml` runs 2 shards x 2
  replicas, one container and one named volume per server. Replicas of a shard get the same
  `--seed`, so the same writes still build the same index.
- The image holds only the server: numpy, grpcio, protobuf, and the package. Embedding
  happens in whatever writes to the cluster (the `vsearch` command on the host), so the
  model and ONNX Runtime stay out of it. About 325 MB on disk, 73 MB to download.
- Dependencies are installed in their own layer, from `pyproject.toml` with an empty package
  in place, so changing the code rebuilds in seconds instead of reinstalling numpy and gRPC.
  `.dockerignore` lets in only `pyproject.toml` and `vsearch/`; a `.env` with API keys can
  never end up in the image.
- The server runs as an ordinary user, not root. It listens on 0.0.0.0 inside its container
  so its neighbours and the published port can reach it, and the compose file publishes the
  ports on 127.0.0.1 only, since the service has no authentication.
- `init: true` puts a small init process in front of the server. Python running as PID 1
  ignores SIGTERM unless it installs a handler, so `docker stop` would wait 10 s and then kill
  it; with an init process the signal stops it at once. (Killing it was never unsafe: every
  acknowledged write is already in the fsynced log.)
- `restart: unless-stopped`: when the server process inside a container was killed, Docker
  started the container again within a second, and the server recovered its 60 vectors from
  its volume (restart count 1, nothing lost). A container stopped on purpose stays stopped.
- First build failed: all four services had `build: .` and the same image name, and Compose
  built them in parallel, racing to name the image. Now one service builds it and the others
  reuse it with `pull_policy: never`, which also stops Docker from ever pulling an image
  called `vsearch-server` from a public registry.
- `vsearch cluster attach` records servers that something else runs. Everything that reads
  and writes works as before (status, ingest, search, memory, MCP, ops), while starting,
  stopping, and restarting are refused, since they belong to Docker. The ops agent's restart
  tool says the same to the model, and its log tool says the logs are kept by Docker.
- Found by running it: `repair` failed in Docker. The coordinator tells the lagging replica
  to copy from its twin and passes the twin's address, which was `127.0.0.1:50054`: right from
  the host, but inside a container 127.0.0.1 is the container itself. Each server now has two
  addresses: the one clients use, and the one its peers use (`shard-1-replica-1:50051` on the
  compose network). Kafka has the same split between listeners and advertised listeners. A
  test attaches a server with a wrong peer address and checks that repair fails, which shows
  that the peer address is the one handed over.
- With Docker the work divides. Docker restarts a server that dies; it does not know that
  the restarted replica missed writes while it was down. Run against the containers, the ops
  agent found "replicas of shard 1 are out of sync (seq 3 vs 4)", repaired it over the
  compose network, and the check came back healthy. Kubernetes has the same gap, which is
  why databases on Kubernetes ship an operator.
- `scripts/docker_smoke.sh` builds and starts the cluster, attaches, ingests, searches, has
  Docker restart a server, checks with `ops check`, and searches again; CI runs it on every
  push. It uses a Compose project of its own, so it never touches a cluster started with plain
  `docker compose up`, and it removes its containers and volumes when it ends.
- Not done: `cluster detach` forgets what was ingested along with the addresses, so attaching
  the same servers again starts without that record (ingest then refuses, because the servers
  are not empty). The agent cannot read container logs.
- Each check was verified by removing it in a copy of the code and confirming a test fails:
  repair handing over the peer address, refusing to stop or restart attached servers, the
  agent's refusal to restart them, and attach checking that every server answers.
