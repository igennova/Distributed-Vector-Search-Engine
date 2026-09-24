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