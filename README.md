# Distributed Vector Search Engine

A small, educational distributed ANN (approximate nearest neighbor) search engine, built
from scratch to understand indexing, sharding, routing, replication, and distributed query
execution. **Not** a Qdrant/Milvus clone — the goal is understanding, not feature parity.

## The learning contract
- **I implement the search algorithms** (`search.py`) — that's where the learning is.
- The dataset oracle, benchmark, and tests are provided so my progress is always measurable.
- Rule: I don't move to the next phase until the current one is benchmarked and I can explain *why* it works.

## Setup
```bash
cd vector-search-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 0 — exact baseline (START HERE)
1. Implement `brute_force_search` in `search.py`.
2. `python test_search.py`   → must print "All tests passed".
3. `python benchmark.py`     → brute force is the oracle, so **recall@10 must be 1.000**.
   Note the latency/QPS — this is the bar every index has to beat on speed while keeping recall high.

## Roadmap
- [x] Phase 0 — brute-force cosine baseline + benchmark harness
- [ ] Phase 1 — HNSW index (graph-based ANN), single node. Goal: recall ≥ 0.95 at a fraction of brute-force latency.
- [ ] Phase 2 — persistence (serialize/load the index, mmap)
- [ ] Phase 3 — gRPC API around a single shard
- [ ] Phase 4 — sharding: N shard processes + a coordinator (scatter-gather)
- [ ] Phase 5 — parallel fan-out + top-K merge across shards
- [ ] Phase 6 — replication
- [ ] Phase 7 — node-failure handling
- [ ] Phase 8 — WAL / snapshots for durability
- [ ] Phase 9 — benchmark recall/latency/QPS across growing dataset sizes
- [ ] Phase 10 — Docker / Kubernetes deployment

## DECISIONS.md
For each phase, write down: the tradeoff faced, what I chose, and why. That file doubles as
the resume/interview evidence — and it keeps the reasoning mine, not the AI's.
```
