"""
Sharded search: vectors are spread across several HNSW shards, and a coordinator
fans each query out to every shard and merges their results (scatter-gather).

The coordinator can reach its shards in three ways:
  - "sequential": query one shard after another
  - "threads":    one thread per shard (CPU-bound search is limited by the GIL)
  - "processes":  one long-lived worker process per shard that owns that shard's
                  index, so searches run truly in parallel and only the query and
                  the k results cross process boundaries
"""
import heapq
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import numpy as np

from .hnsw import HNSW

MODES = ("sequential", "threads", "processes")


class Shard:
    """One partition of the data: an HNSW index plus a map from its local ids to global ids."""

    def __init__(self, **hnsw_params):
        self.index = HNSW(**hnsw_params)
        self.global_ids = []          # local id (insert order) -> global id

    def add(self, global_id, vector):
        self.index.insert(vector)
        self.global_ids.append(global_id)

    def search(self, query, k):
        """This shard's top-k as (distance, global_id) pairs."""
        return [(dist, self.global_ids[local_id])
                for dist, local_id in self.index.search_with_distances(query, k)]

    def __len__(self):
        return len(self.global_ids)


def _serve_shard(conn, hnsw_params):
    """Worker-process loop: own one shard and answer requests arriving on `conn`."""
    shard = Shard(**hnsw_params)
    while True:
        op, *args = conn.recv()
        if op == "add":
            global_ids, vectors = args
            for global_id, vector in zip(global_ids, vectors):
                shard.add(global_id, vector)
            conn.send(len(shard))
        elif op == "search":
            query, k = args
            conn.send(shard.search(query, k))
        elif op == "stop":
            conn.close()
            return


class ShardProcess:
    """Coordinator-side handle to a shard that lives in its own worker process."""

    def __init__(self, ctx, hnsw_params):
        self.conn, child_conn = ctx.Pipe()
        self.process = ctx.Process(target=_serve_shard, args=(child_conn, hnsw_params),
                                   daemon=True)
        self.process.start()
        child_conn.close()

    def stop(self):
        self.conn.send(("stop",))
        self.process.join(timeout=5)


class Coordinator:
    """Routes vectors to shards and answers queries by scatter-gather + top-k merge."""

    def __init__(self, num_shards, mode="sequential", seed=None, **hnsw_params):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.size = 0
        self._closed = False
        shard_params = [dict(hnsw_params, seed=None if seed is None else seed + i)
                        for i in range(num_shards)]

        if mode == "processes":
            # "spawn" gives every worker a fresh interpreter and behaves the same on every OS.
            ctx = mp.get_context("spawn")
            self.shards = [ShardProcess(ctx, params) for params in shard_params]
        else:
            self.shards = [Shard(**params) for params in shard_params]
        self._pool = ThreadPoolExecutor(max_workers=num_shards) if mode == "threads" else None

    def add(self, vectors):
        # Round-robin: global id g goes to shard g % num_shards. Similarity search has no
        # lookup key to route on, so every query has to visit every shard anyway.
        batches = [([], []) for _ in self.shards]
        for v in vectors:
            ids, vecs = batches[self.size % len(self.shards)]
            ids.append(self.size)
            vecs.append(v)
            self.size += 1

        if self.mode == "processes":
            # Hand every worker its batch before waiting on any, so the shards build in parallel.
            for shard, (ids, vecs) in zip(self.shards, batches):
                shard.conn.send(("add", ids, np.asarray(vecs, dtype=np.float32)))
            for shard in self.shards:
                shard.conn.recv()
        else:
            for shard, (ids, vecs) in zip(self.shards, batches):
                for global_id, v in zip(ids, vecs):
                    shard.add(global_id, v)
        return self

    def search(self, query, k=10):
        # Scatter: every shard returns its FULL local top-k, not k / num_shards,
        # because the global top-k can all live in a single shard.
        if self.mode == "sequential":
            per_shard = [shard.search(query, k) for shard in self.shards]
        elif self.mode == "threads":
            futures = [self._pool.submit(shard.search, query, k) for shard in self.shards]
            per_shard = [f.result() for f in futures]
        else:
            # Send the query to every worker before reading any reply, so all shards
            # search at the same time.
            for shard in self.shards:
                shard.conn.send(("search", query, k))
            per_shard = [shard.conn.recv() for shard in self.shards]

        # Gather + merge: the global top-k is simply the k closest candidates overall.
        merged = heapq.nsmallest(k, chain.from_iterable(per_shard))
        return [global_id for _, global_id in merged]

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.mode == "processes":
            for shard in self.shards:
                shard.stop()
        if self._pool is not None:
            self._pool.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
