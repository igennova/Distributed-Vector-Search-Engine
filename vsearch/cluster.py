"""
Sharded search: vectors are spread across several HNSW shards, and a coordinator
fans each query out to every shard and merges their results (scatter-gather).

The coordinator can reach its shards in three ways:
  - "sequential": query one shard after another
  - "threads":    one thread per shard (CPU-bound search is limited by the GIL)
  - "processes":  one long-lived worker process per shard that owns that shard's
                  index, so searches run truly in parallel and only the query and
                  the k results cross process boundaries

Each vector can carry a payload: any JSON-serializable dict (the text it was made from, its
source, ...). Shards store payloads as JSON text and hand them back with search results.
"""
import heapq
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import NamedTuple

import numpy as np

from .hnsw import HNSW

MODES = ("sequential", "threads", "processes")


class Hit(NamedTuple):
    """One search result."""
    id: int
    distance: float
    payload: dict | None      # what was stored with the vector, or None


def encode_payloads(payloads, count):
    """Payloads as they are stored and sent: one JSON string per vector, "" for no payload."""
    if payloads is None:
        return [""] * count
    if len(payloads) != count:
        raise ValueError(f"got {len(payloads)} payloads for {count} vectors")
    return ["" if payload is None else json.dumps(payload, ensure_ascii=False)
            for payload in payloads]


def decode_payload(text):
    return json.loads(text) if text else None


def route_round_robin(vectors, num_shards, start_id=0, payloads=None):
    """
    Split vectors across shards round-robin by global id (global id g goes to shard
    g % num_shards). Similarity search has no lookup key to route on, so every query
    has to visit every shard anyway; round-robin just keeps the shards balanced.
    Returns one (global_ids, vectors, payloads) triple per shard; `payloads` are the
    encoded payloads, one per vector, and travel with their vectors.
    """
    batches = [([], [], []) for _ in range(num_shards)]
    for offset, v in enumerate(vectors):
        global_id = start_id + offset
        ids, vecs, texts = batches[global_id % num_shards]
        ids.append(global_id)
        vecs.append(v)
        texts.append(payloads[offset] if payloads is not None else "")
    return batches


def merge_top_k_scored(per_shard, k):
    """Global top-k, closest first, from each shard's results.

    Results are tuples starting with (distance, global_id), so they sort by distance and
    then by id. Each shard must return its full local top-k (not k / num_shards), because
    the global top-k can all live in a single shard.
    """
    return heapq.nsmallest(k, chain.from_iterable(per_shard))


def merge_top_k(per_shard, k):
    """Global top-k ids from each shard's (distance, global_id) results."""
    return [global_id for _, global_id in merge_top_k_scored(per_shard, k)]


class Shard:
    """One partition of the data: an HNSW index, plus each vector's global id and payload."""

    def __init__(self, **hnsw_params):
        self.index = HNSW(**hnsw_params)
        self.global_ids = []          # local id (insert order) -> global id
        self.payloads = []            # local id -> payload as JSON text, "" for none

    def add(self, global_id, vector, payload=""):
        self.index.insert(vector)
        self.global_ids.append(global_id)
        self.payloads.append(payload)

    def add_batch(self, global_ids, vectors, payloads=None):
        payloads = payloads or [""] * len(global_ids)
        for global_id, vector, payload in zip(global_ids, vectors, payloads):
            self.add(int(global_id), vector, payload)

    def search(self, query, k):
        """This shard's top-k as (distance, global_id) pairs."""
        return [(dist, self.global_ids[local_id])
                for dist, local_id in self.index.search_with_distances(query, k)]

    def search_with_payloads(self, query, k):
        """This shard's top-k as (distance, global_id, payload_text) triples."""
        return [(dist, self.global_ids[local_id], self.payloads[local_id])
                for dist, local_id in self.index.search_with_distances(query, k)]

    def __len__(self):
        return len(self.global_ids)


def _serve_shard(conn, hnsw_params):
    """Worker-process loop: own one shard and answer requests arriving on `conn`."""
    shard = Shard(**hnsw_params)
    while True:
        op, *args = conn.recv()
        if op == "add":
            global_ids, vectors, payloads = args
            shard.add_batch(global_ids, vectors, payloads)
            conn.send(len(shard))
        elif op == "search":
            query, k, with_payloads = args
            conn.send(shard.search_with_payloads(query, k) if with_payloads
                      else shard.search(query, k))
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

    def add(self, vectors, payloads=None):
        """Index vectors, optionally each with a payload (a JSON-serializable dict, or None)."""
        batches = route_round_robin(vectors, len(self.shards), start_id=self.size,
                                    payloads=encode_payloads(payloads, len(vectors)))
        self.size += len(vectors)

        if self.mode == "processes":
            # Hand every worker its batch before waiting on any, so the shards build in parallel.
            for shard, (ids, vecs, texts) in zip(self.shards, batches):
                shard.conn.send(("add", ids, np.asarray(vecs, dtype=np.float32), texts))
            for shard in self.shards:
                shard.conn.recv()
        else:
            for shard, (ids, vecs, texts) in zip(self.shards, batches):
                shard.add_batch(ids, vecs, texts)
        return self

    def search(self, query, k=10):
        """Ids of the k nearest vectors."""
        return merge_top_k(self._scatter(query, k, with_payloads=False), k)

    def search_hits(self, query, k=10):
        """The k nearest vectors as Hit(id, distance, payload), closest first."""
        merged = merge_top_k_scored(self._scatter(query, k, with_payloads=True), k)
        return [Hit(global_id, dist, decode_payload(text)) for dist, global_id, text in merged]

    def _scatter(self, query, k, with_payloads):
        """Ask every shard for its full local top-k."""
        if self.mode == "sequential":
            return [shard.search_with_payloads(query, k) if with_payloads
                    else shard.search(query, k) for shard in self.shards]
        if self.mode == "threads":
            futures = [self._pool.submit(shard.search_with_payloads if with_payloads
                                         else shard.search, query, k) for shard in self.shards]
            return [f.result() for f in futures]
        # Send the query to every worker before reading any reply, so all shards search
        # at the same time.
        for shard in self.shards:
            shard.conn.send(("search", query, k, with_payloads))
        return [shard.conn.recv() for shard in self.shards]

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
