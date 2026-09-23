"""
Sharded search: vectors are spread across several HNSW shards, and a coordinator
fans each query out to every shard and merges their results (scatter-gather).
"""
import heapq
from hnsw import HNSW


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


class Coordinator:
    """Routes vectors to shards and answers queries by scatter-gather + top-k merge."""

    def __init__(self, num_shards, seed=None, **hnsw_params):
        self.shards = [Shard(seed=None if seed is None else seed + i, **hnsw_params)
                       for i in range(num_shards)]
        self.size = 0

    def _shard_for(self, global_id):
        # Round-robin keeps shards balanced. Similarity search has no lookup key to
        # route on, so every query has to visit every shard anyway.
        return self.shards[global_id % len(self.shards)]

    def add(self, vectors):
        for v in vectors:
            self._shard_for(self.size).add(self.size, v)
            self.size += 1
        return self

    def search(self, query, k=10):
        # Scatter: every shard returns its FULL local top-k, not k / num_shards,
        # because the global top-k can all live in a single shard.
        # (Shards are queried one after another here; querying them concurrently
        # is what turns sharding into a latency win.)
        candidates = []
        for shard in self.shards:
            candidates.extend(shard.search(query, k))

        # Gather + merge: the global top-k is simply the k closest candidates overall.
        return [global_id for _, global_id in heapq.nsmallest(k, candidates)]
