"""
Long-term memory for an agent, kept in the cluster. A memory is one fact in plain words
("prefers short answers"), embedded and stored with its text, tags, and the time it was
written, and recalled later by meaning.

    remember(text)           write a fact down; returns its id
    recall(query)            the facts that matter for a query, best first
    forget(id)               take a fact out of use

Each owner's memories form one namespace, so they live on one shard and a recall asks only
that shard. A cluster holds either documents or memories, never both.

Facts change, so a memory can name the memories it replaces, and recall leaves replaced ones
out. Newer facts usually matter more, so recall adds a bonus for recency on top of
similarity. The engine has no delete: replacing and forgetting work by writing a record that
names its target, and recall skips targets. A forgotten fact is no longer recalled, but its
text is still on disk, in the write-ahead log and the snapshots.
"""
import time
import uuid
from pathlib import Path
from typing import NamedTuple

import grpc
import numpy as np

from .client import ReplicasOutOfSyncError, ShardUnavailableError
from .documents import DEFAULT_EMBEDDER
from .embedding import get_embedder
from .state import UserError, connect, read_state, writable, write_state

DUPLICATE_SIMILARITY = 0.97   # a new fact this close to a stored one is the same fact
RECENCY_WEIGHT = 0.03         # the most a fact's age can add to its score
HALF_LIFE_DAYS = 30           # the recency bonus halves every 30 days
CANDIDATES = 4                # recall re-ranks k x 4 nearest facts by recency
MAX_LOOKUP = 1000             # most records one bookkeeping lookup returns


class Memory(NamedTuple):
    id: str
    text: str
    tags: list
    created: float            # seconds since the epoch
    similarity: float         # to the query, 0 to 1
    score: float              # similarity plus the recency bonus; what recall sorts by


def describe_age(seconds):
    """'just now', '5 minutes ago', '3 hours ago', '12 days ago'."""
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size:
            count = int(seconds // size)
            return f"{count} {unit}{'s' if count > 1 else ''} ago"
    return "just now"


class MemoryStore:
    """The memories of one owner in the cluster at `home`. Keeps its coordinator and
    embedder between calls; close() releases them."""

    def __init__(self, home, owner="default", embedder=None, embedders=get_embedder,
                 clock=time.time):
        if not isinstance(owner, str) or not owner.strip() or len(owner) > 128:
            raise UserError("an owner is a name of 1 to 128 characters")
        self.home = Path(home)
        self.owner = owner
        self.default_embedder = embedder        # for a memory this store creates
        self._embedders = embedders
        self._clock = clock
        self._embedder_cache = {}
        self._coordinator = None
        self._addresses = None

    # --- the three operations ----------------------------------------------------------------

    def remember(self, text, tags=(), replaces=()):
        """Store a fact. Returns (id, stored): stored is False when an almost identical fact
        is already remembered, in which case its id is returned instead."""
        text = (text or "").strip()
        if not text:
            raise UserError("there is nothing to remember: the text is empty")
        state = self._state()
        replies = writable(state)
        dataset = self._dataset(state, replies)
        coordinator = self._connect(state)
        replaces = list(dict.fromkeys(replaces))
        missing = set(replaces) - set(self._live_ids(state, replaces))
        if missing:
            raise UserError(f"no current memory has the id(s) {sorted(missing)}; "
                            "recall shows the ids")

        vector = self._embedder(dataset).embed_documents([text])[0]
        if not replaces:
            for hit in self._search(coordinator, vector, 1, {"type": "memory"}):
                same = 1.0 - hit.distance >= DUPLICATE_SIMILARITY
                if same and not self._replaced(state, [hit.payload["memory_id"]]):
                    return hit.payload["memory_id"], False

        memory_id = "m" + uuid.uuid4().hex[:10]
        self._write(coordinator, vector, {
            "type": "memory", "memory_id": memory_id, "text": text,
            "tags": sorted(set(tags)), "created": self._clock(), "replaces": replaces})
        return memory_id, True

    def recall(self, query, k=5, tags=None):
        """The k facts that best match the query, as Memory tuples, best first. Replaced and
        forgotten facts are left out; newer facts get a bonus of up to RECENCY_WEIGHT."""
        state = self._state()
        dataset = state.get("dataset")
        if not dataset:
            return []
        self._check_kind(dataset)
        where = {"type": "memory"}
        if tags:
            where["tags"] = list(tags)
        query_vector = self._embedder(dataset).embed_query(query)
        hits = self._search(self._connect(state), query_vector, k * CANDIDATES, where)
        replaced = self._replaced(state, [hit.payload["memory_id"] for hit in hits])
        now = self._clock()
        memories = []
        for hit in hits:
            payload = hit.payload
            if payload["memory_id"] in replaced:
                continue
            similarity = 1.0 - hit.distance
            age_days = max(0.0, now - payload["created"]) / 86400
            score = similarity + RECENCY_WEIGHT * 0.5 ** (age_days / HALF_LIFE_DAYS)
            memories.append(Memory(payload["memory_id"], payload["text"], payload["tags"],
                                   payload["created"], similarity, score))
        return sorted(memories, key=lambda memory: -memory.score)[:k]

    def forget(self, memory_id):
        """Stop recalling a fact. Its text stays on disk: the engine cannot delete."""
        state = self._state()
        replies = writable(state)
        dataset = self._dataset(state, replies)
        if memory_id not in self._live_ids(state, [memory_id]):
            raise UserError(f"no current memory has the id {memory_id!r}")
        # The record only has to be found by what it replaces, so its vector does not matter.
        self._write(self._connect(state), np.ones(dataset["dim"], dtype=np.float32), {
            "type": "forget", "memory_id": "f" + uuid.uuid4().hex[:10],
            "created": self._clock(), "replaces": [memory_id]})

    def close(self):
        if self._coordinator is not None:
            self._coordinator.close()
            self._coordinator = self._addresses = None

    # --- bookkeeping -------------------------------------------------------------------------

    def _live_ids(self, state, ids):
        """Those of `ids` that name a memory that is stored and not replaced or forgotten."""
        if not ids:
            return []
        found = [hit.payload["memory_id"] for hit in self._lookup(
            state, {"type": "memory", "memory_id": list(ids)})]
        replaced = self._replaced(state, found)
        return [memory_id for memory_id in found if memory_id not in replaced]

    def _replaced(self, state, ids):
        """Which of `ids` a later memory or a forget record names as replaced."""
        if not ids:
            return set()
        records = self._lookup(state, {"replaces": list(ids)})
        return {target for hit in records for target in hit.payload["replaces"]} & set(ids)

    def _lookup(self, state, where):
        """Every record matching the filter (up to MAX_LOOKUP), whatever its vector. With a
        filter, a shard compares the query with each matching record, so the query vector
        only orders the answer."""
        dataset = state.get("dataset")
        if not dataset:
            return []
        query = np.ones(dataset["dim"], dtype=np.float32)
        return self._search(self._connect(state), query, MAX_LOOKUP, where)

    def _search(self, coordinator, vector, k, where):
        try:
            return coordinator.search_hits(vector, k, where=where, namespace=self.owner)
        except ShardUnavailableError as err:
            raise UserError(f"{err}; see: vsearch status") from None

    def _write(self, coordinator, vector, payload):
        coordinator.refresh()          # another process may have written since the last time
        try:
            coordinator.add(np.asarray([vector], dtype=np.float32), [payload],
                            namespace=self.owner)
        except (grpc.RpcError, ReplicasOutOfSyncError) as err:
            raise UserError(f"the memory was not stored: {err}. Check `vsearch status`, then "
                            "`vsearch repair`") from None

    def _state(self):
        state = read_state(self.home)
        if state.get("dataset"):
            self._check_kind(state["dataset"])
        return state

    def _check_kind(self, dataset):
        if dataset["name"] != "memory":
            raise UserError(f"this cluster holds {dataset['name']}; keep memories in a cluster "
                            "of their own (vsearch --home data/memory cluster up)")

    def _dataset(self, state, replies):
        """The cluster's memory record, created on the first write."""
        if state.get("dataset"):
            return state["dataset"]
        if any(reply.size for reply in replies):
            raise UserError("this cluster already holds vectors; keep memories in a cluster "
                            "of their own")
        embedder = self._embedder_by_name(self.default_embedder or DEFAULT_EMBEDDER)
        state["dataset"] = {"name": "memory", "embedder": embedder.name, "dim": embedder.dim}
        write_state(self.home, state)
        return state["dataset"]

    def _embedder(self, dataset):
        return self._embedder_by_name(dataset["embedder"])

    def _embedder_by_name(self, name):
        if name not in self._embedder_cache:
            self._embedder_cache[name] = self._embedders(name)
        return self._embedder_cache[name]

    def _connect(self, state):
        addresses = [[server["port"] for server in group] for group in state["servers"]]
        if addresses != self._addresses:
            self.close()
            self._coordinator, self._addresses = connect(state), addresses
        return self._coordinator
