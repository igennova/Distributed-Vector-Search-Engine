"""
Durable shards: a snapshot of the whole index plus a write-ahead log (WAL) of every
write since that snapshot.

Write path:  append the write to the WAL (and fsync) -> apply it in memory -> reply.
Recovery:    load the snapshot -> replay WAL records newer than it -> cut off a torn tail.
Checkpoint:  write a new snapshot (atomically) -> empty the WAL.

WAL record layout (little-endian):
    [payload length: u32][crc32 of payload: u32]
    payload = [seq: u64][count: u32][dim: u32][global ids: count x i64][vectors: count*dim x f32]
"""
import json
import os
import struct
import zlib
from pathlib import Path

import numpy as np

from .cluster import Shard
from .hnsw import HNSW

SNAPSHOT_FORMAT = 1
_HEADER = struct.Struct("<II")         # payload length, crc32
_RECORD_START = struct.Struct("<QII")  # seq, count, dim


class OutOfOrderWrite(ValueError):
    """A write arrived with a sequence number other than this shard's next one."""


def _fsync_dir(path):
    """Make a rename inside `path` durable: the directory entry itself must reach disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_snapshot(shard, last_seq, path):
    """Write the shard's full state to `path` atomically.

    The snapshot goes to a temporary file, is fsynced, then renamed over the old one.
    A rename either happens completely or not at all, so a crash mid-save leaves the
    previous snapshot intact.
    """
    path = Path(path)
    index = shard.index
    arrays = {
        "vectors": (np.stack(index.vectors).astype(np.float32) if index.vectors
                    else np.zeros((0, 0), dtype=np.float32)),
        "global_ids": np.asarray(shard.global_ids, dtype=np.int64),
    }
    # Each layer's adjacency lists, flattened: node ids, where each node's neighbors
    # start in `neighbors`, and the neighbors themselves.
    for layer, adjacency in enumerate(index.graph):
        nodes = list(adjacency)
        arrays[f"layer{layer}_nodes"] = np.asarray(nodes, dtype=np.int64)
        arrays[f"layer{layer}_offsets"] = np.cumsum(
            [0] + [len(adjacency[n]) for n in nodes], dtype=np.int64)
        arrays[f"layer{layer}_neighbors"] = np.asarray(
            [m for n in nodes for m in adjacency[n]], dtype=np.int64)
    meta = {
        "format": SNAPSHOT_FORMAT,
        "last_seq": last_seq,
        "M": index.M,
        "ef_construction": index.ef_construction,
        "ef_search": index.ef_search,
        "entry_point": index.entry_point,
        "top_layer": index.top_layer,
        "num_layers": len(index.graph),
        # The random generator decides each new node's level. Restoring its exact state
        # lets a recovered replica keep building the same graph as its twin.
        "rng_state": index.rng.bit_generator.state,
    }
    arrays["meta"] = np.array(json.dumps(meta))

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def load_snapshot(path, ef_search=None):
    """Rebuild a Shard from a snapshot. Returns (shard, last_seq).

    ef_search is a query-time setting, so it can be overridden; everything that shaped
    the graph (M, ef_construction, levels) comes from the snapshot.
    """
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta["format"] != SNAPSHOT_FORMAT:
            raise ValueError(f"unsupported snapshot format {meta['format']}")

        index = HNSW(M=meta["M"], ef_construction=meta["ef_construction"],
                     ef_search=meta["ef_search"] if ef_search is None else ef_search)
        index.rng.bit_generator.state = meta["rng_state"]
        index.vectors = list(data["vectors"])
        for layer in range(meta["num_layers"]):
            nodes = data[f"layer{layer}_nodes"].tolist()
            offsets = data[f"layer{layer}_offsets"].tolist()
            neighbors = data[f"layer{layer}_neighbors"].tolist()
            index.graph.append({node: neighbors[offsets[i]:offsets[i + 1]]
                                for i, node in enumerate(nodes)})
        index.entry_point = meta["entry_point"]
        index.top_layer = meta["top_layer"]

        shard = Shard()
        shard.index = index
        shard.global_ids = data["global_ids"].tolist()
    return shard, meta["last_seq"]


def _read_records(path):
    """Parse every intact record. Returns (records, bytes_of_intact_records).

    Parsing stops at the first record that is cut short or fails its checksum: that is
    what a crash in the middle of an append leaves behind. Like Postgres and SQLite, a bad
    record is treated as the end of the log.
    """
    if not path.exists():
        return [], 0
    data = path.read_bytes()
    records, pos = [], 0
    while pos + _HEADER.size <= len(data):
        length, crc = _HEADER.unpack_from(data, pos)
        start, end = pos + _HEADER.size, pos + _HEADER.size + length
        if end > len(data) or length < _RECORD_START.size:
            break
        payload = data[start:end]
        if zlib.crc32(payload) != crc:
            break
        seq, count, dim = _RECORD_START.unpack_from(payload)
        if length != _RECORD_START.size + 8 * count + 4 * count * dim:
            break
        offset = _RECORD_START.size
        ids = np.frombuffer(payload, dtype="<i8", count=count, offset=offset)
        vectors = np.frombuffer(payload, dtype="<f4", count=count * dim,
                                offset=offset + 8 * count).reshape(count, dim)
        records.append((seq, ids.tolist(), vectors.copy()))
        pos = end
    return records, pos


class WriteAheadLog:
    """Append-only log of writes. On open, reads back the intact records and cuts off a torn tail."""

    def __init__(self, path, fsync=True):
        self.path = Path(path)
        self.fsync = fsync
        self.recovered, intact_bytes = _read_records(self.path)
        if self.path.exists() and self.path.stat().st_size > intact_bytes:
            with open(self.path, "r+b") as f:
                f.truncate(intact_bytes)
                f.flush()
                os.fsync(f.fileno())
        self.file = open(self.path, "ab")

    def append(self, seq, global_ids, vectors):
        ids = np.asarray(global_ids, dtype="<i8")
        vectors = np.ascontiguousarray(vectors, dtype="<f4")
        count, dim = vectors.shape
        payload = _RECORD_START.pack(seq, count, dim) + ids.tobytes() + vectors.tobytes()
        self.file.write(_HEADER.pack(len(payload), zlib.crc32(payload)) + payload)
        self.file.flush()             # hand the bytes to the OS
        if self.fsync:
            os.fsync(self.file.fileno())   # and make the OS put them on disk now

    def reset(self):
        """Empty the log; called once a snapshot covers every record in it."""
        self.file.truncate(0)
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


class DurableShard:
    """A Shard whose acknowledged writes survive a crash, stored in `data_dir`.

    snapshot_every: take a snapshot (and empty the log) once this many vectors have been
    written since the last one; 0 means never. Fewer snapshots mean cheaper writes but a
    longer log to replay on restart.
    """

    def __init__(self, data_dir, fsync=True, snapshot_every=1000, **hnsw_params):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = self.data_dir / "snapshot.npz"
        self.snapshot_every = snapshot_every

        if self.snapshot_path.exists():
            self.shard, self.last_seq = load_snapshot(self.snapshot_path,
                                                      ef_search=hnsw_params.get("ef_search"))
        else:
            self.shard, self.last_seq = Shard(**hnsw_params), 0
        self.snapshot_seq = self.last_seq

        # Replay whatever the log holds beyond the snapshot. Records at or below the
        # snapshot's seq are already in it: that happens when a crash lands between
        # writing a snapshot and emptying the log.
        self.wal = WriteAheadLog(self.data_dir / "wal.log", fsync=fsync)
        self.replayed_records = 0
        self.vectors_since_snapshot = 0
        for seq, ids, vectors in self.wal.recovered:
            if seq <= self.last_seq:
                continue
            if seq != self.last_seq + 1:
                raise RuntimeError(f"write-ahead log jumps from seq {self.last_seq} to {seq}")
            self.shard.add_batch(ids, vectors)
            self.last_seq = seq
            self.replayed_records += 1
            self.vectors_since_snapshot += len(ids)
        self.wal.recovered = None

    def add_batch(self, global_ids, vectors, seq=None):
        """Log and apply one write. If `seq` is given it must be exactly the next one, so the
        same write gets the same number on every replica (what log catch-up relies on)."""
        expected = self.last_seq + 1
        if seq is not None and seq != expected:
            raise OutOfOrderWrite(
                f"write seq {seq} does not follow this replica's last seq {self.last_seq}")
        seq = expected
        self.wal.append(seq, global_ids, vectors)   # durable before anything else happens
        self.shard.add_batch(global_ids, vectors)
        self.last_seq = seq
        self.vectors_since_snapshot += len(global_ids)
        if self.snapshot_every and self.vectors_since_snapshot >= self.snapshot_every:
            self.checkpoint()

    def checkpoint(self):
        """Snapshot everything written so far, then empty the log."""
        save_snapshot(self.shard, self.last_seq, self.snapshot_path)
        self.wal.reset()
        self.snapshot_seq = self.last_seq
        self.vectors_since_snapshot = 0

    def records_after(self, after_seq):
        """Log records with seq > after_seq, or None if the log no longer holds all of them
        (they were folded into the snapshot and the log was emptied)."""
        if after_seq < self.snapshot_seq:
            return None
        records, _ = _read_records(self.wal.path)
        return [record for record in records if record[0] > after_seq]

    def snapshot_bytes(self):
        """The current snapshot file and the last seq it covers, taking one first if needed."""
        if not self.snapshot_path.exists():
            self.checkpoint()
        return self.snapshot_path.read_bytes(), self.snapshot_seq

    def install_snapshot(self, data):
        """Replace this shard's whole state with a snapshot copied from a peer."""
        tmp = self.snapshot_path.with_name(self.snapshot_path.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.snapshot_path)
        _fsync_dir(self.data_dir)
        # Any log records left over are older than the new snapshot, so recovery would skip
        # them anyway; emptying the log just keeps it tidy.
        self.wal.reset()
        self.shard, self.last_seq = load_snapshot(self.snapshot_path,
                                                  ef_search=self.shard.index.ef_search)
        self.snapshot_seq = self.last_seq
        self.vectors_since_snapshot = 0

    def search(self, query, k):
        return self.shard.search(query, k)

    def __len__(self):
        return len(self.shard)

    def close(self):
        self.wal.close()
