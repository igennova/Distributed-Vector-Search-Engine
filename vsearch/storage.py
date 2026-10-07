"""
Durable shards: a snapshot of the whole index plus a write-ahead log (WAL) of every
write since that snapshot.

Write path:  append the write to the WAL (and fsync) -> apply it in memory -> reply.
Recovery:    load the snapshot -> replay WAL records newer than it -> cut off a torn tail.
Checkpoint:  write a new snapshot (atomically) -> empty the WAL.

WAL record layout (little-endian):
    [body length: u32][crc32 of body: u32]
    body = [seq: u64][count: u32][dim: u32][global ids: count x i64][vectors: count*dim x f32]
           then, only when some vector in the record has a payload:
           count x ([length: u32][payload: UTF-8 JSON text])

A record whose vectors carry no payloads is byte-for-byte what it was before payloads existed,
so logs written by older versions still read back.
"""
import json
import os
import struct
import zlib
from pathlib import Path

import numpy as np

from .cluster import Shard
from .hnsw import HNSW

SNAPSHOT_FORMAT = 3   # 3: adds payloads; 2: vectors stored at unit length; 1: as inserted
_HEADER = struct.Struct("<II")         # body length, crc32
_RECORD_START = struct.Struct("<QII")  # seq, count, dim
_U32 = struct.Struct("<I")


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
        "vectors": np.asarray(index.vectors, dtype=np.float32),
        "global_ids": np.asarray(shard.global_ids, dtype=np.int64),
    }
    # Payloads are variable-length text: one byte blob plus where each one starts.
    encoded = [payload.encode("utf-8") for payload in shard.payloads]
    arrays["payload_offsets"] = np.cumsum([0] + [len(b) for b in encoded], dtype=np.int64)
    arrays["payload_bytes"] = np.frombuffer(b"".join(encoded), dtype=np.uint8)
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
        "neighbor_selection": index.neighbor_selection,
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
        if meta["format"] not in (1, 2, SNAPSHOT_FORMAT):
            raise ValueError(f"unsupported snapshot format {meta['format']}")

        index = HNSW(M=meta["M"], ef_construction=meta["ef_construction"],
                     ef_search=meta["ef_search"] if ef_search is None else ef_search,
                     # snapshots written before this setting existed used "closest"
                     neighbor_selection=meta.get("neighbor_selection", "closest"))
        index.rng.bit_generator.state = meta["rng_state"]
        # From format 2 on, vectors are stored at unit length: keep them bit for bit, so a
        # restored replica keeps building exactly the same graph as its twin.
        index.set_vectors(data["vectors"], normalized=meta["format"] >= 2)
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
        if "payload_offsets" in data.files:
            offsets = data["payload_offsets"].tolist()
            blob = data["payload_bytes"].tobytes()
            shard.payloads = [blob[offsets[i]:offsets[i + 1]].decode("utf-8")
                              for i in range(len(offsets) - 1)]
        else:                                    # snapshot from before payloads existed
            shard.payloads = [""] * len(shard.global_ids)
    return shard, meta["last_seq"]


def _parse_payloads(body, pos, count):
    """The payload section of a record body starting at `pos`, or None if it is malformed."""
    if pos == len(body):
        return [""] * count                      # no payload section: none of them has one
    payloads = []
    for _ in range(count):
        if pos + _U32.size > len(body):
            return None
        (size,) = _U32.unpack_from(body, pos)
        pos += _U32.size
        if pos + size > len(body):
            return None
        payloads.append(body[pos:pos + size].decode("utf-8"))
        pos += size
    return payloads if pos == len(body) else None


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
        body = data[start:end]
        if zlib.crc32(body) != crc:
            break
        seq, count, dim = _RECORD_START.unpack_from(body)
        offset = _RECORD_START.size
        vectors_end = offset + 8 * count + 4 * count * dim
        if length < vectors_end:
            break
        ids = np.frombuffer(body, dtype="<i8", count=count, offset=offset)
        vectors = np.frombuffer(body, dtype="<f4", count=count * dim,
                                offset=offset + 8 * count).reshape(count, dim)
        payloads = _parse_payloads(body, vectors_end, count)
        if payloads is None:
            break
        records.append((seq, ids.tolist(), vectors.copy(), payloads))
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

    def append(self, seq, global_ids, vectors, payloads=None):
        ids = np.asarray(global_ids, dtype="<i8")
        vectors = np.ascontiguousarray(vectors, dtype="<f4")
        count, dim = vectors.shape
        body = _RECORD_START.pack(seq, count, dim) + ids.tobytes() + vectors.tobytes()
        if payloads and any(payloads):
            encoded = [payload.encode("utf-8") for payload in payloads]
            body += b"".join(_U32.pack(len(b)) + b for b in encoded)
        self.file.write(_HEADER.pack(len(body), zlib.crc32(body)) + body)
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
        for seq, ids, vectors, payloads in self.wal.recovered:
            if seq <= self.last_seq:
                continue
            if seq != self.last_seq + 1:
                raise RuntimeError(f"write-ahead log jumps from seq {self.last_seq} to {seq}")
            self.shard.add_batch(ids, vectors, payloads)
            self.last_seq = seq
            self.replayed_records += 1
            self.vectors_since_snapshot += len(ids)
        self.wal.recovered = None

    def add_batch(self, global_ids, vectors, payloads=None, seq=None):
        """Log and apply one write. If `seq` is given it must be exactly the next one, so the
        same write gets the same number on every replica (what log catch-up relies on)."""
        expected = self.last_seq + 1
        if seq is not None and seq != expected:
            raise OutOfOrderWrite(
                f"write seq {seq} does not follow this replica's last seq {self.last_seq}")
        seq = expected
        self.wal.append(seq, global_ids, vectors, payloads)   # durable before anything else
        self.shard.add_batch(global_ids, vectors, payloads)
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

    def search_with_payloads(self, query, k):
        return self.shard.search_with_payloads(query, k)

    def __len__(self):
        return len(self.shard)

    def close(self):
        self.wal.close()
