"""
What durability costs: log appends with and without fsync, writes into a shard with the
log in front of them, and restart time from a long log vs from a snapshot.

Run:  python -m benchmarks.bench_persistence
"""
import tempfile
import time
from pathlib import Path

import numpy as np
from vsearch.cluster import Shard
from vsearch.dataset import make_dataset
from vsearch.storage import DurableShard, WriteAheadLog

try:
    import fcntl
    FULLFSYNC = getattr(fcntl, "F_FULLFSYNC", None)   # macOS: flush the drive's own cache too
except ImportError:
    FULLFSYNC = None

PARAMS = dict(M=16, ef_construction=100, ef_search=50, seed=0)
DIM = 128


def log_appends_per_second(mode, n=2_000):
    """One 128-dim vector per record, no index work: the raw cost of the log."""
    vector = np.zeros((1, DIM), dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        wal = WriteAheadLog(Path(d) / "wal.log", fsync=(mode == "fsync"))
        start = time.perf_counter()
        for seq in range(1, n + 1):
            wal.append(seq, [seq], vector)
            if mode == "full":
                fcntl.fcntl(wal.file.fileno(), FULLFSYNC)
        elapsed = time.perf_counter() - start
        wal.close()
    return n / elapsed


def shard_writes_per_second(vectors, durable, fsync=True):
    """One vector per write into a shard: the log plus the HNSW insert."""
    with tempfile.TemporaryDirectory() as d:
        shard = (DurableShard(d, fsync=fsync, snapshot_every=0, **PARAMS) if durable
                 else Shard(**PARAMS))
        start = time.perf_counter()
        for i, vector in enumerate(vectors):
            shard.add_batch([i], vector[None, :])
        elapsed = time.perf_counter() - start
        if durable:
            shard.close()
    return len(vectors) / elapsed


def restart_times(vectors, batch=100):
    """Reopen a shard whose writes are all in the log, then again after a snapshot."""
    with tempfile.TemporaryDirectory() as d:
        shard = DurableShard(d, fsync=False, snapshot_every=0, **PARAMS)
        for start in range(0, len(vectors), batch):
            shard.add_batch(list(range(start, start + batch)), vectors[start:start + batch])
        shard.close()
        log_mb = (Path(d) / "wal.log").stat().st_size / 1e6

        t0 = time.perf_counter()
        shard = DurableShard(d, fsync=False, snapshot_every=0, **PARAMS)   # replays every record
        from_log = time.perf_counter() - t0

        shard.checkpoint()
        shard.close()
        snapshot_mb = (Path(d) / "snapshot.npz").stat().st_size / 1e6

        t0 = time.perf_counter()
        DurableShard(d, fsync=False, snapshot_every=0, **PARAMS).close()   # loads the snapshot
        from_snapshot = time.perf_counter() - t0
    return from_log, log_mb, from_snapshot, snapshot_mb


def main():
    vectors, _ = make_dataset(n_vectors=2_500)       # one shard's worth of a 10k, 4-shard index

    print("Log appends (one 128-dim vector per record, no index work)")
    modes = [("no fsync", "off"), ("fsync", "fsync")]
    if FULLFSYNC is not None:
        modes.append(("F_FULLFSYNC (macOS)", "full"))
    for label, mode in modes:
        rate = log_appends_per_second(mode)
        print(f"  {label:<22}{rate:>10,.0f} appends/s   {1e6 / rate:>8.1f} µs each")

    print("\nWrites into a shard (one vector per write: log + HNSW insert)")
    sample = vectors[:1_000]
    for label, durable, fsync in [("in memory only", False, False),
                                  ("durable, no fsync", True, False),
                                  ("durable, fsync", True, True)]:
        rate = shard_writes_per_second(sample, durable, fsync)
        print(f"  {label:<22}{rate:>10,.0f} writes/s")

    from_log, log_mb, from_snapshot, snapshot_mb = restart_times(vectors)
    print(f"\nRestart of a {len(vectors):,}-vector shard")
    print(f"  replay the whole log  {from_log * 1000:>8.1f} ms   (log: {log_mb:.2f} MB)")
    print(f"  load a snapshot       {from_snapshot * 1000:>8.1f} ms   (snapshot: {snapshot_mb:.2f} MB)")


if __name__ == "__main__":
    main()
