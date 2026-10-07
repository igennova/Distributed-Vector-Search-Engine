"""
Catching up a replica over gRPC: copying just the missing writes from a peer's log, for
growing gaps, vs copying the peer's whole snapshot.

Run:  python -m benchmarks.bench_resync
"""
import tempfile
import time
from pathlib import Path

import grpc
from vsearch.dataset import make_dataset
from vsearch.protos import shard_pb2, shard_pb2_grpc
from vsearch.server import serve_shard
from vsearch.storage import DurableShard, WriteAheadLog

PARAMS = dict(M=16, ef_construction=100, ef_search=50, seed=0)
N, BATCH = 2_500, 50           # one shard of a 10k, 4-shard index, written 50 vectors at a time


def time_sync(follower_dir, leader_address):
    """Serve a follower that recovers from follower_dir, and time a SyncFrom the leader."""
    follower = DurableShard(follower_dir, **PARAMS)
    server, port = serve_shard(follower)
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = shard_pb2_grpc.ShardServiceStub(channel)
            start = time.perf_counter()
            result = stub.SyncFrom(shard_pb2.SyncFromRequest(peer=leader_address), timeout=600)
            return time.perf_counter() - start, result
    finally:
        server.stop(grace=None)
        follower.close()


def main():
    vectors, _ = make_dataset(n_vectors=N)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        leader = DurableShard(tmp / "leader", snapshot_every=0, **PARAMS)
        for start in range(0, N, BATCH):
            leader.add_batch(list(range(start, start + BATCH)), vectors[start:start + BATCH])
        records = leader.records_after(0)
        server, port = serve_shard(leader)
        leader_address = f"127.0.0.1:{port}"

        print(f"Catching up a replica of a {N:,}-vector shard over gRPC (localhost)")
        print(f"  {'missed':<14}{'method':<10}{'records':>8}{'time':>10}")
        for missed in [50, 500, 2_000]:
            # A follower that has every write except the last `missed` vectors.
            follower_dir = tmp / f"follower-{missed}"
            follower_dir.mkdir()
            wal = WriteAheadLog(follower_dir / "wal.log", fsync=False)
            for seq, ids, vecs, payloads in records[:(N - missed) // BATCH]:
                wal.append(seq, ids, vecs, payloads)
            wal.close()
            elapsed, result = time_sync(follower_dir, leader_address)
            print(f"  {missed:>5,} vectors {result.method:<10}{result.records_applied:>8}"
                  f"{elapsed:>9.3f}s")

        # Once the leader has snapshotted, its log is empty, so an empty follower has to
        # copy the snapshot.
        leader.checkpoint()
        snapshot_mb = (tmp / "leader" / "snapshot.npz").stat().st_size / 1e6
        elapsed, result = time_sync(tmp / "follower-empty", leader_address)
        print(f"  {'everything':<14}{result.method:<10}{result.records_applied:>8}"
              f"{elapsed:>9.3f}s   ({snapshot_mb:.1f} MB copied)")

        server.stop(grace=None)
        leader.close()


if __name__ == "__main__":
    main()
