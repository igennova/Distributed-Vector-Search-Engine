"""
A shard as a network service: one process owns one shard and serves Add and Search
over gRPC, plus the calls replicas use to resync with each other.

Run:  python -m vsearch.server --port 50051 --seed 0
      python -m vsearch.server --port 50051 --seed 0 --data-dir data/shard-0   # survives restarts
"""
import argparse
import threading
from concurrent import futures

import grpc
import numpy as np

from .cluster import Shard
from .protos import shard_pb2, shard_pb2_grpc
from .storage import DurableShard, OutOfOrderWrite

SNAPSHOT_CHUNK_BYTES = 1024 * 1024    # stream snapshots in pieces well under gRPC's 4 MB limit
SYNC_TIMEOUT = 600.0                  # catching up can mean rebuilding a large part of a shard


class ShardServicer(shard_pb2_grpc.ShardServiceServicer):
    def __init__(self, shard):
        self.shard = shard
        self.durable = isinstance(shard, DurableShard)
        # Insert mutates the HNSW graph, so a search must not run in the middle of one.
        # One lock per shard keeps that safe; requests to a shard run one at a time.
        self.lock = threading.Lock()

    def Add(self, request, context):
        vectors = np.asarray(request.values, dtype=np.float32).reshape(-1, request.dim)
        with self.lock:
            # A durable shard logs the batch before applying it, so by the time this
            # reply goes out the write can no longer be lost.
            if not request.seq:
                self.shard.add_batch(list(request.global_ids), vectors)
            elif not self.durable:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                              "numbered writes need a durable shard (--data-dir)")
            else:
                try:
                    self.shard.add_batch(list(request.global_ids), vectors, seq=request.seq)
                except OutOfOrderWrite as err:
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"{err}; run repair()")
            return shard_pb2.AddResponse(size=len(self.shard))

    def Search(self, request, context):
        query = np.asarray(request.query, dtype=np.float32)
        with self.lock:
            hits = self.shard.search(query, request.k)
        return shard_pb2.SearchResponse(
            hits=[shard_pb2.Hit(global_id=global_id, distance=dist) for dist, global_id in hits])

    def Status(self, request, context):
        with self.lock:
            return shard_pb2.StatusResponse(
                size=len(self.shard), durable=self.durable,
                last_seq=self.shard.last_seq if self.durable else 0,
                snapshot_seq=self.shard.snapshot_seq if self.durable else 0)

    def FetchLog(self, request, context):
        self._require_durable(context)
        with self.lock:                       # copy what's needed, then stream without the lock
            records = self.shard.records_after(request.after_seq)
        if records is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          f"log no longer reaches back to seq {request.after_seq + 1}; "
                          "fetch the snapshot instead")
        for seq, ids, vectors in records:
            yield shard_pb2.LogRecord(seq=seq, global_ids=ids, dim=vectors.shape[1],
                                      values=vectors.ravel().tolist())

    def FetchSnapshot(self, request, context):
        self._require_durable(context)
        with self.lock:
            data, last_seq = self.shard.snapshot_bytes()
        for start in range(0, len(data), SNAPSHOT_CHUNK_BYTES):
            yield shard_pb2.SnapshotChunk(data=data[start:start + SNAPSHOT_CHUNK_BYTES],
                                          last_seq=last_seq)

    def SyncFrom(self, request, context):
        self._require_durable(context)
        with self.lock:
            try:
                method, applied = catch_up(self.shard, request.peer)
            except grpc.RpcError as err:
                context.abort(grpc.StatusCode.UNAVAILABLE,
                              f"could not sync from {request.peer}: {err.details()}")
            return shard_pb2.SyncFromResponse(method=method, records_applied=applied,
                                              last_seq=self.shard.last_seq)

    def _require_durable(self, context):
        if not self.durable:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "resync needs a durable shard (--data-dir)")


def catch_up(shard, peer, attempts=3):
    """Bring a durable shard up to date with the replica at `peer`.

    Tries the peer's log first: it only sends the writes this shard is missing. If the peer
    has already folded those writes into its snapshot, copies the snapshot and then asks for
    the log again. Returns (method, log_records_applied), method being "log" or "snapshot".
    """
    method = "log"
    with grpc.insecure_channel(peer) as channel:
        stub = shard_pb2_grpc.ShardServiceStub(channel)
        for _ in range(attempts):
            try:
                applied = 0
                for record in stub.FetchLog(shard_pb2.FetchLogRequest(after_seq=shard.last_seq),
                                            timeout=SYNC_TIMEOUT):
                    vectors = np.asarray(record.values, dtype=np.float32).reshape(-1, record.dim)
                    shard.add_batch(list(record.global_ids), vectors, seq=record.seq)
                    applied += 1
                return method, applied
            except grpc.RpcError as err:
                if err.code() != grpc.StatusCode.FAILED_PRECONDITION:
                    raise
            chunks = stub.FetchSnapshot(shard_pb2.FetchSnapshotRequest(), timeout=SYNC_TIMEOUT)
            shard.install_snapshot(b"".join(chunk.data for chunk in chunks))
            method = "snapshot"
    raise RuntimeError(f"could not catch up with {peer} after {attempts} attempts")


def open_shard(data_dir=None, fsync=True, snapshot_every=1000, **hnsw_params):
    """An in-memory Shard, or a DurableShard recovered from data_dir when one is given."""
    if data_dir is None:
        return Shard(**hnsw_params)
    return DurableShard(data_dir, fsync=fsync, snapshot_every=snapshot_every, **hnsw_params)


def serve_shard(shard, port=0, host="127.0.0.1"):
    """Serve an existing shard in the background. Returns (server, bound_port).

    port=0 lets the OS pick a free port. host defaults to loopback so the unauthenticated
    service is not exposed to the network; pass host="0.0.0.0" to accept remote clients.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    shard_pb2_grpc.add_ShardServiceServicer_to_server(ShardServicer(shard), server)
    bound_port = server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server, bound_port


def start_server(port=0, host="127.0.0.1", data_dir=None, fsync=True, snapshot_every=1000,
                 **hnsw_params):
    """Open (or recover) a shard and serve it in the background. Returns (server, bound_port)."""
    shard = open_shard(data_dir, fsync, snapshot_every, **hnsw_params)
    return serve_shard(shard, port, host)


def main():
    parser = argparse.ArgumentParser(description="Serve one shard over gRPC.")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--neighbor-selection", choices=["heuristic", "closest"],
                        default="heuristic", help="how the graph picks each node's links")
    parser.add_argument("--data-dir", default=None,
                        help="keep a snapshot + write-ahead log here; without it, data lives in memory only")
    parser.add_argument("--fsync", choices=["always", "off"], default="always",
                        help="'always' forces every log write to disk before replying")
    parser.add_argument("--snapshot-every", type=int, default=1000,
                        help="vectors written between snapshots (0 = never)")
    args = parser.parse_args()

    shard = open_shard(args.data_dir, args.fsync == "always", args.snapshot_every,
                       M=args.M, ef_construction=args.ef_construction,
                       ef_search=args.ef_search, seed=args.seed,
                       neighbor_selection=args.neighbor_selection)
    if isinstance(shard, DurableShard):
        print(f"recovered {len(shard)} vectors from {args.data_dir} (snapshot at seq "
              f"{shard.snapshot_seq}, replayed {shard.replayed_records} log records)", flush=True)
    server, port = serve_shard(shard, args.port, args.host)
    print(f"shard serving on {args.host}:{port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
