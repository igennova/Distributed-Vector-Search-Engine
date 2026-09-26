"""
Coordinator for shards running as separate gRPC services, plus a helper that launches
a local cluster of shard servers.
"""
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import grpc
import numpy as np

from .cluster import merge_top_k, route_round_robin
from .protos import shard_pb2, shard_pb2_grpc

# gRPC rejects messages over 4 MB by default; keep each Add request well under that.
MAX_ADD_BYTES = 2 * 1024 * 1024


class GrpcCoordinator:
    """Routes vectors to remote shards and answers queries by scatter-gather + top-k merge."""

    def __init__(self, addresses, timeout=5.0, add_timeout=600.0):
        self.channels = [grpc.insecure_channel(address) for address in addresses]
        self.stubs = [shard_pb2_grpc.ShardServiceStub(channel) for channel in self.channels]
        self.timeout = timeout            # per-search deadline: a stuck shard can't hang a query
        self.add_timeout = add_timeout    # inserts build graph, so they get a longer deadline
        self.size = 0
        self.shard_sizes = [0] * len(addresses)   # as reported back by each shard

    def add(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors) == 0:
            return self
        dim = vectors.shape[1]
        per_request = max(1, MAX_ADD_BYTES // (dim * vectors.itemsize))
        batches = route_round_robin(vectors, len(self.stubs), start_id=self.size)
        self.size += len(vectors)

        # Each shard receives its vectors in order, one chunk per request. Chunk i is sent
        # to every shard before waiting on any, so the shards build in parallel.
        longest = max(len(ids) for ids, _ in batches)
        for start in range(0, longest, per_request):
            calls = []
            for shard_index, (stub, (ids, vecs)) in enumerate(zip(self.stubs, batches)):
                chunk_ids = ids[start:start + per_request]
                if not chunk_ids:
                    continue
                chunk = np.asarray(vecs[start:start + per_request], dtype=np.float32)
                request = shard_pb2.AddRequest(global_ids=chunk_ids, dim=dim,
                                               values=chunk.ravel().tolist())
                calls.append((shard_index, stub.Add.future(request, timeout=self.add_timeout)))
            for shard_index, call in calls:
                self.shard_sizes[shard_index] = call.result().size
        return self

    def search(self, query, k=10):
        request = shard_pb2.SearchRequest(query=np.asarray(query, dtype=np.float32).tolist(), k=k)
        # Fire every request before waiting on any. The coordinator only waits on the
        # network here (the GIL is released while waiting), so no extra processes are needed.
        calls = [stub.Search.future(request, timeout=self.timeout) for stub in self.stubs]
        per_shard = [[(hit.distance, hit.global_id) for hit in call.result().hits]
                     for call in calls]
        return merge_top_k(per_shard, k)

    def close(self):
        for channel in self.channels:
            channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def _free_ports(n):
    sockets = []
    try:
        for _ in range(n):
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            sockets.append(s)
        return [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()


@contextmanager
def local_grpc_cluster(num_shards, seed=None, **hnsw_params):
    """Run num_shards shard servers as separate local processes; yields their addresses."""
    repo_root = Path(__file__).resolve().parent.parent
    ports = _free_ports(num_shards)
    processes = []
    try:
        for i, port in enumerate(ports):
            cmd = [sys.executable, "-m", "vsearch.server", "--port", str(port)]
            for key, value in hnsw_params.items():
                cmd += [f"--{key.replace('_', '-')}", str(value)]
            if seed is not None:
                cmd += ["--seed", str(seed + i)]
            processes.append(subprocess.Popen(cmd, cwd=repo_root, stdout=subprocess.DEVNULL))

        addresses = [f"127.0.0.1:{port}" for port in ports]
        for address in addresses:
            with grpc.insecure_channel(address) as channel:
                grpc.channel_ready_future(channel).result(timeout=20)
        yield addresses
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=10)
