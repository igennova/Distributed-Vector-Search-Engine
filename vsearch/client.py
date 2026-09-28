"""
Coordinator for shards running as separate gRPC services, plus a helper that launches
a local cluster of shard servers.

Each shard can have several replicas (identical copies on different servers):
  - writes go to every replica of the shard, so the copies never drift apart
  - reads go to one replica per shard, rotating between replicas to spread the load
  - if a replica fails or times out, the read is retried on another replica of the same
    shard, and the failed replica is tried last for a while (a simple circuit breaker)
"""
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import grpc
import numpy as np

from .cluster import merge_top_k, route_round_robin
from .protos import shard_pb2, shard_pb2_grpc

# gRPC rejects messages over 4 MB by default; keep each Add request well under that.
MAX_ADD_BYTES = 2 * 1024 * 1024

# Errors meaning "this server can't answer right now", worth retrying on another replica.
RETRYABLE = (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED)


class ShardUnavailableError(RuntimeError):
    """Every replica of one or more shards failed, so the results would be incomplete."""

    def __init__(self, shards):
        super().__init__(f"no replica answered for shard(s) {shards}")
        self.shards = shards


class _Replica:
    def __init__(self, address):
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = shard_pb2_grpc.ShardServiceStub(self.channel)
        self.down_until = 0.0     # time.monotonic() until which this replica is tried last


class GrpcCoordinator:
    """Routes vectors to remote shards and answers queries by scatter-gather + top-k merge.

    `shards` has one entry per shard: a single address, or a list of replica addresses.
    """

    def __init__(self, shards, timeout=5.0, add_timeout=600.0, retry_after=5.0):
        self.replicas = [[_Replica(address) for address in ([s] if isinstance(s, str) else s)]
                         for s in shards]
        self.timeout = timeout            # per-search deadline: a stuck server can't hang a query
        self.add_timeout = add_timeout    # inserts build graph, so they get a longer deadline
        self.retry_after = retry_after    # seconds a failed replica is deprioritized
        self.size = 0
        self.shard_sizes = [0] * len(self.replicas)   # as reported back by the servers
        self.failovers = 0                            # searches retried on another replica
        self._next_replica = [0] * len(self.replicas) # round-robin position per shard

    def add(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors) == 0:
            return self
        dim = vectors.shape[1]
        per_request = max(1, MAX_ADD_BYTES // (dim * vectors.itemsize))
        batches = route_round_robin(vectors, len(self.replicas), start_id=self.size)
        self.size += len(vectors)

        # Write-all: every chunk goes to every replica of its shard, so the copies stay
        # identical. Chunk i is sent everywhere before waiting on any, so all servers build
        # in parallel. If any replica fails, the write fails (and those replicas may now
        # differ until they are rebuilt).
        longest = max(len(ids) for ids, _ in batches)
        for start in range(0, longest, per_request):
            calls = []
            for shard, (ids, vecs) in enumerate(batches):
                chunk_ids = ids[start:start + per_request]
                if not chunk_ids:
                    continue
                chunk = np.asarray(vecs[start:start + per_request], dtype=np.float32)
                request = shard_pb2.AddRequest(global_ids=chunk_ids, dim=dim,
                                               values=chunk.ravel().tolist())
                for replica in self.replicas[shard]:
                    calls.append((shard, replica.stub.Add.future(request, timeout=self.add_timeout)))

            reported = {}
            for shard, call in calls:
                reported.setdefault(shard, set()).add(call.result().size)
            for shard, sizes in reported.items():
                if len(sizes) > 1:
                    raise RuntimeError(f"replicas of shard {shard} diverged: sizes {sorted(sizes)}")
                self.shard_sizes[shard] = sizes.pop()
        return self

    def search(self, query, k=10):
        """Global top-k ids. Raises ShardUnavailableError if any shard has no live replica."""
        ids, missing = self.search_partial(query, k)
        if missing:
            raise ShardUnavailableError(missing)
        return ids

    def search_partial(self, query, k=10):
        """Like search, but skips shards whose replicas are all down instead of failing.

        Returns (ids, missing_shards). A non-empty missing_shards means the results come
        from the remaining shards only and may lack some true nearest neighbors.
        """
        request = shard_pb2.SearchRequest(query=np.asarray(query, dtype=np.float32).tolist(), k=k)
        now = time.monotonic()
        orders = [self._replica_order(shard, now) for shard in range(len(self.replicas))]
        tried = [0] * len(self.replicas)

        # One request per shard, all sent before waiting on any (parallel fan-out). Shards
        # whose replica fails are retried together on their next replica, still in parallel.
        pending = {shard: order[0].stub.Search.future(request, timeout=self.timeout)
                   for shard, order in enumerate(orders)}
        per_shard, missing = [], []
        while pending:
            retries = {}
            for shard, call in pending.items():
                replica = orders[shard][tried[shard]]
                try:
                    hits = call.result().hits
                except grpc.RpcError as err:
                    if err.code() not in RETRYABLE:
                        raise
                    replica.down_until = time.monotonic() + self.retry_after
                    tried[shard] += 1
                    if tried[shard] < len(orders[shard]):
                        self.failovers += 1
                        retries[shard] = orders[shard][tried[shard]].stub.Search.future(
                            request, timeout=self.timeout)
                    else:
                        missing.append(shard)
                    continue
                replica.down_until = 0.0
                per_shard.append([(hit.distance, hit.global_id) for hit in hits])
            pending = retries
        return merge_top_k(per_shard, k), sorted(missing)

    def _replica_order(self, shard, now):
        """Replicas to try for one shard: rotate the starting replica, recently failed ones last."""
        replicas = self.replicas[shard]
        start = self._next_replica[shard]
        self._next_replica[shard] = (start + 1) % len(replicas)
        rotated = replicas[start:] + replicas[:start]
        # Failed replicas are tried last rather than never, so a server that comes back
        # is used again, and a shard is only given up on once every copy has failed.
        return ([r for r in rotated if r.down_until <= now] +
                [r for r in rotated if r.down_until > now])

    def close(self):
        for group in self.replicas:
            for replica in group:
                replica.channel.close()

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


class LocalCluster:
    """Shard servers running as local processes. addresses[shard][replica] is "host:port"."""

    def __init__(self, addresses, processes):
        self.addresses = addresses
        self.processes = processes

    def kill(self, shard, replica=0):
        """Kill one server abruptly (SIGKILL), as if its machine had crashed."""
        process = self.processes[shard][replica]
        process.kill()
        process.wait()


@contextmanager
def local_grpc_cluster(num_shards, replicas=1, seed=None, **hnsw_params):
    """Run num_shards x replicas shard servers as separate local processes.

    Every replica of a shard gets the same seed, so the same writes build identical indexes.
    """
    repo_root = Path(__file__).resolve().parent.parent
    ports = iter(_free_ports(num_shards * replicas))
    processes, addresses = [], []
    try:
        for shard in range(num_shards):
            processes.append([])
            addresses.append([])
            for _ in range(replicas):
                port = next(ports)
                cmd = [sys.executable, "-m", "vsearch.server", "--port", str(port)]
                for key, value in hnsw_params.items():
                    cmd += [f"--{key.replace('_', '-')}", str(value)]
                if seed is not None:
                    cmd += ["--seed", str(seed + shard)]
                processes[shard].append(
                    subprocess.Popen(cmd, cwd=repo_root, stdout=subprocess.DEVNULL))
                addresses[shard].append(f"127.0.0.1:{port}")

        for group in addresses:
            for address in group:
                with grpc.insecure_channel(address) as channel:
                    grpc.channel_ready_future(channel).result(timeout=20)
        yield LocalCluster(addresses, processes)
    finally:
        all_processes = [p for group in processes for p in group]
        for process in all_processes:
            if process.poll() is None:
                process.terminate()
        for process in all_processes:
            process.wait(timeout=10)
