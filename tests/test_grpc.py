"""Tests for shards served over gRPC."""
import socket

import grpc
import numpy as np
import pytest

from vsearch.client import GrpcCoordinator, local_grpc_cluster
from vsearch.cluster import Coordinator
from vsearch.server import start_server

PARAMS = dict(M=8, ef_construction=32, ef_search=32)


@pytest.fixture
def shard_addresses():
    """Four shard servers in this process, seeded 0..3 to match Coordinator(seed=0)."""
    servers, addresses = [], []
    for i in range(4):
        server, port = start_server(port=0, seed=i, **PARAMS)
        servers.append(server)
        addresses.append(f"127.0.0.1:{port}")
    yield addresses
    for server in servers:
        server.stop(grace=None)


def test_grpc_results_match_local_coordinator(shard_addresses):
    # Moving shards behind the network must not change a single result.
    rng = np.random.default_rng(3)
    data = rng.standard_normal((200, 16)).astype(np.float32)
    queries = rng.standard_normal((10, 16)).astype(np.float32)

    local = Coordinator(4, seed=0, **PARAMS).add(data)
    with GrpcCoordinator(shard_addresses) as remote:
        remote.add(data)
        assert remote.shard_sizes == [50, 50, 50, 50]
        for q in queries:
            assert remote.search(q, 5) == local.search(q, 5)


def test_add_larger_than_grpc_message_limit(shard_addresses):
    # 1,100 vectors x 1,024 dims x 4 bytes is ~4.5 MB, over gRPC's 4 MB default limit.
    # A single request would be rejected, so this passes only if the upload is chunked.
    rng = np.random.default_rng(4)
    data = rng.standard_normal((1_100, 1_024)).astype(np.float32)
    with GrpcCoordinator(shard_addresses[:1]) as remote:
        remote.add(data)
        assert remote.shard_sizes == [1_100]


def test_unreachable_shard_fails_fast():
    with socket.socket() as s:            # a port nothing is listening on
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    with GrpcCoordinator([f"127.0.0.1:{dead_port}"], timeout=2.0) as remote:
        with pytest.raises(grpc.RpcError) as err:
            remote.search(np.zeros(16, dtype=np.float32), 5)
    assert err.value.code() == grpc.StatusCode.UNAVAILABLE


def test_shards_as_separate_processes():
    rng = np.random.default_rng(5)
    data = rng.standard_normal((200, 16)).astype(np.float32)
    query = rng.standard_normal(16).astype(np.float32)

    local = Coordinator(2, seed=0, **PARAMS).add(data)
    with local_grpc_cluster(2, seed=0, **PARAMS) as addresses, \
            GrpcCoordinator(addresses) as remote:
        remote.add(data)
        assert remote.search(query, 5) == local.search(query, 5)
