"""Tests for replicated shards and failover over gRPC."""
import grpc
import numpy as np
import pytest

from vsearch.client import GrpcCoordinator, ShardUnavailableError, local_grpc_cluster
from vsearch.cluster import Coordinator
from vsearch.server import start_server

PARAMS = dict(M=8, ef_construction=32, ef_search=32)
NUM_SHARDS, REPLICAS = 3, 2


@pytest.fixture
def cluster():
    """3 shards x 2 replicas in this process: servers[shard][replica], addresses likewise.

    Both replicas of a shard get the shard's seed, matching Coordinator(seed=0).
    """
    servers, addresses = [], []
    for shard in range(NUM_SHARDS):
        servers.append([])
        addresses.append([])
        for _ in range(REPLICAS):
            server, port = start_server(port=0, seed=shard, **PARAMS)
            servers[shard].append(server)
            addresses[shard].append(f"127.0.0.1:{port}")
    yield servers, addresses
    for group in servers:
        for server in group:
            server.stop(grace=None)


def _data():
    rng = np.random.default_rng(7)
    return (rng.standard_normal((300, 16)).astype(np.float32),
            rng.standard_normal((10, 16)).astype(np.float32))


def test_every_replica_holds_the_full_shard(cluster):
    _, addresses = cluster
    data, queries = _data()
    local = Coordinator(NUM_SHARDS, seed=0, **PARAMS).add(data)
    with GrpcCoordinator(addresses) as coord:
        coord.add(data)
        assert coord.shard_sizes == [100, 100, 100]

    # Query each set of copies on its own: every copy must give the full answer.
    for replica in range(REPLICAS):
        with GrpcCoordinator([group[replica] for group in addresses]) as one_copy_each:
            for q in queries:
                assert one_copy_each.search(q, 5) == local.search(q, 5)


def test_search_fails_over_when_a_replica_dies(cluster):
    servers, addresses = cluster
    data, queries = _data()
    local = Coordinator(NUM_SHARDS, seed=0, **PARAMS).add(data)
    with GrpcCoordinator(addresses) as coord:
        coord.add(data)
        servers[1][0].stop(grace=None)            # one copy of shard 1 goes away
        for q in queries:                         # round-robin sends some reads to it
            assert coord.search(q, 5) == local.search(q, 5)
        assert coord.failovers >= 1


def test_writes_fail_if_any_replica_is_down(cluster):
    servers, addresses = cluster
    data, _ = _data()
    servers[0][1].stop(grace=None)
    with GrpcCoordinator(addresses) as coord:
        with pytest.raises(grpc.RpcError) as err:
            coord.add(data)
    assert err.value.code() == grpc.StatusCode.UNAVAILABLE


def test_losing_every_replica_of_a_shard(cluster):
    servers, addresses = cluster
    data, queries = _data()
    with GrpcCoordinator(addresses) as coord:
        coord.add(data)
        for server in servers[2]:
            server.stop(grace=None)

        with pytest.raises(ShardUnavailableError) as err:
            coord.search(queries[0], 5)
        assert err.value.shards == [2]

        ids, missing = coord.search_partial(queries[0], 5)
        assert missing == [2]
        assert len(ids) == 5
        assert all(i % NUM_SHARDS != 2 for i in ids)     # nothing from the lost shard


def test_failover_after_a_server_process_crashes():
    data, queries = _data()
    local = Coordinator(2, seed=0, **PARAMS).add(data)
    with local_grpc_cluster(2, replicas=2, seed=0, **PARAMS) as cluster, \
            GrpcCoordinator(cluster.addresses) as coord:
        coord.add(data)
        cluster.kill(shard=0, replica=0)          # SIGKILL: no clean shutdown
        for q in queries:
            assert coord.search(q, 5) == local.search(q, 5)


class _Down(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE


class _Call:
    """A stand-in for an Add sent to one replica: fails at once, or answers when waited on."""
    def __init__(self, waited, name, fails):
        self.waited, self.name, self.fails = waited, name, fails

    def result(self):
        if self.fails:
            raise _Down()
        self.waited.append(self.name)
        return type("Reply", (), {"size": 3})()


def test_a_failed_write_waits_for_the_other_replicas_before_raising():
    # One replica fails at once while the others are still applying the write. Raising at
    # that moment would let the caller close the channels and cancel the writes still in
    # flight, so one dead server would also leave healthy shards with replicas that disagree.
    # (CI caught this: on a slower machine a healthy shard's second replica missed the write.)
    waited = []
    coordinator = GrpcCoordinator([["127.0.0.1:1", "127.0.0.1:2"], ["127.0.0.1:3", "127.0.0.1:4"]])
    for shard, group in enumerate(coordinator.replicas):
        for replica, target in enumerate(group):
            name, fails = f"{shard}/{replica}", (shard, replica) == (0, 0)
            target.stub = type("Stub", (), {})()
            target.stub.Add = type("Add", (), {"future": staticmethod(
                lambda request, timeout, name=name, fails=fails: _Call(waited, name, fails))})()
    coordinator._last_seqs = [4, 4]
    with pytest.raises(grpc.RpcError):
        coordinator.add(np.ones((4, 8), dtype=np.float32))
    assert waited == ["0/1", "1/0", "1/1"]       # every live replica was waited for
    assert coordinator._last_seqs == [None, None]
    coordinator.close()
