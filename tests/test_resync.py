"""Replica resync: a replica that missed writes, or lost its disk, catches up from its twin."""
import shutil

import grpc
import numpy as np
import pytest

from vsearch.client import GrpcCoordinator, ReplicasOutOfSyncError, local_grpc_cluster
from vsearch.cluster import Coordinator
from vsearch.protos import shard_pb2, shard_pb2_grpc
from vsearch.server import start_server

PARAMS = dict(M=8, ef_construction=32, ef_search=32)


def _data():
    rng = np.random.default_rng(11)
    return (rng.standard_normal((900, 16)).astype(np.float32),
            rng.standard_normal((10, 16)).astype(np.float32))


def test_replica_that_missed_a_write_catches_up_from_the_log(tmp_path):
    data, queries = _data()
    first, second, third = data[:300], data[300:600], data[600:]
    local = Coordinator(2, seed=0, **PARAMS).add(data)

    with local_grpc_cluster(2, replicas=2, seed=0, data_dir=tmp_path, **PARAMS) as cluster, \
            GrpcCoordinator(cluster.addresses) as coord:
        coord.add(first)
        cluster.kill(shard=0, replica=1)
        with pytest.raises(grpc.RpcError):
            coord.add(second)                  # lands on every replica except the dead one
        cluster.restart(shard=0, replica=1)    # back from its own disk, one write behind

        with pytest.raises(ReplicasOutOfSyncError):
            coord.add(third)                   # refused until the replicas agree again

        synced = coord.repair()
        assert [(s.shard, s.method, s.records_applied) for s in synced] == [(0, "log", 1)]

        coord.add(third)                       # writes work again
        cluster.kill(shard=0, replica=0)       # only the repaired copy is left for shard 0
        for q in queries:                      # the "failed" second write rolled forward
            assert coord.search(q, 5) == local.search(q, 5)


def test_replica_that_lost_its_disk_is_rebuilt_from_a_snapshot(tmp_path):
    data, queries = _data()
    local = Coordinator(2, seed=0, **PARAMS).add(data)

    # With snapshot_every=100 each server snapshots after its first write and empties its
    # log, so an empty replica can't be caught up from the log alone.
    with local_grpc_cluster(2, replicas=2, seed=0, data_dir=tmp_path, snapshot_every=100,
                            **PARAMS) as cluster, GrpcCoordinator(cluster.addresses) as coord:
        coord.add(data)
        cluster.kill(shard=0, replica=1)
        shutil.rmtree(tmp_path / "shard-0-replica-1")      # its disk is gone
        cluster.restart(shard=0, replica=1)                # comes back empty

        synced = coord.repair()
        assert [(s.shard, s.method) for s in synced] == [(0, "snapshot")]

        cluster.kill(shard=0, replica=0)
        for q in queries:
            assert coord.search(q, 5) == local.search(q, 5)


def test_repair_does_nothing_when_replicas_agree(tmp_path):
    data, _ = _data()
    with local_grpc_cluster(1, replicas=2, seed=0, data_dir=tmp_path, **PARAMS) as cluster, \
            GrpcCoordinator(cluster.addresses) as coord:
        coord.add(data[:100])
        assert coord.repair() == []


def test_server_rejects_a_write_with_the_wrong_seq(tmp_path):
    server, port = start_server(port=0, data_dir=tmp_path, seed=0, **PARAMS)
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = shard_pb2_grpc.ShardServiceStub(channel)
            request = shard_pb2.AddRequest(global_ids=[0], dim=2, values=[1.0, 0.0], seq=5)
            with pytest.raises(grpc.RpcError) as err:
                stub.Add(request, timeout=5)
        assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    finally:
        server.stop(grace=None)
