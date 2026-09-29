"""Shard servers that crash or restart recover their data from disk."""
import numpy as np

from vsearch.client import GrpcCoordinator, local_grpc_cluster
from vsearch.cluster import Coordinator

PARAMS = dict(M=8, ef_construction=32, ef_search=32)


def _data():
    rng = np.random.default_rng(9)
    return (rng.standard_normal((300, 16)).astype(np.float32),
            rng.standard_normal((10, 16)).astype(np.float32))


def test_restarted_replica_serves_its_data_again(tmp_path):
    data, queries = _data()
    local = Coordinator(2, seed=0, **PARAMS).add(data)

    with local_grpc_cluster(2, replicas=2, seed=0, data_dir=tmp_path, **PARAMS) as cluster, \
            GrpcCoordinator(cluster.addresses) as coord:
        coord.add(data)
        cluster.kill(shard=0, replica=0)       # one copy crashes...
        cluster.restart(shard=0, replica=0)    # ...and comes back, reloading from disk
        cluster.kill(shard=0, replica=1)       # now its twin dies: only the restarted copy is left
        for q in queries:
            assert coord.search(q, 5) == local.search(q, 5)


def test_whole_cluster_restart_keeps_the_data(tmp_path):
    data, queries = _data()
    local = Coordinator(2, seed=0, **PARAMS).add(data)

    with local_grpc_cluster(2, replicas=2, seed=0, data_dir=tmp_path, **PARAMS) as cluster, \
            GrpcCoordinator(cluster.addresses) as coord:
        coord.add(data)

    # Every server is gone. Start a fresh cluster on the same data directories.
    with local_grpc_cluster(2, replicas=2, seed=0, data_dir=tmp_path, **PARAMS) as cluster, \
            GrpcCoordinator(cluster.addresses) as coord:
        for q in queries:
            assert coord.search(q, 5) == local.search(q, 5)
