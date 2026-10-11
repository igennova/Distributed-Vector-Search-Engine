"""Namespaces: each one is a separate set of vectors, kept on a single shard, so a search
inside it asks only that shard."""
from collections import Counter

import numpy as np
import pytest

from vsearch.client import GrpcCoordinator, ShardUnavailableError
from vsearch.cluster import MODES, NAMESPACE_FIELD, Coordinator, shard_for_namespace
from vsearch.dataset import normalize
from vsearch.server import start_server

PARAMS = dict(M=8, ef_construction=32, ef_search=32)


def _vectors(n, seed):
    return np.random.default_rng(seed).standard_normal((n, 16)).astype(np.float32)


def _fill(coord):
    """alice: ids 0-99, bob: 100-179, no namespace: 180-299. Returns all the vectors."""
    alice, bob, shared = _vectors(100, 1), _vectors(80, 2), _vectors(120, 3)
    coord.add(alice, [{"n": i % 5} for i in range(100)], namespace="alice")
    coord.add(bob, namespace="bob")                       # no payloads of their own
    coord.add(shared, [{"n": i % 5} for i in range(120)])
    return np.concatenate([alice, bob, shared])


def _nearest(vectors, ids, query, k):
    ids = np.asarray(ids)
    return ids[np.argsort(-(normalize(vectors[ids]) @ query), kind="stable")[:k]].tolist()


def test_a_namespace_always_maps_to_the_same_shard():
    # Pinned values: if this mapping ever changed, searches would look for existing
    # namespaces on the wrong shard.
    assert [shard_for_namespace(name, 4) for name in ("alice", "bob", "eve")] == [3, 0, 2]
    spread = Counter(shard_for_namespace(f"user-{i}", 4) for i in range(200))
    assert len(spread) == 4 and min(spread.values()) > 35      # namespaces spread evenly


@pytest.mark.parametrize("mode", MODES)
def test_a_search_inside_a_namespace_sees_only_that_namespace(mode):
    with Coordinator(4, mode=mode, seed=0, **PARAMS) as coord:
        vectors = _fill(coord)
        for q in vectors[:5] + 0.1:
            assert coord.search(q, 10, namespace="alice") == _nearest(vectors, range(100), q, 10)
            assert coord.search(q, 10, namespace="bob") == _nearest(vectors, range(100, 180), q, 10)
        # A filter narrows it further.
        wanted = [i for i in range(100) if i % 5 == 2]
        assert coord.search(vectors[0], 5, where={"n": 2}, namespace="alice") == \
            _nearest(vectors, wanted, vectors[0], 5)
        # Without a namespace a search covers everything, namespaced or not.
        assert coord.search(vectors[150], 1) == [150] and coord.search(vectors[250], 1) == [250]
        assert coord.search(vectors[0], 5, namespace="nobody") == []

        hit = coord.search_hits(vectors[3], 1, namespace="alice")[0]
        assert hit.id == 3 and hit.payload == {"n": 3, NAMESPACE_FIELD: "alice"}
        assert coord.search_hits(vectors[100], 1, namespace="bob")[0].payload == {NAMESPACE_FIELD: "bob"}


def test_a_namespace_is_stored_on_one_shard():
    coord = Coordinator(4, seed=0, **PARAMS)
    _fill(coord)
    # 120 un-namespaced vectors go round-robin: 30 per shard. alice adds 100 to shard 3,
    # bob 80 to shard 0.
    assert [len(shard) for shard in coord.shards] == [30 + 80, 30, 30, 30 + 100]


def test_the_reserved_field_and_bad_names_are_rejected():
    coord = Coordinator(2, **PARAMS)
    vectors = _vectors(2, 4)
    with pytest.raises(ValueError):
        coord.add(vectors, [{NAMESPACE_FIELD: "alice"}, None])     # must go through namespace=
    with pytest.raises(ValueError):
        coord.search(vectors[0], 1, where={NAMESPACE_FIELD: "alice"})
    for bad in ("", 7, "x" * 500):
        with pytest.raises(ValueError):
            coord.add(vectors, namespace=bad)
    assert coord.size == 0


@pytest.fixture
def servers():
    started = [start_server(port=0, seed=i, **PARAMS) for i in range(4)]
    yield started
    for server, _ in started:
        server.stop(grace=None)


def test_grpc_namespaces_match_the_local_coordinator(servers):
    local = Coordinator(4, seed=0, **PARAMS)
    vectors = _fill(local)
    with GrpcCoordinator([f"127.0.0.1:{port}" for _, port in servers]) as remote:
        _fill(remote)
        assert remote.shard_for("alice") == 3
        for q in vectors[:5] + 0.1:
            for namespace in ("alice", "bob", None):
                assert remote.search_hits(q, 5, namespace=namespace) == \
                    local.search_hits(q, 5, namespace=namespace)


def test_a_namespace_search_needs_only_its_own_shard(servers):
    with GrpcCoordinator([f"127.0.0.1:{port}" for _, port in servers], timeout=2.0) as coord:
        vectors = _fill(coord)
        expected = coord.search(vectors[5], 5, namespace="alice")
        for shard, (server, _) in enumerate(servers):
            if shard != coord.shard_for("alice"):
                server.stop(grace=None)                   # three of the four shards are gone

        assert coord.search(vectors[5], 5, namespace="alice") == expected
        with pytest.raises(ShardUnavailableError):
            coord.search(vectors[5], 5)                   # a search of everything needs them all
        with pytest.raises(ShardUnavailableError) as err:
            coord.search(vectors[5], 5, namespace="bob")  # bob's shard is one of the dead ones
        assert err.value.shards == [coord.shard_for("bob")]
