"""Payloads: the data stored with a vector comes back with search results, and survives
everything the vector survives (restarts, snapshots, resync)."""
import json

import grpc
import numpy as np
import pytest

from vsearch.client import GrpcCoordinator, local_grpc_cluster
from vsearch.cluster import MODES, Coordinator, Hit, encode_payloads
from vsearch.protos import shard_pb2, shard_pb2_grpc
from vsearch.server import start_server
from vsearch.storage import DurableShard, load_snapshot

PARAMS = dict(M=8, ef_construction=32, ef_search=32)


def _data(n=120, dim=16):
    """Vectors plus payloads; every third vector has none, the rest mix unicode and nesting."""
    vectors = np.random.default_rng(21).standard_normal((n, dim)).astype(np.float32)
    payloads = [None if i % 3 == 0 else
                {"text": f"passage {i} — café ☕", "source": f"doc-{i % 4}.md", "tags": ["a", i]}
                for i in range(n)]
    return vectors, payloads


@pytest.fixture
def shard_addresses():
    """Three shard servers in this process, seeded 0..2 to match Coordinator(seed=0)."""
    servers, addresses = [], []
    for i in range(3):
        server, port = start_server(port=0, seed=i, **PARAMS)
        servers.append(server)
        addresses.append(f"127.0.0.1:{port}")
    yield addresses
    for server in servers:
        server.stop(grace=None)


@pytest.mark.parametrize("mode", MODES)
def test_search_hits_returns_each_vectors_payload(mode):
    vectors, payloads = _data()
    with Coordinator(3, mode=mode, seed=0, **PARAMS) as coord:
        coord.add(vectors, payloads)
        hits = coord.search_hits(vectors[7], k=5)

    assert isinstance(hits[0], Hit)
    assert hits[0].id == 7 and hits[0].distance < 1e-5           # the vector itself comes first
    assert hits[0].payload == {"text": "passage 7 — café ☕", "source": "doc-3.md", "tags": ["a", 7]}
    assert all(hit.payload == payloads[hit.id] for hit in hits)  # including None where there is none


def test_payload_count_must_match_the_vectors():
    vectors, payloads = _data()
    with pytest.raises(ValueError):
        Coordinator(2, **PARAMS).add(vectors, payloads[:5])


def test_grpc_payloads_match_the_local_coordinator(shard_addresses):
    vectors, payloads = _data()
    local = Coordinator(3, seed=0, **PARAMS).add(vectors, payloads)
    with GrpcCoordinator(shard_addresses) as remote:
        remote.add(vectors, payloads)
        for q in vectors[:10]:
            assert remote.search_hits(q, 5) == local.search_hits(q, 5)
            assert remote.search(q, 5) == local.search(q, 5)     # plain search is unchanged


def test_server_rejects_a_payload_count_that_does_not_match(shard_addresses):
    with grpc.insecure_channel(shard_addresses[0]) as channel:
        stub = shard_pb2_grpc.ShardServiceStub(channel)
        request = shard_pb2.AddRequest(global_ids=[0, 1], dim=2, values=[1, 0, 0, 1],
                                       payloads=["{}"])
        with pytest.raises(grpc.RpcError) as err:
            stub.Add(request, timeout=5)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def _filled_shard(path, checkpoint):
    vectors, payloads = _data()
    texts = encode_payloads(payloads, len(vectors))
    shard = DurableShard(path, snapshot_every=0, seed=0, **PARAMS)
    shard.add_batch(list(range(60)), vectors[:60], texts[:60])
    shard.add_batch(list(range(60, 120)), vectors[60:], texts[60:])
    if checkpoint:
        shard.checkpoint()
    return shard, vectors


def _results(shard, vectors):
    return [shard.search_with_payloads(q, 5) for q in vectors[:10]]


@pytest.mark.parametrize("checkpoint, replayed", [(False, 2), (True, 0)])
def test_payloads_survive_a_restart(tmp_path, checkpoint, replayed):
    # Without a checkpoint they come back from the log; with one, from the snapshot.
    shard, vectors = _filled_shard(tmp_path, checkpoint)
    before = _results(shard, vectors)
    assert any(payload for hits in before for _, _, payload in hits)
    shard.close()

    recovered = DurableShard(tmp_path, snapshot_every=0, seed=0, **PARAMS)
    assert recovered.replayed_records == replayed
    assert _results(recovered, vectors) == before


def test_installing_a_peers_snapshot_brings_its_payloads(tmp_path):
    peer, vectors = _filled_shard(tmp_path / "peer", checkpoint=True)
    data, _ = peer.snapshot_bytes()
    copy = DurableShard(tmp_path / "copy", snapshot_every=0, seed=0, **PARAMS)
    copy.install_snapshot(data)
    assert _results(copy, vectors) == _results(peer, vectors)


def test_a_record_without_payloads_keeps_the_original_layout(tmp_path):
    # Logs written before payloads existed have to keep reading back, so a record whose
    # vectors have no payloads must be exactly: header + (seq, count, dim) + ids + vectors.
    vectors, _ = _data(n=10)
    shard = DurableShard(tmp_path, snapshot_every=0, seed=0, **PARAMS)
    shard.add_batch(list(range(10)), vectors)
    shard.close()
    assert (tmp_path / "wal.log").stat().st_size == 8 + 16 + 10 * 8 + 10 * 16 * 4


def test_a_snapshot_from_before_payloads_still_loads(tmp_path):
    shard, vectors = _filled_shard(tmp_path, checkpoint=True)
    plain = [shard.search(q, 5) for q in vectors[:10]]
    shard.close()

    # Rewrite the snapshot the way format 2 looked: no payload arrays.
    path = tmp_path / "snapshot.npz"
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files if not name.startswith("payload_")}
    meta = json.loads(str(arrays["meta"]))
    meta["format"] = 2
    arrays["meta"] = np.array(json.dumps(meta))
    with open(path, "wb") as f:
        np.savez(f, **arrays)

    old, _ = load_snapshot(path)
    assert old.payloads == [""] * 120
    assert [old.search(q, 5) for q in vectors[:10]] == plain


def test_resync_carries_payloads(tmp_path):
    vectors, payloads = _data()
    local = Coordinator(1, seed=0, **PARAMS).add(vectors, payloads)

    with local_grpc_cluster(1, replicas=2, seed=0, data_dir=tmp_path, **PARAMS) as cluster, \
            GrpcCoordinator(cluster.addresses) as coord:
        coord.add(vectors[:60], payloads[:60])
        cluster.kill(shard=0, replica=1)
        with pytest.raises(grpc.RpcError):
            coord.add(vectors[60:], payloads[60:])     # reaches replica 0 only
        cluster.restart(shard=0, replica=1)
        assert [sync.method for sync in coord.repair()] == ["log"]

        cluster.kill(shard=0, replica=0)               # only the repaired copy is left
        for q in vectors[60:70]:
            assert coord.search_hits(q, 5) == local.search_hits(q, 5)
