"""Filtered search: only matching vectors come back, whichever way the shard answers, and
filters keep working across shards, over gRPC, and after a restart."""
import grpc
import numpy as np
import pytest

from vsearch import cluster
from vsearch.client import GrpcCoordinator
from vsearch.cluster import MODES, Coordinator, Shard, encode_payloads, filter_strategy
from vsearch.dataset import normalize
from vsearch.filtering import normalize_filter
from vsearch.protos import shard_pb2, shard_pb2_grpc
from vsearch.server import start_server
from vsearch.storage import DurableShard

PARAMS = dict(M=8, ef_construction=32, ef_search=32)
COLORS = ["red", "green", "blue", "red", "red", "green"]      # red is half of the data


def _data(n=600, dim=16):
    vectors = np.random.default_rng(11).standard_normal((n, dim)).astype(np.float32)
    payloads = [{"color": COLORS[i % 6], "size": i % 4, "tags": ["even" if i % 2 == 0 else "odd", "all"]}
                for i in range(n)]
    return vectors, payloads


def _matches(payload, where):
    def one(field, wanted):
        wanted = wanted if isinstance(wanted, list) else [wanted]
        have = payload[field] if isinstance(payload[field], list) else [payload[field]]
        return any(item in wanted for item in have)
    return all(one(field, wanted) for field, wanted in where.items())


def _truth(vectors, payloads, query, k, where):
    """Ids of the exact k nearest vectors among those matching `where`."""
    ids = np.array([i for i, payload in enumerate(payloads) if _matches(payload, where)])
    sims = normalize(vectors[ids]) @ (query / np.linalg.norm(query))
    return ids[np.argsort(-sims, kind="stable")[:k]].tolist()


# --- one shard --------------------------------------------------------------------------------

def test_few_matches_are_compared_directly_and_many_go_through_the_graph():
    assert filter_strategy(matches=100, size=100_000, ef=100) == "exact"
    assert filter_strategy(matches=90_000, size=100_000, ef=100) == "graph"
    # A larger shard switches at a smaller share of its vectors.
    assert filter_strategy(matches=30_000, size=100_000, ef=100) == "exact"       # 30%
    assert filter_strategy(matches=300_000, size=1_000_000, ef=100) == "graph"    # 30%


@pytest.mark.parametrize("strategy, advantage", [("exact", 10**9), ("graph", 0)])
def test_a_shard_filters_the_same_whichever_way_it_answers(monkeypatch, strategy, advantage):
    monkeypatch.setattr(cluster, "EXACT_ADVANTAGE", advantage)       # force one strategy
    vectors, payloads = _data()
    shard = Shard(seed=0, **PARAMS)
    shard.add_batch(list(range(len(vectors))), vectors, encode_payloads(payloads, len(vectors)))
    where = {"color": "red", "size": [0, 1]}
    assert filter_strategy(1, len(shard), 32) == strategy

    hits = total = 0
    for q in vectors[:30] + 0.1:
        got = [global_id for _, global_id in shard.search(q, 10, normalize_filter(where))]
        assert len(got) == 10 and all(_matches(payloads[i], where) for i in got)
        hits += len(set(got) & set(_truth(vectors, payloads, q, 10, where)))
        total += 10
    assert hits / total > (0.999 if strategy == "exact" else 0.9)

    # Fewer matches than k: all of them come back. No matches: nothing does.
    rare = normalize_filter({"color": "blue", "size": 2, "tags": "even"})
    expected = [i for i, p in enumerate(payloads) if p["color"] == "blue" and p["size"] == 2]
    assert sorted(g for _, g in shard.search(vectors[0], 500, rare)) == expected
    assert shard.search(vectors[0], 10, normalize_filter({"color": "purple"})) == []
    assert [text for _, _, text in shard.search_with_payloads(vectors[0], 3, rare)] == \
        [shard.payloads[g] for _, g in shard.search(vectors[0], 3, rare)]


# --- across shards, over the network, after a restart ---------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_filtered_search_across_shards(mode):
    vectors, payloads = _data()
    where = {"color": ["green", "blue"], "tags": "odd"}
    with Coordinator(3, mode=mode, seed=0, **PARAMS) as coord:
        coord.add(vectors, payloads)
        for q in vectors[:10] + 0.1:
            assert coord.search(q, 10, where=where) == _truth(vectors, payloads, q, 10, where)
        hits = coord.search_hits(vectors[1], 5, where={"size": 1})
        assert len(hits) == 5 and all(hit.payload["size"] == 1 for hit in hits)
        assert coord.search(vectors[0], 10, where={"color": "purple"}) == []
        assert len(coord.search(vectors[0], 10)) == 10        # no filter: unchanged


@pytest.fixture
def shard_addresses():
    servers, addresses = [], []
    for i in range(3):
        server, port = start_server(port=0, seed=i, **PARAMS)
        servers.append(server)
        addresses.append(f"127.0.0.1:{port}")
    yield addresses
    for server in servers:
        server.stop(grace=None)


def test_grpc_filtered_search_matches_the_local_coordinator(shard_addresses):
    vectors, payloads = _data()
    where = {"color": "red", "size": [0, 3]}
    local = Coordinator(3, seed=0, **PARAMS).add(vectors, payloads)
    with GrpcCoordinator(shard_addresses) as remote:
        remote.add(vectors, payloads)
        for q in vectors[:10] + 0.1:
            assert remote.search_hits(q, 5, where=where) == local.search_hits(q, 5, where=where)
            assert remote.search(q, 5, where=where) == _truth(vectors, payloads, q, 5, where)
        assert remote.search_partial(vectors[0], 5, where=where)[1] == []
        with pytest.raises(ValueError):
            remote.search(vectors[0], 5, where={"score": 0.5})       # refused before any request


@pytest.mark.parametrize("bad", ["not json", '{"score": [0.5]}', '["color"]'])
def test_server_rejects_a_bad_filter(shard_addresses, bad):
    with grpc.insecure_channel(shard_addresses[0]) as channel:
        stub = shard_pb2_grpc.ShardServiceStub(channel)
        with pytest.raises(grpc.RpcError) as err:
            stub.Search(shard_pb2.SearchRequest(query=[1.0] * 16, k=3, filter=bad), timeout=5)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize("checkpoint", [False, True])
def test_filters_work_after_a_restart(tmp_path, checkpoint):
    # The payload index is not stored; it is rebuilt from the recovered payloads.
    vectors, payloads = _data(200)
    where = normalize_filter({"color": "green", "tags": "even"})
    shard = DurableShard(tmp_path, snapshot_every=0, seed=0, **PARAMS)
    shard.add_batch(list(range(120)), vectors[:120], encode_payloads(payloads[:120], 120))
    if checkpoint:
        shard.checkpoint()
    before = [shard.search(q, 5, where) for q in vectors[:10]]
    shard.close()

    recovered = DurableShard(tmp_path, snapshot_every=0, seed=0, **PARAMS)
    assert [recovered.search(q, 5, where) for q in vectors[:10]] == before
    # Vectors added after the index was built are picked up too.
    recovered.add_batch(list(range(120, 200)), vectors[120:], encode_payloads(payloads[120:], 80))
    found = {g for _, g in recovered.search(vectors[150], 200, where)}
    assert found == {i for i in range(200) if payloads[i]["color"] == "green" and i % 2 == 0}
