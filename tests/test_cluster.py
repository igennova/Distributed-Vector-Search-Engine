"""Tests for sharded search (coordinator + shards)."""
import numpy as np
import pytest
from vsearch.cluster import Coordinator
from vsearch.dataset import exact_neighbors


def test_every_vector_lands_in_exactly_one_shard():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((103, 16)).astype(np.float32)
    coord = Coordinator(4, M=8, ef_construction=32, ef_search=32, seed=0).add(data)

    all_ids = sorted(gid for shard in coord.shards for gid in shard.global_ids)
    assert all_ids == list(range(103))
    sizes = [len(s) for s in coord.shards]
    assert max(sizes) - min(sizes) <= 1          # round-robin keeps shards balanced


def test_top_k_all_in_one_shard_is_still_found():
    # Place the query's 10 true nearest neighbours on the SAME shard. If each shard
    # returned only k / num_shards results, most of them would be lost in the merge.
    num_shards, k = 4, 10
    rng = np.random.default_rng(1)
    data = rng.standard_normal((400, 16)).astype(np.float32)
    query = rng.standard_normal(16).astype(np.float32)
    for j in range(k):
        data[j * num_shards] = query + 0.01 * rng.standard_normal(16)   # ids 0,4,8,... -> shard 0

    expected = set(exact_neighbors(data, query[None, :], k)[0].tolist())
    assert expected == {j * num_shards for j in range(k)}               # setup sanity check

    coord = Coordinator(num_shards, M=8, ef_construction=64, ef_search=64, seed=0).add(data)
    assert set(coord.search(query, k)) == expected


def test_sharded_recall_is_high():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((600, 32)).astype(np.float32)
    queries = rng.standard_normal((30, 32)).astype(np.float32)
    coord = Coordinator(3, M=8, ef_construction=64, ef_search=32, seed=0).add(data)

    truth = exact_neighbors(data, queries, 10)
    hits = sum(len(set(coord.search(q, 10)) & set(t.tolist())) for q, t in zip(queries, truth))
    recall = hits / (10 * len(queries))
    assert recall > 0.85, f"recall too low: {recall:.3f}"


@pytest.mark.parametrize("mode", ["threads", "processes"])
def test_parallel_modes_return_same_results_as_sequential(mode):
    # Parallel fan-out must change only how fast results arrive, never what they are.
    rng = np.random.default_rng(3)
    data = rng.standard_normal((200, 16)).astype(np.float32)
    queries = rng.standard_normal((10, 16)).astype(np.float32)
    params = dict(M=8, ef_construction=32, ef_search=32, seed=0)

    with Coordinator(4, **params) as sequential, Coordinator(4, mode=mode, **params) as parallel:
        sequential.add(data)
        parallel.add(data)
        for q in queries:
            assert parallel.search(q, 5) == sequential.search(q, 5)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        Coordinator(2, mode="gpu")
