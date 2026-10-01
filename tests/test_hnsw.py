"""Tests for HNSW: the graph-search primitives on a hand-built graph, and a built index's recall."""
import numpy as np
import pytest
from vsearch.hnsw import HNSW

# Six points fanned out by angle in 2D, connected as a chain (each node linked to its
# immediate neighbors). A chain has no local minima, so a greedy walk from any start
# reaches the true nearest node.
POINTS = np.array([
    [1.0, 0.0],   # 0
    [1.0, 0.3],   # 1
    [1.0, 0.7],   # 2
    [1.0, 1.0],   # 3
    [0.6, 1.0],   # 4
    [0.2, 1.0],   # 5
], dtype=np.float32)

CHAIN = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}


def _index():
    h = HNSW()
    h.vectors = POINTS
    h.graph = [CHAIN]
    h.entry_point = 0
    h.top_layer = 0
    return h


def test_greedy_descend_reaches_true_nearest():
    h = _index()
    query = np.array([0.65, 1.0], dtype=np.float32)
    expected = int(np.argmin([h._distance(query, v) for v in POINTS]))
    assert h._greedy_descend(query, entry=0, layer=0) == expected


def test_search_layer_returns_true_top_k():
    h = _index()
    query = np.array([0.65, 1.0], dtype=np.float32)
    k = 3
    expected = set(int(i) for i in np.argsort([h._distance(query, v) for v in POINTS])[:k])
    assert set(h._search_layer(query, entry=0, layer=0, ef=k)) == expected


def test_search_layer_ef1_matches_greedy():
    h = _index()
    query = np.array([0.65, 1.0], dtype=np.float32)
    greedy = h._greedy_descend(query, entry=0, layer=0)
    assert h._search_layer(query, entry=0, layer=0, ef=1) == [greedy]


def test_built_index_has_high_recall():
    # Build a real multi-layer index and check search recall against exact top-k.
    rng = np.random.default_rng(0)
    data = rng.standard_normal((300, 32)).astype(np.float32)
    queries = rng.standard_normal((50, 32)).astype(np.float32)

    index = HNSW(M=8, ef_construction=64, ef_search=32, seed=1).build(data)

    def exact_top_k(q, k):
        return set(int(i) for i in np.argsort([index._distance(q, v) for v in data])[:k])

    hits = total = 0
    for q in queries:
        got = set(index.search(q, k=10))
        hits += len(got & exact_top_k(q, 10))
        total += 10
    recall = hits / total
    assert recall > 0.85, f"recall too low: {recall:.3f}"


def _fan(*degrees):
    """Unit vectors in 2D at the given angles; the base node (id 0) sits at 0 degrees."""
    h = HNSW()
    h.vectors = np.array([[np.cos(np.radians(d)), np.sin(np.radians(d))] for d in (0, *degrees)],
                         dtype=np.float32)
    ids = list(range(1, len(degrees) + 1))
    return h, ids, 1.0 - h.vectors[ids] @ h.vectors[0]


def test_heuristic_prefers_neighbors_in_different_directions():
    # Candidates at 10 and 12 degrees are almost the same direction; -40 degrees is not.
    h, ids, dists = _fan(10, 12, -40)

    h.neighbor_selection = "closest"
    assert h._select_neighbors(ids, dists, 2) == [1, 2]      # the two nearest, side by side

    h.neighbor_selection = "heuristic"
    assert h._select_neighbors(ids, dists, 2) == [1, 3]      # 12 deg is reachable via 10 deg


def test_heuristic_may_use_fewer_slots_than_allowed():
    h, ids, dists = _fan(10, 12, -40)
    assert h._select_neighbors(ids, dists, 3) == [1, 2, 3]   # no more candidates than slots: keep all
    h, ids, dists = _fan(10, 12, 14, -40)
    assert h._select_neighbors(ids, dists, 3) == [1, 4]      # 12 and 14 deg are redundant: left out


def test_unknown_neighbor_selection_is_rejected():
    with pytest.raises(ValueError):
        HNSW(neighbor_selection="random")
