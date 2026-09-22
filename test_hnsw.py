"""
Tests for the HNSW graph-search primitives on a small hand-built graph.
Run:  python test_hnsw.py
"""
import numpy as np
from hnsw import HNSW

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


if __name__ == "__main__":
    test_greedy_descend_reaches_true_nearest()
    test_search_layer_returns_true_top_k()
    test_search_layer_ef1_matches_greedy()
    test_built_index_has_high_recall()
    print("All HNSW tests passed ✅")
