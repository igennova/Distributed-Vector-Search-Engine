"""
Tiny hand-checkable tests for Phase 0.  (PROVIDED.)
Run:  python test_search.py
Make these pass before touching the benchmark.
"""
import numpy as np
from search import brute_force_search


def test_nearest_is_identical_vector():
    # query equals vector index 1 exactly -> nearest neighbor must be index 1
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    query = np.array([0, 1, 0], dtype=np.float32)
    out = list(np.asarray(brute_force_search(vectors, query, k=1)))
    assert out[0] == 1, f"expected first result 1, got {out}"


def test_returns_k_in_similarity_order():
    # cosine(query,v0)=1.0, cosine(query,v1)=0.994, cosine(query,v2)=0.0
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    query = np.array([1, 0], dtype=np.float32)
    out = list(np.asarray(brute_force_search(vectors, query, k=2)))
    assert out == [0, 1], f"expected [0, 1], got {out}"


if __name__ == "__main__":
    test_nearest_is_identical_vector()
    test_returns_k_in_similarity_order()
    print("All tests passed ✅  — now run: python benchmark.py")
