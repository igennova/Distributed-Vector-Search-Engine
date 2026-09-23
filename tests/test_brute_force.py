"""Tests for exact brute-force search on small, hand-checkable inputs."""
import numpy as np
from vsearch.brute_force import brute_force_search


def test_nearest_is_identical_vector():
    # query equals vector 1 exactly -> nearest neighbor must be index 1
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
