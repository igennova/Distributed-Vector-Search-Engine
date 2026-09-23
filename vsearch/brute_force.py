"""Exact nearest-neighbor search by cosine similarity (the baseline every index is measured against)."""
import numpy as np


def brute_force_search(vectors, query, k=10):
    """
    Return the indices of the k vectors most similar to `query` by cosine similarity,
    most similar first. Compares the query against every vector, so it is always exact
    and costs O(n) per query.

    Args:
        vectors: np.ndarray, shape (n, dim)
        query:   np.ndarray, shape (dim,)
        k:       number of neighbors to return
    """
    scores = []
    for i, v in enumerate(vectors):
        sim = np.dot(query, v) / (np.linalg.norm(query) * np.linalg.norm(v))
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scores[:k]]
