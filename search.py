"""
Search implementations.  ← THIS is the file YOU write.

Phase 0: implement brute_force_search below.
Later phases (you'll add these as new functions/classes): HNSWIndex, IVFIndex, etc.
"""
import numpy as np


def brute_force_search(vectors, query, k=10):
    """
    Return the indices of the k most similar vectors to `query`, by COSINE similarity,
    sorted most-similar-first.

    Args:
        vectors: np.ndarray, shape (n, dim)
        query:   np.ndarray, shape (dim,)
        k:       int, number of neighbors to return

    Returns:
        array-like of k integer indices into `vectors`.

    Why this matters: this is your correctness oracle. Every later index (HNSW, IVF)
    will be scored on how many of ITS results match this exact answer — that's recall.
    If this is wrong, every recall number you ever print is wrong.

    Hint:
      cosine(a, b) = dot(a, b) / (||a|| * ||b||)
      Start with a readable for-loop (compute similarity to every vector, pick top k).
      Once the tests pass, try a vectorized version and compare the speed in benchmark.py.
    """
    raise NotImplementedError("Phase 0: implement brute_force_search (see the hint above).")
