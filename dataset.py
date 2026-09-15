"""
Dataset + ground-truth oracle.  (PROVIDED — you don't need to edit this.)

- make_dataset(): synthetic vectors + queries to search over.
- exact_neighbors(): the TRUE nearest neighbors by exact cosine, computed with a
  fully-vectorized numpy pass. This is the "answer key" your recall is measured against.
  Note it's deliberately separate from search.py's brute_force_search: this one is the
  trusted oracle, yours is the implementation you're learning to write.
"""
import numpy as np


def make_dataset(n_vectors=10_000, dim=128, n_queries=100, seed=42):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    return vectors, queries


def normalize(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def exact_neighbors(vectors, queries, k=10):
    """True top-k neighbor indices per query, by cosine similarity, most-similar-first."""
    v = normalize(vectors)
    q = normalize(queries)
    sims = q @ v.T                                  # (n_queries, n_vectors)
    topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(queries.shape[0])[:, None]
    order = np.argsort(-sims[rows, topk], axis=1)   # sort the k by real similarity
    return topk[rows, order]                        # (n_queries, k)
