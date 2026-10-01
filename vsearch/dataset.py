"""
Data to search over, and the exact ground truth used to score every index.

- make_dataset(): random vectors plus queries to search with.
- load_glove(): real word vectors (Stanford GloVe 6B), with their words.
- glove_benchmark_split(): GloVe base vectors plus held-out query words.
- exact_neighbors(): the true top-k neighbors by cosine similarity, computed with
  vectorized numpy. Recall is measured against this.
"""
import io
import zipfile
from pathlib import Path

import numpy as np

GLOVE_ZIP = "glove.6B.zip"     # from https://nlp.stanford.edu/projects/glove/ (862 MB)


def make_dataset(n_vectors=10_000, dim=128, n_queries=100, seed=42):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    return vectors, queries


def load_glove(data_dir="data", dim=100):
    """Stanford GloVe 6B word vectors as (words, vectors), most frequent words first.

    The first call reads the chosen dimension out of glove.6B.zip in data_dir and caches it
    as a .npy file plus a word list, which later calls load in about a second.
    """
    data_dir = Path(data_dir)
    vectors_path = data_dir / f"glove.6B.{dim}d.npy"
    words_path = data_dir / f"glove.6B.{dim}d.words.txt"
    if not vectors_path.exists():
        _cache_glove(data_dir / GLOVE_ZIP, dim, vectors_path, words_path)
    words = words_path.read_text(encoding="utf-8").split("\n")
    return words, np.load(vectors_path)


def _cache_glove(zip_path, dim, vectors_path, words_path):
    words, rows = [], []
    with zipfile.ZipFile(zip_path) as archive, archive.open(f"glove.6B.{dim}d.txt") as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            word, *values = line.rstrip("\n").split(" ")
            if len(values) != dim:
                raise ValueError(f"expected {dim} values for {word!r}, got {len(values)}")
            words.append(word)
            rows.append(np.array(values, dtype=np.float32))
    np.save(vectors_path, np.stack(rows))
    words_path.write_text("\n".join(words), encoding="utf-8")


def glove_benchmark_split(n_base, n_queries=1_000, data_dir="data", seed=0):
    """GloVe vectors for benchmarking: (base_words, base_vectors, query_vectors).

    The queries are n_queries random words held out of the base; the base is the n_base most
    frequent of the remaining words. The queries don't depend on n_base, so results at
    different sizes are measured with the same queries.
    """
    words, vectors = load_glove(data_dir)
    query_ids = np.random.default_rng(seed).choice(len(words), size=n_queries, replace=False)
    held_out = np.zeros(len(words), dtype=bool)
    held_out[query_ids] = True
    base_ids = np.flatnonzero(~held_out)[:n_base]
    return [words[i] for i in base_ids], vectors[base_ids], vectors[query_ids]


def normalize(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def exact_neighbors(vectors, queries, k=10, batch=256):
    """True top-k neighbor indices per query, by cosine similarity, most-similar-first.

    Queries are processed in batches so the similarity matrix stays small for big datasets.
    """
    v = normalize(vectors)
    out = []
    for start in range(0, len(queries), batch):
        q = normalize(queries[start:start + batch])
        sims = q @ v.T                                  # (batch, n_vectors)
        topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(len(q))[:, None]
        order = np.argsort(-sims[rows, topk], axis=1)   # sort the k by real similarity
        out.append(topk[rows, order])
    return np.concatenate(out)                          # (n_queries, k)
