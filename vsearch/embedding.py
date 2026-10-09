"""
Embedders turn text into vectors, so that texts about the same thing end up close together.

Every embedder has the same small interface:

    name                      what it is called on the command line and in cluster.json
    dim                       length of the vectors it produces
    embed_documents(texts)    -> (len(texts), dim) float32 array, one row per text
    embed_query(text)         -> (dim,) float32 vector for a search query

Documents and queries go through separate methods because some models embed them
differently (for example with a different instruction in front of a query).

A collection must be searched with the embedder it was built with: vectors from two
different embedders live in unrelated spaces, and distances between them mean nothing.
"""
import re
import zlib

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
# Words too common to say anything about a text's topic.
_STOPWORDS = frozenset("""
    a an and are as at be been but by can do does for from had has have how if in into is it
    its no not of on or so such than that the their then there these they this to was we
    were what when where which who why will with you your
""".split())


class HashingEmbedder:
    """Bag-of-words vectors made with the hashing trick: each word is hashed to one of
    `dim` positions and counted there. No model and nothing to download.

    Two texts are close when they share words, not when they share meaning: "car" and
    "automobile" are unrelated to it. That is enough to exercise and test the whole
    pipeline, and it is the baseline a real embedding model has to beat.
    """
    name = "hash"

    def __init__(self, dim=512):
        self.dim = dim

    def _tokens(self, text):
        for token in _TOKEN.findall(text.lower()):
            if token in _STOPWORDS:
                continue
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]                 # "replicas" and "replica" count as one word
            yield token

    def _embed(self, text):
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in self._tokens(text):
            # crc32 gives the same value in every process; Python's hash() of a string does not.
            code = zlib.crc32(token.encode("utf-8"))
            # A second bit of the hash picks the sign, so words that collide on a position
            # cancel out on average instead of piling up.
            vector[code % self.dim] += 1.0 if (code >> 16) & 1 else -1.0
        # Dampen repeats: a word used ten times is not ten times as important.
        vector = np.sign(vector) * np.log1p(np.abs(vector))
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def embed_documents(self, texts):
        return np.stack([self._embed(text) for text in texts]) if len(texts) else \
            np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text):
        return self._embed(text)


EMBEDDERS = {"hash": HashingEmbedder}


def get_embedder(name):
    """Create the embedder with this name."""
    try:
        return EMBEDDERS[name]()
    except KeyError:
        raise ValueError(f"unknown embedder {name!r}; choose from {sorted(EMBEDDERS)}") from None
