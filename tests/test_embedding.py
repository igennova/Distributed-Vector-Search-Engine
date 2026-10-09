"""Embedders: the vectors have the promised shape, and texts sharing words are close."""
import subprocess
import sys

import numpy as np
import pytest

from vsearch.embedding import HashingEmbedder, get_embedder


def test_vectors_are_unit_length_float32():
    embedder = get_embedder("hash")
    vectors = embedder.embed_documents(["a replica catches up from its twin", "snapshots and logs"])
    assert vectors.shape == (2, embedder.dim) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    assert embedder.embed_query("replica").shape == (embedder.dim,)
    assert embedder.embed_documents([]).shape == (0, embedder.dim)


def test_texts_that_share_words_are_closer():
    embedder = HashingEmbedder()
    query = embedder.embed_query("how does a replica catch up")
    related, unrelated = embedder.embed_documents([
        "A replica that missed writes will catch up from the log of its twin.",
        "Cosine distance compares the direction of two vectors.",
    ])
    assert query @ related > 0.3 > query @ unrelated


def test_plurals_and_capitals_match():
    embedder = HashingEmbedder()
    assert np.allclose(embedder.embed_query("Replicas"), embedder.embed_query("replica"))


def test_text_without_words_is_the_zero_vector():
    assert not HashingEmbedder().embed_query("the and of ...").any()


def test_the_same_text_gets_the_same_vector_in_another_process():
    # A stored vector must still match its text after a restart. Python salts hash() per
    # process, so run a second interpreter with a different salt and compare.
    text = "write-ahead log and snapshots"
    code = ("from vsearch.embedding import HashingEmbedder;"
            f"print(HashingEmbedder().embed_query({text!r}).tobytes().hex())")
    other = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           check=True, env={"PYTHONHASHSEED": "12345"})
    assert bytes.fromhex(other.stdout.strip()) == HashingEmbedder().embed_query(text).tobytes()


def test_unknown_embedder_is_rejected():
    with pytest.raises(ValueError):
        get_embedder("nope")
