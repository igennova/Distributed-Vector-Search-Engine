"""Filters: what a filter may contain, the payload index that answers it, and the two ways
the HNSW index searches among matching vectors only."""
import numpy as np
import pytest

from vsearch.cluster import encode_payloads
from vsearch.dataset import normalize
from vsearch.filtering import PayloadIndex, decode_filter, encode_filter, normalize_filter
from vsearch.hnsw import HNSW

PARAMS = dict(M=8, ef_construction=32, ef_search=32)


def _data(n=600, dim=16):
    return np.random.default_rng(11).standard_normal((n, dim)).astype(np.float32)


# --- the filter itself ---------------------------------------------------------------------

def test_a_filter_is_normalized_to_lists_of_values():
    assert normalize_filter({"color": "red", "size": [1, 2]}) == {"color": ["red"], "size": [1, 2]}
    assert normalize_filter(None) is None and normalize_filter({}) is None
    assert decode_filter(encode_filter({"color": "red"})) == {"color": ["red"]}
    assert encode_filter(None) == "" and decode_filter("") is None


@pytest.mark.parametrize("bad", [
    {"score": 0.5},              # floats are not indexed
    {"text": "x" * 500},         # nor are long strings
    {"meta": {"a": 1}},          # nor nested values
    {"color": []},               # nothing to match
    ["color", "red"],            # not a dict
])
def test_a_filter_the_index_cannot_answer_is_rejected(bad):
    with pytest.raises(ValueError):
        normalize_filter(bad)


def test_payload_index_combines_fields_with_and_and_values_with_or():
    index = PayloadIndex()
    payloads = encode_payloads([
        {"color": "red", "size": 1, "tags": ["a", "b"]},
        {"color": "blue", "size": 1, "flag": True},
        None,
        {"color": "red", "size": 2, "text": "y" * 500, "score": 0.5, "meta": {"color": "red"}},
        {"color": "green", "size": 1, "flag": 1},
    ], 5)
    index.catch_up(payloads[:3])
    index.catch_up(payloads)                 # picks up where it stopped

    def ids(where):
        return np.flatnonzero(index.matching(normalize_filter(where), 5)).tolist()

    assert ids({"color": "red"}) == [0, 3]
    assert ids({"color": "red", "size": 1}) == [0]
    assert ids({"color": ["red", "blue"], "size": 1}) == [0, 1]
    assert ids({"tags": "b"}) == [0]                 # a list field matches on any item
    assert ids({"flag": True}) == [1] and ids({"flag": 1}) == [4]     # True is not 1
    assert ids({"color": "purple"}) == [] and ids({"missing": 1}) == []


# --- searching among allowed nodes --------------------------------------------------------

def test_filtered_graph_search_returns_only_allowed_nodes():
    vectors = _data(800)
    index = HNSW(seed=0, **PARAMS).build(vectors)
    allowed = np.arange(800) % 3 == 0
    unit, hits, total = normalize(vectors), 0, 0
    for q in vectors[:40] + 0.1:
        got = [i for _, i in index.search_with_distances(q, 10, allowed)]
        assert len(got) == 10 and all(allowed[i] for i in got)
        ids = np.flatnonzero(allowed)
        expected = ids[np.argsort(-(unit[ids] @ q))[:10]]
        hits += len(set(got) & set(expected.tolist()))
        total += 10
    assert hits / total > 0.9


def test_filtered_graph_search_keeps_walking_until_it_has_enough_matches():
    # Six nodes are allowed: the query's own vector and the five least like it. Having found
    # the first, the walk must not stop at the next (farther) node as an unfiltered search
    # would; it has to pass through nodes that are not allowed to reach the other five.
    vectors = _data(800)
    index = HNSW(seed=0, **PARAMS).build(vectors)
    order = np.argsort(normalize(vectors) @ normalize(vectors[:1])[0])
    wanted = [0] + order[:5].tolist()
    allowed = np.zeros(800, dtype=bool)
    allowed[wanted] = True
    got = [i for _, i in index.search_with_distances(vectors[0], 10, allowed)]
    assert got[0] == 0 and sorted(got) == sorted(wanted)


def test_exact_among_is_the_true_answer():
    vectors = _data(500)
    index = HNSW(seed=0, **PARAMS).build(vectors)
    ids = np.arange(0, 500, 7)
    for q in vectors[:20] + 0.1:
        expected = ids[np.argsort(-(normalize(vectors[ids]) @ q), kind="stable")[:10]].tolist()
        scored = index.exact_among(q, ids, 10)
        assert [i for _, i in scored] == expected
        assert [d for d, _ in scored] == sorted(d for d, _ in scored)
    assert index.exact_among(vectors[0], [], 10) == []
    assert len(index.exact_among(vectors[0], [3, 4], 10)) == 2       # fewer than k exist
