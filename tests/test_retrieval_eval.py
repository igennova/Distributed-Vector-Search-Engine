"""The retrieval eval: its questions match the corpus, and its scores are computed right."""
from collections import Counter

import pytest

from evals.retrieval_eval import CORPUS, evaluate, load_questions
from vsearch.embedding import HashingEmbedder
from vsearch.ingest import read_passages


def test_every_question_points_at_a_real_section_and_the_kinds_are_balanced():
    questions = load_questions()
    sections = {passage["heading"] for passage in read_passages(CORPUS)}
    assert {question["section"] for question in questions} <= sections
    assert Counter(question["kind"] for question in questions) == {"keyword": 20, "paraphrase": 20}
    assert len({question["question"] for question in questions}) == len(questions)


def test_retrieval_scores_on_a_tiny_corpus(tmp_path):
    corpus = tmp_path / "notes.md"
    corpus.write_text("# Cats\n\nCats purr and sleep all day.\n\n# Rockets\n\nRockets burn fuel to reach orbit.\n")
    questions = [{"kind": "keyword", "question": "do cats purr", "section": "Cats"},
                 {"kind": "keyword", "question": "what fuel do rockets burn", "section": "Rockets"},
                 {"kind": "paraphrase", "question": "spacecraft propulsion", "section": "Rockets"}]
    results = evaluate(HashingEmbedder(), questions, corpus=corpus)
    assert results["keyword"] == {"n": 2, "hit@1": 1.0, "hit@5": 1.0, "mrr": 1.0}
    assert results["all"]["n"] == 3
    with pytest.raises(ValueError, match="Dogs"):
        evaluate(HashingEmbedder(), [{"kind": "keyword", "question": "x", "section": "Dogs"}],
                 corpus=corpus)
