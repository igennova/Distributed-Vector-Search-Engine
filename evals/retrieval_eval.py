"""
How often does search find the passage that answers a question? A fixed set of 40
questions about this project's design notes (a frozen copy in evals/corpus), each labelled
with the section that answers it. Half reuse that section's own words ("keyword"); half
were written to avoid them ("paraphrase"), which is where matching by meaning should beat
matching by words.

  hit@1   the top passage comes from the right section
  hit@5   one of the top five does
  MRR     mean of 1 / (rank of the first passage from the right section), 0 if not in the top 10

Passages are ranked by exact cosine similarity, so this measures the embedder and the
chunking, not the approximate index (whose recall the benchmarks measure).

The questions were written by the author of the notes, who knows their wording; scores on
someone else's questions would likely be lower.

Run:  python -m evals.retrieval_eval
"""
import json
import sys
from pathlib import Path

import numpy as np

from vsearch.embedding import EmbedderUnavailable, get_embedder
from vsearch.ingest import embedding_text, read_passages

HERE = Path(__file__).parent
CORPUS = HERE / "corpus" / "DECISIONS.md"
QUESTIONS = HERE / "questions.jsonl"
SETUPS = [  # embedder, longest passage, embed the headings with the text
    ("hash", 1000, True), ("bge-small", 1000, True),
    ("bge-small", 500, True), ("bge-small", 1000, False),
]


def load_questions(path=QUESTIONS):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate(embedder, questions, max_chars=1000, with_headings=True, corpus=CORPUS):
    """Scores for one setup: {kind: {"hit@1", "hit@5", "mrr", "n"}}, kinds plus "all"."""
    passages = read_passages(corpus, max_chars=max_chars, overlap=min(150, max_chars // 6))
    texts = [embedding_text(p) if with_headings else p["text"] for p in passages]
    vectors = embedder.embed_documents(texts)
    sections = [p["heading"] for p in passages]
    missing = {q["section"] for q in questions} - set(sections)
    if missing:
        raise ValueError(f"questions name sections the corpus does not have: {sorted(missing)}")

    ranks = []
    for question in questions:
        scores = vectors @ embedder.embed_query(question["question"])
        order = np.argsort(-scores, kind="stable")[:10]
        rank = next((i + 1 for i, passage in enumerate(order)
                     if sections[passage] == question["section"]), None)
        ranks.append((question["kind"], rank))

    results = {}
    for kind in sorted({kind for kind, _ in ranks}) + ["all"]:
        mine = [rank for k, rank in ranks if kind in ("all", k)]
        results[kind] = {
            "n": len(mine),
            "hit@1": sum(rank == 1 for rank in mine) / len(mine),
            "hit@5": sum(rank is not None and rank <= 5 for rank in mine) / len(mine),
            "mrr": sum(1 / rank for rank in mine if rank) / len(mine),
        }
    return results


def main():
    questions = load_questions()
    print(f"{len(questions)} questions on {CORPUS.name}\n")
    print(f"{'embedder':<11}{'passage':>8}{'headings':>10}  "
          + "".join(f"{kind:>24}" for kind in ("keyword", "paraphrase", "all")))
    print(f"{'':<31}" + "   hit@1  hit@5   MRR " * 3)
    print("-" * 103)
    embedders = {}
    for name, max_chars, with_headings in SETUPS:
        try:
            embedder = embedders.setdefault(name, get_embedder(name))
            results = evaluate(embedder, questions, max_chars, with_headings)
        except EmbedderUnavailable as err:
            print(f"{name:<11} skipped: {err}")
            continue
        cells = "".join(f"   {r['hit@1']:>5.0%}  {r['hit@5']:>5.0%}  {r['mrr']:>4.2f}"
                        for r in (results["keyword"], results["paraphrase"], results["all"]))
        print(f"{name:<11}{max_chars:>8}{'yes' if with_headings else 'no':>10}  {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
