"""Ingestion: files become passages, and a question finds the passage that answers it."""
import pytest

from vsearch.client import GrpcCoordinator
from vsearch.cluster import Coordinator
from vsearch.embedding import HashingEmbedder
from vsearch.ingest import (embedding_text, find_documents, fingerprint, ingest_passages,
                            read_passages, search_passages)
from vsearch.server import start_server

PARAMS = dict(M=8, ef_construction=32, ef_search=32)

GUIDE = """\
# Storage

## Write-ahead log

Every write is appended to a log file before it is applied. After a crash the server
replays the log and loses nothing.

## Snapshots

A snapshot saves the whole graph to disk, so recovery does not have to replay every write.
"""

NOTES = "Replicas answer searches in turn. When one fails, the coordinator retries on its twin.\n"


@pytest.fixture
def docs(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text(GUIDE)
    (tmp_path / "docs" / "notes.txt").write_text(NOTES)
    (tmp_path / "docs" / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "docs" / ".cache").mkdir()
    (tmp_path / "docs" / ".cache" / "hidden.md").write_text("# hidden")
    return tmp_path / "docs"


def test_find_documents_takes_text_files_once_and_skips_hidden_folders(docs):
    found = find_documents([docs, docs / "guide.md"])
    assert [path.name for path in found] == ["guide.md", "notes.txt"]
    assert all(path.is_absolute() for path in found)
    # A file named explicitly is taken whatever its suffix or folder.
    assert find_documents([docs / ".cache" / "hidden.md"])[0].name == "hidden.md"
    with pytest.raises(FileNotFoundError):
        find_documents([docs / "missing"])


def test_fingerprint_changes_with_the_contents(docs):
    before = fingerprint(docs / "notes.txt")
    assert fingerprint(docs / "notes.txt") == before
    (docs / "notes.txt").write_text(NOTES + "One more line.\n")
    assert fingerprint(docs / "notes.txt") != before


def test_passages_carry_their_source_and_headings(docs):
    passages = read_passages(docs / "guide.md", source="guide.md")
    assert [(p["source"], p["heading"], p["chunk"]) for p in passages] == [
        ("guide.md", "Storage > Write-ahead log", 0), ("guide.md", "Storage > Snapshots", 1)]
    assert passages[0]["text"].startswith("Every write is appended")
    assert embedding_text(passages[0]).startswith("Storage > Write-ahead log\n\nEvery write")

    plain = read_passages(docs / "notes.txt")
    assert plain[0]["heading"] == "" and embedding_text(plain[0]) == plain[0]["text"]


def _ingest_all(coordinator, embedder, docs):
    return sum(ingest_passages(coordinator, embedder, read_passages(path, source=path.name))
               for path in find_documents([docs]))


def test_a_question_finds_the_passage_that_answers_it(docs):
    embedder = HashingEmbedder()
    with Coordinator(2, seed=0, **PARAMS) as coordinator:
        assert _ingest_all(coordinator, embedder, docs) == 3
        best = search_passages(coordinator, embedder, "what happens to the log after a crash?", k=3)[0]
        assert best.payload["heading"] == "Storage > Write-ahead log"
        assert "replays the log" in best.payload["text"]

        best = search_passages(coordinator, embedder, "which replica does the coordinator retry on", k=1)[0]
        assert best.payload["source"] == "notes.txt"


def test_ingest_and_search_over_grpc(docs):
    embedder = HashingEmbedder()
    servers = [start_server(port=0, seed=i, **PARAMS) for i in range(2)]
    try:
        with GrpcCoordinator([f"127.0.0.1:{port}" for _, port in servers]) as coordinator:
            assert _ingest_all(coordinator, embedder, docs) == 3
            hits = search_passages(coordinator, embedder, "saves the whole graph to disk", k=2)
            assert hits[0].payload == {
                "text": "A snapshot saves the whole graph to disk, so recovery does not have to "
                        "replay every write.",
                "source": "guide.md", "heading": "Storage > Snapshots", "chunk": 1}
    finally:
        for server, _ in servers:
            server.stop(grace=None)


def test_ingesting_nothing_is_fine():
    with Coordinator(1, **PARAMS) as coordinator:
        assert ingest_passages(coordinator, HashingEmbedder(), []) == 0
