"""End-to-end tests of the command line, against a tiny stand-in for the GloVe data."""
import numpy as np
import pytest

from vsearch.cli import main
from vsearch.dataset import normalize


@pytest.fixture
def glove_dir(tmp_path):
    """A cache like load_glove() writes: 300 'words' (w0..w299) with random 16-dim vectors."""
    vectors = np.random.default_rng(0).standard_normal((300, 16)).astype(np.float32)
    np.save(tmp_path / "glove.6B.100d.npy", vectors)
    (tmp_path / "glove.6B.100d.words.txt").write_text("\n".join(f"w{i}" for i in range(300)))
    return tmp_path


def run(home, *argv):
    return main(["--home", str(home), *argv])


def result_words(output):
    """The words from `vsearch similar` output lines such as '  1. w17   0.642'."""
    return [line.split()[1] for line in output.splitlines() if line.strip()[:1].isdigit()]


def test_cluster_lifecycle(tmp_path, glove_dir, capsys):
    home = tmp_path / "cluster"
    indexed = normalize(np.load(glove_dir / "glove.6B.100d.npy")[:200])
    expected = [f"w{i}" for i in np.argsort(-(indexed @ indexed[5]))[1:4]]   # true neighbors of w5
    try:
        assert run(home, "cluster", "up", "--shards", "2", "--replicas", "2") == 0
        assert run(home, "load", "glove", "--limit", "200", "--glove-dir", str(glove_dir)) == 0
        capsys.readouterr()
        assert run(home, "similar", "w5", "-k", "3") == 0
        assert result_words(capsys.readouterr().out) == expected

        # A crashed replica doesn't stop searches, and shows up as down.
        assert run(home, "cluster", "kill", "0", "1") == 0
        assert run(home, "similar", "w5", "-k", "3") == 0
        assert result_words(capsys.readouterr().out) == expected
        assert run(home, "status") == 0
        assert "down" in capsys.readouterr().out

        # Restarted, it recovers its data from disk, so there is nothing to repair.
        assert run(home, "cluster", "restart", "0", "1") == 0
        assert run(home, "repair") == 0
        assert "in sync" in capsys.readouterr().out

        # Stopping the cluster keeps the data; starting it again brings everything back.
        assert run(home, "cluster", "down") == 0
        assert run(home, "cluster", "up") == 0
        capsys.readouterr()
        assert run(home, "similar", "w5", "-k", "3") == 0
        assert result_words(capsys.readouterr().out) == expected
    finally:
        run(home, "cluster", "down", "--wipe")


def test_helpful_errors(tmp_path, glove_dir, capsys):
    home = tmp_path / "cluster"
    assert run(home, "status") == 1                       # no cluster yet
    assert "vsearch cluster up" in capsys.readouterr().err
    try:
        assert run(home, "cluster", "up", "--shards", "1", "--replicas", "1") == 0
        assert run(home, "cluster", "up") == 1            # already running
        assert run(home, "similar", "w5") == 1            # nothing loaded
        assert "vsearch load glove" in capsys.readouterr().err
        assert run(home, "cluster", "kill", "3", "0") == 1   # no such server
    finally:
        run(home, "cluster", "down", "--wipe")


GUIDE = """\
# Storage

## Write-ahead log

Every write is appended to a log file before it is applied. After a crash the server
replays the log and loses nothing.

## Snapshots

A snapshot saves the whole graph to disk, so recovery does not have to replay every write.
"""


def result_sources(output):
    """Where each passage came from, from `vsearch search` lines such as
    '  1. 0.512  docs/guide.md > Storage > Snapshots'."""
    return [line.split(None, 2)[2] for line in output.splitlines() if line.strip()[:1].isdigit()]


def test_ingest_and_search_documents(tmp_path, glove_dir, capsys, monkeypatch):
    home = tmp_path / "cluster"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(GUIDE)
    (docs / "notes.txt").write_text("Replicas answer searches in turn.\n")
    monkeypatch.chdir(tmp_path)
    question = "what happens to the log after a crash?"
    try:
        assert run(home, "search", question) == 1          # no cluster yet
        assert run(home, "cluster", "up", "--shards", "2", "--replicas", "2") == 0
        assert run(home, "search", question) == 1          # nothing ingested
        assert "vsearch ingest" in capsys.readouterr().err

        assert run(home, "ingest", "docs") == 0
        assert "ingested 3 passages" in capsys.readouterr().out
        assert run(home, "search", question, "-k", "2") == 0
        output = capsys.readouterr().out
        assert result_sources(output)[0] == "docs/guide.md > Storage > Write-ahead log"
        assert "replays the log" in " ".join(output.split())     # the snippet is wrapped

        # The same files again add nothing; a changed file is reported, not duplicated.
        assert run(home, "ingest", "docs") == 0
        assert "ingested 0 passages" in capsys.readouterr().out
        (docs / "notes.txt").write_text("Something new.\n")
        (docs / "more.md").write_text("# Sharding\n\nVectors are spread over shards by id.\n")
        assert run(home, "ingest", "docs") == 0
        output = capsys.readouterr().out
        assert "notes.txt: changed" in output and "ingested 1 passages" in output
        assert run(home, "status") == 0
        assert "ingested: 4 passages from 3 file(s)" in capsys.readouterr().out

        # Words and documents do not mix in one cluster.
        assert run(home, "similar", "w5") == 1
        assert run(home, "load", "glove", "--glove-dir", str(glove_dir)) == 1
        assert run(home, "ingest", "missing-folder") == 1
        capsys.readouterr()

        # Passages are payloads, so they survive a crash and a full restart like the vectors.
        assert run(home, "cluster", "kill", "1", "0") == 0
        assert run(home, "cluster", "down") == 0
        assert run(home, "cluster", "up") == 0
        capsys.readouterr()
        assert run(home, "search", "spread over shards", "--full") == 0
        output = capsys.readouterr().out
        assert result_sources(output)[0] == "docs/more.md > Sharding"
        assert "Vectors are spread over shards by id." in output
    finally:
        run(home, "cluster", "down", "--wipe")


def test_documents_cannot_be_ingested_into_a_word_cluster(tmp_path, glove_dir, capsys):
    home = tmp_path / "cluster"
    (tmp_path / "a.md").write_text("Some text.\n")
    try:
        assert run(home, "cluster", "up", "--shards", "1", "--replicas", "1") == 0
        assert run(home, "load", "glove", "--limit", "50", "--glove-dir", str(glove_dir)) == 0
        assert run(home, "ingest", str(tmp_path / "a.md")) == 1
        assert run(home, "search", "text") == 1
        assert "vsearch similar" in capsys.readouterr().err
    finally:
        run(home, "cluster", "down", "--wipe")
