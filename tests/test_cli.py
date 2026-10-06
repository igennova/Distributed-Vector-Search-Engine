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
