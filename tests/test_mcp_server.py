"""The engine's MCP tools, against a real local cluster: in process for the details, and
once as the separate process a client would start."""
import json
import re
import subprocess
import sys

import pytest

from vsearch.cli import main
from vsearch.client import REPO_ROOT
from vsearch.mcp import INVALID_PARAMS
from vsearch.mcp_server import build_server

GUIDE = """\
# Storage

## Write-ahead log

Every write is appended to a log file before it is applied. After a crash the server
replays the log and loses nothing.

## Snapshots

A snapshot saves the whole graph to disk, so recovery does not have to replay every write.
"""


def run(home, *argv):
    return main(["--home", str(home), *argv])


def vectors_per_server(status_text):
    """The vector count of every server that is up, from the status tool's text."""
    return [int(count.replace(",", "")) for count in re.findall(r"up \(([\d,]+) vectors", status_text)]


def call(server, name, **arguments):
    """Call a tool; returns (text, is_error)."""
    reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": name, "arguments": arguments}})
    result = reply["result"]
    return result["content"][0]["text"], result["isError"]


@pytest.fixture
def cluster(tmp_path, monkeypatch, capsys):
    """A 2-shard, 2-replica cluster and some documents; the working directory is tmp_path."""
    home = tmp_path / "cluster"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text(GUIDE)
    (tmp_path / "docs" / "notes.txt").write_text("Replicas answer searches in turn.\n")
    (tmp_path / "extra.md").write_text("# Sharding\n\nVectors are spread over shards by id.\n")
    monkeypatch.chdir(tmp_path)
    assert run(home, "cluster", "up", "--shards", "2", "--replicas", "2") == 0
    capsys.readouterr()
    yield home
    run(home, "cluster", "down", "--wipe")


@pytest.fixture
def server(cluster):
    server = build_server(cluster, embedder="hash")
    yield server
    server.close()


def test_ingest_then_search_through_the_tools(server, tmp_path):
    text, is_error = call(server, "search", question="what happens after a crash?")
    assert is_error and "ingest tool" in text                 # tells the model what to do next
    text, _ = call(server, "status")
    assert "4 of 4 servers up (2 shards)." in text and "Nothing is loaded yet." in text

    text, is_error = call(server, "ingest", paths=[str(tmp_path / "docs")])
    assert not is_error and text.startswith("Stored 3 passages.")
    assert "docs/guide.md: 2 passages" in text
    text, _ = call(server, "ingest", paths=[str(tmp_path / "docs")])
    assert text.startswith("Stored 0 passages.") and "2 file(s) were already indexed" in text

    text, is_error = call(server, "search", question="what happens to the log after a crash?", k=2)
    assert not is_error and "asked all 2 shards" in text
    first = text.split("[1] ")[1].split("[2] ")[0]
    assert first.startswith("source: docs/guide.md | section: Storage > Write-ahead log | similarity: 0.")
    assert "replays the log and loses nothing." in first       # the whole passage, not a snippet
    assert "[2] " in text and "[3] " not in text

    text, _ = call(server, "search", question="log after a crash", source="docs/notes.txt")
    assert "source: docs/notes.txt" in text and "guide.md" not in text
    text, is_error = call(server, "search", question="log", source="docs/missing.md")
    assert not is_error and text.startswith("No passages matched")


def test_namespaces_and_status(server, cluster, tmp_path, capsys):
    call(server, "ingest", paths=[str(tmp_path / "docs")])
    text, _ = call(server, "ingest", paths=[str(tmp_path / "extra.md")], namespace="alice")
    assert text.startswith("Stored 1 passages into namespace 'alice'.")

    text, _ = call(server, "search", question="how are vectors spread", namespace="alice")
    assert "1 passages" in text and "asked shard 1 of 2" in text and "source: extra.md" in text
    text, _ = call(server, "search", question="how are vectors spread", namespace="bob")
    assert text.startswith("No passages matched (asked shard")

    text, _ = call(server, "status")
    assert "Ingested: 4 passages from 3 file(s) in 1 namespace(s), embedder: hash." in text
    assert "Namespaces: alice" in text
    assert "Files: docs/guide.md, docs/notes.txt, extra.md (namespace alice)" in text
    assert len(vectors_per_server(text)) == 4 and "DOWN" not in text

    # With a replica down, searches still work and status says which one is gone.
    assert run(cluster, "cluster", "kill", "1", "0") == 0
    text, _ = call(server, "status")
    assert "3 of 4 servers up" in text and "shard 1: replica 0 DOWN" in text
    text, is_error = call(server, "search", question="what happens to the log after a crash?")
    assert not is_error and "Write-ahead log" in text
    # Writes need every replica, and the tool says so instead of half-writing.
    (tmp_path / "late.md").write_text("# Late\n\nAdded while a replica was down.\n")
    text, is_error = call(server, "ingest", paths=[str(tmp_path / "late.md")])
    assert is_error and "a server is down" in text


def test_the_server_keeps_working_when_the_command_line_writes_in_between(server, cluster, tmp_path):
    # The server's coordinator lives for the whole session; the command line uses another one.
    call(server, "ingest", paths=[str(tmp_path / "docs" / "guide.md")])
    assert run(cluster, "ingest", str(tmp_path / "docs" / "notes.txt")) == 0
    text, is_error = call(server, "ingest", paths=[str(tmp_path / "extra.md")])
    assert not is_error and text.startswith("Stored 1 passages.")
    text, _ = call(server, "status")
    assert "Ingested: 4 passages from 3 file(s)" in text
    assert sum(vectors_per_server(text)) == 4 * 2            # 4 passages, on 2 replicas each
    for question, source in [("replicas answer searches in turn", "docs/notes.txt"),
                             ("vectors spread over shards", "extra.md")]:
        assert f"[1] source: {source}" in call(server, "search", question=question)[0]


def test_bad_calls_are_explained(server, tmp_path):
    text, is_error = call(server, "search", question="x", k=100)
    assert is_error and "'k' must be at most 20" in text
    text, is_error = call(server, "search")
    assert is_error and "missing required argument 'question'" in text
    text, is_error = call(server, "ingest", paths=[str(tmp_path / "nowhere")])
    assert is_error and text.startswith("no such file or directory")    # a plain explanation,
    assert "unexpectedly" not in text                                    # not a crash report


def test_read_only_offers_no_ingest(cluster):
    server = build_server(cluster, read_only=True)
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert [tool["name"] for tool in listed] == ["search", "status"]
    reply = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                           "params": {"name": "ingest", "arguments": {"paths": ["x"]}}})
    assert reply["error"]["code"] == INVALID_PARAMS
    assert all(tool["description"] and tool["inputSchema"]["type"] == "object" for tool in listed)


def test_without_a_cluster_the_tools_say_so(tmp_path):
    server = build_server(tmp_path / "nothing-here")
    text, is_error = call(server, "status")
    assert is_error and "vsearch cluster up" in text and "unexpectedly" not in text
    assert main(["--home", str(tmp_path / "nothing-here"), "mcp"]) == 1    # and the command refuses


def test_a_client_can_drive_the_server_as_a_separate_process(cluster, tmp_path):
    # What an MCP client does: start `vsearch mcp`, write requests to its stdin, read replies
    # from its stdout, one JSON object per line.
    process = subprocess.Popen(
        [sys.executable, "-m", "vsearch", "--home", str(cluster), "mcp", "--embedder", "hash"],
        cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)

    def send(method, params=None, request_id=None):
        message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if request_id is not None:
            message["id"] = request_id
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()
        if request_id is not None:
            reply = json.loads(process.stdout.readline())
            assert reply["id"] == request_id
            return reply["result"]

    try:
        started = send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "test", "version": "0"}}, 1)
        assert started["serverInfo"]["name"] == "vsearch" and "tools" in started["capabilities"]
        send("notifications/initialized")
        assert [tool["name"] for tool in send("tools/list", request_id=2)["tools"]] == \
            ["search", "status", "ingest"]
        stored = send("tools/call", {"name": "ingest",
                                     "arguments": {"paths": [str(tmp_path / "docs")]}}, 3)
        assert stored["content"][0]["text"].startswith("Stored 3 passages.")
        found = send("tools/call", {"name": "search",
                                    "arguments": {"question": "what saves the whole graph to disk?"}}, 4)
        assert not found["isError"] and "section: Storage > Snapshots" in found["content"][0]["text"]
    finally:
        process.stdin.close()                    # the client hangs up: the server exits
        assert process.wait(timeout=10) == 0
        assert process.stdout.read() == ""       # nothing but replies was ever written to stdout
        process.stdout.close()
        process.stderr.close()
