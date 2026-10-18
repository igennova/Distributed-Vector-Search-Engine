"""Attached clusters: servers that something else runs (Docker in practice; in-process servers
here). vsearch searches, writes, and repairs them, but never starts, stops, or restarts them."""
import grpc
import numpy as np
import pytest

from vsearch.cli import main
from vsearch.mcp import ToolError
from vsearch.ops import Operations, run_runbook
from vsearch.server import start_server
from vsearch.state import connect, read_state

PARAMS = dict(M=8, ef_construction=32, ef_search=32)


def run(home, *argv):
    return main(["--home", str(home), *argv])


class Orchestrator:
    """Plays the part of Docker: runs 2 shards x 2 durable replicas, and can stop and start
    any of them on the same port with the same disk."""

    def __init__(self, root):
        self.root = root
        self.ports, self.servers = {}, {}
        for shard in range(2):
            for replica in range(2):
                self.start(shard, replica)

    def start(self, shard, replica):
        server, port = start_server(port=self.ports.get((shard, replica), 0), seed=shard,
                                    data_dir=self.root / f"s{shard}r{replica}",
                                    snapshot_every=0, **PARAMS)
        self.ports[shard, replica], self.servers[shard, replica] = port, server

    def stop(self, shard, replica):
        self.servers.pop((shard, replica)).stop(grace=None).wait()

    def shards(self):
        return [",".join(f"127.0.0.1:{self.ports[shard, replica]}" for replica in range(2))
                for shard in range(2)]

    def close(self):
        for server in self.servers.values():
            server.stop(grace=None)


@pytest.fixture
def orchestrator(tmp_path):
    orchestrator = Orchestrator(tmp_path / "volumes")
    yield orchestrator
    orchestrator.close()


def test_an_attached_cluster_is_searched_like_any_other(orchestrator, tmp_path, capsys, monkeypatch):
    home = tmp_path / "attached"
    (tmp_path / "notes.md").write_text("# Storage\n\nEvery write goes to the log first.\n")
    monkeypatch.chdir(tmp_path)
    assert run(home, "cluster", "attach", *orchestrator.shards()) == 0
    output = capsys.readouterr().out
    assert "attached 4 servers in 2 shards, run by docker" in output and output.count(" up ") == 4
    state = read_state(home)
    assert state["managed_by"] == "docker"
    assert state["servers"][1][0] == {"host": "127.0.0.1", "port": orchestrator.ports[1, 0], "pid": None}

    assert run(home, "ingest", "notes.md", "--embedder", "hash") == 0
    assert run(home, "search", "where does a write go first") == 0
    assert "notes.md > Storage" in capsys.readouterr().out


def test_starting_and_stopping_belong_to_the_orchestrator(orchestrator, tmp_path, capsys):
    home = tmp_path / "attached"
    assert run(home, "cluster", "attach", *orchestrator.shards()) == 0
    capsys.readouterr()
    for argv in (["cluster", "up"], ["cluster", "down"], ["cluster", "down", "--wipe"],
                 ["cluster", "kill", "0", "0"], ["cluster", "restart", "0", "0"]):
        assert run(home, *argv) == 1
        assert "run by docker" in capsys.readouterr().err
    assert run(home, "cluster", "attach", *orchestrator.shards()) == 1    # already attached
    assert all(server is not None for server in orchestrator.servers.values())

    # Detaching forgets the cluster, not the servers.
    assert run(home, "cluster", "detach") == 0
    assert not (home / "cluster.json").exists() and len(orchestrator.servers) == 4


def test_attach_checks_every_server_answers(orchestrator, tmp_path, capsys):
    home = tmp_path / "attached"
    assert run(home, "cluster", "attach", "--timeout", "1", "127.0.0.1:1",
               orchestrator.shards()[1]) == 1
    assert "shard 0 replica 0 does not answer at 127.0.0.1:1" in capsys.readouterr().err
    assert not (home / "cluster.json").exists()
    for bad in ("localhost", "localhost:port", ":50051", "127.0.0.1:5=nohost"):
        assert run(home, "cluster", "attach", bad) == 1
        assert "expected HOST:PORT" in capsys.readouterr().err, bad


def test_the_orchestrator_restarts_and_the_agent_repairs(orchestrator, tmp_path, capsys):
    home = tmp_path / "attached"
    assert run(home, "cluster", "attach", *orchestrator.shards()) == 0
    vectors = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)
    with connect(read_state(home)) as coordinator:
        coordinator.add(vectors[:2])
        orchestrator.stop(0, 1)                      # a server dies during a write
        with pytest.raises(grpc.RpcError):
            coordinator.add(vectors[2:])             # only its twin gets this one

    # The agent cannot restart it: that is the orchestrator's job, and it says so.
    ops = Operations(home)
    assert "run by docker, which restarts servers itself" in run_runbook(ops)
    assert ops.problems() == ["shard 0 replica 1 is down"]
    with pytest.raises(ToolError, match="kept by docker"):
        ops.read_server_log(0, 1)

    # The orchestrator brings it back from its own disk, but it is behind its twin.
    # Restarting is not enough; the agent's repair is what makes the replicas agree.
    orchestrator.start(0, 1)
    assert ops.problems() == ["replicas of shard 0 are out of sync (replica 0 at seq 2, "
                              "replica 1 at seq 1)"]
    capsys.readouterr()
    assert run(home, "ops", "check") == 0
    output = capsys.readouterr().out
    assert "-> repair()  [ok]" in output and "the cluster is healthy" in output


@pytest.mark.parametrize("peer_of_twin, repaired", [("localhost", True), ("bad", False)])
def test_repair_tells_a_replica_to_copy_from_its_twins_peer_address(
        orchestrator, tmp_path, capsys, peer_of_twin, repaired):
    # The coordinator reaches the servers at one address; the servers reach each other at
    # another. Repair has to hand the lagging replica the second one.
    home = tmp_path / "attached"
    twin = orchestrator.ports[0, 0]
    peer = f"localhost:{twin}" if peer_of_twin == "localhost" else "127.0.0.1:1"
    shards = [f"127.0.0.1:{twin}={peer},127.0.0.1:{orchestrator.ports[0, 1]}",
              orchestrator.shards()[1]]
    assert run(home, "cluster", "attach", *shards) == 0
    assert read_state(home)["servers"][0][0]["peer"] == peer
    vectors = np.random.default_rng(1).standard_normal((6, 16)).astype(np.float32)
    with connect(read_state(home)) as coordinator:
        coordinator.add(vectors[:2])
        orchestrator.stop(0, 1)
        with pytest.raises(grpc.RpcError):
            coordinator.add(vectors[2:])
    orchestrator.start(0, 1)

    ops = Operations(home)
    if repaired:
        # Reported by the address you know it by; the copy itself went over the peer address.
        assert f"caught up from 127.0.0.1:{twin}" in ops.repair() and ops.problems() == []
    else:
        with pytest.raises(ToolError, match="could not sync from 127.0.0.1:1"):
            ops.repair()
