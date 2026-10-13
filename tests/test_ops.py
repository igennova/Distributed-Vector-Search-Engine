"""Operating a cluster: the tools and the runbook against real faults on a real local
cluster, the agent loop driven by a scripted model, and the limits both are held to."""
import json
import shutil
import sys

import grpc
import numpy as np
import pytest

from vsearch.agent import AgentRun, ScriptedModel, ToolCall, Turn, run_agent
from vsearch.cli import main
from vsearch.mcp import ToolError
from vsearch.ops import OPERATOR_PROMPT, Journal, Operations, run_runbook, task_for
from vsearch.state import connect, read_state


def run(home, *argv):
    return main(["--home", str(home), *argv])


@pytest.fixture
def cluster(tmp_path, monkeypatch, capsys):
    """2 shards x 2 replicas holding a few passages; the working directory is tmp_path."""
    home = tmp_path / "cluster"
    (tmp_path / "docs.md").write_text("# One\n\nFirst passage.\n\n# Two\n\nSecond passage.\n\n"
                                      "# Three\n\nThird passage.\n\n# Four\n\nFourth passage.\n")
    monkeypatch.chdir(tmp_path)
    assert run(home, "cluster", "up", "--shards", "2", "--replicas", "2") == 0
    assert run(home, "ingest", "docs.md", "--embedder", "hash") == 0
    capsys.readouterr()
    yield home
    run(home, "cluster", "down", "--wipe")


def kill(home, shard, replica):
    assert run(home, "cluster", "kill", str(shard), str(replica)) == 0


def lose_a_write(home, *down):
    """Kill the given (shard, replica) servers in the middle of a coordinator's life, so its
    next write reaches only the replicas that are still up and leaves the others behind.
    (A coordinator that starts with a server down refuses to write at all.)"""
    vectors = np.random.default_rng(0).standard_normal((8, 512)).astype(np.float32)
    with connect(read_state(home)) as coordinator:
        coordinator.add(vectors[:2])                  # everyone gets this one
        for shard, replica in down:
            kill(home, shard, replica)
        with pytest.raises(grpc.RpcError):
            coordinator.add(vectors[2:])              # only the survivors get this one


def call(name, **arguments):
    return ToolCall(f"call-{name}", name, arguments)


# --- the tools -------------------------------------------------------------------------------

def test_a_healthy_cluster_has_no_problems(cluster):
    ops = Operations(cluster)
    assert ops.problems() == []
    status = ops.cluster_status()
    assert "shard 0: replica 0 up (2 vectors, seq 1), replica 1 up (2 vectors, seq 1)" in status
    assert "No problems found" in status
    assert ops.probe_search().startswith("A search across all 2 shards answered")
    assert "Nothing to repair" in ops.repair()
    assert "listening" in ops.read_server_log(0, 0) or ops.read_server_log(0, 0)

    with pytest.raises(ToolError, match="already up"):
        ops.restart_server(0, 0)
    with pytest.raises(ToolError, match="no shard 7"):
        ops.restart_server(7, 0)
    with pytest.raises(ToolError, match="no shard 0 replica 9"):
        ops.read_server_log(0, 9)


def test_problems_name_what_is_wrong(cluster):
    kill(cluster, 1, 0)
    ops = Operations(cluster)
    assert ops.problems() == ["shard 1 replica 0 is down"]
    assert "replica 0 DOWN" in ops.cluster_status()
    assert ops.probe_search().startswith("A search across all 2 shards")      # its twin answers

    kill(cluster, 1, 1)
    assert ops.problems() == ["shard 1 replica 0 is down", "shard 1 replica 1 is down",
                              "shard 1 has no live replica: searches that need it fail"]
    assert "no answer from shard(s) [1]" in ops.probe_search()


# --- the runbook on real faults ----------------------------------------------------------------

def test_runbook_restarts_a_dead_replica(cluster):
    kill(cluster, 0, 1)
    ops = Operations(cluster, Journal(cluster / "journal.jsonl"))
    report = run_runbook(ops)
    assert ops.problems() == []
    assert "restarted shard 0 replica 1" in report
    assert ops.journal.actions() == ["cluster_status", "restart_server", "repair", "probe_search"]

    # The journal on disk says the same, one JSON object per line.
    lines = [json.loads(line) for line in (cluster / "journal.jsonl").read_text().splitlines()]
    assert [line["tool"] for line in lines] == ops.journal.actions()
    assert lines[1]["arguments"] == {"shard": 0, "replica": 1} and lines[1]["error"] is False
    assert "recovered 2 vectors" in lines[1]["result"]


def test_runbook_brings_a_replica_that_missed_writes_up_to_date(cluster):
    lose_a_write(cluster, (0, 1), (1, 1))             # replicas 0 move ahead
    ops = Operations(cluster)
    report = run_runbook(ops)
    assert "repaired replicas that were behind" in report
    assert ops.problems() == []
    health = ops.health()
    assert [server.size for server in health] == [6, 6, 6, 6]     # 2 passages + 4 vectors each
    assert len({server.last_seq for server in health}) == 1


def test_restart_alone_leaves_replicas_out_of_sync(cluster):
    # The reason repair exists: after its restart the replica is up, but behind.
    lose_a_write(cluster, (0, 1))
    ops = Operations(cluster)
    ops.restart_server(0, 1)
    assert ops.problems() == ["replicas of shard 0 are out of sync (replica 0 at seq 3, "
                              "replica 1 at seq 2)"]
    assert "via log (1 log records), now at seq 3" in ops.repair()
    assert ops.problems() == []


def test_runbook_rebuilds_a_replica_that_lost_its_disk(cluster):
    kill(cluster, 1, 0)
    shutil.rmtree(cluster / "shard-1-replica-0")
    ops = Operations(cluster)
    run_runbook(ops)
    assert ops.problems() == []
    assert [server.size for server in ops.health()] == [2, 2, 2, 2]
    restart, repair = (entry["result"] for entry in ops.journal.entries
                       if entry["tool"] in ("restart_server", "repair"))
    assert "recovered 0 vectors" in restart          # it came back empty,
    assert "caught up from" in repair                # and got everything from its twin


# --- limits ------------------------------------------------------------------------------------

def test_dry_run_changes_nothing(cluster):
    kill(cluster, 0, 0)
    ops = Operations(cluster, dry_run=True)
    assert ops.restart_server(0, 0) == "DRY RUN: would restart shard 0 replica 0."
    assert ops.problems() == ["shard 0 replica 0 is down"]
    run_runbook(ops)
    assert ops.problems() == ["shard 0 replica 0 is down"]


def test_a_server_that_cannot_start_is_not_restarted_forever(cluster):
    kill(cluster, 0, 0)
    (cluster / "shard-0-replica-0" / "snapshot.npz").write_bytes(b"this is not a snapshot")
    ops = Operations(cluster, max_restarts=2, start_timeout=3)
    for _ in range(2):
        with pytest.raises(ToolError, match="did not answer within 3 s"):
            ops.restart_server(0, 0)
    with pytest.raises(ToolError, match="already restarted 2 times"):
        ops.restart_server(0, 0)
    assert "Traceback" in ops.read_server_log(0, 0, lines=60)     # the log says why
    # The runbook has nothing else to try: it reports the failure and the cluster stays degraded.
    report = run_runbook(Operations(cluster, start_timeout=3))
    assert "could not restart shard 0 replica 0" in report
    assert Operations(cluster).problems() == ["shard 0 replica 0 is down"]


# --- the agent loop ----------------------------------------------------------------------------

def test_the_loop_runs_what_the_model_asks_and_hands_back_the_results(cluster):
    kill(cluster, 1, 1)
    ops = Operations(cluster)

    def after_status(outcomes):
        assert "replica 1 DOWN" in outcomes[0].text and not outcomes[0].is_error
        return Turn("Replica 1 of shard 1 is down; restarting it.",
                    [call("restart_server", shard=1, replica=1)])

    model = ScriptedModel([
        Turn("", [call("cluster_status")]),
        after_status,
        Turn("", [call("cluster_status"), call("probe_search")]),      # two tools in one turn
        Turn("Shard 1 replica 1 was down. I restarted it; all servers answer now.", []),
    ])
    result = run_agent(model, ops.tools(), task_for(ops.problems()), OPERATOR_PROMPT,
                       journal=ops.journal)

    assert result == AgentRun("Shard 1 replica 1 was down. I restarted it; all servers answer now.",
                              finished=True, steps=4, calls=4)
    assert ops.problems() == []
    assert model.task.startswith("A health check of the cluster found:\n- shard 1 replica 1 is down")
    assert model.tool_names == ["cluster_status", "restart_server", "repair", "read_server_log",
                                "probe_search"]
    assert model.seen[0] is None and len(model.seen[3]) == 2
    assert "recovered 2 vectors" in model.seen[2][0].text
    # The journal interleaves what the model said with what was done.
    kinds = [(entry["kind"], entry.get("tool")) for entry in ops.journal.entries]
    assert kinds == [("model", None), ("tool", "cluster_status"), ("model", None),
                     ("tool", "restart_server"), ("model", None), ("tool", "cluster_status"),
                     ("tool", "probe_search"), ("model", None)]


def test_a_model_that_misbehaves_gets_errors_back_and_cannot_do_harm(cluster):
    ops = Operations(cluster)
    model = ScriptedModel([
        Turn("", [call("wipe_everything")]),                         # no such tool
        Turn("", [call("restart_server", shard="zero", replica=0)]),  # wrong type
        Turn("", [call("restart_server", shard=0, replica=0)]),       # the server is up
        Turn("", [call("read_server_log", shard=0, replica=0, lines=5000)]),
        Turn("Nothing was wrong.", []),
    ])
    run_agent(model, ops.tools(), "check", OPERATOR_PROMPT)
    errors = [outcomes[0] for outcomes in model.seen[1:]]
    assert all(outcome.is_error for outcome in errors)
    assert "no tool named 'wipe_everything'" in errors[0].text
    assert "'shard' must be of type integer" in errors[1].text
    assert "already up" in errors[2].text
    assert "'lines' must be at most 100" in errors[3].text
    assert ops.problems() == [] and all(server.size == 2 for server in ops.health())


def test_the_step_limit_stops_a_model_that_never_finishes(cluster):
    ops = Operations(cluster)
    model = ScriptedModel([Turn("still looking", [call("cluster_status")])] * 50)
    result = run_agent(model, ops.tools(), "check", OPERATOR_PROMPT, max_steps=4)
    assert result == AgentRun("still looking", finished=False, steps=4, calls=4)
    assert len(model.seen) == 4


# --- the command line ----------------------------------------------------------------------------

def test_ops_check_on_the_command_line(cluster, capsys, monkeypatch):
    assert run(cluster, "ops", "check") == 0
    assert "the cluster is healthy: nothing to do" in capsys.readouterr().out

    kill(cluster, 1, 0)
    capsys.readouterr()
    assert run(cluster, "ops", "check", "--dry-run") == 0
    output = capsys.readouterr().out
    assert "found 1 problem(s):\n  - shard 1 replica 0 is down" in output
    assert "DRY RUN: would restart shard 1 replica 0." in output and "nothing was changed" in output

    assert run(cluster, "ops", "check") == 0
    output = capsys.readouterr().out
    assert "-> restart_server(shard=1, replica=0)  [ok]" in output
    assert "checked: the cluster is healthy" in output
    journal = [json.loads(line) for line in (cluster / "ops-journal.jsonl").read_text().splitlines()]
    assert journal[-1]["kind"] == "verdict" and journal[-1]["healthy"] is True

    # A fault it cannot fix: exit code 1, and it says what is still wrong.
    kill(cluster, 0, 1)
    (cluster / "shard-0-replica-1" / "snapshot.npz").write_bytes(b"garbage")
    monkeypatch.setattr(Operations.__init__, "__defaults__", (None, False, 2, 3))   # short timeout
    assert run(cluster, "ops", "check") == 1
    assert "checked: still wrong:\n  - shard 0 replica 1 is down" in capsys.readouterr().out

    # Watching: the first round tries and fails; the second sees the same problem and leaves it.
    assert run(cluster, "ops", "watch", "--interval", "0.1", "--rounds", "3") == 0
    output = capsys.readouterr().out
    assert output.count("found 1 problem(s)") == 1 and "not retrying until something changes" in output

    # A model brain without what it needs says so and changes nothing.
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert run(cluster, "ops", "check", "--brain", "claude") == 1
    assert "pip install anthropic" in capsys.readouterr().err
    assert run(cluster, "ops", "check", "--brain", "openai") == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_ops_check_with_a_model_brain(cluster, capsys, openai_api):
    # The whole path with a model deciding: the stand-in API plays a model that looks,
    # restarts the dead replica, looks again, and reports.
    kill(cluster, 0, 0)
    capsys.readouterr()
    openai_api.replies += [
        openai_api.reply(tool_calls=[("c1", "cluster_status", "{}")], finish_reason="tool_calls"),
        openai_api.reply("Replica 0 of shard 0 is down. Restarting it.",
                         tool_calls=[("c2", "restart_server", '{"shard": 0, "replica": 0}')],
                         finish_reason="tool_calls"),
        openai_api.reply(tool_calls=[("c3", "cluster_status", "{}")], finish_reason="tool_calls"),
        openai_api.reply("Shard 0 replica 0 was down; I restarted it and the cluster is healthy."),
    ]
    assert run(cluster, "ops", "check", "--brain", "openai", "--model", "test-model") == 0
    output = capsys.readouterr().out
    assert "-> restart_server(shard=0, replica=0)  [ok]" in output
    assert "  Replica 0 of shard 0 is down. Restarting it." in output      # said while working
    assert "[test-model: 4 turns, 3 tool calls, 400 input + 40 output tokens]" in output
    assert "report:\n  Shard 0 replica 0 was down; I restarted it" in output
    assert "checked: the cluster is healthy" in output
    # What the model was told: the operator's instructions, then the problems found.
    sent = openai_api.requests[0]["body"]["messages"]
    assert sent[0]["content"] == OPERATOR_PROMPT
    assert sent[1]["content"] == task_for(["shard 0 replica 0 is down"])
    assert "DOWN" in openai_api.requests[1]["body"]["messages"][-1]["content"]


def test_a_report_is_not_believed_without_checking(cluster, capsys, openai_api):
    # The model claims success without doing anything. The exit code comes from the check.
    kill(cluster, 1, 1)
    capsys.readouterr()
    openai_api.replies += [openai_api.reply("Everything is fixed and healthy now.")]
    assert run(cluster, "ops", "check", "--brain", "openai") == 1
    output = capsys.readouterr().out
    assert "Everything is fixed and healthy now." in output
    assert "checked: still wrong:\n  - shard 1 replica 1 is down" in output
