"""
Operating a cluster: the actions that look at it and repair it, a journal of every action
taken, and the fixed procedure ("runbook") a script would follow.

The same actions serve two callers: the runbook below, and a model deciding for itself
(agent.py). Both go through Operations, so both are held to the same limits and leave the
same journal:

  - nothing here can delete data or stop a server
  - one server is restarted at most `max_restarts` times per run
  - with dry_run, actions that would change something are described instead of done
"""
import json
import re
import time
from collections import Counter

import grpc
import numpy as np

from .health import cluster_health, describe_health, find_problems
from .mcp import Tool, ToolError
from .processes import log_path, pick, spawn, wait_for
from .state import UserError, connect, managed_by, read_state, write_state

MAX_LOG_LINES = 100
SERVER = {
    "type": "object",
    "properties": {"shard": {"type": "integer", "minimum": 0},
                   "replica": {"type": "integer", "minimum": 0}},
    "required": ["shard", "replica"],
}
NO_ARGUMENTS = {"type": "object", "properties": {}}

# What a model is told before it is handed the tools: how the system behaves, a set of
# working rules, and how to end. Two sets of rules are kept, to measure against each other
# (evals/ops_eval.py): "careful" looks before and after every change; "lean" acts on the
# listed problems at once and verifies at the end, which takes fewer turns.
_SYSTEM = """\
You operate a small distributed vector search cluster. Its data is split over several \
shards, and each shard has one or more replicas: identical copies on different servers. The \
cluster is healthy when every server answers and the replicas of each shard have applied \
the same writes.

Your job is to find out what is wrong, fix what your tools can fix, confirm the result, and \
report.

How the system behaves:
- A server that is down keeps its data on disk. Restarting it recovers that data.
- A replica that was down while writes happened is behind its twin after a restart. The \
repair tool copies the missing writes from the most advanced replica of the same shard. \
Repair only reaches replicas that are up, so restart first.
- Searches fail for a shard only when all of its replicas are down.
"""

_RULES = {
    "careful": """\
Rules:
- Look before you act: start with cluster_status, and check it again after every change.
- If a restart fails, read that server's log before doing anything else. Do not retry blindly.
- You cannot delete data, stop servers, or change configuration. If the fix needs any of \
that, say so in your report instead of trying something else.
- Stop when the cluster is healthy, or when nothing more can be done with your tools.
""",
    "lean": """\
Rules:
- Act on the problems listed in the task straight away. You may ask for several tools in \
one turn; they run in the order you give. Repair does nothing when replicas already agree, \
so it is safe to ask for it right after restarts.
- When you have made your changes, call cluster_status once to verify them.
- If a restart fails, read that server's log before doing anything else. Do not retry blindly.
- You cannot delete data, stop servers, or change configuration. If the fix needs any of \
that, say so in your report instead of trying something else.
- Stop when the cluster is healthy, or when nothing more can be done with your tools.
""",
}

_FINISH = """
Finish with a short report in plain words: what was wrong, what you did, the state now, and \
anything a person still needs to do. End the report with exactly one of these lines:
Outcome: fixed
Outcome: needs a person
Outcome: nothing wrong"""

PROMPTS = {name: _SYSTEM + "\n" + rules + _FINISH for name, rules in _RULES.items()}
OPERATOR_PROMPT = PROMPTS["careful"]
OUTCOMES = ("fixed", "needs a person", "nothing wrong")
_OUTCOME_LINE = re.compile(r"^\W*outcome\W*:\W*(fixed|needs a person|nothing wrong)\W*$",
                           re.IGNORECASE | re.MULTILINE)


def claimed_outcome(report):
    """What a report says happened (one of OUTCOMES), or None if it does not say."""
    found = _OUTCOME_LINE.findall(report or "")
    return found[-1].lower() if found else None


def task_for(problems):
    listed = "\n".join(f"- {problem}" for problem in problems)
    return f"A health check of the cluster found:\n{listed}\n\nInvestigate and repair it."


class Journal:
    """A record of what was looked at, decided, and done: one JSON object per line."""

    def __init__(self, path=None, on_entry=None):
        self.path = path
        self.entries = []
        self._on_entry = on_entry          # called with each entry, e.g. to print it

    def write(self, kind, **fields):
        entry = {"time": round(time.time(), 3), "kind": kind, **fields}
        self.entries.append(entry)
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if self._on_entry:
            self._on_entry(entry)

    def actions(self):
        """The names of the tools that were run, in order."""
        return [entry["tool"] for entry in self.entries if entry["kind"] == "tool"]


class Operations:
    def __init__(self, home, journal=None, dry_run=False, max_restarts=2, start_timeout=20):
        self.home = home
        self.journal = journal or Journal()
        self.dry_run = dry_run
        self.max_restarts = max_restarts
        self.start_timeout = start_timeout
        self._restarts = Counter()

    # --- looking -----------------------------------------------------------------------------

    def health(self):
        return cluster_health(read_state(self.home))

    def problems(self):
        """What is wrong right now; empty when the cluster is healthy."""
        return find_problems(self.health())

    def cluster_status(self):
        health = self.health()
        problems = find_problems(health)
        verdict = ("Problems:\n" + "\n".join(f"- {problem}" for problem in problems)
                   if problems else "No problems found: every server answers and replicas agree.")
        return f"{describe_health(health)}\n{verdict}"

    def read_server_log(self, shard, replica, lines=40):
        state = read_state(self.home)
        self._pick(shard, replica, state)
        if managed_by(state):
            raise ToolError(f"the logs of shard {shard} replica {replica} are kept by "
                            f"{managed_by(state)}, not here (e.g. docker compose logs); "
                            "report that a person should read them.")
        path = log_path(self.home, shard, replica)
        if not path.exists():
            return f"shard {shard} replica {replica} has no log file yet."
        tail = path.read_text(errors="replace").splitlines()[-min(lines, MAX_LOG_LINES):]
        return "\n".join(tail) if tail else "(the log is empty)"

    def probe_search(self):
        """Send one real search through the coordinator, the way a user's search would go."""
        state = read_state(self.home)
        dataset = state.get("dataset")
        if not dataset:
            return "The cluster holds no vectors yet, so there is nothing to search."
        query = np.ones(dataset.get("dim", 100), dtype=np.float32)
        with connect(state) as coordinator:
            started = time.perf_counter()
            ids, missing = coordinator.search_partial(query, 1)
            elapsed = (time.perf_counter() - started) * 1000
        shards = len(state["servers"])
        if missing:
            return (f"A search got no answer from shard(s) {missing}: {shards - len(missing)} of "
                    f"{shards} shards answered, so results are incomplete.")
        return f"A search across all {shards} shards answered in {elapsed:.0f} ms."

    # --- changing ----------------------------------------------------------------------------

    def restart_server(self, shard, replica):
        state = read_state(self.home)
        server = self._pick(shard, replica, state)
        name = f"shard {shard} replica {replica}"
        current = next(s for s in cluster_health(state) if (s.shard, s.replica) == (shard, replica))
        if current.up:
            raise ToolError(f"{name} is already up; there is nothing to restart.")
        if managed_by(state):
            raise ToolError(f"{name} is run by {managed_by(state)}, which restarts servers "
                            "itself; this agent cannot. Check its status again shortly, or "
                            "report that it is not coming back.")
        if current.process:
            raise ToolError(f"{name} does not answer but its process is still running. This "
                            "agent cannot stop processes: report it for a person to look at.")
        if self._restarts[shard, replica] >= self.max_restarts:
            raise ToolError(f"{name} was already restarted {self.max_restarts} times in this run "
                            "and did not stay up. Do not try again: read its log and report it.")
        if self.dry_run:
            return f"DRY RUN: would restart {name}."
        self._restarts[shard, replica] += 1
        spawn(self.home, state, shard, replica)
        write_state(self.home, state)
        try:
            wait_for([(shard, replica, server)], self.home, timeout=self.start_timeout)
        except UserError:
            raise ToolError(f"{name} was started but did not answer within "
                            f"{self.start_timeout} s. Read its log to see why.") from None
        back = next(s for s in self.health() if (s.shard, s.replica) == (shard, replica))
        return (f"{name} restarted and answering: it recovered {back.size:,} vectors from its "
                f"disk and is at seq {back.last_seq}.")

    def repair(self):
        health = self.health()
        behind = []
        for shard in sorted({server.shard for server in health}):
            alive = [server for server in health if server.shard == shard and server.up]
            newest = max((server.last_seq for server in alive), default=0)
            behind += [f"shard {shard} replica {server.replica} (seq {server.last_seq} of {newest})"
                       for server in alive if server.last_seq < newest]
        if not behind:
            return "Nothing to repair: the replicas that are up already agree."
        if self.dry_run:
            return "DRY RUN: would bring up to date: " + "; ".join(behind) + "."
        try:
            with connect(read_state(self.home)) as coordinator:
                synced = coordinator.repair()
        except grpc.RpcError as err:
            raise ToolError(f"repair failed: {err.code().name}: {err.details()}") from None
        return "\n".join(
            f"shard {sync.shard}: {sync.replica} caught up from {sync.source} via {sync.method} "
            f"({sync.records_applied} log records), now at seq {sync.last_seq}" for sync in synced)

    # --- as tools ----------------------------------------------------------------------------

    def tools(self):
        """The actions as tools a model can call. Every call is written to the journal."""
        return [
            self._tool("cluster_status",
                       "Show every shard server (up or down, vectors held, last write applied) "
                       "and list what is wrong. Call this first, and again after changing "
                       "anything to confirm the change worked.",
                       NO_ARGUMENTS, self.cluster_status),
            self._tool("restart_server",
                       "Start a shard server that is down. It recovers its data from its own "
                       "disk. Only for servers that are down; it fails on one that is up.",
                       SERVER, self.restart_server, mutating=True),
            self._tool("repair",
                       "Bring replicas that missed writes up to date by copying from the most "
                       "advanced replica of the same shard. Needs the lagging replica to be "
                       "up, so restart servers first.",
                       NO_ARGUMENTS, self.repair, mutating=True),
            self._tool("read_server_log",
                       "Read the last lines of one server's log, to find out why it is down "
                       "or why a restart failed.",
                       {"type": "object",
                        "properties": {**SERVER["properties"],
                                       "lines": {"type": "integer", "minimum": 1,
                                                 "maximum": MAX_LOG_LINES}},
                        "required": ["shard", "replica"]},
                       self.read_server_log),
            self._tool("probe_search",
                       "Run one real search across the cluster to check that every shard "
                       "answers, as a user's search would need.",
                       NO_ARGUMENTS, self.probe_search),
        ]

    def _tool(self, name, description, schema, function, mutating=False):
        def journaled(**arguments):
            try:
                result = function(**arguments)
            except (ToolError, UserError) as err:
                self.journal.write("tool", tool=name, arguments=arguments, result=str(err),
                                   error=True)
                raise ToolError(str(err)) from None
            self.journal.write("tool", tool=name, arguments=arguments, result=result, error=False)
            return result
        return Tool(name, description, schema, journaled, mutating)

    def _pick(self, shard, replica, state=None):
        try:
            return pick(state or read_state(self.home), shard, replica)
        except UserError as err:
            raise ToolError(str(err)) from None


def run_runbook(ops):
    """The fixed procedure: restart whatever is down, repair, check. No judgement: it does
    the same thing whatever it finds, and has nothing to try when a step fails."""
    tools = {tool.name: tool.handler for tool in ops.tools()}
    tools["cluster_status"]()
    notes = []
    for server in ops.health():
        if not server.up:
            try:
                tools["restart_server"](shard=server.shard, replica=server.replica)
                notes.append(f"restarted shard {server.shard} replica {server.replica}")
            except ToolError as err:
                notes.append(f"could not restart shard {server.shard} replica {server.replica}: "
                             f"{str(err).rstrip('.')}")
    try:
        result = tools["repair"]()
        if not result.startswith("Nothing to repair"):
            notes.append("repaired replicas that were behind")
    except ToolError as err:
        notes.append(str(err))
    tools["probe_search"]()
    outcome = ("needs a person" if any(note.startswith("could not") for note in notes)
               else "fixed" if notes else "nothing wrong")
    return ("Runbook: " + ("; ".join(notes) if notes else "nothing needed doing") +
            f".\nOutcome: {outcome}")
