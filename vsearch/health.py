"""
The health of a local cluster, as facts and as a list of problems. One definition of
"healthy" shared by the status display, the operations agent, and the check that the
agent really fixed what it says it fixed.
"""
from typing import NamedTuple

from .processes import is_running
from .state import address, server_status, servers


class ServerHealth(NamedTuple):
    shard: int
    replica: int
    address: str
    up: bool              # answers requests
    process: bool         # its process exists (an unanswering server may be starting or stuck)
    size: int             # vectors held; 0 when down
    last_seq: int         # last write applied; 0 when down
    snapshot_seq: int


def cluster_health(state):
    """The state of every server, in shard and replica order."""
    health = []
    for shard, replica, server in servers(state):
        reply = server_status(server)
        health.append(ServerHealth(
            shard, replica, address(server), reply is not None, is_running(server),
            reply.size if reply else 0, reply.last_seq if reply else 0,
            reply.snapshot_seq if reply else 0))
    return health


def find_problems(health):
    """What is wrong, one sentence each; an empty list means the cluster is healthy:
    every server answers, and the replicas of each shard have applied the same writes."""
    problems = []
    for shard in sorted({server.shard for server in health}):
        group = [server for server in health if server.shard == shard]
        alive = [server for server in group if server.up]
        for server in group:
            if server.up:
                continue
            if server.process:
                problems.append(f"shard {shard} replica {server.replica} does not answer, "
                                "although its process is running")
            else:
                problems.append(f"shard {shard} replica {server.replica} is down")
        if not alive:
            problems.append(f"shard {shard} has no live replica: searches that need it fail")
        elif len({server.last_seq for server in alive}) > 1:
            seqs = ", ".join(f"replica {server.replica} at seq {server.last_seq}" for server in alive)
            problems.append(f"replicas of shard {shard} are out of sync ({seqs})")
    return problems


def describe_health(health):
    """The cluster's servers as text, one line per shard."""
    lines = []
    for shard in sorted({server.shard for server in health}):
        parts = []
        for server in (s for s in health if s.shard == shard):
            if server.up:
                parts.append(f"replica {server.replica} up ({server.size:,} vectors, "
                             f"seq {server.last_seq})")
            else:
                parts.append(f"replica {server.replica} DOWN ({server.address})")
        lines.append(f"shard {shard}: " + ", ".join(parts))
    return "\n".join(lines)
