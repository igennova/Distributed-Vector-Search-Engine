"""
A local cluster's state: cluster.json in the cluster's home directory records the servers'
ports and process ids and what has been loaded into them. The command line and the MCP
server both start from it.
"""
import json
import os
from pathlib import Path

import grpc

from .client import GrpcCoordinator
from .protos import shard_pb2, shard_pb2_grpc


class UserError(Exception):
    """Something to tell the user about, without a traceback."""


def load_env_file(path=".env"):
    """Read KEY=VALUE lines from a file into the environment, for secrets such as API keys
    that must not be typed on a command line or committed. A variable that is already set
    wins. Returns the names that were set, never the values."""
    path = Path(path)
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        name, equals, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not equals or not name or name.startswith("#"):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def read_state(home):
    path = home / "cluster.json"
    if not path.exists():
        raise UserError(f"no cluster in {home}; start one with: vsearch cluster up")
    return json.loads(path.read_text())


def write_state(home, state):
    tmp = home / "cluster.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, home / "cluster.json")     # atomic: never leaves a half-written state file


def address(server):
    return f"127.0.0.1:{server['port']}"


def servers(state):
    """Every server as (shard, replica, server record)."""
    return [(shard, replica, server) for shard, group in enumerate(state["servers"])
            for replica, server in enumerate(group)]


def shard_addresses(state):
    return [[address(server) for server in group] for group in state["servers"]]


def connect(state):
    """A coordinator for the cluster's servers."""
    return GrpcCoordinator(shard_addresses(state))


def server_status(server):
    """A server's StatusResponse, or None if it does not answer."""
    try:
        with grpc.insecure_channel(address(server)) as channel:
            return shard_pb2_grpc.ShardServiceStub(channel).Status(shard_pb2.StatusRequest(),
                                                                   timeout=1.0)
    except grpc.RpcError:
        return None


def writable(state):
    """Every server's status, after checking that all of them are up to take a write."""
    replies = [server_status(server) for _, _, server in servers(state)]
    if any(reply is None for reply in replies):
        raise UserError("a server is down, and writes need every replica; see: vsearch status")
    return replies
