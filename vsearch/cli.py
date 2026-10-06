"""
Command-line interface for running and querying a local cluster.

    vsearch cluster up --shards 4 --replicas 2    start shard servers in the background
    vsearch load glove --limit 100000             index the most frequent GloVe words
    vsearch similar king                          nearest words, with their similarity
    vsearch status                                state, size, and write position of every server
    vsearch cluster kill 0 1                      crash shard 0's replica 1
    vsearch cluster restart 0 1                   start it again (it recovers from disk)
    vsearch repair                                bring replicas that fell behind back in sync
    vsearch cluster down                          stop the servers (their data stays on disk)

Everything lives in the --home directory (default data/cluster): cluster.json with the
servers' ports and process ids, one data directory per server, and the servers' logs.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import grpc

from .client import (REPO_ROOT, GrpcCoordinator, ReplicasOutOfSyncError, ShardUnavailableError,
                     free_ports, server_command, wait_until_ready)
from .dataset import load_glove
from .protos import shard_pb2, shard_pb2_grpc

LOAD_BATCH = 10_000
_spawned = []     # keep handles to servers started by this process until it exits


class CliError(Exception):
    """Something to tell the user about, without a traceback."""


# --- cluster state -------------------------------------------------------------------------

def _read_state(home):
    path = home / "cluster.json"
    if not path.exists():
        raise CliError(f"no cluster in {home}; start one with: vsearch cluster up")
    return json.loads(path.read_text())


def _write_state(home, state):
    tmp = home / "cluster.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, home / "cluster.json")     # atomic: never leaves a half-written state file


def _address(server):
    return f"127.0.0.1:{server['port']}"


def _servers(state):
    return [(shard, replica, server) for shard, group in enumerate(state["servers"])
            for replica, server in enumerate(group)]


def _pick(state, shard, replica):
    try:
        return state["servers"][shard][replica]
    except IndexError:
        raise CliError(f"there is no shard {shard} replica {replica} in this cluster") from None


def _coordinator(state):
    return GrpcCoordinator([[_address(server) for server in group] for group in state["servers"]])


# --- server processes ----------------------------------------------------------------------

def _is_running(server):
    """True if the recorded pid is still our shard server on the recorded port.

    Process ids are reused by the OS, so a pid is never signalled without checking that it
    still belongs to the server that was started with it.
    """
    pid = server["pid"]
    if pid is None:
        return False
    try:
        os.waitpid(pid, os.WNOHANG)          # collect it if it was our own child and has exited
    except ChildProcessError:
        pass
    # -ww: never truncate the command line, whatever the terminal width.
    command = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True).stdout
    return "vsearch.server" in command and f"--port {server['port']}" in command


def _spawn(home, state, shard, replica):
    server = state["servers"][shard][replica]
    name = f"shard-{shard}-replica-{replica}"
    (home / "logs").mkdir(parents=True, exist_ok=True)
    command = server_command(server["port"], seed=state["seed"] + shard, data_dir=home / name,
                             **state["server_args"])
    with open(home / "logs" / f"{name}.log", "ab") as log:
        # A new session detaches the server from this terminal, so it keeps running after
        # the command returns.
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    _spawned.append(process)
    server["pid"] = process.pid


def _wait_for(servers, home):
    for shard, replica, server in servers:
        try:
            wait_until_ready(_address(server))
        except grpc.FutureTimeoutError:
            raise CliError(f"shard {shard} replica {replica} did not start; see "
                           f"{home / 'logs' / f'shard-{shard}-replica-{replica}.log'}") from None


def _stop(servers, sig):
    running = [server for server in servers if _is_running(server)]
    for server in running:
        os.kill(server["pid"], sig)
    deadline = time.monotonic() + 10
    while any(_is_running(server) for server in running) and time.monotonic() < deadline:
        time.sleep(0.05)
    return len(running)


# --- commands ------------------------------------------------------------------------------

def cluster_up(args):
    home = args.home
    if (home / "cluster.json").exists():
        state = _read_state(home)
        if any(_is_running(server) for _, _, server in _servers(state)):
            raise CliError("this cluster is already running; see: vsearch status")
        print(f"starting the existing cluster in {home} again (servers recover from disk)")
    else:
        home.mkdir(parents=True, exist_ok=True)
        ports = iter(free_ports(args.shards * args.replicas))
        state = {
            "seed": args.seed,
            "server_args": {"M": args.M, "ef_construction": args.ef_construction,
                            "ef_search": args.ef_search, "snapshot_every": args.snapshot_every},
            "servers": [[{"port": next(ports), "pid": None} for _ in range(args.replicas)]
                        for _ in range(args.shards)],
            "dataset": None,
        }
    for shard, replica, _ in _servers(state):
        _spawn(home, state, shard, replica)
    _write_state(home, state)
    _wait_for(_servers(state), home)
    return status(args)


def cluster_down(args):
    state = _read_state(args.home)
    stopped = _stop([server for _, _, server in _servers(state)], signal.SIGTERM)
    print(f"stopped {stopped} server(s)")
    if args.wipe:
        shutil.rmtree(args.home)
        print(f"deleted {args.home}")
    else:
        print("their data is still on disk; `vsearch cluster up` starts them again")
    return 0


def cluster_kill(args):
    state = _read_state(args.home)
    server = _pick(state, args.shard, args.replica)
    if not _stop([server], signal.SIGKILL):
        raise CliError(f"shard {args.shard} replica {args.replica} is not running")
    print(f"killed shard {args.shard} replica {args.replica} (SIGKILL, no clean shutdown)")
    return 0


def cluster_restart(args):
    state = _read_state(args.home)
    server = _pick(state, args.shard, args.replica)
    if _is_running(server):
        raise CliError(f"shard {args.shard} replica {args.replica} is already running")
    _spawn(args.home, state, args.shard, args.replica)
    _write_state(args.home, state)
    _wait_for([(args.shard, args.replica, server)], args.home)
    print(f"shard {args.shard} replica {args.replica} is back on {_address(server)}")
    return 0


def _server_status(server):
    """A server's StatusResponse, or None if it does not answer."""
    try:
        with grpc.insecure_channel(_address(server)) as channel:
            return shard_pb2_grpc.ShardServiceStub(channel).Status(shard_pb2.StatusRequest(),
                                                                   timeout=1.0)
    except grpc.RpcError:
        return None


def status(args):
    state = _read_state(args.home)
    print(f"{'shard':>5} {'replica':>7}  {'address':<17}{'state':<6}{'vectors':>9}"
          f"{'last seq':>10}{'snapshot seq':>14}")
    for shard, replica, server in _servers(state):
        reply = _server_status(server)
        if reply is None:
            print(f"{shard:>5} {replica:>7}  {_address(server):<17}{'down':<6}")
        else:
            print(f"{shard:>5} {replica:>7}  {_address(server):<17}{'up':<6}{reply.size:>9,}"
                  f"{reply.last_seq:>10}{reply.snapshot_seq:>14}")
    dataset = state.get("dataset")
    if dataset:
        print(f"loaded: the {dataset['limit']:,} most frequent {dataset['name']} words")
    return 0


def load(args):
    state = _read_state(args.home)
    replies = [_server_status(server) for _, _, server in _servers(state)]
    if any(reply is None for reply in replies):
        raise CliError("a server is down, and writes need every replica; see: vsearch status")
    if any(reply.size for reply in replies):
        raise CliError("this cluster already holds vectors; to start over: "
                       "vsearch cluster down --wipe")

    words, vectors = load_glove(args.glove_dir)
    limit = min(args.limit, len(words))
    started = time.perf_counter()
    try:
        with _coordinator(state) as coordinator:
            for start in range(0, limit, LOAD_BATCH):
                coordinator.add(vectors[start:min(start + LOAD_BATCH, limit)])
                print(f"  indexed {min(start + LOAD_BATCH, limit):,} / {limit:,} words")
    except (grpc.RpcError, ReplicasOutOfSyncError) as err:
        raise CliError(f"load stopped: {err}. Check `vsearch status`, then `vsearch repair`") from None
    state["dataset"] = {"name": "glove", "limit": limit,
                        "dir": str(Path(args.glove_dir).resolve())}
    _write_state(args.home, state)
    print(f"loaded {limit:,} words in {time.perf_counter() - started:.1f}s")
    return 0


def similar(args):
    state = _read_state(args.home)
    dataset = state.get("dataset")
    if not dataset:
        raise CliError("nothing is loaded yet; run: vsearch load glove")
    words, vectors = load_glove(dataset["dir"])
    word = args.word.lower()                   # the GloVe 6B vocabulary is lowercase
    try:
        word_id = words.index(word)
    except ValueError:
        raise CliError(f"{word!r} is not in the GloVe vocabulary") from None

    started = time.perf_counter()
    try:
        with _coordinator(state) as coordinator:
            # Ask for one extra: if the word itself is indexed, it comes back first.
            scored = coordinator.search_scored(vectors[word_id], args.k + 1)
            failovers = coordinator.failovers
    except ShardUnavailableError as err:
        raise CliError(f"{err}; see: vsearch status") from None
    elapsed_ms = (time.perf_counter() - started) * 1000

    neighbors = [(dist, global_id) for dist, global_id in scored if global_id != word_id][:args.k]
    print(f"words closest to {word!r}:")
    for rank, (dist, global_id) in enumerate(neighbors, start=1):
        print(f"{rank:>3}. {words[global_id]:<20} {1.0 - dist:.3f}")
    note = f", {failovers} failover(s)" if failovers else ""
    print(f"({elapsed_ms:.1f} ms across {len(state['servers'])} shards{note})")
    return 0


def repair(args):
    state = _read_state(args.home)
    with _coordinator(state) as coordinator:
        synced = coordinator.repair()
    if not synced:
        print("all reachable replicas are in sync")
    for sync in synced:
        print(f"shard {sync.shard}: {sync.replica} caught up from {sync.source} via "
              f"{sync.method} ({sync.records_applied} log records), now at seq {sync.last_seq}")
    return 0


# --- argument parsing ----------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(prog="vsearch", description="Run and query a local cluster.")
    parser.add_argument("--home", type=Path, default=Path("data/cluster"),
                        help="directory holding the cluster's state, data, and logs")
    commands = parser.add_subparsers(dest="command", required=True)

    cluster = commands.add_parser("cluster", help="start, stop, crash, or restart servers")
    actions = cluster.add_subparsers(dest="action", required=True)

    up = actions.add_parser("up", help="start the shard servers in the background")
    up.add_argument("--shards", type=int, default=4)
    up.add_argument("--replicas", type=int, default=2)
    up.add_argument("--seed", type=int, default=0)
    up.add_argument("--M", type=int, default=16)
    up.add_argument("--ef-construction", type=int, default=100)
    up.add_argument("--ef-search", type=int, default=100)
    up.add_argument("--snapshot-every", type=int, default=1000)
    up.set_defaults(run=cluster_up)

    down = actions.add_parser("down", help="stop the servers; their data stays on disk")
    down.add_argument("--wipe", action="store_true", help="also delete all of the cluster's data")
    down.set_defaults(run=cluster_down)

    for name, run, text in [("kill", cluster_kill, "crash one server with SIGKILL"),
                            ("restart", cluster_restart, "start a stopped server again")]:
        action = actions.add_parser(name, help=text)
        action.add_argument("shard", type=int)
        action.add_argument("replica", type=int)
        action.set_defaults(run=run)

    commands.add_parser("status", help="state, size, and write position of every server") \
        .set_defaults(run=status)
    commands.add_parser("repair", help="bring replicas that fell behind back in sync") \
        .set_defaults(run=repair)

    loader = commands.add_parser("load", help="index a dataset")
    loader.add_argument("dataset", choices=["glove"])
    loader.add_argument("--limit", type=int, default=100_000, help="how many words to index")
    loader.add_argument("--glove-dir", default="data", help="where glove.6B.zip or its cache is")
    loader.set_defaults(run=load)

    finder = commands.add_parser("similar", help="words closest in meaning to a word")
    finder.add_argument("word")
    finder.add_argument("-k", type=int, default=10)
    finder.set_defaults(run=similar)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.home = args.home.resolve()
    try:
        return args.run(args)
    except CliError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
