"""
The shard server processes of a local cluster: start one, tell whether a recorded process
is still that server, and stop it. Used by the command line and the operations agent.
"""
import os
import subprocess
import time

import grpc

from .client import REPO_ROOT, server_command, wait_until_ready
from .state import UserError, address

_spawned = []     # keep handles to servers started by this process until it exits


def pick(state, shard, replica):
    """The record of one server, or a UserError if the cluster has no such server."""
    if shard < 0 or replica < 0:
        raise UserError(f"there is no shard {shard} replica {replica} in this cluster")
    try:
        return state["servers"][shard][replica]
    except IndexError:
        raise UserError(f"there is no shard {shard} replica {replica} in this cluster") from None


def log_path(home, shard, replica):
    return home / "logs" / f"shard-{shard}-replica-{replica}.log"


def is_running(server):
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


def spawn(home, state, shard, replica):
    """Start one server in the background and record its pid in `state`."""
    server = state["servers"][shard][replica]
    name = f"shard-{shard}-replica-{replica}"
    (home / "logs").mkdir(parents=True, exist_ok=True)
    command = server_command(server["port"], seed=state["seed"] + shard, data_dir=home / name,
                             **state["server_args"])
    with open(log_path(home, shard, replica), "ab") as log:
        # A new session detaches the server from this terminal, so it keeps running after
        # the command returns.
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    _spawned.append(process)
    server["pid"] = process.pid


def wait_for(targets, home, timeout=20):
    """Wait until each (shard, replica, server) answers; UserError naming its log if not."""
    for shard, replica, server in targets:
        try:
            wait_until_ready(address(server), timeout=timeout)
        except grpc.FutureTimeoutError:
            raise UserError(f"shard {shard} replica {replica} did not start; see "
                            f"{log_path(home, shard, replica)}") from None


def stop(targets, sig):
    """Signal the given servers that are running and wait for them to exit. Returns how many."""
    running = [server for server in targets if is_running(server)]
    for server in running:
        os.kill(server["pid"], sig)
    deadline = time.monotonic() + 10
    while any(is_running(server) for server in running) and time.monotonic() < deadline:
        time.sleep(0.05)
    return len(running)
