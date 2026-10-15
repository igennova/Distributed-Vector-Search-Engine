"""
Faults to break a cluster with, for the operations eval. Each scenario breaks a freshly
started 2-shard, 2-replica cluster in one known way and says what a good outcome is.
"""
import shutil
import signal
from typing import Callable, NamedTuple

import grpc
import numpy as np

from vsearch.processes import stop
from vsearch.state import connect, read_state


class Scenario(NamedTuple):
    name: str
    description: str
    inject: Callable          # inject(home): break the cluster
    fixable: bool             # can the agent's tools make it healthy again?
    broken_server: tuple | None = None    # the server whose log explains an unfixable fault
    stale_report: list | None = None      # problems to report although nothing is wrong


def kill(home, shard, replica):
    """Crash one server (SIGKILL: no clean shutdown)."""
    stop([read_state(home)["servers"][shard][replica]], signal.SIGKILL)


def _miss_a_write(home):
    """Kill shard 0 replica 1 while a coordinator is writing, so the write reaches only its
    twin. (A coordinator that starts with a server down refuses to write at all.)"""
    dim = read_state(home)["dataset"]["dim"]
    vectors = np.random.default_rng(0).standard_normal((6, dim)).astype(np.float32)
    with connect(read_state(home)) as coordinator:
        coordinator.add(vectors[:2])
        kill(home, 0, 1)
        try:
            coordinator.add(vectors[2:])
        except grpc.RpcError:
            pass


def _lose_disk(home):
    kill(home, 1, 1)
    shutil.rmtree(home / "shard-1-replica-1")


def _corrupt_snapshot(home):
    kill(home, 0, 1)
    (home / "shard-0-replica-1" / "snapshot.npz").write_bytes(b"this is not a snapshot")


SCENARIOS = [
    Scenario("dead_replica", "one replica killed",
             lambda home: kill(home, 1, 0), fixable=True),
    Scenario("missed_writes", "one replica killed during a write, so it is behind after a restart",
             _miss_a_write, fixable=True),
    Scenario("lost_disk", "one replica killed and its data directory deleted",
             _lose_disk, fixable=True),
    Scenario("two_down", "one replica killed in each shard",
             lambda home: (kill(home, 0, 0), kill(home, 1, 1)), fixable=True),
    Scenario("shard_down", "both replicas of one shard killed: its searches fail",
             lambda home: (kill(home, 1, 0), kill(home, 1, 1)), fixable=True),
    Scenario("corrupt_snapshot", "one replica killed and its snapshot file overwritten",
             _corrupt_snapshot, fixable=False, broken_server=(0, 1)),
    Scenario("false_alarm", "nothing is wrong, but the report says a replica is down",
             lambda home: None, fixable=True, stale_report=["shard 0 replica 1 is down"]),
]
BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
