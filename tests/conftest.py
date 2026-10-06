"""Suite-wide checks."""
import os
import subprocess

import pytest


@pytest.fixture(scope="session", autouse=True)
def no_leftover_servers():
    """Fail the run if any shard server started by the tests is still alive at the end."""
    yield
    processes = subprocess.run(["ps", "-ww", "-eo", "pid=,ppid=,command="],
                               capture_output=True, text=True).stdout.splitlines()
    mine = [line.strip() for line in processes
            if "vsearch.server" in line and line.split()[1] == str(os.getpid())]
    assert not mine, "shard servers still running after the tests:\n" + "\n".join(mine)
