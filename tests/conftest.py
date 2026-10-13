"""Suite-wide checks and shared stand-ins."""
import json
import os
import subprocess
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

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


@pytest.fixture(autouse=True)
def no_real_model_calls(monkeypatch):
    """No test may reach a real model API or use a real key. The key file is not read, keys
    in the environment are removed, and the OpenAI address points at a port nothing listens
    on. Tests of the model clients install their own stand-ins on top of this."""
    monkeypatch.setattr("vsearch.cli.load_env_file", lambda *args: [], raising=False)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")


def _openai_reply(content=None, tool_calls=None, finish_reason="stop"):
    """A Chat Completions reply; tool_calls are (id, name, arguments as JSON text)."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [{"id": call_id, "type": "function",
                                  "function": {"name": name, "arguments": arguments}}
                                 for call_id, name, arguments in tool_calls]
    return 200, {"choices": [{"message": message, "finish_reason": finish_reason}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


@pytest.fixture
def openai_api(monkeypatch):
    """A local HTTP server standing in for the OpenAI API: it serves the queued `replies`
    (status, body) at /chat/completions and records every request."""
    api = types.SimpleNamespace(requests=[], replies=[], reply=_openai_reply)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            api.requests.append({"path": self.path, "auth": self.headers["Authorization"], "body": body})
            status, payload = api.replies.pop(0)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    yield api
    server.shutdown()
    server.server_close()
