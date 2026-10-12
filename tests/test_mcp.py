"""The MCP protocol layer: a session over stdio, tool calls, and the two kinds of error."""
import io
import json

import pytest

from vsearch.mcp import (INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND, PARSE_ERROR,
                         PROTOCOL_VERSIONS, McpServer, Tool, ToolError, check_arguments)

ADD_SCHEMA = {"type": "object",
              "properties": {"a": {"type": "integer"}, "b": {"type": "integer", "maximum": 10},
                             "tags": {"type": "array", "items": {"type": "string"}},
                             "loud": {"type": "boolean"}},
              "required": ["a"]}


def _add(a, b=0, tags=None, loud=False):
    if a < 0:
        raise ToolError("a must not be negative; pass a positive number")
    if a == 99:
        raise RuntimeError("boom")
    print("a stray print from inside a tool")              # must not reach the protocol stream
    return f"sum is {a + b}"


@pytest.fixture
def server():
    return McpServer("test-server", "1.2.3", [Tool("add", "Add two numbers.", ADD_SCHEMA, _add)],
                     instructions="Use add for sums.")


def request(method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def call(server, arguments, name="add"):
    return server.handle(request("tools/call", {"name": name, "arguments": arguments}))["result"]


def test_initialize_agrees_on_a_version_and_describes_the_server(server):
    reply = server.handle(request("initialize", {"protocolVersion": "2024-11-05",
                                                 "capabilities": {}, "clientInfo": {"name": "t"}}))
    assert reply["jsonrpc"] == "2.0" and reply["id"] == 1
    assert reply["result"] == {"protocolVersion": "2024-11-05",
                               "capabilities": {"tools": {"listChanged": False}},
                               "serverInfo": {"name": "test-server", "version": "1.2.3"},
                               "instructions": "Use add for sums."}
    # A revision this server does not know: it answers with the newest one it does.
    newer = server.handle(request("initialize", {"protocolVersion": "2999-01-01"}))
    assert newer["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]


def test_tools_are_listed_with_their_schema(server):
    assert server.handle(request("tools/list"))["result"] == {"tools": [
        {"name": "add", "description": "Add two numbers.", "inputSchema": ADD_SCHEMA}]}
    assert server.handle(request("ping", request_id="abc")) == {"jsonrpc": "2.0", "id": "abc",
                                                                "result": {}}


def test_a_tool_call_returns_its_text(server):
    assert call(server, {"a": 2, "b": 3}) == {"content": [{"type": "text", "text": "sum is 5"}],
                                              "isError": False}


def test_a_failing_tool_is_a_result_the_model_can_read_not_a_protocol_error(server):
    failed = call(server, {"a": -1})
    assert failed["isError"] and "pass a positive number" in failed["content"][0]["text"]
    crashed = call(server, {"a": 99})                      # a bug in the tool
    assert crashed["isError"] and "RuntimeError: boom" in crashed["content"][0]["text"]
    assert call(server, {"a": 1})["isError"] is False      # and the server carries on


@pytest.mark.parametrize("arguments, problem", [
    ({}, "missing required argument 'a'"),
    ({"a": "2"}, "'a' must be of type integer"),
    ({"a": True}, "'a' must be of type integer"),          # a boolean is not an integer
    ({"a": 1, "b": 11}, "'b' must be at most 10"),
    ({"a": 1, "c": 1}, "unknown argument 'c'"),
    ({"a": 1, "tags": ["x", 2]}, "every item of 'tags' must be of type string"),
    ([1, 2], "must be an object"),
])
def test_bad_arguments_are_explained_to_the_model(server, arguments, problem):
    assert problem in check_arguments(ADD_SCHEMA, arguments)
    if isinstance(arguments, dict):
        result = call(server, arguments)
        assert result["isError"] and problem in result["content"][0]["text"]
    assert check_arguments(ADD_SCHEMA, {"a": 1, "tags": ["x"], "loud": True}) is None


def test_requests_the_server_cannot_serve_are_protocol_errors(server):
    assert server.handle(request("resources/list"))["error"]["code"] == METHOD_NOT_FOUND
    unknown = server.handle(request("tools/call", {"name": "nope", "arguments": {}}))
    assert unknown["error"]["code"] == INVALID_PARAMS and "id" in unknown
    assert server.handle({"jsonrpc": "2.0", "id": 5})["error"]["code"] == INVALID_REQUEST
    assert server.handle("hello")["error"]["code"] == INVALID_REQUEST


def test_notifications_get_no_reply(server):
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert server.handle({"jsonrpc": "2.0", "method": "tools/list"}) is None    # no id: no reply


def test_a_batch_is_answered_as_a_batch(server):
    replies = server.handle([request("ping", request_id=1),
                             {"jsonrpc": "2.0", "method": "notifications/initialized"},
                             request("tools/list", request_id=2)])
    assert [reply["id"] for reply in replies] == [1, 2]
    assert server.handle([{"jsonrpc": "2.0", "method": "notifications/initialized"}]) is None


def test_a_session_over_stdio_is_one_json_message_per_line(server, capsys):
    lines = [json.dumps(request("initialize", {"protocolVersion": "2025-06-18"}, request_id=0)),
             json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
             "",                                            # blank lines are ignored
             "this is not json",
             json.dumps(request("tools/call", {"name": "add", "arguments": {"a": 4, "b": 1}},
                                request_id=1)),
             json.dumps(request("tools/call", {"name": "add", "arguments": {"a": 1}}, request_id=2),
                        ensure_ascii=False)]
    out = io.StringIO()
    server.serve(io.StringIO("\n".join(lines) + "\n"), out)

    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert len(replies) == 4 and out.getvalue().endswith("\n")
    assert replies[0]["result"]["protocolVersion"] == "2025-06-18"
    assert replies[1] == {"jsonrpc": "2.0", "id": None,
                          "error": {"code": PARSE_ERROR, "message": "the message is not valid JSON"}}
    assert replies[2]["result"]["content"][0]["text"] == "sum is 5"
    # The tool's own print went to stderr, not into the protocol stream.
    captured = capsys.readouterr()
    assert "stray print" in captured.err and "stray print" not in out.getvalue()
