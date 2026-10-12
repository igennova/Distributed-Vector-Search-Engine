"""
A Model Context Protocol (MCP) server over stdio: the part that speaks the protocol, with
no knowledge of what its tools do.

MCP lets an AI application call tools that another program offers. The two sides exchange
JSON-RPC 2.0 messages, one JSON object per line: the client writes to this process's
stdin, and replies go to stdout.

    client -> {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "search", "arguments": {"question": "..."}}}
    server -> {"jsonrpc": "2.0", "id": 1,
               "result": {"content": [{"type": "text", "text": "..."}], "isError": false}}

A session is: initialize (agree on a protocol version, say what this server offers), then
any number of tools/list and tools/call. A message without an id is a notification and
gets no reply.

Two kinds of failure are kept apart, as the protocol asks. A tool that ran and failed
returns a normal result with isError set, so the model reads the message and can try
something else. A malformed request (unknown method, unknown tool) is a JSON-RPC error,
which is for the client program, not the model.
"""
import json
import sys
from contextlib import redirect_stdout
from typing import Callable, NamedTuple

# Revisions of the protocol this server can speak, newest first. It only uses tools, which
# work the same way in all of them.
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
               "array": list, "object": dict}


class ToolError(Exception):
    """A tool could not do what was asked. The message goes back to the model, so it should
    say what went wrong and, where possible, what to do instead."""


class Tool(NamedTuple):
    name: str
    description: str          # what the model reads to decide when to call the tool
    input_schema: dict        # JSON Schema of the arguments (an object)
    handler: Callable         # handler(**arguments) -> text shown to the model


class _RequestError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def check_arguments(schema, arguments):
    """Check arguments against a tool's schema: required ones are present, none are unknown,
    and each has the declared type and range. Returns the problem as a sentence, or None.
    This covers the part of JSON Schema the tools here use, not all of it."""
    if not isinstance(arguments, dict):
        return "the arguments must be an object"
    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in arguments:
            return f"missing required argument {name!r}"
    for name, value in arguments.items():
        if name not in properties:
            return f"unknown argument {name!r}; this tool takes {sorted(properties)}"
        rule = properties[name]
        expected = _JSON_TYPES[rule["type"]]
        # bool is a kind of int in Python, but not in JSON.
        if not isinstance(value, expected) or (isinstance(value, bool) and rule["type"] != "boolean"):
            return f"argument {name!r} must be of type {rule['type']}"
        if "minimum" in rule and value < rule["minimum"]:
            return f"argument {name!r} must be at least {rule['minimum']}"
        if "maximum" in rule and value > rule["maximum"]:
            return f"argument {name!r} must be at most {rule['maximum']}"
        if rule["type"] == "array" and "items" in rule:
            item_type = _JSON_TYPES[rule["items"]["type"]]
            if not all(isinstance(item, item_type) for item in value):
                return f"every item of {name!r} must be of type {rule['items']['type']}"
    return None


class McpServer:
    def __init__(self, name, version, tools, instructions=None, on_close=None):
        self.name = name
        self.version = version
        self.tools = {tool.name: tool for tool in tools}
        self.instructions = instructions      # general guidance for the model, sent once
        self._on_close = on_close             # releases whatever the tools hold open

    # --- one message -----------------------------------------------------------------------

    def handle(self, message):
        """Answer one decoded JSON-RPC message. Returns the reply, or None for a notification."""
        if isinstance(message, list):         # a batch (protocol revision 2025-03-26)
            if not message:
                return _error(None, INVALID_REQUEST, "empty batch")
            replies = [reply for reply in map(self.handle, message) if reply is not None]
            return replies or None
        if not isinstance(message, dict) or not isinstance(message.get("method"), str):
            request_id = message.get("id") if isinstance(message, dict) else None
            return _error(request_id, INVALID_REQUEST, "not a JSON-RPC request")
        if "id" not in message:               # a notification: act on it, never reply
            return None
        params = message.get("params") or {}
        try:
            result = self._dispatch(message["method"], params)
        except _RequestError as err:
            return _error(message["id"], err.code, str(err))
        return {"jsonrpc": "2.0", "id": message["id"], "result": result}

    def _dispatch(self, method, params):
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [{"name": tool.name, "description": tool.description,
                               "inputSchema": tool.input_schema}
                              for tool in self.tools.values()]}
        if method == "tools/call":
            return self._call_tool(params)
        raise _RequestError(METHOD_NOT_FOUND, f"method not found: {method}")

    def _initialize(self, params):
        # Use the client's revision if this server speaks it; otherwise offer the newest one
        # it does speak, and the client decides whether to go on.
        requested = params.get("protocolVersion")
        version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        result = {"protocolVersion": version,
                  "capabilities": {"tools": {"listChanged": False}},
                  "serverInfo": {"name": self.name, "version": self.version}}
        if self.instructions:
            result["instructions"] = self.instructions
        return result

    def _call_tool(self, params):
        tool = self.tools.get(params.get("name"))
        if tool is None:
            raise _RequestError(INVALID_PARAMS, f"unknown tool: {params.get('name')!r}")
        arguments = params.get("arguments") or {}
        problem = check_arguments(tool.input_schema, arguments)
        if problem:
            return _tool_result(f"Cannot call {tool.name}: {problem}.", is_error=True)
        try:
            return _tool_result(tool.handler(**arguments))
        except ToolError as err:
            return _tool_result(str(err), is_error=True)
        except Exception as err:              # a bug in a tool must not end the session
            return _tool_result(f"{tool.name} failed unexpectedly: {type(err).__name__}: {err}",
                                is_error=True)

    # --- the session -----------------------------------------------------------------------

    def serve(self, stdin=None, stdout=None):
        """Read messages from stdin and write replies to stdout until stdin closes."""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        # stdout carries the protocol and nothing else: a stray print from any library would
        # corrupt the stream, so everything else written to sys.stdout goes to stderr.
        try:
            with redirect_stdout(sys.stderr):
                for line in stdin:
                    if not line.strip():
                        continue
                    try:
                        message = json.loads(line)
                    except ValueError:
                        reply = _error(None, PARSE_ERROR, "the message is not valid JSON")
                    else:
                        reply = self.handle(message)
                    if reply is not None:
                        # One line per message; json.dumps never emits a raw newline.
                        stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                        stdout.flush()
        finally:
            self.close()

    def close(self):
        if self._on_close is not None:
            self._on_close()


def _error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_result(text, is_error=False):
    return {"content": [{"type": "text", "text": text}], "isError": is_error}
