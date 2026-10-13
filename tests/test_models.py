"""The model clients, against stand-ins: OpenAI's API as a local HTTP server (so the real
request and response path is exercised), and the anthropic SDK as a stand-in module. They
check the shapes this code sends and reads, not the real services."""
import json
import os
import sys
import types

import pytest

from vsearch.agent import (AgentRun, ClaudeModel, ModelError, ModelUnavailable, OpenAIModel,
                           run_agent)
from vsearch.mcp import Tool, ToolError
from vsearch.state import load_env_file

SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


def _weather(city):
    if city == "Atlantis":
        raise ToolError("no such city")
    return f"sunny in {city}"


TOOLS = [Tool("weather", "The weather in a city.", SCHEMA, _weather)]


# --- OpenAI, against the local stand-in from conftest.py -----------------------------------------

def test_openai_model_runs_a_tool_and_returns_its_result(openai_api):
    openai_api.replies += [
        openai_api.reply(tool_calls=[("call_1", "weather", '{"city": "Paris"}'),
                           ("call_2", "weather", '{"city": "Atlantis"}'),
                           ("call_3", "weather", '{"city": ')], finish_reason="tool_calls"),
        openai_api.reply("Paris is sunny; Atlantis does not exist."),
    ]
    model = OpenAIModel("some-model")
    result = run_agent(model, TOOLS, "weather in Paris and Atlantis?", "You report weather.")

    assert result == AgentRun("Paris is sunny; Atlantis does not exist.", finished=True, steps=2, calls=3)
    assert (model.input_tokens, model.output_tokens) == (200, 20)
    first, second = openai_api.requests
    assert first["path"] == "/v1/chat/completions" and first["auth"] == "Bearer sk-test-not-a-real-key"
    assert first["body"]["model"] == "some-model"
    assert first["body"]["messages"] == [{"role": "system", "content": "You report weather."},
                                         {"role": "user", "content": "weather in Paris and Atlantis?"}]
    assert first["body"]["tools"] == [{"type": "function", "function": {
        "name": "weather", "description": "The weather in a city.", "parameters": SCHEMA}}]

    # The second request carries the model's turn back with its calls, then one result per call.
    sent = second["body"]["messages"]
    assert sent[2]["role"] == "assistant" and [call["id"] for call in sent[2]["tool_calls"]] == \
        ["call_1", "call_2", "call_3"]
    assert sent[3:] == [
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny in Paris"},
        {"role": "tool", "tool_call_id": "call_2", "content": "ERROR: no such city"},
        {"role": "tool", "tool_call_id": "call_3", "content": "ERROR: the arguments were not valid JSON"},
    ]


def test_openai_model_explains_failures_without_showing_the_key(openai_api):
    key = os.environ["OPENAI_API_KEY"]
    openai_api.replies += [
        (401, {"error": {"message": f"Incorrect API key provided: {key}"}}),
        (429, {"error": {"message": "You exceeded your current quota."}}),
        (500, {"oops": True}),
        openai_api.reply("half an ans", finish_reason="length"),
    ]
    model = OpenAIModel()
    for expected, message in [(ModelUnavailable, "rejected the API key"), (ModelError, "exceeded your current quota"),
                              (ModelError, r"error \(500\)"), (ModelError, "cut off")]:
        model.begin("system", "task", TOOLS)
        with pytest.raises(expected, match=message) as err:
            model.step(None)
        assert key not in str(err.value)


def test_openai_model_needs_a_key_and_a_reachable_api(monkeypatch):
    with pytest.raises(ModelUnavailable, match="OPENAI_API_KEY"):
        OpenAIModel()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")      # the address is a dead port
    model = OpenAIModel(timeout=2)
    model.begin("system", "task", [])
    with pytest.raises(ModelError, match="could not reach the OpenAI API"):
        model.step(None)


# --- a stand-in for the anthropic SDK ----------------------------------------------------------

def _fake_anthropic(responses, requests):
    """A module shaped like the parts of the anthropic SDK that ClaudeModel uses."""
    module = types.ModuleType("anthropic")
    for name in ("AuthenticationError", "RateLimitError", "APIStatusError", "APIConnectionError"):
        setattr(module, name, type(name, (Exception,), {}))

    class Anthropic:
        def __init__(self):
            self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self._create))

        def _create(self, **request):
            requests.append(json.loads(json.dumps(request, default=vars)))
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    module.Anthropic = Anthropic
    return module


def _block(**fields):
    return types.SimpleNamespace(**fields)


def _response(blocks, stop_reason):
    return types.SimpleNamespace(content=blocks, stop_reason=stop_reason,
                                 usage=types.SimpleNamespace(input_tokens=100, output_tokens=20))


def test_claude_model_sends_tools_and_returns_results_in_the_api_shapes(monkeypatch):
    requests = []
    responses = [
        _response([_block(type="thinking", thinking="", signature="sig"),
                   _block(type="text", text="Checking."),
                   _block(type="tool_use", id="tu_1", name="weather", input={"city": "Paris"})], "tool_use"),
        _response([_block(type="text", text="Sunny.")], "end_turn"),
    ]
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(responses, requests))
    model = ClaudeModel()
    result = run_agent(model, TOOLS, "weather in Paris?", "You report weather.")

    assert result == AgentRun("Sunny.", finished=True, steps=2, calls=1)
    assert (model.input_tokens, model.output_tokens) == (200, 40)
    first, second = requests
    assert first["model"] == "claude-opus-5-5" and first["system"] == "You report weather."
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["messages"] == [{"role": "user", "content": "weather in Paris?"}]
    assert first["tools"] == [{"name": "weather", "description": "The weather in a city.",
                               "input_schema": SCHEMA}]
    # The second request carries the model's whole first turn back (reasoning block included),
    # then the tool's result under the id of the call it answers.
    assert [block["type"] for block in second["messages"][1]["content"]] == ["thinking", "text", "tool_use"]
    assert second["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "sunny in Paris", "is_error": False}]}


def test_claude_model_reports_why_it_could_not_produce_a_turn(monkeypatch):
    requests, responses = [], []
    fake = _fake_anthropic(responses, requests)
    responses += [_response([], "refusal"), _response([_block(type="text", text="half")], "max_tokens"),
                  fake.AuthenticationError(), fake.APIConnectionError()]
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    model = ClaudeModel()
    for expected, message in [(ModelError, "declined"), (ModelError, "cut off"),
                              (ModelUnavailable, "ANTHROPIC_API_KEY"), (ModelError, "could not reach")]:
        model.begin("system", "task", [])
        with pytest.raises(expected, match=message):
            model.step(None)

    monkeypatch.setitem(sys.modules, "anthropic", None)       # the package is not installed
    with pytest.raises(ModelUnavailable, match="pip install anthropic"):
        ClaudeModel()


# --- the key file --------------------------------------------------------------------------------

def test_env_file_sets_what_is_missing_and_never_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("ALREADY_SET", "from the shell")
    for name in ("PLAIN", "QUOTED", "EXPORTED"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text('# a comment\nPLAIN=abc=def\nQUOTED = "with spaces"\nexport EXPORTED=\'x\'\n'
                    "ALREADY_SET=from the file\n\nnot a setting\n")
    assert load_env_file(path) == ["PLAIN", "QUOTED", "EXPORTED"]        # names, never values
    assert (os.environ["PLAIN"], os.environ["QUOTED"], os.environ["EXPORTED"]) == \
        ("abc=def", "with spaces", "x")
    assert os.environ["ALREADY_SET"] == "from the shell"
    assert load_env_file(tmp_path / "missing") == []
