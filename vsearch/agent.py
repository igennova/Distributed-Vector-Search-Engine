"""
An agent: a loop in which a model decides what to do next.

    ask the model  ->  it answers with text and the tools it wants to run
    run the tools  ->  hand the results back to the model
    repeat until the model answers without asking for a tool, or the step limit is hit

The loop does not know which model it is talking to. A model is anything with

    begin(system, task, tools)       start a conversation
    step(outcomes) -> Turn           the model's next turn, given the results of the tools
                                     it asked for in its last one (None on the first step)

OpenAIModel and ClaudeModel talk to those companies' APIs. ScriptedModel replays a fixed
script, so the loop and the tools can be tested without a model, a network, or an API key.
"""
import json
import os
import urllib.error
import urllib.request
from typing import NamedTuple

from .mcp import run_tool

OPENAI_MODEL = "gpt-5.4-mini"
OPENAI_URL = "https://api.openai.com/v1"
CLAUDE_MODEL = "claude-opus-5-5"
# Asks the API to re-run a request on another model if a safety classifier declines it.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ToolCall(NamedTuple):
    id: str
    name: str
    arguments: dict
    problem: str | None = None    # set when the model's request could not even be read


class Turn(NamedTuple):
    text: str             # what the model said
    calls: list           # the ToolCalls it wants run; empty means it is finished


class ToolOutcome(NamedTuple):
    call: ToolCall
    text: str
    is_error: bool


class AgentRun(NamedTuple):
    report: str           # the model's final words
    finished: bool        # False if the step limit stopped it
    steps: int            # turns the model took
    calls: int            # tools it ran


class ModelUnavailable(RuntimeError):
    """The model cannot be used here: a missing package or missing credentials."""


class ModelError(RuntimeError):
    """The model could not produce a turn (declined, cut off, or the API failed)."""


def run_agent(model, tools, task, system, max_steps=12, journal=None):
    """Let the model work on the task with the tools until it stops asking for them."""
    by_name = {tool.name: tool for tool in tools}
    model.begin(system, task, tools)
    outcomes, calls, text = None, 0, ""
    for step in range(1, max_steps + 1):
        turn = model.step(outcomes)
        text = turn.text or text
        if journal:
            journal.write("model", step=step, text=turn.text,
                          calls=[{"tool": call.name, "arguments": call.arguments}
                                 for call in turn.calls])
        if not turn.calls:
            return AgentRun(turn.text, True, step, calls)
        outcomes = []
        for call in turn.calls:
            calls += 1
            if call.problem:
                outcomes.append(ToolOutcome(call, call.problem, True))
                continue
            if call.name not in by_name:
                outcomes.append(ToolOutcome(
                    call, f"There is no tool named {call.name!r}. The tools are "
                          f"{sorted(by_name)}.", True))
                continue
            result, is_error = run_tool(by_name[call.name], call.arguments)
            outcomes.append(ToolOutcome(call, result, is_error))
    # Out of steps. Whatever the model was in the middle of is not trusted as a conclusion.
    return AgentRun(text, False, max_steps, calls)


class ScriptedModel:
    """Replays a script instead of thinking. Each entry is a Turn, or a function that takes
    the outcomes of the previous turn's tools and returns a Turn."""

    def __init__(self, script):
        self._script = list(script)
        self.seen = []                         # the outcomes it was handed, turn by turn

    def begin(self, system, task, tools):
        self.system, self.task = system, task
        self.tool_names = [tool.name for tool in tools]

    def step(self, outcomes):
        self.seen.append(outcomes)
        entry = self._script.pop(0)
        return entry(outcomes) if callable(entry) else entry


class OpenAIModel:
    """An OpenAI model through the Chat Completions API. One HTTPS endpoint, called with
    the standard library, so there is nothing to install. Needs OPENAI_API_KEY, from the
    environment or a .env file."""

    def __init__(self, model=OPENAI_MODEL, timeout=120):
        self._key = os.environ.get("OPENAI_API_KEY")
        if not self._key:
            raise ModelUnavailable("the OpenAI brain needs OPENAI_API_KEY: set it in the "
                                   "environment or in a .env file in the current directory")
        self._url = os.environ.get("OPENAI_BASE_URL", OPENAI_URL).rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.input_tokens = self.output_tokens = 0

    def begin(self, system, task, tools):
        self._tools = [{"type": "function",
                        "function": {"name": tool.name, "description": tool.description,
                                     "parameters": tool.input_schema}} for tool in tools]
        self._messages = [{"role": "system", "content": system},
                          {"role": "user", "content": task}]

    def step(self, outcomes):
        for outcome in outcomes or []:
            # One message per tool call, tied to it by id. There is no error flag in this
            # API, so a failure is marked in the text.
            self._messages.append({"role": "tool", "tool_call_id": outcome.call.id,
                                   "content": ("ERROR: " if outcome.is_error else "") + outcome.text})
        body = {"model": self.model, "messages": self._messages}
        if self._tools:
            body["tools"] = self._tools
        reply = self._post(body)

        usage = reply.get("usage") or {}
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        choice = reply["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ModelError("the model's answer was cut off at the output limit")
        if choice.get("finish_reason") == "content_filter":
            raise ModelError("the model declined to continue with this request")
        message = choice["message"]
        # The model's turn goes back as it came, so its tool calls keep their ids.
        self._messages.append({key: message[key] for key in ("role", "content", "tool_calls")
                               if message.get(key) is not None or key == "content"})
        calls = []
        for requested in message.get("tool_calls") or []:
            function = requested["function"]
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                problem = None if isinstance(arguments, dict) else "the arguments must be an object"
            except ValueError:
                arguments, problem = {}, "the arguments were not valid JSON"
            calls.append(ToolCall(requested["id"], function["name"],
                                  arguments if problem is None else {}, problem))
        return Turn(message.get("content") or "", calls)

    def _post(self, body):
        request = urllib.request.Request(
            self._url, data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as err:
            try:
                detail = json.load(err)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                detail = err.reason
            detail = str(detail).replace(self._key, "<key>")     # never echo the key back
            if err.code == 401:
                raise ModelUnavailable(f"OpenAI rejected the API key: {detail}") from None
            raise ModelError(f"the OpenAI API returned an error ({err.code}): {detail}") from None
        except (urllib.error.URLError, TimeoutError) as err:
            raise ModelError(f"could not reach the OpenAI API: {getattr(err, 'reason', err)}") from None


class ClaudeModel:
    """A Claude model through the Messages API (needs the `anthropic` package and
    credentials, e.g. ANTHROPIC_API_KEY)."""

    def __init__(self, model=CLAUDE_MODEL, max_tokens=16000):
        try:
            import anthropic
        except ImportError:
            raise ModelUnavailable(
                "the Claude brain needs the anthropic package: pip install anthropic") from None
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()   # credentials come from the environment
        self.model = model
        self.max_tokens = max_tokens
        self.input_tokens = self.output_tokens = 0

    def begin(self, system, task, tools):
        self._system = system
        self._tools = [{"name": tool.name, "description": tool.description,
                        "input_schema": tool.input_schema} for tool in tools]
        self._messages = [{"role": "user", "content": task}]

    def step(self, outcomes):
        if outcomes:
            # Every result of one turn goes back in a single message.
            self._messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": outcome.call.id, "content": outcome.text,
                 "is_error": outcome.is_error} for outcome in outcomes]})
        anthropic = self._anthropic
        try:
            response = self._client.beta.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=self._system,
                tools=self._tools, messages=self._messages,
                betas=[FALLBACK_BETA], fallbacks="default")
        except anthropic.AuthenticationError:
            raise ModelUnavailable("the Claude API rejected the credentials; set a valid "
                                   "ANTHROPIC_API_KEY") from None
        except anthropic.RateLimitError:
            raise ModelError("the Claude API is rate limiting this key; try again later") from None
        except anthropic.APIStatusError as err:
            raise ModelError(f"the Claude API returned an error ({err.status_code}): "
                             f"{err.message}") from None
        except anthropic.APIConnectionError:
            raise ModelError("could not reach the Claude API") from None

        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        if response.stop_reason == "refusal":
            raise ModelError("the model declined to continue with this request")
        if response.stop_reason == "max_tokens":
            raise ModelError("the model's answer was cut off at the output limit")
        # The whole turn goes back unchanged next time, including the blocks that are
        # neither text nor tool calls (the model's own reasoning).
        self._messages.append({"role": "assistant", "content": response.content})
        text = "\n".join(block.text for block in response.content if block.type == "text")
        calls = [ToolCall(block.id, block.name, block.input)
                 for block in response.content if block.type == "tool_use"]
        return Turn(text, calls)
