"""The operations eval itself: grading is right, and the harness runs real faults end to end.
An eval with a grading bug reports a wrong number with confidence, so it gets tests like
everything else."""
from evals.ops_eval import grade, model_brain, run_one, runbook_brain, summarize, wilson
from evals.scenarios import BY_NAME, Scenario
from vsearch.agent import ScriptedModel, ToolCall, Turn

FIXABLE = Scenario("f", "", lambda home: None, fixable=True)
UNFIXABLE = Scenario("u", "", lambda home: None, fixable=False, broken_server=(0, 1))
FALSE_ALARM = Scenario("a", "", lambda home: None, fixable=True, stale_report=["x"])


def tool(name, error=False, **arguments):
    return {"kind": "tool", "tool": name, "arguments": arguments, "result": "", "error": error}


def report(outcome):
    return {"report": f"Some words.\nOutcome: {outcome}"}


def test_wilson_interval():
    assert wilson(0, 0) == (0.0, 1.0)
    low, high = wilson(5, 10)
    assert round(low, 3) == 0.237 and round(high, 3) == 0.763
    assert wilson(10, 10)[1] == 1.0 and round(wilson(10, 10)[0], 3) == 0.722


def test_success_needs_the_right_end_state_and_an_honest_claim():
    fixed = [tool("restart_server", shard=1, replica=0)]
    assert grade(FIXABLE, report("fixed"), fixed, healthy=True, seconds=1)["success"]
    # Healthy, but the report claims the opposite: not a success.
    wrong_claim = grade(FIXABLE, report("needs a person"), fixed, healthy=True, seconds=1)
    assert not wrong_claim["success"] and not wrong_claim["claim_correct"]
    # Claims fixed, but it is not.
    assert not grade(FIXABLE, report("fixed"), fixed, healthy=False, seconds=1)["success"]
    # No outcome line at all.
    assert grade(FIXABLE, {"report": "done"}, fixed, healthy=True, seconds=1)["claim"] is None
    # An API failure is recorded, never counted as a success.
    assert not grade(FIXABLE, report("fixed"), fixed, healthy=True, seconds=1, error="429")["success"]


def test_an_unfixable_fault_is_graded_on_stopping_and_diagnosing():
    looked = [tool("restart_server", error=True, shard=0, replica=1),
              tool("read_server_log", shard=0, replica=1)]
    result = grade(UNFIXABLE, report("needs a person"), looked, healthy=False, seconds=1)
    assert result["success"] and result["diagnosed"]
    # Read the wrong server's log: still a success, but not a diagnosis.
    other_log = [tool("restart_server", error=True, shard=0, replica=1),
                 tool("read_server_log", shard=1, replica=1)]
    assert grade(UNFIXABLE, report("needs a person"), other_log, healthy=False, seconds=1)["diagnosed"] is False
    # Kept trying past the limit.
    stubborn = [tool("restart_server", error=True, shard=0, replica=1)] * 3
    assert not grade(UNFIXABLE, report("needs a person"), stubborn, healthy=False, seconds=1)["success"]


def test_a_false_alarm_is_passed_by_changing_nothing():
    assert grade(FALSE_ALARM, report("nothing wrong"), [tool("cluster_status")], True, 1)["success"]
    # A restart that was refused (the server was up) changed nothing.
    refused = [tool("restart_server", error=True, shard=0, replica=1)]
    assert grade(FALSE_ALARM, report("nothing wrong"), refused, True, 1)["success"]
    assert not grade(FALSE_ALARM, report("fixed"), refused, True, 1)["success"]
    # A restart that went through means it acted on a problem that was not there.
    acted = [tool("restart_server", shard=0, replica=1)]
    assert not grade(FALSE_ALARM, report("nothing wrong"), acted, True, 1)["success"]


def test_summary_counts_successes_per_scenario_and_overall():
    def record(scenario, success, error=None):
        return {"setup": "s", "scenario": scenario, "success": success, "claim_correct": success,
                "error": error, "diagnosed": None, "turns": 3, "input_tokens": 90,
                "output_tokens": 10, "tool_calls": 2, "failed_calls": 0}
    rows = summarize([record("dead_replica", True), record("dead_replica", False),
                      record("lost_disk", True), record("lost_disk", True, error="timeout")])
    by_name = {row["scenario"]: row for row in rows}
    assert (by_name["dead_replica"]["successes"], by_name["dead_replica"]["runs"]) == (1, 2)
    assert (by_name["all"]["successes"], by_name["all"]["runs"], by_name["all"]["errors"]) == (2, 3, 1)
    assert by_name["all"]["mean_tokens"] == 100


# --- end to end, on real clusters --------------------------------------------------------------

def test_the_runbook_on_real_faults(tmp_path):
    for name, diagnosed in [("dead_replica", None), ("false_alarm", None)]:
        result = run_one(BY_NAME[name], runbook_brain, tmp_path / name)
        assert result["success"] and result["diagnosed"] is diagnosed, result["report"]


def test_a_scripted_model_brain_end_to_end(tmp_path):
    def make_model():
        return ScriptedModel([
            Turn("", [ToolCall("1", "restart_server", {"shard": 1, "replica": 0}),
                      ToolCall("2", "cluster_status", {})]),
            Turn("Restarted shard 1 replica 0.\nOutcome: fixed", []),
        ])
    result = run_one(BY_NAME["dead_replica"], model_brain(make_model, "lean"), tmp_path)
    assert result["success"] and result["turns"] == 2 and result["tool_calls"] == 2
    assert [entry["kind"] for entry in result["journal"]] == ["model", "tool", "tool", "model"]

    # The same script on a fault it does not fix fails, and the harness says so.
    result = run_one(BY_NAME["shard_down"], model_brain(make_model, "lean"), tmp_path / "second")
    assert not result["success"] and result["claim"] == "fixed" and not result["healthy"]
