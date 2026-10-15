"""
How well does the operations agent repair a cluster? Each run starts a fresh 2-shard,
2-replica cluster, breaks it in a known way (evals/scenarios.py), lets a brain work on it,
and grades the result with code:

  end state    the cluster is healthy afterwards (or, for a fault the tools cannot fix,
               it is left alone after at most the allowed restarts)
  claim        the report's last line (Outcome: fixed / needs a person / nothing wrong)
               matches what actually happened
  success      both of the above
  diagnosed    for the unfixable fault: the brain read the log of the broken server

and records the cost: model turns, tool calls, calls that failed (a restart that did not
come up, or one refused by a guardrail), tokens, and time.
Every run's journal is kept, to read what happened in the runs that failed.

Run:  python -m evals.ops_eval --brain runbook --runs 3
      python -m evals.ops_eval --brain openai --prompt lean --runs 10
      python -m evals.ops_eval --summary evals/results/*.jsonl
"""
import argparse
import contextlib
import io
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

# gRPC logs a line at info level for every process started while a channel is open; the
# health checks start `ps` many times per run.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

from vsearch.agent import OPENAI_MODEL, ModelError, OpenAIModel, run_agent
from vsearch.cli import main as vsearch
from vsearch.ops import PROMPTS, Journal, Operations, claimed_outcome, run_runbook, task_for
from vsearch.state import load_env_file

from .scenarios import BY_NAME

RESULTS = Path(__file__).parent / "results"
CORPUS = Path(__file__).parent / "corpus" / "DECISIONS.md"
START_TIMEOUT = 8          # seconds a restarted server gets to answer (small clusters start fast)


# --- brains: what decides the steps ------------------------------------------------------------

def runbook_brain(ops, problems):
    return {"report": run_runbook(ops)}


def model_brain(make_model, prompt, max_steps=12):
    """A brain backed by a model. make_model() returns a fresh model for each run."""
    def decide(ops, problems):
        model = make_model()
        run = run_agent(model, ops.tools(), task_for(problems), PROMPTS[prompt],
                        max_steps=max_steps, journal=ops.journal)
        return {"report": run.report, "finished": run.finished, "turns": run.steps,
                "input_tokens": getattr(model, "input_tokens", 0),
                "output_tokens": getattr(model, "output_tokens", 0)}
    return decide


# --- one run -----------------------------------------------------------------------------------

def _quiet(*argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return vsearch(list(map(str, argv)))


def run_one(scenario, decide, workdir):
    """Break a fresh cluster with the scenario, let `decide` work on it, and grade it."""
    home = Path(workdir) / "cluster"
    if _quiet("--home", home, "cluster", "up", "--shards", 2, "--replicas", 2) != 0:
        raise RuntimeError("could not start a cluster")
    try:
        _quiet("--home", home, "ingest", CORPUS, "--embedder", "hash")
        scenario.inject(home)
        ops = Operations(home, Journal(), start_timeout=START_TIMEOUT)
        problems = scenario.stale_report or ops.problems()

        started = time.perf_counter()
        error = None
        try:
            outcome = decide(ops, problems)
        except ModelError as err:                # the API failed: not the agent's doing
            outcome, error = {"report": ""}, str(err)
        seconds = time.perf_counter() - started

        healthy = not ops.problems()
        return grade(scenario, outcome, ops.journal.entries, healthy, seconds, error)
    finally:
        _quiet("--home", home, "cluster", "down", "--wipe")


def grade(scenario, outcome, journal, healthy, seconds, error=None):
    tools = [entry for entry in journal if entry["kind"] == "tool"]
    failed = [entry for entry in tools if entry["error"]]
    restarts = [entry for entry in tools if entry["tool"] == "restart_server"]
    claim = claimed_outcome(outcome["report"])
    if not scenario.fixable:
        truth = "needs a person"
    elif scenario.stale_report:
        truth = "nothing wrong"
    else:
        truth = "fixed"
    end_state_ok = healthy if scenario.fixable else (
        not healthy and len(restarts) <= 2)
    if scenario.stale_report:                    # nothing to fix: changing nothing is right
        end_state_ok = healthy and not any(entry["tool"] == "restart_server" and not entry["error"]
                                           for entry in tools)
    diagnosed = None
    if scenario.broken_server:
        shard, replica = scenario.broken_server
        diagnosed = any(entry["tool"] == "read_server_log" and not entry["error"]
                        and entry["arguments"].get("shard") == shard
                        and entry["arguments"].get("replica") == replica for entry in tools)
    return {
        "scenario": scenario.name, "error": error,
        "success": error is None and end_state_ok and claim == truth,
        "end_state_ok": end_state_ok, "healthy": healthy,
        "claim": claim, "expected_claim": truth, "claim_correct": claim == truth,
        "diagnosed": diagnosed,
        "turns": outcome.get("turns"), "finished": outcome.get("finished", True),
        "tool_calls": len(tools), "failed_calls": len(failed),
        "input_tokens": outcome.get("input_tokens", 0),
        "output_tokens": outcome.get("output_tokens", 0),
        "seconds": round(seconds, 2), "report": outcome["report"], "journal": journal,
    }


# --- many runs -----------------------------------------------------------------------------------

def wilson(successes, n, z=1.96):
    """95% confidence interval for a success rate: with few runs, k/n alone overstates
    how much is known."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(records):
    """One row per (setup, scenario), plus an overall row per setup."""
    rows = []
    setups = sorted({record["setup"] for record in records})
    for setup in setups:
        mine = [record for record in records if record["setup"] == setup and not record["error"]]
        groups = [(name, [r for r in mine if r["scenario"] == name]) for name in BY_NAME]
        groups = [(name, group) for name, group in groups if group] + [("all", mine)]
        for name, group in groups:
            k, n = sum(r["success"] for r in group), len(group)
            low, high = wilson(k, n)
            turns = [r["turns"] for r in group if r["turns"] is not None]
            diagnosed = [r["diagnosed"] for r in group if r["diagnosed"] is not None]
            rows.append({
                "setup": setup, "scenario": name, "runs": n, "successes": k,
                "rate": k / n if n else 0.0, "interval": (low, high),
                "claims_correct": sum(r["claim_correct"] for r in group),
                "diagnosed": f"{sum(diagnosed)}/{len(diagnosed)}" if diagnosed else "",
                "median_turns": statistics.median(turns) if turns else None,
                "mean_tokens": statistics.mean(r["input_tokens"] + r["output_tokens"] for r in group),
                "calls_per_run": statistics.mean(r["tool_calls"] for r in group),
                "failed_per_run": statistics.mean(r["failed_calls"] for r in group),
                "errors": sum(1 for r in records if r["setup"] == setup and r["error"]
                              and (name == "all" or r["scenario"] == name)),
            })
    return rows


def print_summary(rows):
    print(f"{'setup':<26}{'scenario':<18}{'success':>9}{'95% interval':>15}{'claims':>8}"
          f"{'diag':>6}{'turns':>7}{'calls':>7}{'failed':>8}{'tokens':>9}")
    print("-" * 113)
    for row in rows:
        low, high = row["interval"]
        turns = f"{row['median_turns']:g}" if row["median_turns"] is not None else "-"
        print(f"{row['setup']:<26}{row['scenario']:<18}{row['successes']:>4}/{row['runs']:<4}"
              f"{low:>8.0%}-{high:<5.0%}{row['claims_correct']:>4}/{row['runs']:<3}"
              f"{row['diagnosed']:>6}{turns:>7}{row['calls_per_run']:>7.1f}{row['failed_per_run']:>8.1f}"
              f"{row['mean_tokens']:>9,.0f}" + (f"  ({row['errors']} API errors)" if row["errors"] else ""))
        if row["scenario"] == "all":
            print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--brain", choices=["runbook", "openai"], default="runbook")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="careful")
    parser.add_argument("--model", default=OPENAI_MODEL)
    parser.add_argument("--runs", type=int, default=3, help="runs per scenario")
    parser.add_argument("--scenarios", default=",".join(BY_NAME),
                        help="comma-separated scenario names")
    parser.add_argument("--summary", nargs="*", type=Path,
                        help="only summarize these results files")
    args = parser.parse_args(argv)

    if args.summary is not None:
        records = [json.loads(line) for path in args.summary for line in path.read_text().splitlines()]
        print_summary(summarize(records))
        return 0

    if args.brain == "runbook":
        decide, setup = runbook_brain, "runbook"
    else:
        load_env_file()
        decide = model_brain(lambda: OpenAIModel(args.model), args.prompt)
        setup = f"{args.model}/{args.prompt}"
    scenarios = [BY_NAME[name] for name in args.scenarios.split(",")]
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}-{setup.replace('/', '-')}.jsonl"
    records = []
    print(f"{setup}: {len(scenarios)} scenarios x {args.runs} runs -> {out}")
    for run in range(args.runs):
        for scenario in scenarios:
            with tempfile.TemporaryDirectory() as workdir:
                record = {"setup": setup, "run": run, **run_one(scenario, decide, workdir)}
            records.append(record)
            with open(out, "a") as f:
                f.write(json.dumps(record) + "\n")
            mark = "error" if record["error"] else ("ok" if record["success"] else "FAIL")
            print(f"  run {run + 1} {scenario.name:<18} {mark:<5} turns={record['turns']} "
                  f"tokens={record['input_tokens'] + record['output_tokens']:,} "
                  f"claim={record['claim']}", flush=True)
    print()
    print_summary(summarize(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
