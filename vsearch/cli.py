"""
Command-line interface for running and querying a local cluster.

    vsearch cluster up --shards 4 --replicas 2    start shard servers in the background
    vsearch load glove --limit 100000             index the most frequent GloVe words
    vsearch similar king                          nearest words, with their similarity
    vsearch ingest docs/ README.md                split documents into passages and index them
    vsearch search "how does recovery work?"      the passages closest to a question
    vsearch search "..." --where source=README.md   only passages whose fields match
    vsearch ingest notes/ --namespace alice       keep one user's documents apart, on one shard
    vsearch search "..." --namespace alice        search only that user's documents
    vsearch mcp                                   serve search/ingest/status to an MCP client
    vsearch memory remember "prefers short answers" --owner rahul     an agent's long-term memory
    vsearch memory recall "how should I answer?" --owner rahul
    vsearch ops check                             find what is wrong with the cluster and fix it
    vsearch ops check --brain openai              the same, with a model deciding what to do
    vsearch ops watch --interval 30               keep checking, and fix problems as they appear
    vsearch status                                state, size, and write position of every server
    vsearch cluster kill 0 1                      crash shard 0's replica 1
    vsearch cluster restart 0 1                   start it again (it recovers from disk)
    vsearch repair                                bring replicas that fell behind back in sync
    vsearch cluster down                          stop the servers (their data stays on disk)

Everything lives in the --home directory (default data/cluster): cluster.json with the
servers' ports and process ids, one data directory per server, and the servers' logs.
"""
import argparse
import json
import shutil
import signal
import sys
import textwrap
import time
from pathlib import Path

import grpc

from .client import ReplicasOutOfSyncError, ShardUnavailableError, free_ports, wait_until_ready
from .dataset import load_glove
from .documents import DEFAULT_EMBEDDER, describe, find_passages, holds, ingest_files
from .embedding import EMBEDDERS, EmbedderUnavailable
from . import __version__
from .agent import (CLAUDE_MODEL, OPENAI_MODEL, ClaudeModel, ModelError, ModelUnavailable,
                    OpenAIModel, run_agent)
from .mcp_server import build_memory_server, build_server
from .memory import MemoryStore, describe_age
from .ops import PROMPTS, Journal, Operations, claimed_outcome, run_runbook, task_for
from .processes import is_running, pick, spawn, stop, wait_for
from .state import (UserError, address, connect, load_env_file, read_state, refuse_if_managed,
                    server_status, servers, writable, write_state)

LOAD_BATCH = 10_000
SNIPPET_CHARS = 240


# --- commands ------------------------------------------------------------------------------

def cluster_up(args):
    home = args.home
    if (home / "cluster.json").exists():
        state = read_state(home)
        refuse_if_managed(state, "start")
        if any(is_running(server) for _, _, server in servers(state)):
            raise UserError("this cluster is already running; see: vsearch status")
        print(f"starting the existing cluster in {home} again (servers recover from disk)")
    else:
        home.mkdir(parents=True, exist_ok=True)
        ports = iter(free_ports(args.shards * args.replicas))
        state = {
            "seed": args.seed,
            "server_args": {"M": args.M, "ef_construction": args.ef_construction,
                            "ef_search": args.ef_search, "snapshot_every": args.snapshot_every},
            "servers": [[{"port": next(ports), "pid": None} for _ in range(args.replicas)]
                        for _ in range(args.shards)],
            "dataset": None,
        }
    for shard, replica, _ in servers(state):
        spawn(home, state, shard, replica)
    write_state(home, state)
    wait_for(servers(state), home)
    return status(args)


def cluster_attach(args):
    """Record servers that something else runs (Docker, another machine) as this cluster."""
    home = args.home
    if (home / "cluster.json").exists():
        raise UserError(f"{home} already has a cluster; use another --home, or "
                        "`vsearch cluster detach` first")
    groups = []
    for shard in args.shards:
        group = []
        for replica in shard.split(","):
            reach, _, peer = replica.strip().partition("=")
            host, colon, port = reach.rpartition(":")
            if not colon or not host or not port.isdigit() or (peer and ":" not in peer):
                raise UserError(f"expected HOST:PORT or HOST:PORT=PEER_HOST:PEER_PORT, "
                                f"got {replica!r}")
            server = {"host": host, "port": int(port), "pid": None}
            if peer:
                server["peer"] = peer
            group.append(server)
        groups.append(group)
    state = {"seed": None, "server_args": {}, "servers": groups, "dataset": None,
             "managed_by": args.managed_by}
    for shard, replica, server in servers(state):
        try:
            wait_until_ready(address(server), timeout=args.timeout)
        except grpc.FutureTimeoutError:
            raise UserError(f"shard {shard} replica {replica} does not answer at "
                            f"{address(server)}") from None
    home.mkdir(parents=True, exist_ok=True)
    write_state(home, state)
    print(f"attached {sum(map(len, groups))} servers in {len(groups)} shards, run by "
          f"{args.managed_by}")
    return status(args)


def cluster_detach(args):
    state = read_state(args.home)
    if not state.get("managed_by"):
        raise UserError("this cluster was started by vsearch; stop it with: vsearch cluster down")
    (args.home / "cluster.json").unlink()
    print(f"forgot the attached cluster in {args.home}; its servers and their data are untouched")
    return 0


def cluster_down(args):
    state = read_state(args.home)
    refuse_if_managed(state, "stop")
    stopped = stop([server for _, _, server in servers(state)], signal.SIGTERM)
    print(f"stopped {stopped} server(s)")
    if args.wipe:
        shutil.rmtree(args.home)
        print(f"deleted {args.home}")
    else:
        print("their data is still on disk; `vsearch cluster up` starts them again")
    return 0


def cluster_kill(args):
    state = read_state(args.home)
    refuse_if_managed(state, "kill")
    server = pick(state, args.shard, args.replica)
    if not stop([server], signal.SIGKILL):
        raise UserError(f"shard {args.shard} replica {args.replica} is not running")
    print(f"killed shard {args.shard} replica {args.replica} (SIGKILL, no clean shutdown)")
    return 0


def cluster_restart(args):
    state = read_state(args.home)
    refuse_if_managed(state, "restart")
    server = pick(state, args.shard, args.replica)
    if is_running(server):
        raise UserError(f"shard {args.shard} replica {args.replica} is already running")
    spawn(args.home, state, args.shard, args.replica)
    write_state(args.home, state)
    wait_for([(args.shard, args.replica, server)], args.home)
    print(f"shard {args.shard} replica {args.replica} is back on {address(server)}")
    return 0


def status(args):
    state = read_state(args.home)
    print(f"{'shard':>5} {'replica':>7}  {'address':<17}{'state':<6}{'vectors':>9}"
          f"{'last seq':>10}{'snapshot seq':>14}")
    for shard, replica, server in servers(state):
        reply = server_status(server)
        if reply is None:
            print(f"{shard:>5} {replica:>7}  {address(server):<17}{'down':<6}")
        else:
            print(f"{shard:>5} {replica:>7}  {address(server):<17}{'up':<6}{reply.size:>9,}"
                  f"{reply.last_seq:>10}{reply.snapshot_seq:>14}")
    if state.get("dataset"):
        print(describe(state["dataset"]))
    return 0


def load(args):
    state = read_state(args.home)
    replies = writable(state)
    if any(reply.size for reply in replies):
        raise UserError("this cluster already holds vectors; to start over: "
                       "vsearch cluster down --wipe")

    words, vectors = load_glove(args.glove_dir)
    limit = min(args.limit, len(words))
    started = time.perf_counter()
    try:
        with connect(state) as coordinator:
            for start in range(0, limit, LOAD_BATCH):
                coordinator.add(vectors[start:min(start + LOAD_BATCH, limit)])
                print(f"  indexed {min(start + LOAD_BATCH, limit):,} / {limit:,} words")
    except (grpc.RpcError, ReplicasOutOfSyncError) as err:
        raise UserError(f"load stopped: {err}. Check `vsearch status`, then `vsearch repair`") from None
    state["dataset"] = {"name": "glove", "limit": limit,
                        "dir": str(Path(args.glove_dir).resolve())}
    write_state(args.home, state)
    print(f"loaded {limit:,} words in {time.perf_counter() - started:.1f}s")
    return 0


def similar(args):
    state = read_state(args.home)
    dataset = state.get("dataset")
    if not dataset:
        raise UserError("nothing is loaded yet; run: vsearch load glove")
    if dataset["name"] != "glove":
        raise UserError(holds(dataset))
    words, vectors = load_glove(dataset["dir"])
    word = args.word.lower()                   # the GloVe 6B vocabulary is lowercase
    try:
        word_id = words.index(word)
    except ValueError:
        raise UserError(f"{word!r} is not in the GloVe vocabulary") from None

    started = time.perf_counter()
    try:
        with connect(state) as coordinator:
            # Ask for one extra: if the word itself is indexed, it comes back first.
            scored = coordinator.search_scored(vectors[word_id], args.k + 1)
            failovers = coordinator.failovers
    except ShardUnavailableError as err:
        raise UserError(f"{err}; see: vsearch status") from None
    elapsed_ms = (time.perf_counter() - started) * 1000

    neighbors = [(dist, global_id) for dist, global_id in scored if global_id != word_id][:args.k]
    print(f"words closest to {word!r}:")
    for rank, (dist, global_id) in enumerate(neighbors, start=1):
        print(f"{rank:>3}. {words[global_id]:<20} {1.0 - dist:.3f}")
    note = f", {failovers} failover(s)" if failovers else ""
    print(f"({elapsed_ms:.1f} ms across {len(state['servers'])} shards{note})")
    return 0


def ingest(args):
    state = read_state(args.home)
    replies = writable(state)
    started = time.perf_counter()
    with connect(state) as coordinator:
        report = ingest_files(
            args.home, state, coordinator, replies, args.paths, embedder_name=args.embedder,
            namespace=args.namespace, chunk_chars=args.chunk_chars, overlap=args.overlap,
            on_file=lambda path, outcome: print(f"  {path}: {outcome}"))
    note = f", {report.unchanged} unchanged file(s) skipped" if report.unchanged else ""
    into = ""
    if args.namespace:
        into = (f" into namespace {args.namespace!r} "
                f"(shard {report.shard} of {len(state['servers'])})")
    print(f"ingested {report.added:,} passages{into} in {time.perf_counter() - started:.1f}s{note}")
    return 0


def _parse_where(items):
    """--where FIELD=VALUE options as a filter. A value that reads as a JSON integer or
    boolean is taken as one (chunk=3, done=true); anything else is text. To match text
    that looks like a number, quote it: year='"2024"'."""
    where = {}
    for item in items or []:
        field, equals, raw = item.partition("=")
        if not field or not equals:
            raise UserError(f"--where takes FIELD=VALUE, got {item!r}")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        if not isinstance(value, (str, int)):      # a float, null, a list: keep the text
            value = raw
        where.setdefault(field, []).append(value)
    return where


def search(args):
    state = read_state(args.home)
    with connect(state) as coordinator:
        found = find_passages(state, coordinator, args.question, args.k,
                              where=_parse_where(args.where), namespace=args.namespace)
    hits = found.hits

    print(f"passages closest to {args.question!r}:")
    if not hits:
        print("  (nothing matches)")
    for rank, hit in enumerate(hits, start=1):
        passage = hit.payload or {}
        where = " > ".join(part for part in (passage.get("source"), passage.get("heading")) if part)
        print(f"{rank:>3}. {1.0 - hit.distance:.3f}  {where}")
        text = passage.get("text", "")
        if args.full:
            print(textwrap.indent(text, "     "))
        else:
            snippet = " ".join(text.split())
            if len(snippet) > SNIPPET_CHARS:
                snippet = snippet[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " ..."
            print(textwrap.fill(snippet, width=100, initial_indent="     ",
                                subsequent_indent="     "))
    note = f", {found.failovers} failover(s)" if found.failovers else ""
    shards = len(state["servers"])
    asked = f"on shard {found.shard} of {shards}" if args.namespace else f"across {shards} shards"
    print(f"(embed {found.embed_ms:.1f} ms, search {found.search_ms:.1f} ms {asked}{note})")
    return 0


def mcp(args):
    read_state(args.home)                      # fail now, not on the first tool call
    if args.memory:
        build_memory_server(args.home, args.owner, embedder=args.embedder).serve()
    else:
        build_server(args.home, read_only=args.read_only, embedder=args.embedder).serve()
    return 0


def _with_memory(args, use):
    store = MemoryStore(args.home, args.owner, embedder=getattr(args, "embedder", None))
    try:
        return use(store)
    finally:
        store.close()


def memory_remember(args):
    memory_id, stored = _with_memory(
        args, lambda store: store.remember(args.text, args.tag or (), args.replaces or ()))
    print(f"remembered as {memory_id}" if stored else f"already remembered as {memory_id}")
    return 0


def memory_recall(args):
    memories = _with_memory(args, lambda store: store.recall(args.query, args.k, args.tag))
    if not memories:
        print("nothing remembered matches")
    now = time.time()
    for memory in memories:
        tags = f"[{', '.join(memory.tags)}] " if memory.tags else ""
        print(f"  {memory.id}  {memory.similarity:.3f}  {describe_age(now - memory.created):<15} "
              f"{tags}{memory.text}")
    return 0


def memory_forget(args):
    _with_memory(args, lambda store: store.forget(args.memory_id))
    print(f"forgot {args.memory_id}: it is no longer recalled (its text stays on disk)")
    return 0


BRAINS = {"openai": OpenAIModel, "claude": ClaudeModel}     # the models that can decide


def _show_entry(entry):
    """Print one journal entry as it is written, so a run can be followed live."""
    if entry["kind"] == "tool":
        arguments = ", ".join(f"{name}={value}" for name, value in entry["arguments"].items())
        mark = "failed" if entry["error"] else "ok"
        print(f"  -> {entry['tool']}({arguments})  [{mark}]")
        for line in entry["result"].splitlines()[:6]:
            print(f"       {line}")
    elif entry["kind"] == "model" and entry["text"] and entry["calls"]:
        print(textwrap.indent(entry["text"], "  "))        # the model thinking out loud


def _check_once(args, journal):
    """One round: look, let the brain work if something is wrong, then verify for ourselves.
    Returns the problems that are left."""
    ops = Operations(args.home, journal, dry_run=args.dry_run)
    problems = ops.problems()
    if not problems:
        return []
    print(f"found {len(problems)} problem(s):")
    for problem in problems:
        print(f"  - {problem}")
    journal.write("check", problems=problems, brain=args.brain, dry_run=args.dry_run)

    if args.brain == "runbook":
        report = run_runbook(ops)
    else:
        model = BRAINS[args.brain](*([args.model] if args.model else []))
        run = run_agent(model, ops.tools(), task_for(problems), PROMPTS[args.prompt],
                        max_steps=args.max_steps, journal=journal)
        report = run.report if run.finished else (
            f"(stopped at the limit of {args.max_steps} steps before finishing)\n{run.report}")
        print(f"  [{model.model}: {run.steps} turns, {run.calls} tool calls, "
              f"{model.input_tokens:,} input + {model.output_tokens:,} output tokens]")
    print(f"\nreport:\n{textwrap.indent(report, '  ')}\n")

    # The report is a claim. Whether the cluster is healthy is checked here, independently.
    remaining = ops.problems()
    claim = claimed_outcome(report)
    journal.write("verdict", healthy=not remaining, problems=remaining, claimed=claim)
    if not args.dry_run and claim == "fixed" and remaining:
        print("note: the report says it is fixed, but the check below disagrees")
    if args.dry_run:
        print("dry run: nothing was changed")
    elif remaining:
        print("checked: still wrong:")
        for problem in remaining:
            print(f"  - {problem}")
    else:
        print("checked: the cluster is healthy")
    return remaining


def ops_check(args):
    read_state(args.home)
    journal = Journal(args.home / "ops-journal.jsonl", on_entry=_show_entry)
    if not Operations(args.home).problems():
        print("the cluster is healthy: nothing to do")
        return 0
    remaining = _check_once(args, journal)
    return 1 if remaining and not args.dry_run else 0


def ops_watch(args):
    read_state(args.home)
    journal = Journal(args.home / "ops-journal.jsonl", on_entry=_show_entry)
    print(f"watching the cluster every {args.interval:g} s (Ctrl-C to stop)")
    gave_up_on = None          # problems the brain already failed to fix
    rounds = 0
    try:
        while args.rounds is None or rounds < args.rounds:
            if rounds:
                time.sleep(args.interval)
            rounds += 1
            problems = Operations(args.home).problems()
            if not problems:
                gave_up_on = None
            elif problems == gave_up_on:
                # Trying the same thing again costs time (and, with a model, money) and
                # would not end differently. Wait for the situation to change.
                continue
            else:
                remaining = _check_once(args, journal)
                gave_up_on = remaining or None
                if remaining and not args.dry_run:
                    print("leaving these for a person; not retrying until something changes")
    except KeyboardInterrupt:
        print("stopped")
    return 0


def repair(args):
    state = read_state(args.home)
    with connect(state) as coordinator:
        synced = coordinator.repair()
    if not synced:
        print("all reachable replicas are in sync")
    for sync in synced:
        print(f"shard {sync.shard}: {sync.replica} caught up from {sync.source} via "
              f"{sync.method} ({sync.records_applied} log records), now at seq {sync.last_seq}")
    return 0


# --- argument parsing ----------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(prog="vsearch", description="Run and query a local cluster.")
    parser.add_argument("--version", action="version", version=f"vsearch {__version__}")
    parser.add_argument("--home", type=Path, default=Path("data/cluster"),
                        help="directory holding the cluster's state, data, and logs")
    commands = parser.add_subparsers(dest="command", required=True)

    cluster = commands.add_parser("cluster", help="start, stop, crash, or restart servers")
    actions = cluster.add_subparsers(dest="action", required=True)

    up = actions.add_parser("up", help="start the shard servers in the background")
    up.add_argument("--shards", type=int, default=4)
    up.add_argument("--replicas", type=int, default=2)
    up.add_argument("--seed", type=int, default=0)
    up.add_argument("--M", type=int, default=16)
    up.add_argument("--ef-construction", type=int, default=100)
    up.add_argument("--ef-search", type=int, default=100)
    up.add_argument("--snapshot-every", type=int, default=1000)
    up.set_defaults(run=cluster_up)

    down = actions.add_parser("down", help="stop the servers; their data stays on disk")
    down.add_argument("--wipe", action="store_true", help="also delete all of the cluster's data")
    down.set_defaults(run=cluster_down)

    attach = actions.add_parser(
        "attach", help="use servers that something else runs, such as Docker")
    attach.add_argument("shards", nargs="+", metavar="HOST:PORT[=PEER][,...]",
                        help="one argument per shard: its replicas' addresses, comma-separated. "
                             "Add =PEER_HOST:PEER_PORT when the servers reach each other by "
                             "another address than you reach them (Docker)")
    attach.add_argument("--managed-by", default="docker",
                        help="what runs the servers, shown in messages (default: docker)")
    attach.add_argument("--timeout", type=float, default=20,
                        help="seconds to wait for every server to answer")
    attach.set_defaults(run=cluster_attach)
    actions.add_parser("detach", help="forget an attached cluster (its servers keep running)") \
        .set_defaults(run=cluster_detach)

    for name, run, text in [("kill", cluster_kill, "crash one server with SIGKILL"),
                            ("restart", cluster_restart, "start a stopped server again")]:
        action = actions.add_parser(name, help=text)
        action.add_argument("shard", type=int)
        action.add_argument("replica", type=int)
        action.set_defaults(run=run)

    commands.add_parser("status", help="state, size, and write position of every server") \
        .set_defaults(run=status)
    commands.add_parser("repair", help="bring replicas that fell behind back in sync") \
        .set_defaults(run=repair)

    loader = commands.add_parser("load", help="index a dataset")
    loader.add_argument("dataset", choices=["glove"])
    loader.add_argument("--limit", type=int, default=100_000, help="how many words to index")
    loader.add_argument("--glove-dir", default="data", help="where glove.6B.zip or its cache is")
    loader.set_defaults(run=load)

    finder = commands.add_parser("similar", help="words closest in meaning to a word")
    finder.add_argument("word")
    finder.add_argument("-k", type=int, default=10)
    finder.set_defaults(run=similar)

    ingester = commands.add_parser("ingest", help="split documents into passages and index them")
    ingester.add_argument("paths", nargs="+", type=Path, metavar="PATH",
                          help="files, or directories to search for .md/.txt/.rst files")
    ingester.add_argument("--embedder", choices=sorted(EMBEDDERS),
                          help=f"how text becomes vectors (default: {DEFAULT_EMBEDDER})")
    ingester.add_argument("--chunk-chars", type=int, default=1000,
                          help="longest passage, in characters")
    ingester.add_argument("--overlap", type=int, default=150,
                          help="characters shared by neighboring passages")
    ingester.add_argument("--namespace", help="keep these documents apart under this name; "
                          "all of a namespace is stored on one shard")
    ingester.set_defaults(run=ingest)

    searcher = commands.add_parser("search", help="passages closest in meaning to a question")
    searcher.add_argument("question")
    searcher.add_argument("-k", type=int, default=5)
    searcher.add_argument("--full", action="store_true", help="print whole passages")
    searcher.add_argument("--namespace", help="search only inside this namespace")
    searcher.add_argument("--where", action="append", metavar="FIELD=VALUE",
                          help="only passages whose field has this value, e.g. "
                               "source=README.md; repeat a field to accept several values")
    searcher.set_defaults(run=search)

    ops = commands.add_parser("ops", help="find and fix problems in the cluster")
    ops_actions = ops.add_subparsers(dest="action", required=True)
    for name, run, text in [("check", ops_check, "look once, fix what is wrong, verify"),
                            ("watch", ops_watch, "keep checking, and fix problems as they appear")]:
        action = ops_actions.add_parser(name, help=text)
        action.add_argument("--brain", choices=["runbook", *BRAINS], default="runbook",
                            help="what decides the steps: a fixed procedure, or a model choosing "
                                 "its own (openai needs OPENAI_API_KEY; claude needs the "
                                 "anthropic package and ANTHROPIC_API_KEY)")
        action.add_argument("--model", help=f"model id (default: {OPENAI_MODEL} for openai, "
                                            f"{CLAUDE_MODEL} for claude)")
        action.add_argument("--prompt", choices=sorted(PROMPTS), default="careful",
                            help="working rules given to the model: check before and after every "
                                 "change (careful), or act at once and verify at the end (lean)")
        action.add_argument("--max-steps", type=int, default=12,
                            help="most turns the model may take in one round")
        action.add_argument("--dry-run", action="store_true",
                            help="describe the changes instead of making them")
        action.set_defaults(run=run)
        if name == "watch":
            action.add_argument("--interval", type=float, default=30.0, help="seconds between checks")
            action.add_argument("--rounds", type=int, help="stop after this many checks")

    server = commands.add_parser(
        "mcp", help="serve this cluster to an MCP client over stdin/stdout")
    server.add_argument("--memory", action="store_true",
                        help="serve remember/recall/forget for one owner instead of documents")
    server.add_argument("--owner", default="default", help="whose memory, with --memory")
    server.add_argument("--read-only", action="store_true",
                        help="offer search and status only, not ingest")
    server.add_argument("--embedder", choices=sorted(EMBEDDERS),
                        help=f"embedder for a collection created through the ingest tool "
                             f"(default: {DEFAULT_EMBEDDER})")
    server.set_defaults(run=mcp)

    memory = commands.add_parser("memory", help="an agent's long-term memory, in its own cluster")
    memory_actions = memory.add_subparsers(dest="action", required=True)
    remember = memory_actions.add_parser("remember", help="store a fact")
    remember.add_argument("text")
    remember.add_argument("--tag", action="append", help="a label to recall by; repeatable")
    remember.add_argument("--replaces", action="append", metavar="ID",
                          help="id of a fact this one makes out of date; repeatable")
    remember.add_argument("--embedder", choices=sorted(EMBEDDERS),
                          help=f"for the first memory in a cluster (default: {DEFAULT_EMBEDDER})")
    remember.set_defaults(run=memory_remember)
    recall = memory_actions.add_parser("recall", help="the facts most relevant to a query")
    recall.add_argument("query")
    recall.add_argument("-k", type=int, default=5)
    recall.add_argument("--tag", action="append", help="only facts with one of these labels")
    recall.set_defaults(run=memory_recall)
    forget = memory_actions.add_parser("forget", help="stop recalling a fact")
    forget.add_argument("memory_id")
    forget.set_defaults(run=memory_forget)
    for action in (remember, recall, forget):
        action.add_argument("--owner", default="default", help="whose memory")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.home = args.home.resolve()
    load_env_file()                            # API keys kept in ./.env, out of the shell history
    try:
        return args.run(args)
    except UserError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except (ModelUnavailable, ModelError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except EmbedderUnavailable as err:
        print(f"error: {err} (or pick the built-in one: vsearch ingest --embedder hash)",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
