"""
A document collection on a local cluster: ingest files as passages and search them, keeping
track in cluster.json of which embedder built the collection and which files are in it.
Shared by the command line and the MCP server.
"""
import time
from pathlib import Path
from typing import NamedTuple

import grpc

from .client import ReplicasOutOfSyncError, ShardUnavailableError
from .cluster import shard_for_namespace
from .embedding import get_embedder
from .ingest import find_documents, fingerprint, ingest_passages, read_passages
from .state import UserError, write_state

DEFAULT_EMBEDDER = "bge-small"


class IngestReport(NamedTuple):
    added: int            # passages stored
    unchanged: int        # files skipped because they are already in, with the same contents
    changed: int          # files skipped because they are in with different contents
    shard: int | None     # the shard holding the namespace, if one was given


class PassageSearch(NamedTuple):
    hits: list            # Hit(id, distance, payload), closest first
    embed_ms: float
    search_ms: float
    shard: int | None     # the one shard asked, for a search inside a namespace
    failovers: int


def display_path(path):
    """A path as the user would write it: relative to where they are, if it is below that."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def describe(dataset):
    """One line saying what a cluster holds."""
    if not dataset:
        return "nothing is loaded yet"
    if dataset["name"] != "documents":
        return f"loaded: the {dataset['limit']:,} most frequent {dataset['name']} words"
    spaces = namespaces(dataset)
    inside = f" in {len(spaces)} namespace(s)" if spaces else ""
    return (f"ingested: {dataset['passages']:,} passages from {len(dataset['sources'])} "
            f"file(s){inside}, embedder: {dataset['embedder']}")


def namespaces(dataset):
    """The namespaces documents have been ingested into, sorted."""
    return sorted({source.get("namespace") for source in dataset["sources"].values()} - {None})


def ingest_files(home, state, coordinator, replies, paths, embedder_name=None, namespace=None,
                 chunk_chars=1000, overlap=150, embedders=get_embedder, on_file=None):
    """Split the files under `paths` into passages and index them. `replies` are the servers'
    statuses (from state.writable); `embedders` turns an embedder's name into the embedder;
    `on_file(path, outcome)` is called for each file that is stored or cannot be."""
    dataset = state.get("dataset")
    if dataset is None:
        if any(reply.size for reply in replies):
            raise UserError("this cluster already holds vectors; to start over: "
                            "vsearch cluster down --wipe")
        embedder = embedders(embedder_name or DEFAULT_EMBEDDER)
        dataset = {"name": "documents", "embedder": embedder.name, "dim": embedder.dim,
                   "passages": 0, "sources": {}}
    elif dataset["name"] != "documents":
        raise UserError("this cluster holds GloVe words, and vectors from two embedders cannot "
                        "share an index; start another cluster with --home")
    elif embedder_name not in (None, dataset["embedder"]):
        raise UserError(f"this cluster was built with the {dataset['embedder']!r} embedder; "
                        "every passage in it has to use the same one")
    else:
        embedder = embedders(dataset["embedder"])

    try:
        files = find_documents(paths)
    except FileNotFoundError as err:
        raise UserError(str(err)) from None
    if not files:
        raise UserError("found no .md, .markdown, .txt, or .rst files there")

    added = unchanged = changed = 0
    try:
        for path in files:
            shown, digest = display_path(path), fingerprint(path)
            # The same file can be ingested into several namespaces, once each.
            key = f"{namespace}::{path}" if namespace else str(path)
            known = dataset["sources"].get(key)
            if known and known["sha256"] == digest:
                unchanged += 1
                continue
            if known:
                # The index has no delete yet, so the old passages cannot be replaced.
                changed += 1
                if on_file:
                    on_file(shown, "changed since it was ingested, skipped "
                                   "(to re-index: vsearch cluster down --wipe)")
                continue
            passages = read_passages(path, source=shown, max_chars=chunk_chars, overlap=overlap)
            count = ingest_passages(coordinator, embedder, passages, namespace=namespace)
            dataset["sources"][key] = {"sha256": digest, "passages": count,
                                       "namespace": namespace, "source": shown}
            dataset["passages"] += count
            added += count
            state["dataset"] = dataset
            write_state(home, state)           # after every file: a crash loses one at most
            if on_file:
                on_file(shown, f"{count} passages")
    except (grpc.RpcError, ReplicasOutOfSyncError) as err:
        raise UserError(f"ingest stopped: {err}. Check `vsearch status`, then "
                        "`vsearch repair`") from None
    except ValueError as err:                  # bad chunk size, overlap, or namespace
        raise UserError(str(err)) from None
    shard = shard_for_namespace(namespace, len(state["servers"])) if namespace else None
    return IngestReport(added, unchanged, changed, shard)


def find_passages(state, coordinator, question, k=5, where=None, namespace=None,
                  embedders=get_embedder):
    """The k passages closest to the question, embedded with the embedder the collection was
    built with. `where` filters on passage fields; `namespace` searches inside one."""
    dataset = state.get("dataset")
    if not dataset:
        raise UserError("nothing is ingested yet; run: vsearch ingest PATH")
    if dataset["name"] != "documents":
        raise UserError("this cluster holds GloVe words; use: vsearch similar WORD")

    embedder = embedders(dataset["embedder"])
    started = time.perf_counter()
    query = embedder.embed_query(question)
    embedded = time.perf_counter()
    failovers = coordinator.failovers
    try:
        hits = coordinator.search_hits(query, k, where=where, namespace=namespace)
    except ShardUnavailableError as err:
        raise UserError(f"{err}; see: vsearch status") from None
    except ValueError as err:                  # a filter the index cannot answer
        raise UserError(str(err)) from None
    finished = time.perf_counter()
    shard = shard_for_namespace(namespace, len(state["servers"])) if namespace else None
    return PassageSearch(hits, (embedded - started) * 1000, (finished - embedded) * 1000, shard,
                         coordinator.failovers - failovers)
