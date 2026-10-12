"""
The engine as MCP tools, so an AI application can use it directly:

    search   find the passages that answer a question
    ingest   add documents to the index
    status   which servers are up, and what is indexed

Run it with `vsearch --home DIR mcp`; the client starts that command and talks to it over
stdin and stdout (see mcp.py for the protocol). The cluster itself is started separately
with `vsearch cluster up`.

Unlike the command line, where every call is a new process, this server lives as long as
the client's session. So the embedding model is loaded once instead of on every search,
and the coordinator keeps its connections and its memory of which replica failed last.
"""
from pathlib import Path

from .documents import describe, find_passages, ingest_files, namespaces
from .embedding import EmbedderUnavailable, get_embedder
from .mcp import McpServer, Tool, ToolError
from .state import UserError, address, connect, read_state, server_status, shard_addresses, writable

MAX_RESULTS = 20
MAX_LISTED = 50

INSTRUCTIONS = (
    "This server searches a local index of documents by meaning. Call `status` to see what "
    "is indexed, and `search` to find the passages relevant to a question. Each passage "
    "comes with the file and section it was taken from; mention them when you use it.")

SEARCH = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description":
                     "What to look for, written as a full question or sentence. The search "
                     "matches meaning, not keywords."},
        "k": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS, "description":
              "How many passages to return. Default 5."},
        "namespace": {"type": "string", "description":
                      "Search only the documents ingested into this namespace. Omit to "
                      "search everything."},
        "source": {"type": "string", "description":
                   "Search only passages from this file, named as `status` and earlier "
                   "results show it, for example docs/guide.md."},
    },
    "required": ["question"],
}

INGEST = {
    "type": "object",
    "properties": {
        "paths": {"type": "array", "items": {"type": "string"}, "description":
                  "Files or directories to add. Directories are searched for .md, "
                  ".markdown, .txt, and .rst files. Use absolute paths."},
        "namespace": {"type": "string", "description":
                      "Keep these documents apart under this name, for one user or "
                      "project. They can then be searched on their own."},
    },
    "required": ["paths"],
}

NO_ARGUMENTS = {"type": "object", "properties": {}}


class Engine:
    """What the tools share for the life of the server: the embedding model and the
    connections to the shards."""

    def __init__(self, home, embedder=None):
        self.home = Path(home)
        self.default_embedder = embedder      # for a collection this server creates
        self._embedders = {}
        self._coordinator = None
        self._addresses = None

    def _embedder(self, name):
        if name not in self._embedders:
            self._embedders[name] = get_embedder(name)     # loads its model on first use
        return self._embedders[name]

    def _connect(self, state):
        """The coordinator for the cluster, kept between calls. A new one is made only if
        the cluster's servers changed (it was wiped and started again)."""
        addresses = shard_addresses(state)
        if addresses != self._addresses:
            self.close()
            self._coordinator, self._addresses = connect(state), addresses
        return self._coordinator

    def close(self):
        if self._coordinator is not None:
            self._coordinator.close()
            self._coordinator = None

    # --- tools -----------------------------------------------------------------------------

    def search(self, question, k=5, namespace=None, source=None):
        state = self._state()
        if not state.get("dataset"):
            raise ToolError("Nothing is indexed yet. Add documents with the ingest tool first.")
        found = self._run(find_passages, state, self._connect(state), question, k,
                          where={"source": source} if source else None, namespace=namespace,
                          embedders=self._embedder)
        shards = len(state["servers"])
        asked = f"shard {found.shard} of {shards}" if namespace else f"all {shards} shards"
        if not found.hits:
            return (f"No passages matched (asked {asked}). If a namespace or source was given, "
                    "check its spelling with the status tool.")
        lines = [f"{len(found.hits)} passages for {question!r} (asked {asked}), closest first:"]
        for rank, hit in enumerate(found.hits, start=1):
            passage = hit.payload or {}
            section = f" | section: {passage['heading']}" if passage.get("heading") else ""
            lines.append(f"\n[{rank}] source: {passage.get('source', 'unknown')}{section} | "
                         f"similarity: {1.0 - hit.distance:.2f}\n{passage.get('text', '')}")
        return "\n".join(lines)

    def ingest(self, paths, namespace=None):
        state = self._state()
        replies = self._run(writable, state)
        coordinator = self._connect(state)
        # Another process (the command line) may have written since this coordinator last
        # did, so it has to ask the shards where they are before writing.
        coordinator.refresh()
        stored = []
        report = self._run(ingest_files, self.home, state, coordinator, replies,
                           [Path(path) for path in paths], embedder_name=self.default_embedder,
                           namespace=namespace, embedders=self._embedder,
                           on_file=lambda path, outcome: stored.append(f"  {path}: {outcome}"))
        into = f" into namespace {namespace!r}" if namespace else ""
        lines = [f"Stored {report.added} passages{into}."]
        if report.unchanged:
            lines.append(f"{report.unchanged} file(s) were already indexed and unchanged: skipped.")
        return "\n".join(lines + stored[:MAX_LISTED])

    def status(self):
        state = self._state()
        lines, up, total = [], 0, 0
        for shard, group in enumerate(state["servers"]):
            replies = [server_status(server) for server in group]
            parts = []
            for replica, (server, reply) in enumerate(zip(group, replies)):
                total += 1
                if reply is None:
                    parts.append(f"replica {replica} DOWN ({address(server)})")
                else:
                    up += 1
                    parts.append(f"replica {replica} up ({reply.size:,} vectors, seq {reply.last_seq})")
            seqs = {reply.last_seq for reply in replies if reply is not None}
            note = "  <- replicas disagree; run `vsearch repair`" if len(seqs) > 1 else ""
            lines.append(f"shard {shard}: " + ", ".join(parts) + note)
        summary = [f"{up} of {total} servers up ({len(state['servers'])} shards)."]
        dataset = state.get("dataset")
        summary.append(describe(dataset).capitalize() + ".")
        if dataset and dataset["name"] == "documents":
            if namespaces(dataset):
                summary.append("Namespaces: " + ", ".join(namespaces(dataset)))
            files = sorted({(source.get("namespace") or "", source.get("source", key))
                            for key, source in dataset["sources"].items()})
            listed = [f"{name} (namespace {space})" if space else name
                      for space, name in files[:MAX_LISTED]]
            more = f" and {len(files) - MAX_LISTED} more" if len(files) > MAX_LISTED else ""
            summary.append("Files: " + ", ".join(listed) + more)
        return "\n".join(summary + lines)

    # --- helpers ---------------------------------------------------------------------------

    def _state(self):
        return self._run(read_state, self.home)

    @staticmethod
    def _run(function, *args, **kwargs):
        """Call into the engine, turning the problems a user could fix into tool errors."""
        try:
            return function(*args, **kwargs)
        except (UserError, EmbedderUnavailable) as err:
            raise ToolError(str(err)) from None


def build_server(home, read_only=False, embedder=None):
    """The MCP server for the cluster in `home`. With read_only, it cannot add documents."""
    engine = Engine(home, embedder=embedder)
    tools = [
        Tool("search",
             "Search the indexed documents and return the passages most relevant to a "
             "question, each with the file and section it came from. Use this to answer "
             "from the documents rather than from memory.",
             SEARCH, engine.search),
        Tool("status",
             "Report which shard servers are up, how many vectors each holds, and what is "
             "indexed: passages, files, and namespaces. Call it to see what can be searched, "
             "or when another tool fails.",
             NO_ARGUMENTS, engine.status),
    ]
    if not read_only:
        tools.append(Tool(
            "ingest",
            "Add documents to the index so they can be searched. Each file is split into "
            "passages and embedded. Files already indexed with the same contents are "
            "skipped, so calling it again is safe.",
            INGEST, engine.ingest))
    return McpServer("vsearch", "0.1.0", tools, instructions=INSTRUCTIONS,
                     on_close=engine.close)
