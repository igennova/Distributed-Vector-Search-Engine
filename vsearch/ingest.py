"""
Turn documents into searchable passages:

    file --read--> text --chunk--> passages --embed--> vectors --add--> shards

Each vector is stored with its passage as the payload, so a search returns the text that
answers the question, not just an id:

    {"text": "...", "source": "docs/guide.md", "heading": "Setup > Install", "chunk": 3}
"""
import hashlib
from pathlib import Path

import numpy as np

from .chunking import chunk_document

DOCUMENT_SUFFIXES = (".md", ".markdown", ".txt", ".rst")
MARKDOWN_SUFFIXES = (".md", ".markdown")
EMBED_BATCH = 32


def find_documents(paths):
    """The text files named by `paths` or found under them, as sorted absolute paths with
    no repeats. Directories are searched recursively, skipping hidden ones (.git, .venv)."""
    found = set()
    for path in map(Path, paths):
        if path.is_dir():
            found.update(
                file.resolve() for file in path.rglob("*")
                if file.is_file() and file.suffix.lower() in DOCUMENT_SUFFIXES
                and not any(part.startswith(".") for part in file.relative_to(path).parts))
        elif path.is_file():
            found.add(path.resolve())              # a file named explicitly is always taken
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    return sorted(found)


def fingerprint(path):
    """A hash of the file's contents, to tell whether it changed since it was ingested."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_passages(path, source=None, max_chars=1000, overlap=150):
    """Read one file and return a payload for each of its passages."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_document(text, max_chars=max_chars, overlap=overlap,
                            headings=path.suffix.lower() in MARKDOWN_SUFFIXES)
    return [{"text": chunk.text, "source": source or str(path), "heading": chunk.heading,
             "chunk": chunk.index} for chunk in chunks]


def embedding_text(passage):
    """The text that is embedded for a passage: its headings, then its text. A passage
    often never repeats the name of the thing it is about; its headings do."""
    if passage["heading"]:
        return f"{passage['heading']}\n\n{passage['text']}"
    return passage["text"]


def ingest_passages(coordinator, embedder, passages, batch=EMBED_BATCH):
    """Embed the passages and store each vector with its passage as the payload."""
    if not passages:
        return 0
    texts = [embedding_text(passage) for passage in passages]
    vectors = np.concatenate([embedder.embed_documents(texts[start:start + batch])
                              for start in range(0, len(texts), batch)])
    coordinator.add(vectors, passages)
    return len(passages)


def search_passages(coordinator, embedder, question, k=5):
    """The k passages closest to the question, as Hits whose payload is the passage."""
    return coordinator.search_hits(embedder.embed_query(question), k)
