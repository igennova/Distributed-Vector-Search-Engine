"""
Split a document into passages: small enough to embed, and precise enough to be a useful
search result.

A document is cut in two steps:
  1. into sections at its Markdown headings, so a passage never mixes two topics;
  2. each section into passages of at most `max_chars` characters, breaking only between
     sentences (or between lines, for lists, tables, and code).

Neighboring passages of a section share `overlap` characters of text, so a fact that sits
on a boundary is still whole in one of them.

The size limit matters because an embedding model reads a fixed number of tokens and
silently ignores the rest: text beyond the limit would be stored but never searchable.
"""
import re
from typing import NamedTuple

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_LINE_STRUCTURED = re.compile(r"^\s*([-*+]\s|\d+[.)]\s|\||>)")     # list item, table row, quote
# A sentence ends at . ! or ? followed by whitespace and something that starts a sentence.
# A wrong guess ("e.g. Foo") only moves a passage boundary; it never loses text.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`*_])")


class Chunk(NamedTuple):
    text: str
    heading: str      # the headings above the passage, outermost first: "Setup > Install"
    index: int        # position of the passage within its document


def _sections(lines, headings):
    """Yield (heading trail, body lines) for each stretch of text under one heading."""
    trail = []        # (level, title) of every heading enclosing the current line
    body = []
    fence = None      # the marker that opened the code block we are inside, if any
    for line in lines:
        marker = _FENCE.match(line)
        if marker and fence is None:
            fence = marker.group(1)
        elif marker and marker.group(1) == fence:
            fence = None
        heading = _HEADING.match(line) if headings and fence is None and not marker else None
        if heading is None:
            body.append(line)
            continue
        yield " > ".join(title for _, title in trail), body
        level = len(heading.group(1))
        trail = [(lvl, title) for lvl, title in trail if lvl < level]
        trail.append((level, heading.group(2)))
        body = []
    yield " > ".join(title for _, title in trail), body


def _blocks(lines):
    """Yield (is_prose, lines) for each paragraph, list, table, or fenced code block."""
    block, fence = [], None
    for line in lines:
        marker = _FENCE.match(line)
        if fence is not None:                      # inside code: keep every line, even blank
            block.append(line.rstrip())
            if marker and marker.group(1) == fence:
                yield False, block
                block, fence = [], None
        elif marker:
            if block:
                yield _is_prose(block), block
            block, fence = [line.rstrip()], marker.group(1)
        elif line.strip():
            block.append(line.rstrip())
        elif block:
            yield _is_prose(block), block
            block = []
    if block:
        yield fence is None and _is_prose(block), block


def _is_prose(block):
    return not any(_LINE_STRUCTURED.match(line) for line in block)


def _hard_split(text, max_chars):
    """Cut a piece that is too long on its own, at spaces where there are any."""
    while len(text) > max_chars:
        cut = text.rfind(" ", 1, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        yield text[:cut]
        text = text[cut:].lstrip(" ")
    if text:
        yield text


def _units(lines, max_chars):
    """Yield (separator, text) for each piece a passage may start or end at: a sentence of
    prose, or a line of a list, table, or code block. The separator is what joins the piece
    to the one before it."""
    for is_prose, block in _blocks(lines):
        if is_prose:
            pieces, joiner = _SENTENCE_END.split(" ".join(line.strip() for line in block)), " "
        else:
            pieces, joiner = block, "\n"
        separator = "\n\n"                         # a blank line between blocks
        for piece in pieces:
            for part in ([piece] if len(piece) <= max_chars else _hard_split(piece, max_chars)):
                yield separator, part
                separator = joiner


def _length(units):
    return sum(len(text) for _, text in units) + sum(len(sep) for sep, _ in units[1:])


def _join(units):
    return "".join(sep + text for sep, text in units)[len(units[0][0]):].strip("\n")


def _pack(units, max_chars, overlap):
    """Group units into passages of at most max_chars. Each new passage starts with the
    last units of the one before it (up to `overlap` characters)."""
    passage = []
    for unit in units:
        if passage and _length(passage + [unit]) > max_chars:
            yield _join(passage)
            tail = []
            for previous in reversed(passage):
                if _length([previous] + tail) > overlap:
                    break
                tail.insert(0, previous)
            while tail and _length(tail + [unit]) > max_chars:     # the new unit comes first
                tail.pop(0)
            passage = tail
        passage.append(unit)
    if passage:
        yield _join(passage)


def chunk_document(text, max_chars=1000, overlap=150, headings=True):
    """Split `text` into a list of Chunks. Pass headings=False for plain text, where a line
    starting with '#' is not a heading."""
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be at least 0 and smaller than max_chars")
    chunks = []
    for heading, body in _sections(text.splitlines(), headings):
        for passage in _pack(_units(body, max_chars), max_chars, overlap):
            if passage.strip():
                chunks.append(Chunk(passage, heading, len(chunks)))
    return chunks
