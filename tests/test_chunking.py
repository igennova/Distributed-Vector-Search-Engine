"""Chunking: passages respect the size limit, follow the document's structure, and between
them lose nothing."""
import re

import pytest

from vsearch.chunking import chunk_document

DOC = """\
# Guide

Intro sentence one. Intro sentence two.

## Install

Run the installer. Then restart the machine.

```bash
# not a heading
pip install thing

vsearch cluster up
```

## Usage

### Search

- first item
- second item

Ask a question. Read the answer.
"""


def _words(text):
    return re.findall(r"\w+", text)


def _long_prose(sentences=120):
    return " ".join(f"Sentence number {i} talks about topic{i} in some detail." for i in range(sentences))


def test_passages_follow_the_headings():
    chunks = chunk_document(DOC)
    assert [(c.heading, c.index) for c in chunks] == [
        ("Guide", 0), ("Guide > Install", 1), ("Guide > Usage > Search", 2)]
    assert chunks[0].text == "Intro sentence one. Intro sentence two."


def test_a_hash_inside_a_code_block_is_not_a_heading():
    install = chunk_document(DOC)[1]
    assert "# not a heading" in install.text
    # Code keeps its lines, including the blank one.
    assert "pip install thing\n\nvsearch cluster up" in install.text


def test_lists_keep_one_item_per_line():
    assert "- first item\n- second item" in chunk_document(DOC)[2].text


def test_plain_text_has_no_headings():
    chunks = chunk_document("# just a line\n\nSome text.", headings=False)
    assert [c.heading for c in chunks] == [""]
    assert "# just a line" in chunks[0].text


@pytest.mark.parametrize("max_chars, overlap", [(200, 0), (200, 60), (1000, 150), (80, 79)])
def test_no_passage_is_longer_than_the_limit_and_nothing_is_lost(max_chars, overlap):
    text = DOC + "\n## Long\n\n" + _long_prose() + "\n\n" + "x" * 2500 + "\n"
    chunks = chunk_document(text, max_chars=max_chars, overlap=overlap)

    assert max(len(c.text) for c in chunks) <= max_chars
    assert [c.index for c in chunks] == list(range(len(chunks)))

    body = "\n".join(line for line in text.splitlines() if not line.startswith("#") or "not a" in line)
    kept = "".join("".join(_words(c.text)) for c in chunks)
    position = 0
    for word in _words(body.replace("x" * 2500, "")):      # every word, in document order
        position = kept.index(word, position)
    assert sum(c.text.count("x") for c in chunks if set(c.text) == {"x"}) >= 2500


def test_passages_break_between_sentences():
    chunks = chunk_document(_long_prose(), max_chars=300, overlap=0)
    assert len(chunks) > 5
    for chunk in chunks:
        assert chunk.text.startswith("Sentence number") and chunk.text.endswith("detail.")


def test_neighboring_passages_overlap():
    chunks = chunk_document(_long_prose(), max_chars=300, overlap=80)
    for before, after in zip(chunks, chunks[1:]):
        last_sentence = before.text.split(". ")[-1]
        assert after.text.startswith(last_sentence)

    separate = chunk_document(_long_prose(), max_chars=300, overlap=0)
    for before, after in zip(separate, separate[1:]):
        assert not after.text.startswith(before.text.split(". ")[-1])


def test_overlap_stays_inside_a_section():
    text = "# One\n\n" + _long_prose(4) + "\n\n# Two\n\nSomething else entirely."
    chunks = chunk_document(text, max_chars=300, overlap=80)
    assert chunks[-1].heading == "Two" and chunks[-1].text == "Something else entirely."


def test_overlap_must_be_smaller_than_a_passage():
    with pytest.raises(ValueError):
        chunk_document("text", max_chars=100, overlap=100)
