"""
Filters: restrict a search to the vectors whose payload matches some conditions.

A filter is a dict of field -> wanted value, or field -> list of wanted values:

    {"source": "guide.md"}                           source is guide.md
    {"source": ["guide.md", "notes.md"], "year": 2024}
                                                     source is either one AND year is 2024

Every field must match (AND); a list matches if any of its values does (OR). A payload
field that holds a list, such as {"tags": ["a", "b"]}, matches if any of its items does.

To answer "which vectors match?" without reading every payload, each shard keeps a
PayloadIndex: for every (field, value) the ids of the vectors that have it, like the index
at the back of a book. Only top-level fields holding a string, an integer, a boolean, or a
list of those are indexed.
"""
import json

import numpy as np

# Long strings are content (a passage of text), not something to filter on, and indexing
# them would keep a second copy of the text in memory.
MAX_INDEXED_CHARS = 128


def _key(field, value):
    """The index key for one field value, or None if such a value is not indexed.
    The type is part of the key because True == 1 in Python."""
    if isinstance(value, str):
        return (field, "str", value) if len(value) <= MAX_INDEXED_CHARS else None
    if isinstance(value, (bool, int)):
        return (field, type(value).__name__, value)
    return None


def normalize_filter(where):
    """Check a filter and return it as {field: [values]}, or None for no filter.
    Raises ValueError for a filter that could never be answered from the index."""
    if not where:
        return None
    if not isinstance(where, dict):
        raise ValueError("a filter must be a dict of field -> value or list of values")
    normalized = {}
    for field, wanted in where.items():
        values = list(wanted) if isinstance(wanted, (list, tuple)) else [wanted]
        if not isinstance(field, str) or not values:
            raise ValueError(f"filter field {field!r} needs a name and at least one value")
        for value in values:
            if _key(field, value) is None:
                raise ValueError(
                    f"cannot filter {field!r} on {value!r}: only strings of up to "
                    f"{MAX_INDEXED_CHARS} characters, integers, and booleans are indexed")
        normalized[field] = values
    return normalized


def encode_filter(where):
    """A filter as it is sent to a shard: JSON text, "" for no filter."""
    normalized = normalize_filter(where)
    return json.dumps(normalized) if normalized else ""


def decode_filter(text):
    return normalize_filter(json.loads(text)) if text else None


class PayloadIndex:
    """(field, value) -> local ids of the vectors whose payload has that value."""

    def __init__(self):
        self._ids = {}
        self.indexed = 0          # payloads read so far; local ids below this are indexed

    def add(self, local_id, payload_text):
        if not payload_text:
            return
        payload = json.loads(payload_text)
        if not isinstance(payload, dict):
            return
        for field, value in payload.items():
            for item in (value if isinstance(value, list) else [value]):
                key = _key(field, item)
                if key is not None:
                    self._ids.setdefault(key, []).append(local_id)

    def catch_up(self, payloads):
        """Index the payloads added since the last call. The index is built on demand, so
        loading a snapshot or replaying a log does not have to maintain it, and shards that
        are never filtered never parse their payloads."""
        for local_id in range(self.indexed, len(payloads)):
            self.add(local_id, payloads[local_id])
        self.indexed = len(payloads)

    def matching(self, where, size):
        """A boolean array of length `size`: True for every vector matching the filter
        (as returned by normalize_filter)."""
        allowed = np.ones(size, dtype=bool)
        for field, values in where.items():
            has_field = np.zeros(size, dtype=bool)
            for value in values:
                has_field[self._ids.get(_key(field, value), [])] = True
            allowed &= has_field
        return allowed
