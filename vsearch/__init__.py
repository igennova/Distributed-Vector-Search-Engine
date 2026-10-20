"""A distributed approximate-nearest-neighbor vector search engine."""
__version__ = "0.1.0"           # keep equal to the version in pyproject.toml (a test checks)

from .brute_force import brute_force_search
from .hnsw import HNSW
from .cluster import Coordinator, Shard

__all__ = ["brute_force_search", "HNSW", "Coordinator", "Shard"]
