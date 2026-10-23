"""Heap-based top-K tracker.

Public API:
    TopKTracker — bounded min-heap of the largest N numeric observations seen so far.
"""

from heap_based_top_k_py.core import TopKTracker

__all__ = ["TopKTracker"]
