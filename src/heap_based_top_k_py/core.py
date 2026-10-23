from __future__ import annotations

import heapq
from typing import Iterable, Iterator


class TopKTracker:
    """Track the largest N numeric observations seen so far.

    Backed by a bounded min-heap of capacity N. When a value arrives and the
    heap already has N entries, the smallest retained value is compared
    against the new one: if the new value is larger, it replaces the smallest;
    otherwise it is ignored. Values equal to the current minimum are ignored,
    which avoids churn and keeps the retained set stable.

    Guarantees:
      - Observation (push): amortized O(log N) time per element.
      - Extraction (top): O(N log N) for producing the result, since we sort a
        copy of the heap's internal list.
      - Memory: at most N values retained.
      - Determinism: top() returns a fresh list each call; iteration order is
        from largest to smallest; ties follow insertion order via Python's
        stable sort on values that share their key.
    """

    __slots__ = ("_heap", "_n", "_seen")

    def __init__(self, n: int) -> None:
        if not isinstance(n, int):
            raise TypeError(f"n must be an int, got {type(n).__name__}")
        if n <= 0:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        self._heap: list[float] = []
        self._n: int = n
        self._seen: int = 0

    def push(self, value: int | float) -> None:
        """Observe a single numeric value.

        Raises TypeError for non-numeric types. booleans are rejected to avoid
        silent promotion surprises (in Python, bool is a subclass of int).
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"value must be int or float (not bool), got {type(value).__name__}"
            )
        self._seen += 1
        heap = self._heap
        if len(heap) < self._n:
            heapq.heappush(heap, value)
            return
        # Bounded: only retain if strictly greater than the current minimum.
        # Ties are dropped to keep the retained set stable and avoid churn.
        if value > heap[0]:
            heapq.heapreplace(heap, value)

    def push_all(self, values: Iterable[int | float]) -> None:
        """Observe an iterable of numeric values.

        Pushes one at a time, which preserves identical behavior to repeated
        push() calls (including rejection of non-numeric types).
        """
        for v in values:
            self.push(v)

    def top(self) -> list[float]:
        """Return the currently retained values, largest to smallest.

        The returned list is a fresh copy; callers may mutate it freely.
        Empty when no values have been observed.
        """
        return sorted(self._heap, reverse=True)

    def __iter__(self) -> Iterator[float]:
        """Iterate retained values from largest to smallest.

        Returns a generator over a snapshot copy, so callers may iterate
        safely while mutating the heap concurrently.
        """
        return iter(self.top())

    def __len__(self) -> int:
        """Number of values currently retained (0 <= len <= N)."""
        return len(self._heap)

    def __repr__(self) -> str:
        return f"TopKTracker(n={self._n!r}, retained={len(self)!r}, seen={self._seen!r})"

    @property
    def n(self) -> int:
        """The bounded capacity (N)."""
        return self._n

    @property
    def seen(self) -> int:
        """Total number of values observed via push/push_all so far."""
        return self._seen
