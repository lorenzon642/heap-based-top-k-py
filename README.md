# heap-based-top-k-py

Maintains a bounded min-heap of the largest N numeric observations seen so far, with O(log N) insertion and O(N log N) extraction.

## Usage

```python
from heap_based_top_k_py import TopKTracker

t = TopKTracker(3)
t.push(5)
t.push(1)
t.push(9)
t.push(2)
t.push(7)

print(t.top())   # [9, 7, 5]
print(list(t))   # [9, 7, 5]
print(len(t))    # 3
print(t.seen)    # 5
print(t.n)       # 3
```

`push(value)` observes a single int or float. `push_all(values)` observes an iterable. `top()` returns a fresh list of the retained values, largest to smallest. `len(t)` is the number of values currently retained (at most N). `t.seen` is the total number of values observed. `t.n` is the capacity.

## Why this exists

The problem: given a long or unbounded stream of numeric observations, identify the largest N seen so far without storing the entire stream. A bounded min-heap of size N gives O(log N) per observation and O(N log N) extraction, with constant memory. This is the right shape when the stream is large, N is modest, and you only need the top values at the end.

The trade-off: extracting the result costs O(N log N) because `top()` sorts a copy of the retained heap. If you need the sorted top-K continuously (after every push), you would pay that cost on every query. This library is built for the case where you push many values and extract the top set occasionally or once.

## Edge case: values equal to the current minimum

When the heap is full and a new value equals the current minimum, the new value is **dropped**, not promoted. This is a deliberate policy choice. Promoting equal-valued entries would churn the heap without changing the set of retained values. Dropping ties keeps the retained set stable across duplicate-heavy inputs. If this behavior is unacceptable for your use case, filter or transform your input before pushing.

## Edge case: booleans

`True` and `False` are rejected by `push` with a `TypeError`, even though `bool` is a subclass of `int` in Python. Allowing booleans would invite silent numeric surprises (e.g., `True` treated as `1`). If you need to track booleans, convert them to integers explicitly first.
