import unittest

from heap_based_top_k_py import TopKTracker


class TestTopKTrackerBasics(unittest.TestCase):
    def test_empty_returns_empty_list(self):
        t = TopKTracker(3)
        self.assertEqual(t.top(), [])
        self.assertEqual(len(t), 0)

    def test_n_property(self):
        t = TopKTracker(5)
        self.assertEqual(t.n, 5)

    def test_repr_includes_state(self):
        t = TopKTracker(3)
        t.push(1)
        r = repr(t)
        self.assertIn("TopKTracker", r)
        self.assertIn("n=3", r)
        self.assertIn("retained=1", r)
        self.assertIn("seen=1", r)


class TestTopKTrackerConstruction(unittest.TestCase):
    def test_zero_n_rejected(self):
        with self.assertRaises(ValueError):
            TopKTracker(0)

    def test_negative_n_rejected(self):
        with self.assertRaises(ValueError):
            TopKTracker(-1)

    def test_non_int_n_rejected(self):
        with self.assertRaises(TypeError):
            TopKTracker(2.5)  # type: ignore[arg-type]


class TestTopKTrackerPush(unittest.TestCase):
    def test_push_single(self):
        t = TopKTracker(3)
        t.push(5)
        self.assertEqual(t.top(), [5])

    def test_push_below_capacity_keeps_all(self):
        t = TopKTracker(5)
        t.push_all([3, 1, 4, 1, 5])
        self.assertEqual(t.top(), [5, 4, 3, 1, 1])

    def test_push_above_capacity_keeps_top_n(self):
        t = TopKTracker(3)
        t.push_all([5, 3, 8, 1, 9, 2, 7])
        self.assertEqual(t.top(), [9, 8, 7])

    def test_push_includes_negatives(self):
        t = TopKTracker(3)
        t.push_all([-5, -1, -10, -3, -2])
        self.assertEqual(t.top(), [-1, -2, -3])

    def test_push_all_empty(self):
        t = TopKTracker(3)
        t.push_all([])
        self.assertEqual(t.top(), [])
        self.assertEqual(t.seen, 0)

    def test_seen_count_independent_of_capacity(self):
        t = TopKTracker(2)
        t.push_all([10, 20, 30, 40, 50])
        self.assertEqual(t.seen, 5)
        self.assertEqual(len(t), 2)

    def test_seen_count_includes_dropped_values(self):
        t = TopKTracker(3)
        t.push_all([5, 1, 2, 1, 3])
        self.assertEqual(t.seen, 5)

    def test_seen_with_manual_pushes(self):
        t = TopKTracker(2)
        t.push(5)
        t.push(1)
        t.push(9)
        t.push(3)
        self.assertEqual(t.seen, 4)

    def test_push_mixed_int_and_float(self):
        t = TopKTracker(3)
        t.push_all([1, 2.5, 1, 3.0, 2])
        # By current strict-greater-than policy, the second 2.5 would be
        # retained if it is strictly greater than the heap's minimum. The
        # retained set is the 3 largest values seen: 3.0, 2.5, 2.
        self.assertEqual(t.top(), [3.0, 2.5, 2])

    def test_push_bool_rejected(self):
        t = TopKTracker(3)
        with self.assertRaises(TypeError):
            t.push(True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            t.push(False)  # type: ignore[arg-type]
        self.assertEqual(t.seen, 0)

    def test_push_non_numeric_rejected(self):
        t = TopKTracker(3)
        with self.assertRaises(TypeError):
            t.push("5")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            t.push(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            t.push([1, 2])  # type: ignore[arg-type]
        self.assertEqual(t.seen, 0)

    def test_push_all_rejects_non_numeric(self):
        t = TopKTracker(3)
        with self.assertRaises(TypeError):
            t.push_all([1, 2, "3"])  # type: ignore[list-item]
        # First two were accepted before the TypeError.
        self.assertEqual(t.seen, 2)
        self.assertEqual(t.top(), [2, 1])


class TestTopKTrackerTopExtraction(unittest.TestCase):
    def test_top_returns_copy(self):
        t = TopKTracker(3)
        t.push_all([1, 2, 3])
        a = t.top()
        a.append(999)
        b = t.top()
        self.assertEqual(b, [3, 2, 1])

    def test_top_idempotent(self):
        t = TopKTracker(3)
        t.push_all([4, 1, 7, 2, 9])
        self.assertEqual(t.top(), [9, 7, 4])
        self.assertEqual(t.top(), [9, 7, 4])
        self.assertEqual(t.top(), [9, 7, 4])


class TestTopKTrackerIteration(unittest.TestCase):
    def test_iter_descending(self):
        t = TopKTracker(3)
        t.push_all([4, 1, 7, 2, 9])
        self.assertEqual(list(t), [9, 7, 4])

    def test_iter_snapshot_is_independent_of_mutation(self):
        t = TopKTracker(3)
        t.push_all([1, 2, 3])
        it = iter(t)
        t.push(4)
        # Iterator reflects the snapshot taken at iter() time, not later pushes.
        self.assertEqual(list(it), [3, 2, 1])

    def test_iter_empty(self):
        t = TopKTracker(3)
        self.assertEqual(list(t), [])


class TestTopKTrackerEdgeCases(unittest.TestCase):
    def test_n_equals_one_keeps_max(self):
        t = TopKTracker(1)
        t.push_all([1, 5, 2, 8, 3])
        self.assertEqual(t.top(), [8])
        self.assertEqual(len(t), 1)

    def test_duplicate_values_above_capacity(self):
        t = TopKTracker(3)
        t.push_all([5, 5, 5, 5, 5])
        self.assertEqual(t.top(), [5, 5, 5])
        self.assertEqual(t.seen, 5)

    def test_equal_to_minimum_is_dropped(self):
        # Policy decision: values equal to the current minimum are NOT promoted
        # in. This keeps the retained set stable and avoids churn. See README.
        t = TopKTracker(3)
        t.push_all([1, 2, 3])  # heap has [1, 2, 3]; min is 1
        t.push(1)  # equal to min -> ignored
        t.push(1)  # equal to min -> ignored
        self.assertEqual(t.top(), [3, 2, 1])
        self.assertEqual(t.seen, 5)

    def test_strictly_greater_replaces_minimum(self):
        t = TopKTracker(3)
        t.push_all([1, 2, 3])  # min is 1
        t.push(2)  # strictly greater than 1 -> promotes
        # After promotion, heap has [2, 2, 3]; the 1 is evicted.
        self.assertEqual(sorted(t.top()), [2, 2, 3])
        self.assertEqual(t.top(), [3, 2, 2])

    def test_push_increasing_sequence(self):
        t = TopKTracker(3)
        for i in range(10):
            t.push(i)
        self.assertEqual(t.top(), [9, 8, 7])

    def test_push_decreasing_sequence(self):
        t = TopKTracker(3)
        for i in range(9, -1, -1):
            t.push(i)
        self.assertEqual(t.top(), [9, 8, 7])

    def test_push_interleaved_pattern(self):
        t = TopKTracker(4)
        t.push_all([10, 1, 9, 2, 8, 3, 7, 4, 6, 5])
        self.assertEqual(t.top(), [10, 9, 8, 7])

    def test_large_input(self):
        t = TopKTracker(5)
        t.push_all(range(1000))
        self.assertEqual(t.top(), [999, 998, 997, 996, 995])
        self.assertEqual(t.seen, 1000)

    def test_large_input_shuffled_indices(self):
        t = TopKTracker(3)
        t.push_all([0, 999, 1, 998, 2, 997, 3, 996, 4, 995])
        self.assertEqual(t.top(), [999, 998, 997])

    def test_single_push_then_top(self):
        t = TopKTracker(10)
        t.push(42)
        self.assertEqual(t.top(), [42])
        self.assertEqual(len(t), 1)
        self.assertEqual(t.seen, 1)

    def test_negative_and_positive_mixed(self):
        t = TopKTracker(4)
        t.push_all([-100, 50, -50, 100, 0, -25, 25])
        self.assertEqual(t.top(), [100, 50, 25, 0])

    def test_float_only(self):
        t = TopKTracker(3)
        t.push_all([1.5, 2.5, 0.5, 3.5, 1.25])
        self.assertEqual(t.top(), [3.5, 2.5, 1.5])


if __name__ == "__main__":
    unittest.main()
