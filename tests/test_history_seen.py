import csv
import io
import unittest

from src.check_history_seen import inspect_ordered_rows


class HistorySeenOrderTest(unittest.TestCase):
    def test_strictly_later_decrease_is_reported_after_sorting(self):
        # Deliberately put the later timestamp first in file order.
        rows = [
            ("u1", "de", "lex1", 30, 6, 0),
            ("u1", "de", "lex1", 10, 5, 1),
            ("u1", "de", "lex1", 20, 7, 2),
            ("u2", "de", "lex1", 20, 2, 3),
            ("u2", "de", "lex1", 10, 2, 4),
        ]
        buffer = io.StringIO()
        examples = []
        counts = inspect_ordered_rows(rows, csv.writer(buffer), examples)
        self.assertEqual(counts["later_rows_with_decrease"], 1)
        self.assertEqual(counts["pairs_with_decrease"], 1)
        self.assertEqual(examples[0]["earlier_row_index"], 2)
        self.assertEqual(examples[0]["later_row_index"], 0)
        self.assertEqual(examples[0]["drop"], 1)

    def test_equal_timestamps_are_not_ordered(self):
        rows = [
            ("u1", "en", "lex1", 10, 5, 0),
            ("u1", "en", "lex1", 10, 3, 1),
        ]
        counts = inspect_ordered_rows(rows, csv.writer(io.StringIO()), [])
        self.assertEqual(counts["later_rows_with_decrease"], 0)
        self.assertEqual(
            counts["same_timestamp_groups_with_different_history_seen"], 1
        )

    def test_earlier_maximum_detects_nonadjacent_drop(self):
        rows = [
            ("u1", "en", "lex1", 10, 10, 0),
            ("u1", "en", "lex1", 20, 9, 1),
            ("u1", "en", "lex1", 30, 9, 2),
        ]
        counts = inspect_ordered_rows(rows, csv.writer(io.StringIO()), [])
        self.assertEqual(counts["later_rows_with_decrease"], 2)
        self.assertEqual(counts["timestamp_groups_with_decrease"], 2)


if __name__ == "__main__":
    unittest.main()
