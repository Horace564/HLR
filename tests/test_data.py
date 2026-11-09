import unittest

from src.data import inspect_row


BASE = {
    "p_recall": "0.5",
    "timestamp": "1362076081",
    "delta": "86400",
    "user_id": "u:test",
    "learning_language": "de",
    "ui_language": "en",
    "lexeme_id": "word-1",
    "lexeme_string": "wort/wort<n>",
    "history_seen": "4",
    "history_correct": "3",
    "session_seen": "2",
    "session_correct": "1",
}


class CleaningRulesTest(unittest.TestCase):
    def test_valid_boundary_targets_and_zero_delta_are_kept(self):
        for p, correct in (("0", "0"), ("1", "2")):
            with self.subTest(p=p):
                row = dict(BASE, p_recall=p, session_correct=correct, delta="0")
                reasons, _ = inspect_row(row)
                self.assertEqual(reasons, [])

    def test_ratio_mismatch_and_invalid_counts_are_independent(self):
        row = dict(BASE, p_recall="0.25", history_correct="5")
        reasons, _ = inspect_row(row)
        self.assertIn("history_correct_gt_seen", reasons)
        self.assertIn("p_recall_ratio_mismatch", reasons)

    def test_missing_and_nonfinite_values_are_excluded(self):
        row = dict(BASE, user_id="", p_recall="NaN", delta="1.5")
        reasons, _ = inspect_row(row)
        self.assertIn("missing:user_id", reasons)
        self.assertIn("invalid_type:p_recall", reasons)
        self.assertIn("invalid_type:delta", reasons)

    def test_session_count_bounds(self):
        row = dict(BASE, session_seen="0", session_correct="1")
        reasons, _ = inspect_row(row)
        self.assertIn("session_seen_nonpositive", reasons)
        self.assertIn("session_correct_gt_seen", reasons)
        self.assertNotIn("p_recall_ratio_mismatch", reasons)


if __name__ == "__main__":
    unittest.main()
