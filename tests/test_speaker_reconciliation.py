"""Synthetic speaker reconciliation tests; no models or recordings."""
import unittest

from speaker_reconciliation import reconcile


def interval(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def word(start, end):
    return {"start": start, "end": end, "word": "Example"}


class ReconciliationTests(unittest.TestCase):
    def test_swapped_labels_and_split_intervals(self):
        a = [interval(0, 2, "A"), interval(2, 4, "B")]
        b = [interval(0, 1, "Y"), interval(1, 2, "Y"), interval(2, 4, "X")]
        words = [word(0.2, 0.4), word(2.1, 2.3)]
        result = reconcile(a, b, words)
        self.assertEqual(result["mapping"], {"X": "B", "Y": "A"})
        self.assertEqual([d["speaker"] for d in result["decisions"]], ["A", "B"])
        self.assertTrue(all(d["reason"] == "agreement" for d in result["decisions"]))
        self.assertEqual(words[0], word(0.2, 0.4))

    def test_unmatched_speaker_and_gap(self):
        result = reconcile([interval(0, 1, "SPEAKER_00")],
                           [interval(0, 1, "X"), interval(2, 3, "Y")],
                           [word(2.1, 2.2), word(4, 5), {}])
        self.assertEqual(result["mapping"]["Y"], "SPEAKER_01")
        self.assertEqual([d["speaker"] for d in result["decisions"]],
                         ["SPEAKER_01", "UNKNOWN", "UNKNOWN"])

    def test_duplicate_spans_do_not_double_count(self):
        result = reconcile([interval(0, 2, "A"), interval(1, 2, "A")],
                           [interval(0, 2, "X")], [word(1, 2)])
        self.assertEqual(result["mapping_overlap_seconds"]["X"]["A"], 2)
        self.assertEqual(result["decisions"][0]["coreml_overlap"], {"A": 1})

    def test_equal_overlap_is_uncertain(self):
        a = [interval(0, 2, "A"), interval(0, 2, "B")]
        result = reconcile(a, [], [word(1, 1.1)])
        self.assertEqual(result["decisions"][0]["speaker"], "UNKNOWN")
        self.assertTrue(result["decisions"][0]["uncertain"])

    def test_disagreement_and_short_word(self):
        a = [interval(0, 2, "A"), interval(2, 4, "B")]
        b = [interval(0, 1.8, "X"), interval(1.8, 4, "Y")]
        d = reconcile(a, b, [word(1.9, 1.95), word(1.98, 2.08)])["decisions"]
        self.assertEqual(d[0]["speaker"], "A")
        self.assertEqual(d[0]["reason"], "coreml_disagreement")
        self.assertEqual(d[1]["speaker"], "B")
        self.assertTrue(d[0]["uncertain"])

    def test_full_secondary_vs_partial_primary(self):
        a = [interval(0, 1, "A"), interval(2, 2.02, "A"), interval(3, 4, "B")]
        b = [interval(0, 1, "X"), interval(2, 4, "Y")]
        d = reconcile(a, b, [word(2, 2.1)])["decisions"][0]
        self.assertEqual(d["speaker"], "B")
        self.assertEqual(d["reason"], "mps_full_coreml_partial")

    def test_cancel(self):
        def cancel():
            raise InterruptedError()
        with self.assertRaises(InterruptedError):
            reconcile([], [], [], cancel)


if __name__ == "__main__":
    unittest.main()
