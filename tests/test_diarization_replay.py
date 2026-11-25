import unittest

from benchmarks.replay import audit, candidates


def span(start, end, speaker):
    return dict(start=start, end=end, speaker=speaker)


def word(start, end):
    return dict(start=start, end=end)


def diagnostics(core, mps, words, assigned):
    return {
        "coreml": {"raw_intervals": {"coreml": core}},
        "mps": {"raw_intervals": {"mps": mps}, "aligned_words": words,
                "assigned_speakers": assigned},
    }


class ReplayTests(unittest.TestCase):
    def test_mps_priority_and_mapped_gap(self):
        mps = [span(0, 4, "A"), span(5, 9, "B")]
        core = [span(0, 2, "X"), span(2, 3, "Y"), span(3, 4, "X"),
                span(4, 9, "Y")]
        data = diagnostics(core, mps, [word(2, 3), word(4, 5)], ["A", "A"])
        self.assertEqual(candidates(data), {"mps_first": ["A", "B"],
                                           "mps_gap_only": ["A", "B"]})

    def test_no_new_speaker_or_missing_timing(self):
        data = diagnostics([span(0, 2, "X"), span(3, 4, "Y")],
                           [span(0, 2, "A")], [word(3, 4), {}], ["A", "A"])
        self.assertEqual(candidates(data), {"mps_first": ["A", "A"],
                                           "mps_gap_only": ["A", "A"]})

    def test_gap_requires_half_word_coverage(self):
        data = diagnostics([span(0, 2, "X"), span(4, 4.2, "X")],
                           [span(0, 2, "A"), span(2, 3, "B")],
                           [word(4, 5)], ["B"])
        self.assertEqual(candidates(data), {"mps_first": ["A"],
                                           "mps_gap_only": ["B"]})

    def test_audit_uses_fixed_baseline_mapping(self):
        result = audit(["x", "x", "y", "y", "y", None],
                       ["A", "A", "A", "B", "B", "A"],
                       ["B", "A", "B", "B", "B", "B"])
        self.assertEqual(result, dict(changed_words=3, changed_evaluated_words=2,
                                     corrected_vs_mps=1, regressed_vs_mps=1))


if __name__ == "__main__":
    unittest.main()
