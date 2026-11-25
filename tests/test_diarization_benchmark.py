import unittest
from benchmarks.diarization import best_mapping, interval_score, word_reference, word_score


def interval(start, end, speaker):
    return dict(start=start, end=end, speaker=speaker)


class BenchmarkTests(unittest.TestCase):
    def test_permutation_and_overlap(self):
        ref = [interval(0, 2, "a"), interval(1, 3, "b")]
        hyp = [interval(0, 2, "y"), interval(1, 3, "x")]
        self.assertEqual(interval_score(ref, hyp, 4)["der"], 0)
        score = interval_score(ref, [interval(0, 2, "y"), interval(3, 4, "x")], 4)
        self.assertEqual(score["missed_seconds"], 2)
        self.assertEqual(score["false_alarm_seconds"], 1)
        self.assertEqual(score["der"], .75)

    def test_mapping_is_global(self):
        self.assertEqual(best_mapping({("x", "a"): 8, ("x", "b"): 7,
                                       ("y", "a"): 6}), {"x": "b", "y": "a"})

    def test_word_exclusions_and_unknowns(self):
        ref = [interval(0, 2, "a"), interval(1, 3, "b")]
        words = [dict(start=0, end=.5), dict(start=1, end=2),
                 dict(start=2, end=2.5), dict(start=3, end=4), {}]
        labels = word_reference(words, ref, 4)
        self.assertEqual(labels, ["a", None, "b", None, None])
        score = word_score(labels, ["x", "x", "UNKNOWN", "y", "y"])
        self.assertEqual((score["correct"], score["evaluated"], score["excluded"]), (1, 2, 3))
        self.assertEqual(score["accuracy"], .5)
        self.assertEqual(score["unknown"], 1)

    def test_duplicate_intervals_not_double_counted(self):
        ref = [interval(0, 1, "a"), interval(0, 1, "a")]
        self.assertEqual(interval_score(ref, [interval(0, 1, "x")], 1)["der"], 0)
        self.assertEqual(word_reference([dict(start=0, end=3)], ref, 3), [None])


if __name__ == "__main__":
    unittest.main()
