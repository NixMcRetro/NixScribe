import unittest
from clustering_probe import Memo


class ClusteringProbeTests(unittest.TestCase):
    def test_single_inference_and_copy_isolation(self):
        calls = []
        def infer():
            calls.append(1)
            return {"data": [1, 2]}
        memo = Memo(infer)
        memo()["data"][0] = 9
        memo()["data"][1] = 8
        self.assertEqual(memo(), {"data": [1, 2]})
        self.assertEqual((len(calls), memo.calls, memo.hits), (1, 1, 2))


if __name__ == "__main__":
    unittest.main()
