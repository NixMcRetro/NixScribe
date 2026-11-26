import unittest
from clustering_probe import Memo, verify_identity


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

    def test_identity_rejects_missing_or_changed_keys(self):
        identity = {"snapshot": "a", "audio_sha256": "b"}
        verify_identity(dict(identity, extra=True), identity)
        for bad in ({}, {"snapshot": "a"}, {"snapshot": "a", "audio_sha256": "c"}):
            with self.assertRaises(ValueError):
                verify_identity(bad, identity)

    def test_preloaded_memo_never_calls_inference(self):
        def forbidden():
            raise AssertionError("Inference called")
        memo = Memo(forbidden)
        memo.value = {"data": [1, 2]}
        memo()["data"].clear()
        self.assertEqual(memo(), {"data": [1, 2]})
        self.assertEqual((memo.calls, memo.hits), (0, 2))


if __name__ == "__main__":
    unittest.main()
