"""Test the GPU backend's contract without importing MLX or downloading weights."""
import unittest

from app_config import ALL_MODEL_IDS, MODEL_IDS, MLX_MODELS, MLX_REPOSITORIES
from mlx_backend import normalize_segments


class MLXAdapterTests(unittest.TestCase):
    def test_model_ids_do_not_collide(self):
        self.assertFalse(MODEL_IDS & MLX_REPOSITORIES.keys())
        self.assertEqual(set(MLX_REPOSITORIES), {key for key, _ in MLX_MODELS})
        self.assertEqual(len(ALL_MODEL_IDS), 10)

    def test_normalizes_to_alignment_contract(self):
        result = normalize_segments({"segments": [
            {"start": 4.0, "end": 8.2, "text": " Second. ", "tokens": [1]},
            {"start": -0.1, "end": 3.0, "text": " First."},
            {"start": 3.1, "end": 3.2, "text": " "},
        ]}, 8.0)
        self.assertEqual(result, {"language": "en", "segments": [
            {"start": 0.0, "end": 3.0, "text": "First."},
            {"start": 4.0, "end": 8.0, "text": "Second."},
        ]})

    def test_invalid_nonempty_timing_not_silently_dropped(self):
        for start, end in [(2, 1), (float("nan"), 1), (9, 10)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                normalize_segments({"segments": [
                    {"start": start, "end": end, "text": "Speech"}
                ]}, 8.0)

    def test_silent_recording(self):
        self.assertEqual(normalize_segments({"segments": []}, 1.0)["segments"], [])

    def test_malformed_result(self):
        with self.assertRaises(ValueError):
            normalize_segments({}, 1.0)


if __name__ == "__main__":
    unittest.main()
