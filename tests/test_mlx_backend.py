"""Test the GPU backend's contract without importing MLX or downloading weights."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from app_config import ALL_MODEL_IDS, MODEL_IDS, MLX_MODELS, MLX_REPOSITORIES
from mlx_backend import normalize_segments, transcribe_mlx


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

    def test_worker_failure_does_not_disclose_private_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def start_worker(command, stdout, stderr):
                stdout.write(b"Synthetic private words and prompt")
                stdout.flush()
                from unittest.mock import Mock
                return Mock(wait=Mock(return_value=7), poll=Mock(return_value=7))
            with patch("mlx_backend.subprocess.Popen", side_effect=start_worker):
                with self.assertRaisesRegex(RuntimeError, "worker failed \\(exit 7\\)") as error:
                    transcribe_mlx(root / "sample.wav", "mlx-large-v3", root, 1.0,
                                   "Synthetic private prompt", lambda: None)
            self.assertNotIn("Synthetic", str(error.exception))

    def test_malformed_result(self):
        with self.assertRaises(ValueError):
            normalize_segments({}, 1.0)


if __name__ == "__main__":
    unittest.main()
