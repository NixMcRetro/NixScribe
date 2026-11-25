"""Offline tests of experimental diarization transport, no torch or weights."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diarization_backend import validate_intervals, run_mps


class DiarizationBackendTests(unittest.TestCase):
    def test_valid_intervals_preserve_labels_and_overlap(self):
        rows = [{"start": 1.5, "end": 4.0, "speaker": "SPEAKER_01"},
                {"start": 0.1, "end": 2.0, "speaker": "SPEAKER_00"}]
        self.assertEqual(validate_intervals(rows), list(reversed(rows)))

    def test_invalid_intervals(self):
        for row in [
            {"start": 1, "end": 0, "speaker": "A"},
            {"start": float("nan"), "end": 2, "speaker": "A"},
            {"start": -1, "end": 2, "speaker": "A"},
            {"start": 1, "end": 2, "speaker": ""},
        ]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_intervals([row])

    def test_no_speech_is_valid_transport(self):
        self.assertEqual(validate_intervals([]), [])

    def test_cancel_before_worker_spawn(self):
        def cancel():
            raise InterruptedError()
        with tempfile.TemporaryDirectory() as tmp, patch("diarization_backend.subprocess.Popen") as spawn:
            with self.assertRaises(InterruptedError):
                run_mps(Path(tmp) / "example.wav", 2, lambda *_: None, cancel)
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
