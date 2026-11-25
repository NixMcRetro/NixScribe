"""Offline tests of cache readiness; no network or model downloads."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_cache import patterns_for, prepare_model, snapshot_ready


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name)

    def write(self, *names):
        for name in names:
            (self.path / name).write_bytes(b"test")

    def test_partial_cpu_snapshot_not_ready(self):
        self.write("config.json", "tokenizer.json", "vocabulary.json")
        self.assertFalse(snapshot_ready(self.path, "distil-large-v3.5"))
        self.write("model.bin")
        self.assertFalse(snapshot_ready(self.path, "distil-large-v3.5"))
        self.write("preprocessor_config.json")
        self.assertTrue(snapshot_ready(self.path, "distil-large-v3.5"))

    def test_smaller_cpu_model(self):
        self.write("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
        self.assertTrue(snapshot_ready(self.path, "tiny.en"))
        (self.path / "model.bin").write_bytes(b"")
        self.assertFalse(snapshot_ready(self.path, "tiny.en"))

    def test_mlx_formats_and_broken_link(self):
        self.write("config.json")
        (self.path / "weights.npz").symlink_to(self.path / "not-downloaded")
        self.assertFalse(snapshot_ready(self.path, "mlx-large-v3"))
        self.write("weights.safetensors")
        self.assertTrue(snapshot_ready(self.path, "mlx-large-v3-turbo"))
        self.assertIn("weights.npz", patterns_for("mlx-large-v3"))

    def test_cache_hit_does_not_spawn_downloader(self):
        events = []
        with patch("model_cache.cached_snapshot", return_value=self.path), \
             patch("model_cache.subprocess.Popen") as spawn:
            self.assertEqual(prepare_model("large-v3", self.path, events.append,
                                           lambda: None), self.path)
            spawn.assert_not_called()
        self.assertEqual(events, ["checking", "cached"])

    def test_cancel_before_cache_check(self):
        with patch("model_cache.cached_snapshot") as cached:
            def cancel():
                raise InterruptedError()
            with self.assertRaises(InterruptedError):
                prepare_model("large-v3", self.path, lambda _: None, cancel)
            cached.assert_not_called()


if __name__ == "__main__":
    unittest.main()
