"""No audio, model loading, or network required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from prompt_handling import (prepare_prompt, require_prompt, PromptError, prompt_metadata,
                             save_profile, load_profile)


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.mlx = patch("prompt_handling._mlx_encoder", return_value=list)
        self.cpu = patch("prompt_handling._cpu_encoder", return_value=list)
        self.mlx.start()
        self.cpu.start()
        self.addCleanup(self.mlx.stop)
        self.addCleanup(self.cpu.stop)

    def test_combination_and_fallback_keep_fields(self):
        mlx = prepare_prompt("mlx-large-v3", "Context", "Names")
        cpu = prepare_prompt("large-v3", "Context", "Names")
        self.assertEqual(mlx["initial_prompt"], "Context\n\nNames")
        self.assertEqual(mlx["hotwords"], "")
        self.assertEqual(cpu["initial_prompt"], "Context")
        self.assertEqual(cpu["hotwords"], "Names")
        self.assertEqual(mlx["effective_tokens"], 15)
        self.assertEqual(cpu["effective_tokens"], 14)
        self.assertNotIn("Context", str(prompt_metadata(mlx)))
        self.assertNotIn("Names", str(prompt_metadata(cpu)))

    def test_exact_boundary_and_no_silent_truncation(self):
        for model in ("mlx-large-v3", "large-v3"):
            self.assertEqual(require_prompt(prepare_prompt(model, "a" * 222, ""))["effective_tokens"], 223)
            plan = prepare_prompt(model, "a" * 223, "")
            self.assertEqual(plan["initial_prompt"], "a" * 223)
            with self.assertRaises(PromptError):
                require_prompt(plan)

    def test_missing_cpu_tokenizer_is_explicit(self):
        with patch("prompt_handling._cpu_encoder", return_value=None):
            plan = prepare_prompt("large-v3", "context", "")
            self.assertFalse(plan["available"])
            self.assertIsNone(plan["effective_tokens"])
            with self.assertRaises(PromptError):
                require_prompt(plan)

    def test_blank_does_not_require_tokenizer(self):
        with patch("prompt_handling._mlx_encoder", side_effect=AssertionError):
            self.assertEqual(require_prompt(prepare_prompt("mlx-large-v3", "", ""))["effective_tokens"], 0)

    def test_errors_never_echo_private_text(self):
        with patch("prompt_handling._mlx_encoder", side_effect=ValueError("PRIVATE")):
            with self.assertRaisesRegex(PromptError, "^Could not count prompt tokens"):
                prepare_prompt("mlx-large-v3", "PRIVATE", "")


class VocabularyProfileTests(unittest.TestCase):
    def test_roundtrip_exact_text_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            fields = {"initial_prompt": "  café\n", "hotwords": "  names, terminology "}
            save_profile(path, fields)
            self.assertEqual(load_profile(path), {"schema": 1, **fields})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_profile(path, {"hotwords": "different"})
            self.assertEqual(path.read_bytes(), before)

    def test_bad_or_oversize_profile_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            for invalid in ('{"schema":2,"initial_prompt":"","hotwords":""}',
                            '{"schema":1,"initial_prompt":7,"hotwords":""}', "x" * 262145):
                path.write_text(invalid)
                with self.assertRaises(PromptError):
                    load_profile(path)


@unittest.skipUnless(importlib.util.find_spec("flask"), "Requires build venv")
class PreviewTests(unittest.TestCase):
    def test_local_preview_and_start_reject_over_budget(self):
        import app
        client = app.app.test_client()
        fields = {"model": "large-v3", "initial_prompt": "a" * 223, "hotwords": ""}
        with patch("prompt_handling._cpu_encoder", return_value=list):
            r = client.post("/api/prompt-preview", json=fields,
                            headers={"X-NixScribe-Action": "prompt-preview"})
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.json["over_limit"])
            with patch.object(app.threading, "Thread") as thread:
                r = client.post("/api/start", json={**fields, "input_file": "/unused.wav",
                                                  "speaker_count": "1", "diarization_device": "cpu"})
                self.assertEqual(r.status_code, 400)
                thread.assert_not_called()

    def test_cross_origin_preview_rejected(self):
        import app
        r = app.app.test_client().post("/api/prompt-preview", json={},
            headers={"X-NixScribe-Action": "prompt-preview", "Origin": "https://example.invalid"})
        self.assertEqual(r.status_code, 403)


if __name__ == "__main__":
    unittest.main()
