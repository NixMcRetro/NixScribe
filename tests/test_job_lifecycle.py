"""Job lifecycle checks using Flask's local test client; no inference or media."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec("flask"), "Flask required; run in build venv")
class JobLifecycleTests(unittest.TestCase):
    def setUp(self):
        import app
        self.app = app
        self.state = dict(app.STATE)
        self.addCleanup(self.restore_state)
        app._update(running=False, cancel_requested=False)
        self.client = app.app.test_client()
        self.payload = {"input_file": "/unused.m4a", "model": "large-v3",
                        "speaker_count": "1", "diarization_device": "cpu"}

    def restore_state(self):
        with self.app.state_lock:
            self.app.STATE.clear()
            self.app.STATE.update(self.state)

    def test_copy_activity_log_uses_utf8_locale(self):
        entries = ["Recording 11.28.55\u202fam.mov", "Unicode: café 日本語"]
        self.app._update(log=entries)
        with patch.object(self.app.sys, "platform", "darwin"), \
             patch.dict(self.app.os.environ, {"LC_ALL": "C", "LANG": "C"}), \
             patch.object(self.app.subprocess, "run") as run:
            response = self.client.post("/api/log/copy", json={},
                                        headers={"X-NixScribe-Action": "copy-log"})
        self.assertEqual(response.status_code, 200)
        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["/usr/bin/pbcopy"])
        self.assertEqual(kwargs["input"], "\n".join(entries) + "\n")
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["env"]["LC_ALL"], "en_US.UTF-8")
        self.assertEqual(kwargs["env"]["LANG"], "C")

    def test_live_preview_does_not_echo_private_text(self):
        terminal = io.StringIO()
        with contextlib.redirect_stdout(terminal):
            capture = self.app.LiveTranscriptCapture(None, max_lines=30)
            with contextlib.redirect_stdout(capture):
                print("Transcript: [0.0 --> 1.0] Synthetic private words.")
        self.assertEqual(terminal.getvalue(), "")
        self.assertIn("Synthetic private words.", self.app.STATE["live_preview"])

    def test_job_claim_rechecked_after_validation(self):
        def another_request_claimed():
            self.app._update(running=True)
            return "large-v3"
        with patch.object(self.app, "preferred_model", side_effect=another_request_claimed), \
             patch.object(self.app.threading, "Thread") as thread:
            response = self.client.post("/api/start", json=self.payload)
        self.assertEqual(response.status_code, 409)
        thread.assert_not_called()

    def test_thread_start_failure_releases_job_claim(self):
        with patch.object(self.app.threading, "Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("test")
            response = self.client.post("/api/start", json=self.payload)
        self.assertEqual(response.status_code, 500)
        self.assertFalse(self.app.STATE["running"])
        self.assertEqual(self.app.STATE["status"], "Error")

    def test_work_cleanup_failure_retains_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            transcript = project / "sample.txt"
            transcript.write_text("Completed transcript")
            work = project / ".nixscribe-work"
            work.mkdir()
            with patch.object(self.app.shutil, "rmtree", side_effect=PermissionError("test")):
                self.app._remove_work_dir(work)
            self.assertEqual(transcript.read_text(), "Completed transcript")
            self.assertTrue(work.exists())

    def test_work_cleanup_does_not_remove_project(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            work = project / ".nixscribe-work"
            work.mkdir()
            self.app._remove_work_dir(work)
            self.assertTrue(project.exists())
            self.assertFalse(work.exists())

if __name__ == "__main__":
    unittest.main()
