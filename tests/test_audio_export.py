"""Test the actual ffmpeg audio conversion, with no model downloads."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave

from app_config import DEFAULT_MODEL, MODEL_IDS, MODELS, MODEL_REPOSITORIES
from audio_export import export_mp3, mp3_destination


class ModelOptionsTests(unittest.TestCase):
    def test_models_unique_and_default_present(self):
        self.assertEqual(len(MODELS), len(MODEL_IDS))
        self.assertEqual(len(MODEL_IDS), 8)
        self.assertIn(DEFAULT_MODEL, MODEL_IDS)

    def test_each_model_has_exact_repository(self):
        self.assertEqual(set(MODEL_REPOSITORIES), MODEL_IDS)
        self.assertEqual(MODEL_REPOSITORIES['large-v3'], 'Systran/faster-whisper-large-v3')
        self.assertEqual(MODEL_REPOSITORIES['large-v3-turbo'], 'mobiuslabsgmbh/faster-whisper-large-v3-turbo')
        self.assertEqual(MODEL_REPOSITORIES['distil-large-v3.5'], 'distil-whisper/distil-large-v3.5-ct2')
        self.assertTrue(all(repo.count('/') == 1 for repo in MODEL_REPOSITORIES.values()))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "ffmpeg and ffprobe required")
class AudioExportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "source.wav"
        with wave.open(str(self.source), "wb") as f:
            f.setnchannels(2)
            f.setsampwidth(2)
            f.setframerate(44100)
            f.writeframes(b"\0" * 44100 * 4 * 3)

    def test_mp3_encoder_and_original_preserved(self):
        original = self.source.read_bytes()
        dest = export_mp3(self.source, self.root / "source.mp3")
        report = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,bit_rate,sample_rate,channels",
            "-of", "json", str(dest)
        ]))
        audio = report["streams"][0]
        self.assertEqual(audio["codec_name"], "mp3")
        self.assertEqual(int(audio["bit_rate"]), 128000)
        self.assertEqual(audio["sample_rate"], "44100")
        self.assertEqual(audio["channels"], 2)
        self.assertEqual(self.source.read_bytes(), original)

    def test_collision_does_not_overwrite(self):
        original = self.root / "source.MP3"
        original.write_bytes(b"original")
        dest = mp3_destination(self.root, "source")
        self.assertEqual(dest.name, "source.audio.mp3")
        export_mp3(self.source, dest)
        self.assertEqual(original.read_bytes(), b"original")
        with self.assertRaises(FileExistsError):
            export_mp3(self.source, dest)

    def test_cancellation_keeps_source_and_no_partial(self):
        dest = self.root / "source.mp3"
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls > 1:
                raise InterruptedError("cancelled")
        with self.assertRaises(InterruptedError):
            export_mp3(self.source, dest, cancel)
        self.assertFalse(dest.exists())
        self.assertEqual(list(self.root.glob(".mp3-*")), [])
        self.assertTrue(self.source.exists())


if __name__ == "__main__":
    unittest.main()
