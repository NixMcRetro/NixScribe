import json
from pathlib import Path
import stat
import tempfile
import unittest

from diarization_diagnostics import write_diagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_private_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics.json"
            write_diagnostics(path, {"words": [{"word": "Example"}]})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(json.loads(path.read_text())["schema_version"], 1)
            with self.assertRaises(FileExistsError):
                write_diagnostics(path, {})
            self.assertEqual(json.loads(path.read_text())["words"], [{"word": "Example"}])
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_invalid_json_leaves_no_partial_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics.json"
            with self.assertRaises(ValueError):
                write_diagnostics(path, {"time": float("nan")})
            self.assertEqual(list(Path(tmp).iterdir()), [])
