import stat
import tempfile
import unittest
from pathlib import Path

from activity_log import export_activity_log
from app_config import ALL_MODEL_IDS, MODEL_DESCRIPTIONS


class ActivityLogTests(unittest.TestCase):
    def test_full_snapshot_private_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            entries = [f"[12:00:00] Entry {i}" for i in range(1000)] + ["Unicode: café"]
            first = export_activity_log(folder, entries)
            second = export_activity_log(folder, ["Next job"])
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_text(encoding="utf-8"), "\n".join(entries) + "\n")
            self.assertEqual(second.read_text(encoding="utf-8"), "Next job\n")
            self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o600)

    def test_no_empty_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with self.assertRaises(ValueError):
                export_activity_log(folder, [])
            self.assertEqual(list(folder.iterdir()), [])

    def test_every_model_has_a_description(self):
        self.assertEqual(set(MODEL_DESCRIPTIONS), ALL_MODEL_IDS)
        self.assertTrue(all(MODEL_DESCRIPTIONS.values()))


if __name__ == "__main__":
    unittest.main()
