"""Test dated project reservation without audio, network or ML dependencies."""
from datetime import date
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from project_paths import create_project_dir


class ProjectPathsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.day = date(2026, 11, 25)

    def test_date_prefix_and_safe_name(self):
        folder = create_project_dir(self.root, "Interview: part 1", job_date=self.day)
        self.assertEqual(folder.name, "2026-11-25 Interview- part 1")
        self.assertTrue(folder.is_dir())

    def test_repeated_jobs_preserve_previous_folder(self):
        first = create_project_dir(self.root, "Recording", job_date=self.day)
        marker = first / "original.txt"
        marker.write_text("keep")
        self.assertEqual(create_project_dir(self.root, "Recording", job_date=self.day).name,
                         "2026-11-25 Recording (2)")
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(create_project_dir(self.root, "Recording", job_date=date(2026, 11, 26)).name,
                         "2026-11-26 Recording")

    def test_concurrent_reservations_are_unique(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda _: create_project_dir(self.root, "Recording",
                                                                job_date=self.day), range(8)))
        self.assertEqual(len(set(paths)), 8)

    def test_legacy_folder_untouched(self):
        old = self.root / "Recording"
        old.mkdir()
        (old / "keep.txt").write_text("keep")
        create_project_dir(self.root, "Recording", job_date=self.day)
        self.assertEqual((old / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
