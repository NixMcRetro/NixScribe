"""Create unique, dated transcription project folders without replacing earlier work."""
from datetime import date
from pathlib import Path
import re


def safe_project_name(stem: str) -> str:
    """Return a Finder-friendly project folder name."""
    name = re.sub(r"[\\/:*?\"<>|]", "-", stem).strip().strip(".")
    name = re.sub(r"\s+", " ", name)
    return name or "Transcription"


def create_project_dir(projects_dir: Path, stem: str, *, job_date: date | None = None) -> Path:
    """Reserve a new folder atomically, using the Mac's local calendar date."""
    day = job_date if job_date is not None else date.today()
    base = f"{day.isoformat()} {safe_project_name(stem)}"
    index = 1
    while True:
        name = base if index == 1 else f"{base} ({index})"
        candidate = projects_dir / name
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            index += 1
