"""Explicit, private snapshots of the current job's activity log."""
from pathlib import Path
import tempfile


def export_activity_log(folder: Path, entries: list[str]) -> Path:
    if not entries:
        raise ValueError("No activity to export")
    path = None
    try:
        # Unique file, mode 0600; never overwrite an existing log.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=folder, prefix="NixScribe-activity-",
            suffix=".txt", delete=False,
        ) as handle:
            path = Path(handle.name)
            handle.write("\n".join(entries) + "\n")
        return path
    except Exception:
        if path is not None:
            path.unlink(missing_ok=True)
        raise
