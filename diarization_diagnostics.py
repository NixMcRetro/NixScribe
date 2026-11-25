"""Private opt-in diagnostics; the caller supplies only approved job fields."""
import json
import os
from pathlib import Path
import tempfile


def write_diagnostics(path: Path, data: dict) -> Path:
    """Write atomically with owner-only permissions, without replacing an export."""
    fd, name = tempfile.mkstemp(prefix=".diagnostics-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({**data, "schema_version": 1}, stream, ensure_ascii=False,
                      allow_nan=False, indent=2)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
