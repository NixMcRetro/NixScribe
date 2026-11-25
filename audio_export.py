"""Local MP3 export, independent of the speech recognition engine."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


def mp3_destination(project: Path, stem: str) -> Path:
    """Do not overwrite existing media, including a retained source MP3."""
    used = {p.name.casefold() for p in project.iterdir()}
    name = f"{stem}.mp3"
    if name.casefold() not in used:
        return project / name
    index = 1
    while True:
        name = f"{stem}.audio.mp3" if index == 1 else f"{stem}.audio-{index}.mp3"
        if name.casefold() not in used:
            return project / name
        index += 1


def export_mp3(source: Path, destination: Path,
               check_cancel: Callable[[], None] = lambda: None) -> Path:
    """Encode the first original audio stream at 128 kbps CBR, without metadata.

    Never encode from the 16 kHz transcription WAV. Publish the result only
    after ffmpeg exits successfully; an existing destination is never replaced.
    """
    if destination.exists():
        raise FileExistsError(destination)
    check_cancel()
    with tempfile.TemporaryDirectory(prefix=".mp3-", dir=destination.parent) as tmp:
        partial = Path(tmp) / "audio.mp3"
        with tempfile.TemporaryFile() as log:
            proc = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                 "-y", "-i", str(source), "-map", "0:a:0", "-vn",
                 "-map_metadata", "-1", "-map_chapters", "-1",
                 "-c:a", "libmp3lame", "-b:a", "128k", str(partial)],
                stdout=subprocess.DEVNULL, stderr=log,
            )
            try:
                while True:
                    check_cancel()
                    try:
                        rc = proc.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                check_cancel()
                if rc != 0 or not partial.is_file() or partial.stat().st_size == 0:
                    log.seek(0)
                    detail = log.read(4096).decode("utf-8", errors="replace").strip()
                    raise RuntimeError("MP3 export failed." + (" " + detail if detail else ""))
                # Atomic publication on the same filesystem, without overwrite.
                os.link(partial, destination)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
    return destination
