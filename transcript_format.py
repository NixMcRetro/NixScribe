"""Formatting WhisperX-aligned TXT and SRT exports, without importing ML packages."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import textwrap
from typing import Any


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float


def _segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(result.get("segments", []), key=lambda s: float(s.get("start") or 0.0))


def _speaker(segment: dict[str, Any]) -> str:
    return str(segment.get("speaker") or "UNKNOWN")


def write_txt(result: dict[str, Any], audio_path: Path, output_dir: Path,
              *, group_turns: bool = True) -> Path:
    """Speaker-labelled paragraphs without timestamps."""
    turns: list[tuple[str, str]] = []
    for seg in _segments(result):
        txt = " ".join(str(seg.get("text") or "").split())
        if not txt:
            continue
        speaker = _speaker(seg)
        if group_turns and turns and turns[-1][0] == speaker:
            turns[-1] = (speaker, turns[-1][1] + " " + txt)
        else:
            turns.append((speaker, txt))
    body = "\n\n".join(f"[{speaker}] {text}" for speaker, text in turns)
    dest = output_dir / f"{audio_path.stem}.txt"
    dest.write_text(body + ("\n" if body else ""), encoding="utf-8")
    return dest


def _safe_time(value: Any, fallback: float) -> float:
    try:
        number = float(value)
        return max(0.0, number) if math.isfinite(number) else fallback
    except (ValueError, TypeError):
        return fallback


def _words(segment: dict[str, Any]) -> list[TimedWord]:
    txt = " ".join(str(segment.get("text") or "").split())
    if not txt:
        return []
    start = _safe_time(segment.get("start"), 0.0)
    end = max(start + 0.01, _safe_time(segment.get("end"), start + 2.0))
    aligned = segment.get("words") or []
    if isinstance(aligned, list) and aligned:
        result = []
        for item in aligned:
            if not isinstance(item, dict):
                result = []
                break
            word = str(item.get("word") or "").strip()
            a = _safe_time(item.get("start"), -1.0)
            b = _safe_time(item.get("end"), -1.0)
            if not word or a < 0 or b < a:
                result = []
                break
            result.append(TimedWord(word, a, b))
        if result and " ".join(w.text for w in result) == txt:
            return result
    # Preserve original text when alignment omits words or timestamps.
    tokens = txt.split(" ")
    weights = [max(1, len(w)) for w in tokens]
    total = sum(weights)
    cursor = start
    output = []
    for index, (word, weight) in enumerate(zip(tokens, weights)):
        finish = end if index == len(tokens) - 1 else cursor + (end - start) * weight / total
        output.append(TimedWord(word, cursor, finish))
        cursor = finish
    return output


def _stamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    h, milliseconds = divmod(milliseconds, 3600000)
    m, milliseconds = divmod(milliseconds, 60000)
    s, milliseconds = divmod(milliseconds, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milliseconds:03d}"


def _lines(words: list[TimedWord], speaker: str, include_speakers: bool, width: int) -> list[str]:
    text = " ".join(w.text for w in words)
    if include_speakers:
        text = f"{speaker}: {text}"
    return textwrap.wrap(text, width=width, break_long_words=True,
                         break_on_hyphens=False, replace_whitespace=False)


def write_srt(result: dict[str, Any], audio_path: Path, output_dir: Path,
              *, include_speakers: bool = True, chars_per_line: int = 42,
              max_lines: int = 2, max_duration: float = 6.0) -> Path:
    """Build compact, word-timed cues while preserving the entire original text."""
    if chars_per_line not in (32, 37, 42, 48):
        raise ValueError("Unsupported subtitle line width")
    if max_lines not in (1, 2) or max_duration not in (3.0, 4.0, 6.0, 8.0):
        raise ValueError("Unsupported subtitle limits")

    cues = []
    for seg in _segments(result):
        speaker = _speaker(seg)
        current: list[TimedWord] = []

        def flush() -> None:
            if not current:
                return
            start = current[0].start
            end = min(max(start + 0.08, current[-1].end), start + max_duration)
            cues.append((start, end, _lines(current, speaker, include_speakers, chars_per_line)))
            current.clear()

        for word in _words(seg):
            trial = current + [word]
            too_long = bool(current and word.end - current[0].start > max_duration)
            too_wide = bool(current and len(_lines(trial, speaker, include_speakers, chars_per_line)) > max_lines)
            pause = bool(current and word.start - current[-1].end > 0.8)
            if too_long or too_wide or pause:
                flush()
            current.append(word)
            if len(_lines(current, speaker, include_speakers, chars_per_line)) > max_lines:
                flush()
            elif word.text.endswith((".", "!", "?")) and word.end - current[0].start >= 2.0:
                flush()
        flush()

    blocks = [f"{n}\n{_stamp(start)} --> {_stamp(end)}\n" + "\n".join(lines)
              for n, (start, end, lines) in enumerate(cues, 1)]
    dest = output_dir / f"{audio_path.stem}.srt"
    dest.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return dest
