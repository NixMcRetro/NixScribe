"""Formatting WhisperX-aligned TXT and SRT exports, without importing ML packages."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import math
import textwrap
from typing import Any


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float
    speaker: str = "UNKNOWN"


def _segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(result.get("segments", []), key=lambda s: float(s.get("start") or 0.0))


def _speaker(segment: dict[str, Any]) -> str:
    return str(segment.get("speaker") or "UNKNOWN")


def _readable_paragraphs(text: str) -> list[str]:
    """Break long speech into paragraphs without rewriting or omitting any words.

    Aim for several sentences per paragraph. Prefer sentence punctuation after
    45 words, softer clause boundaries after 70, and a hard limit of 90 words
    or roughly 600 characters for punctuation-free speech.
    """
    words = text.split()
    output: list[str] = []
    current: list[str] = []
    chars = 0

    for word in words:
        current.append(word)
        chars += len(word) + (1 if len(current) > 1 else 0)
        count = len(current)
        ending = word.rstrip('"\'’”)]}')
        sentence_end = ending.endswith((".", "!", "?"))
        clause_end = ending.endswith((",", ";", ":"))
        if (
            (count >= 45 and sentence_end)
            or (count >= 70 and clause_end)
            or count >= 90
            or chars >= 600
        ):
            output.append(" ".join(current))
            current = []
            chars = 0

    if current:
        output.append(" ".join(current))
    return output


def write_txt(result: dict[str, Any], audio_path: Path, output_dir: Path,
              *, group_turns: bool = True) -> Path:
    """Timestamp-free speaker paragraphs with natural readability breaks."""
    turns: list[tuple[str, str]] = []
    last_end: float | None = None
    for seg in speaker_segments(result):
        txt = " ".join(str(seg.get("text") or "").split())
        if not txt:
            continue
        speaker = _speaker(seg)
        seg_start = _safe_time(seg.get("start"), -1.0)
        long_pause = (
            last_end is not None and seg_start >= 0
            and seg_start - last_end >= 2.5
        )
        if group_turns and turns and turns[-1][0] == speaker and not long_pause:
            turns[-1] = (speaker, turns[-1][1] + " " + txt)
        else:
            turns.append((speaker, txt))
        last_end = _safe_time(seg.get("end"), -1.0)
        if last_end < 0:
            last_end = None

    paragraphs = [
        f"[{speaker}] {paragraph}"
        for speaker, text in turns
        for paragraph in _readable_paragraphs(text)
    ]
    body = "\n\n".join(paragraphs)
    dest = output_dir / f"{audio_path.stem}.txt"
    dest.write_text(body + ("\n" if body else ""), encoding="utf-8")
    return dest


def _safe_time(value: Any, fallback: float) -> float:
    try:
        number = float(value)
        return max(0.0, number) if math.isfinite(number) else fallback
    except (ValueError, TypeError):
        return fallback


def _token_key(token: str) -> str:
    # Match case/punctuation variants without altering the original ASR text.
    key = "".join(c for c in token.casefold() if c.isalnum())
    return key or token


def _words(segment: dict[str, Any]) -> list[TimedWord]:
    tokens = str(segment.get("text") or "").split()
    if not tokens:
        return []
    start = _safe_time(segment.get("start"), 0.0)
    end = max(start + 0.01, _safe_time(segment.get("end"), start + 2.0))
    aligned = segment.get("words") or []
    aligned = [w for w in aligned if isinstance(w, dict) and str(w.get("word") or "").strip()]
    # Exact normalized token matches only. Never substitute alignment text for ASR.
    mapping = {}
    matcher = SequenceMatcher(None, [_token_key(t) for t in tokens],
                              [_token_key(str(w["word"]).strip()) for w in aligned],
                              autojunk=False)
    for i, j, count in matcher.get_matching_blocks():
        for offset in range(count):
            mapping[i + offset] = aligned[j + offset]

    has_word_speakers = any(w.get("speaker") for w in aligned)
    speakers = []
    anchors = {}
    for i in range(len(tokens)):
        item = mapping.get(i, {})
        speaker = item.get("speaker") or ("UNKNOWN" if has_word_speakers else _speaker(segment))
        speakers.append(str(speaker))
        a = _safe_time(item.get("start"), -1.0)
        b = _safe_time(item.get("end"), -1.0)
        # Preserve valid chronological aligned intervals, including small
        # overlaps. Reject reversed or out-of-order timestamps.
        if a >= 0 and b >= a and a >= start and b <= end + 0.05:
            if not anchors or a >= anchors[next(reversed(anchors))][0]:
                anchors[i] = (a, b)

    # Interpolate only missing runs between valid anchors, rather than throwing
    # away every aligned timestamp because one word could not be aligned.
    output = []
    cursor = 0
    while cursor < len(tokens):
        if cursor in anchors:
            a, b = anchors[cursor]
            output.append(TimedWord(tokens[cursor], a, b, speakers[cursor]))
            cursor += 1
            continue
        first = cursor
        while cursor < len(tokens) and cursor not in anchors:
            cursor += 1
        left = output[-1].end if output else start
        right = anchors[cursor][0] if cursor in anchors else end
        left = min(left, right)
        weights = [max(1, len(token)) for token in tokens[first:cursor]]
        total = sum(weights)
        t = left
        for index, weight in enumerate(weights):
            finish = right if index == len(weights) - 1 else t + (right-left)*weight/total
            i = first + index
            output.append(TimedWord(tokens[i], t, finish, speakers[i]))
            t = finish
    return output


def speaker_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Export-ready turns: word speaker evidence takes precedence over segment labels.

    Unknown words stay unknown when word-level evidence exists but is missing.
    Original spelling, punctuation and words are preserved. Do not infer alternating
    speakers merely from punctuation or the requested speaker count.
    """
    turns = []
    for seg in _segments(result):
        current = []
        def flush():
            if current:
                turns.append({
                    "start": current[0].start,
                    "end": max(w.end for w in current),
                    "text": " ".join(w.text for w in current),
                    "speaker": current[0].speaker,
                    "words": [{"word": w.text, "start": w.start, "end": w.end,
                               "speaker": w.speaker} for w in current],
                })
                current.clear()
        for word in _words(seg):
            if current and word.speaker != current[-1].speaker:
                flush()
            current.append(word)
        flush()
    return turns


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


def _fragment_long_word(word: TimedWord, speaker: str, include_speakers: bool,
                        width: int) -> list[TimedWord]:
    """Split exceptionally long tokens without losing characters or word timing."""
    prefix = len(speaker) + 2 if include_speakers else 0
    available = max(1, width - prefix)
    if len(word.text) <= available:
        return [word]
    fragments = []
    for offset in range(0, len(word.text), available):
        ending = min(offset + available, len(word.text))
        start = word.start + (word.end - word.start) * offset / len(word.text)
        end = word.start + (word.end - word.start) * ending / len(word.text)
        fragments.append(TimedWord(word.text[offset:ending], start, end, word.speaker))
    return fragments


def write_srt(result: dict[str, Any], audio_path: Path, output_dir: Path,
              *, include_speakers: bool = True, chars_per_line: int = 42,
              max_lines: int = 2, max_duration: float = 6.0) -> Path:
    """Build compact, word-timed cues while preserving the entire original text."""
    if chars_per_line not in (32, 37, 42, 48):
        raise ValueError("Unsupported subtitle line width")
    if max_lines not in (1, 2) or max_duration not in (3.0, 4.0, 6.0, 8.0):
        raise ValueError("Unsupported subtitle limits")

    # Join nearby same-speaker turns even when the recognizer split a phrase
    # into separate ASR segments. Never join across a speaker change.
    groups: list[tuple[str, list[TimedWord]]] = []
    for seg in speaker_segments(result):
        speaker = _speaker(seg)
        words = [part for word in _words(seg)
                 for part in _fragment_long_word(word, speaker, include_speakers, chars_per_line)]
        if not words:
            continue
        if (groups and groups[-1][0] == speaker
                and words[0].start - groups[-1][1][-1].end <= 1.0):
            groups[-1][1].extend(words)
        else:
            groups.append((speaker, words))

    cues = []
    for speaker, words in groups:
        chunks: list[list[TimedWord]] = []
        current: list[TimedWord] = []

        def fits(trial):
            return (max(w.end for w in trial) - trial[0].start <= max_duration
                    and len(_lines(trial, speaker, include_speakers, chars_per_line)) <= max_lines)

        for word in words:
            if current and (not fits(current + [word]) or word.start - current[-1].end > 1.0):
                chunks.append(current)
                current = []
            current.append(word)
            if word.text.endswith((".", "!", "?")) and word.end - current[0].start >= 2.0:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        # Avoid orphaned short phrases at artificial cue boundaries when a
        # neighbouring cue can accommodate them without exceeding the limits.
        merged: list[list[TimedWord]] = []
        for chunk in chunks:
            if merged:
                prev = merged[-1]
                short = (chunk[-1].end - chunk[0].start < 1.0
                         or prev[-1].end - prev[0].start < 1.0)
                if short and chunk[0].start - prev[-1].end <= 1.0 and fits(prev + chunk):
                    merged[-1] = prev + chunk
                    continue
            merged.append(chunk)
        for chunk in merged:
            start = chunk[0].start
            end = min(max(start + 0.08, max(w.end for w in chunk)), start + max_duration)
            cues.append((start, end, _lines(chunk, speaker, include_speakers, chars_per_line)))

    # Prefer at least one second and roughly 20 spoken characters per second.
    # Extend only into known free time, never beyond the next cue, audio end,
    # configured maximum duration, or 750 ms of trailing padding.
    audio_end = _safe_time(result.get("audio_duration"), -1.0)
    adjusted = []
    for i, (start, end, lines) in enumerate(cues):
        following = cues[i + 1][0] if i + 1 < len(cues) else audio_end
        if following >= end:
            spoken = " ".join(lines)
            if include_speakers:
                spoken = spoken.split(": ", 1)[-1]
            preferred = max(1.0, len(spoken) / 20.0)
            end = max(end, min(start + preferred, start + max_duration,
                               end + 0.75, following))
        adjusted.append((start, end, lines))
    cues = adjusted

    blocks = [f"{n}\n{_stamp(start)} --> {_stamp(end)}\n" + "\n".join(lines)
              for n, (start, end, lines) in enumerate(cues, 1)]
    dest = output_dir / f"{audio_path.stem}.srt"
    dest.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return dest
