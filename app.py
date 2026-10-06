#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import gc
import getpass
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

APP_HOST = "127.0.0.1"
APP_PORT = int(os.environ.get("WHISPERX_UI_PORT", "8765"))
APP_NAME = "NixScribe"
KEYCHAIN_SERVICE = "com.nixmcretro.nixscribe.hf-token"

# PyInstaller exposes bundled resources through sys._MEIPASS. Keeping all resource
# lookups relative to this directory lets the exact same code run from source and
# from the double-clickable macOS .app.
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
BUNDLED_BIN_DIR = RESOURCE_ROOT / "bin"
if BUNDLED_BIN_DIR.is_dir():
    os.environ["PATH"] = f"{BUNDLED_BIN_DIR}{os.pathsep}{os.environ.get('PATH', '')}"

# Reuse the standard Hugging Face cache instead of creating an app-specific copy.
# huggingface_hub (and current oMLX) resolve caches in this order:
# HF_HUB_CACHE -> HF_HOME/hub -> ~/.cache/huggingface/hub.
# We deliberately do not override those variables here, so existing compatible
# model downloads are automatically shared. PyTorch likewise keeps its normal
# cache unless TORCH_HOME has already been configured by the user.

def _huggingface_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _torch_cache() -> Path:
    if os.environ.get("TORCH_HOME"):
        return Path(os.environ["TORCH_HOME"]).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "torch"
    return Path.home() / ".cache" / "torch"

DEFAULT_WORKDIR = Path(
    os.environ.get("WHISPERX_WORKDIR", str(Path.home() / "Music"))
).expanduser()
DEFAULT_PROJECTS_DIR = Path(
    os.environ.get(
        "WHISPERX_PROJECTS_DIR",
        str(Path.home() / "Documents" / "NixScribe Transcriptions"),
    )
).expanduser()
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".aiff", ".aif",
    ".ogg", ".opus", ".mp4", ".mov", ".mkv", ".webm",
}
MIN_WHISPERX = (3, 8, 4)

app = Flask(__name__, template_folder=str(RESOURCE_ROOT / "templates"))
state_lock = threading.RLock()

STATE: dict[str, Any] = {
    "running": False,
    "cancel_requested": False,
    "status": "Idle",
    "stage": "Idle",
    "stage_percent": 0.0,
    "overall_percent": 0.0,
    "detail": "Choose an audio file to begin.",
    "started_at": None,
    "elapsed_seconds": 0.0,
    "input_file": None,
    "projects_dir": str(DEFAULT_PROJECTS_DIR),
    "project_dir": None,
    "files": [],
    "preview": "",
    "live_preview": "",
    "live_position": "Waiting for transcription to start.",
    "live_segment_count": 0,
    "log": [],
    "error": None,
    "detected_speakers": [],
    "settings": {},
}


class JobCancelled(Exception):
    pass


def _keychain_account() -> str:
    return os.environ.get("USER") or getpass.getuser() or "whisperx-user"


def _get_hf_token() -> str | None:
    """Return a Hugging Face token without ever exposing it to the web UI."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token.strip()
    if sys.platform != "darwin":
        return None
    try:
        proc = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-a",
                _keychain_account(),
                "-s",
                KEYCHAIN_SERVICE,
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return None


def _save_hf_token(token: str) -> None:
    if sys.platform != "darwin":
        raise RuntimeError("Keychain storage is available only on macOS.")
    token = token.strip()
    if not token:
        raise ValueError("Token cannot be empty.")
    proc = subprocess.run(
        [
            "/usr/bin/security",
            "add-generic-password",
            "-U",
            "-a",
            _keychain_account(),
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
            token,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "Could not save the token to Keychain.")


def _clear_hf_token() -> None:
    if sys.platform != "darwin":
        return
    subprocess.run(
        [
            "/usr/bin/security",
            "delete-generic-password",
            "-a",
            _keychain_account(),
            "-s",
            KEYCHAIN_SERVICE,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )


def _parse_version_tuple(value: str) -> tuple[int, int, int]:
    nums = [int(x) for x in re.findall(r"\d+", value)[:3]]
    nums += [0] * (3 - len(nums))
    return tuple(nums[:3])  # type: ignore[return-value]


def whisperx_version() -> str | None:
    try:
        return importlib.metadata.version("whisperx")
    except importlib.metadata.PackageNotFoundError:
        return None


def _append_log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    with state_lock:
        STATE["log"].append(f"[{stamp}] {message}")
        STATE["log"] = STATE["log"][-250:]


def _update(**kwargs: Any) -> None:
    with state_lock:
        STATE.update(kwargs)
        if STATE.get("started_at"):
            STATE["elapsed_seconds"] = max(0.0, time.time() - STATE["started_at"])


def _cancel_if_requested() -> None:
    with state_lock:
        if STATE.get("cancel_requested"):
            raise JobCancelled("Cancelled by user")


def stage_callback(stage: str, detail: str, start: float, end: float):
    def callback(percent: float) -> None:
        _cancel_if_requested()
        p = max(0.0, min(100.0, float(percent)))
        overall = start + (end - start) * (p / 100.0)
        _update(
            stage=stage,
            detail=detail,
            stage_percent=round(p, 1),
            overall_percent=round(overall, 1),
        )
    return callback


class LiveTranscriptCapture:
    """Capture WhisperX's verbose per-segment ASR output for the local UI.

    WhisperX 3.8.x exposes a percentage-only progress_callback. Its ASR path also
    prints each decoded VAD segment when verbose=True. Capturing those lines gives
    the UI a genuine live text view without altering WhisperX's decoding logic.
    """

    _TRANSCRIPT_RE = re.compile(
        r"^Transcript:\s*\[\s*([0-9.]+)\s*-->\s*([0-9.]+)\s*\]\s*(.*)$"
    )

    def __init__(self, passthrough: Any, max_lines: int = 30) -> None:
        self.passthrough = passthrough
        self.max_lines = max_lines
        self._buffer = ""
        self._lines: list[str] = []
        self._segment_count = 0

    def write(self, data: str) -> int:
        # Preserve normal terminal output when one exists.
        try:
            if self.passthrough is not None:
                self.passthrough.write(data)
                self.passthrough.flush()
        except Exception:
            pass

        self._buffer += str(data)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line.strip())
        return len(data)

    def flush(self) -> None:
        try:
            if self.passthrough is not None:
                self.passthrough.flush()
        except Exception:
            pass

    def _consume_line(self, line: str) -> None:
        match = self._TRANSCRIPT_RE.match(line)
        if not match:
            return
        start = float(match.group(1))
        end = float(match.group(2))
        text = match.group(3).strip()
        if not text:
            return

        self._segment_count += 1
        display = f"[{_fmt_ts(start)} --> {_fmt_ts(end)}] {text}"
        self._lines.append(display)
        self._lines = self._lines[-self.max_lines:]
        _update(
            live_preview="\n".join(self._lines),
            live_position=f"Currently decoding audio around {_fmt_ts(start)}–{_fmt_ts(end)}",
            live_segment_count=self._segment_count,
        )


def _raw_preview_from_segments(segments: list[dict[str, Any]], max_lines: int = 30) -> str:
    lines: list[str] = []
    for seg in sorted(segments, key=lambda item: float(item.get("start") or 0.0)):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"[{_fmt_ts(seg.get('start'))} --> {_fmt_ts(seg.get('end'))}] {text}")
    return "\n".join(lines[-max_lines:])


def _final_preview_from_segments(segments: list[dict[str, Any]], max_lines: int = 30) -> str:
    lines: list[str] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lines.append(
            f"[{_fmt_ts(seg.get('start'))} --> {_fmt_ts(seg.get('end'))}] "
            f"{_speaker_label(seg)}: {text}"
        )
    return "\n".join(lines[-max_lines:])


def _fmt_ts(seconds: float | None) -> str:
    if seconds is None:
        return "??:??:??.???"
    ms_total = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _speaker_label(seg: dict[str, Any]) -> str:
    return str(seg.get("speaker") or "SPEAKER_00")


def _write_txt(result: dict[str, Any], audio_path: Path, output_dir: Path) -> Path:
    """Write a readable line-by-line transcript with timestamps and speaker labels."""
    output_path = output_dir / f"{audio_path.stem}.txt"
    lines: list[str] = []
    for seg in result.get("segments", []):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        speaker = _speaker_label(seg)
        start = _fmt_ts(seg.get("start"))
        end = _fmt_ts(seg.get("end"))
        lines.append(f"[{start} --> {end}] {speaker}: {text}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def _fmt_srt_ts(seconds: float | None) -> str:
    if seconds is None:
        seconds = 0.0
    ms_total = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _write_srt(result: dict[str, Any], audio_path: Path, output_dir: Path) -> Path:
    """Write an SRT subtitle file with a speaker label in every cue."""
    output_path = output_dir / f"{audio_path.stem}.srt"
    blocks: list[str] = []
    cue = 1
    for seg in result.get("segments", []):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        speaker = _speaker_label(seg)
        start = _fmt_srt_ts(seg.get("start"))
        end = _fmt_srt_ts(seg.get("end"))
        blocks.append(f"{cue}\n{start} --> {end}\n{speaker}: {text}")
        cue += 1
    output_path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return output_path


def _torch_cleanup(torch_module: Any = None) -> None:
    gc.collect()
    if torch_module is None:
        return
    try:
        if hasattr(torch_module, "mps") and torch_module.backends.mps.is_available():
            torch_module.mps.empty_cache()
    except Exception:
        pass
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
    except Exception:
        pass


def _safe_project_name(stem: str) -> str:
    """Return a Finder-friendly project folder name."""
    name = re.sub(r"[\\/:*?\"<>|]", "-", stem).strip().strip(".")
    name = re.sub(r"\s+", " ", name)
    return name or "Transcription"


def _unique_project_dir(projects_dir: Path, stem: str) -> Path:
    base = _safe_project_name(stem)
    candidate = projects_dir / base
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = projects_dir / f"{base} ({n})"
        if not candidate.exists():
            return candidate
        n += 1


def _copy_source_with_progress(source: Path, destination: Path) -> None:
    total = max(1, source.stat().st_size)
    copied = 0
    chunk_size = 8 * 1024 * 1024
    try:
        with source.open("rb") as src, destination.open("wb") as dst:
            while True:
                _cancel_if_requested()
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                dst.write(chunk)
                copied += len(chunk)
                p = min(100.0, copied * 100.0 / total)
                _update(
                    stage="Creating project",
                    detail="Copying the original recording into the project folder…",
                    stage_percent=round(p, 1),
                    overall_percent=round(0.5 + 1.5 * p / 100.0, 1),
                )
        shutil.copystat(source, destination)
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _run_job(config: dict[str, Any]) -> None:
    caffeinate_proc: subprocess.Popen[Any] | None = None
    project_dir: Path | None = None
    try:
        import torch
        import whisperx
        from whisperx.diarize import DiarizationPipeline

        wx_ver = whisperx_version()
        if wx_ver is None or _parse_version_tuple(wx_ver) < MIN_WHISPERX:
            raise RuntimeError(
                "This UI requires WhisperX 3.8.4 or newer because it uses the native "
                "progress callbacks added in that release. Run ./setup.sh or upgrade WhisperX."
            )

        input_path = Path(config["input_file"]).expanduser().resolve()
        projects_dir = Path(config["projects_dir"]).expanduser().resolve()

        if not input_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        projects_dir.mkdir(parents=True, exist_ok=True)
        project_dir = _unique_project_dir(projects_dir, input_path.stem)
        project_dir.mkdir(parents=False, exist_ok=False)
        _update(projects_dir=str(projects_dir), project_dir=str(project_dir))
        _append_log(f"Project: {project_dir}")

        working_input = input_path
        if bool(config.get("copy_original", True)):
            copied_input = project_dir / input_path.name
            _copy_source_with_progress(input_path, copied_input)
            working_input = copied_input
            _append_log(f"Original recording copied into project: {copied_input.name}")
        else:
            _append_log("Original recording left in place; project will contain transcript files only")

        hf_token = _get_hf_token()
        if not hf_token:
            raise RuntimeError(
                "No Hugging Face token is configured. Save one in the app settings (or set "
                "HF_TOKEN when running from source) for pyannote speaker diarization."
            )

        # Keep the Mac awake only while a transcription job is running.
        try:
            caffeinate_proc = subprocess.Popen(
                ["/usr/bin/caffeinate", "-i"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _append_log("caffeinate enabled for this job")
        except Exception:
            _append_log("Could not start caffeinate; continuing anyway")

        _update(
            status="Running",
            stage="Loading audio",
            detail="Decoding audio with ffmpeg and loading Whisper large-v3…",
            stage_percent=0.0,
            overall_percent=2.0,
            error=None,
        )
        _append_log(f"Input: {input_path}")
        _append_log(f"Projects location: {projects_dir}")
        _append_log(f"Hugging Face cache: {_huggingface_hub_cache()}")
        _append_log(f"PyTorch cache: {_torch_cache()}")
        _append_log(
            f"Profile: model={config['model']}, compute={config['compute_type']}, "
            f"batch={config['batch_size']}, speakers={config['speaker_count'] or 'Auto'}"
        )
        _cancel_if_requested()

        audio = whisperx.load_audio(str(working_input))
        _update(overall_percent=3.0, stage_percent=100.0)
        _cancel_if_requested()

        asr_options = {
            "beam_size": 5,
            "best_of": 5,
            "patience": 1.0,
            "length_penalty": 1.0,
            "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": False,
            "initial_prompt": config.get("initial_prompt") or None,
            "hotwords": config.get("hotwords") or None,
            "suppress_tokens": [-1],
            "suppress_numerals": False,
        }

        _update(
            stage="Loading model",
            detail=f"Loading {config['model']} on CPU ({config['compute_type']})…",
            stage_percent=0.0,
            overall_percent=4.0,
        )
        _append_log("Loading Whisper ASR model")

        model = whisperx.load_model(
            config["model"],
            device="cpu",
            compute_type=config["compute_type"],
            language="en",
            task="transcribe",
            asr_options=asr_options,
            vad_method="pyannote",
            vad_options={"chunk_size": 30, "vad_onset": 0.500, "vad_offset": 0.363},
            threads=int(config.get("threads", 0)) or 4,
            use_auth_token=hf_token,
        )
        _update(stage_percent=100.0, overall_percent=7.0)
        _cancel_if_requested()

        _append_log("Transcription started")
        transcribe_cb = stage_callback(
            "Transcribing",
            "Whisper large-v3 is decoding speech…",
            7.0,
            66.0,
        )
        live_capture = LiveTranscriptCapture(sys.stdout, max_lines=30)
        with contextlib.redirect_stdout(live_capture):
            result = model.transcribe(
                audio,
                batch_size=int(config["batch_size"]),
                chunk_size=30,
                verbose=True,
                progress_callback=transcribe_cb,
                interleaved_context=bool(config.get("interleaved_context", True)),
            )
        _update(
            live_preview=_raw_preview_from_segments(result.get("segments", [])),
            live_position="Transcription pass complete — refining timestamps…",
            live_segment_count=len(result.get("segments", [])),
        )
        _append_log(f"Transcription finished: {len(result.get('segments', []))} ASR segments")
        del model
        _torch_cleanup(torch)
        _cancel_if_requested()

        _update(
            stage="Loading aligner",
            detail="Loading English forced-alignment model…",
            stage_percent=0.0,
            overall_percent=67.0,
        )
        align_model, metadata = whisperx.load_align_model(language_code="en", device="cpu")
        _update(stage_percent=100.0, overall_percent=69.0)
        _append_log("Forced alignment started")

        align_cb = stage_callback(
            "Aligning words",
            "Snapping words and sentences to precise timestamps…",
            69.0,
            82.0,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            "cpu",
            interpolate_method="nearest",
            return_char_alignments=False,
            progress_callback=align_cb,
        )
        _append_log(f"Alignment finished: {len(result.get('word_segments', []))} word timestamps")
        _update(
            live_preview=_raw_preview_from_segments(result.get("segments", [])),
            live_position="Timestamps aligned — assigning speakers…",
        )
        del align_model
        _torch_cleanup(torch)
        _cancel_if_requested()

        speaker_count = int(config["speaker_count"]) if config.get("speaker_count") else None
        min_speakers = speaker_count
        max_speakers = speaker_count

        _update(
            stage="Loading diarization",
            detail="Loading pyannote Community-1 speaker diarization…",
            stage_percent=0.0,
            overall_percent=83.0,
        )
        diarize_model = DiarizationPipeline(
            model_name="pyannote/speaker-diarization-community-1",
            token=hf_token,
            device="cpu",
        )
        _update(stage_percent=2.0, overall_percent=84.0)
        _append_log("Speaker diarization started")

        diarize_cb = stage_callback(
            "Identifying speakers",
            "Detecting speaker turns and assigning speaker IDs…",
            84.0,
            97.0,
        )
        diarize_segments = diarize_model(
            str(working_input),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            progress_callback=diarize_cb,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
        _update(
            live_preview=_final_preview_from_segments(result.get("segments", [])),
            live_position="Speaker assignment complete — writing TXT and SRT…",
        )
        del diarize_model
        _torch_cleanup(torch)
        _cancel_if_requested()

        speakers = sorted(
            {
                seg.get("speaker")
                for seg in result.get("segments", [])
                if seg.get("speaker") is not None
            }
        )
        _append_log(f"Diarization finished: {len(speakers)} speaker label(s): {', '.join(speakers) or 'none'}")

        _update(
            stage="Writing files",
            detail="Writing timestamped speaker TXT and SRT…",
            stage_percent=10.0,
            overall_percent=97.3,
        )
        result["language"] = "en"
        txt_path = _write_txt(result, input_path, project_dir)
        _update(stage_percent=55.0, overall_percent=98.5)
        srt_path = _write_srt(result, input_path, project_dir)
        _update(stage_percent=85.0, overall_percent=99.2)

        output_files = [str(txt_path), str(srt_path)]
        preview = txt_path.read_text(encoding="utf-8", errors="replace")
        if len(preview) > 120_000:
            preview = preview[:120_000] + "\n\n[Preview truncated in UI; full file is on disk.]\n"

        _update(
            running=False,
            cancel_requested=False,
            status="Complete",
            stage="Complete",
            detail=f"Finished. Wrote {len(output_files)} files.",
            stage_percent=100.0,
            overall_percent=100.0,
            files=output_files,
            preview=preview,
            live_preview=_final_preview_from_segments(result.get("segments", [])),
            live_position="Complete — final speaker-labelled transcript is ready.",
            live_segment_count=len(result.get("segments", [])),
            detected_speakers=speakers,
            projects_dir=str(projects_dir),
            project_dir=str(project_dir),
        )
        _append_log(f"Transcript: {txt_path}")
        _append_log(f"Subtitles: {srt_path}")
        _append_log("Job complete")

    except JobCancelled:
        if project_dir is not None and project_dir.exists():
            try:
                shutil.rmtree(project_dir)
                _append_log("Removed incomplete project folder")
            except Exception:
                _append_log("Could not remove the incomplete project folder")
        _append_log("Job cancelled")
        _update(
            running=False,
            cancel_requested=False,
            status="Cancelled",
            stage="Cancelled",
            detail="The transcription was cancelled.",
            error=None,
        )
    except Exception as exc:
        if project_dir is not None and project_dir.exists():
            try:
                shutil.rmtree(project_dir)
                _append_log("Removed incomplete project folder after error")
            except Exception:
                _append_log("Could not remove the incomplete project folder after error")
        error_text = f"{type(exc).__name__}: {exc}"
        _append_log(error_text)
        _append_log(traceback.format_exc())
        _update(
            running=False,
            cancel_requested=False,
            status="Error",
            stage="Error",
            detail=error_text,
            error=error_text,
        )
    finally:
        if caffeinate_proc is not None:
            try:
                caffeinate_proc.terminate()
                caffeinate_proc.wait(timeout=2)
            except Exception:
                try:
                    caffeinate_proc.kill()
                except Exception:
                    pass
        _torch_cleanup()


def _scan_audio_files() -> list[str]:
    roots = [DEFAULT_WORKDIR, DEFAULT_WORKDIR / "wav"]
    seen: set[Path] = set()
    files: list[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            for p in root.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    rp = p.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        files.append(rp)
        except OSError:
            continue
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return [str(p) for p in files[:250]]


@app.get("/")
def index():
    wx = whisperx_version()
    return render_template(
        "index.html",
        default_workdir=str(DEFAULT_WORKDIR),
        default_projects_dir=str(DEFAULT_PROJECTS_DIR),
        whisperx_version=wx or "not installed",
        hf_token_set=bool(_get_hf_token()),
    )


@app.get("/api/status")
def status():
    with state_lock:
        data = dict(STATE)
        if data.get("started_at") and data.get("running"):
            data["elapsed_seconds"] = max(0.0, time.time() - float(data["started_at"]))
    data["whisperx_version"] = whisperx_version()
    data["hf_token_set"] = bool(_get_hf_token())
    return jsonify(data)


@app.post("/api/token")
def save_token():
    payload = request.get_json(force=True, silent=True) or {}
    action = str(payload.get("action", "save")).strip().lower()
    try:
        if action == "clear":
            _clear_hf_token()
            return jsonify({"ok": True, "hf_token_set": bool(_get_hf_token())})
        token = str(payload.get("token", "")).strip()
        if not token.startswith("hf_"):
            return jsonify({"ok": False, "error": "That does not look like a Hugging Face token (expected hf_…)."}), 400
        _save_hf_token(token)
        return jsonify({"ok": True, "hf_token_set": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/files")
def files():
    return jsonify({"files": _scan_audio_files()})


@app.post("/api/browse")
def browse():
    # Native macOS chooser, because a local browser cannot reveal an arbitrary full POSIX path.
    script = 'POSIX path of (choose file with prompt "Choose audio to transcribe")'
    try:
        proc = subprocess.run(
            ["/usr/bin/osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            return jsonify({"ok": False, "cancelled": True})
        chosen = proc.stdout.strip()
        return jsonify({"ok": True, "path": chosen})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/browse-projects")
def browse_projects():
    script = 'POSIX path of (choose folder with prompt "Choose where transcription projects should be saved")'
    try:
        proc = subprocess.run(
            ["/usr/bin/osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            return jsonify({"ok": False, "cancelled": True})
        chosen = proc.stdout.strip()
        return jsonify({"ok": True, "path": chosen})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/start")
def start_job():
    payload = request.get_json(force=True, silent=False) or {}
    with state_lock:
        if STATE.get("running"):
            return jsonify({"ok": False, "error": "A transcription is already running."}), 409

    input_file = str(payload.get("input_file", "")).strip()
    projects_dir = str(payload.get("projects_dir", str(DEFAULT_PROJECTS_DIR))).strip()
    if not input_file:
        return jsonify({"ok": False, "error": "Choose an audio file first."}), 400
    if not projects_dir:
        return jsonify({"ok": False, "error": "Choose where transcription projects should be saved."}), 400

    speaker_raw = str(payload.get("speaker_count", "2")).strip().lower()
    try:
        speaker_count = None if speaker_raw in {"", "auto"} else int(speaker_raw)
        if speaker_count is not None and not 1 <= speaker_count <= 10:
            raise ValueError
    except ValueError:
        return jsonify({"ok": False, "error": "Speaker count must be Auto or a number from 1 to 10."}), 400

    config = {
        "input_file": input_file,
        "projects_dir": projects_dir,
        "copy_original": bool(payload.get("copy_original", True)),
        "model": str(payload.get("model", "large-v3")),
        "compute_type": str(payload.get("compute_type", "float32")),
        "batch_size": max(1, min(16, int(payload.get("batch_size", 4)))),
        "threads": max(0, min(32, int(payload.get("threads", 0)))),
        "speaker_count": speaker_count,
        "interleaved_context": bool(payload.get("interleaved_context", True)),
        "initial_prompt": str(payload.get("initial_prompt", "")).strip(),
        "hotwords": str(payload.get("hotwords", "")).strip(),
    }

    if config["model"] not in {"large-v3", "large-v3-turbo"}:
        return jsonify({"ok": False, "error": "Unsupported model selection."}), 400
    if config["compute_type"] not in {"float32", "int8"}:
        return jsonify({"ok": False, "error": "Unsupported compute type."}), 400

    _update(
        running=True,
        cancel_requested=False,
        status="Starting",
        stage="Starting",
        stage_percent=0.0,
        overall_percent=0.0,
        detail="Preparing transcription job…",
        started_at=time.time(),
        elapsed_seconds=0.0,
        input_file=input_file,
        projects_dir=projects_dir,
        project_dir=None,
        files=[],
        preview="",
        live_preview="",
        live_position="Preparing transcription…",
        live_segment_count=0,
        log=[],
        error=None,
        detected_speakers=[],
        settings=config,
    )
    threading.Thread(target=_run_job, args=(config,), daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/cancel")
def cancel_job():
    with state_lock:
        if not STATE.get("running"):
            return jsonify({"ok": False, "error": "No active job."}), 409
        STATE["cancel_requested"] = True
        STATE["detail"] = "Cancellation requested; stopping at the next safe progress checkpoint…"
    return jsonify({"ok": True})


@app.post("/api/open-output")
def open_output():
    payload = request.get_json(force=True, silent=True) or {}
    path = Path(str(payload.get("path") or STATE.get("project_dir") or STATE.get("projects_dir") or DEFAULT_PROJECTS_DIR)).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.Popen(["/usr/bin/open", str(path)])
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/health")
def health():
    wx = whisperx_version()
    issues: list[str] = []
    if wx is None:
        issues.append("WhisperX is not installed in this Python environment.")
    elif _parse_version_tuple(wx) < MIN_WHISPERX:
        issues.append(f"WhisperX {wx} is too old; 3.8.4+ is required for progress callbacks.")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
    except Exception:
        issues.append("ffmpeg is not available in PATH.")
    if not _get_hf_token():
        issues.append("No Hugging Face token is configured; diarization will not work.")
    return jsonify({"ok": not issues, "issues": issues, "whisperx_version": wx})


def _open_browser() -> None:
    try:
        webbrowser.open(f"http://{APP_HOST}:{APP_PORT}")
    except Exception:
        pass


if __name__ == "__main__":
    print(f"{APP_NAME}: http://{APP_HOST}:{APP_PORT}")
    print(f"Work directory: {DEFAULT_WORKDIR}")
    if os.environ.get("WHISPERX_UI_NO_BROWSER") != "1":
        threading.Timer(0.8, _open_browser).start()
    app.run(host=APP_HOST, port=APP_PORT, debug=False, threaded=True, use_reloader=False)
