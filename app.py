#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import gc
import getpass
import importlib.metadata
import json
import math
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
from datetime import date
from typing import Any, Callable

# Set offline policy before importing any ML dependencies in research mode.
if __name__ == "__main__" and sys.argv[1:2] == ["--clustering-probe"]:
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      PYANNOTE_METRICS_ENABLED="0", PYTORCH_ENABLE_MPS_FALLBACK="0")

from flask import Flask, jsonify, render_template, request, send_file, abort
from app_config import (DEFAULT_MODEL, MODEL_IDS, MODELS, MODEL_REPOSITORIES, VERSION,
                        ALL_MODEL_IDS, MLX_MODELS, MLX_REPOSITORIES,
                        MLX_CPU_FALLBACK, preferred_model, apple_silicon)
from mlx_backend import transcribe_mlx
from model_cache import prepare_model, repository_for
from prompt_handling import (PromptError, prepare_prompt, require_prompt,
                             prompt_fields, prompt_metadata)
from audio_export import export_mp3, mp3_destination
from activity_log import export_activity_log
from app_config import MODEL_DESCRIPTIONS
from project_paths import create_project_dir
from diarization_backend import (DEVICES as DIARIZATION_DEVICES, default_device,
                                 run_coreml, run_mps, fallback_devices, validate_intervals)
from speaker_reconciliation import apply_reconciliation
from diarization_diagnostics import write_diagnostics

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
        str(Path.home() / "Documents" / "NixScribe"),
    )
).expanduser()
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".aiff", ".aif",
    ".ogg", ".opus",
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv",
    ".mpeg", ".mpg", ".ts", ".mts", ".m2ts", ".3gp",
}
MIN_WHISPERX = (3, 8, 4)

app = Flask(__name__, template_folder=str(RESOURCE_ROOT / "templates"))
state_lock = threading.RLock()
NATIVE_QUIT: Callable[[], None] | None = None

STATE: dict[str, Any] = {
    "running": False,
    "cancel_requested": False,
    "status": "Idle",
    "stage": "Idle",
    "stage_percent": 0.0,
    "overall_percent": 0.0,
    "detail": "Choose an audio or video file to begin.",
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
    "recognition": None,
    "diarization_actual": None,
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
        # Keep the complete current job until another job is started.


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
    return str(seg.get("speaker") or "UNKNOWN")


def _write_txt(
    result: dict[str, Any], audio_path: Path, output_dir: Path, *,
    group_turns: bool = True,
) -> Path:
    from transcript_format import write_txt
    return write_txt(result, audio_path, output_dir, group_turns=group_turns)

def _fmt_srt_ts(seconds: float | None) -> str:
    if seconds is None:
        seconds = 0.0
    ms_total = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _write_srt(
    result: dict[str, Any], audio_path: Path, output_dir: Path, *,
    include_speakers: bool = True,
    chars_per_line: int = 42,
    max_lines: int = 2,
    max_duration: float = 6.0,
) -> Path:
    from transcript_format import write_srt
    return write_srt(
        result, audio_path, output_dir,
        include_speakers=include_speakers,
        chars_per_line=chars_per_line,
        max_lines=max_lines,
        max_duration=max_duration,
    )

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


def _probe_media_duration(path: Path) -> float | None:
    """Return container duration in seconds when ffprobe can determine it."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            duration = float(proc.stdout.strip())
            if duration > 0:
                return duration
    except Exception:
        pass
    return None


def _extract_audio_to_wav(source: Path, destination: Path) -> None:
    """Extract the first audio stream to a temporary 16 kHz mono PCM WAV."""
    duration = _probe_media_duration(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        "-nostats",
        str(destination),
    ]

    _update(
        stage="Preparing audio",
        detail="Extracting the audio stream to a temporary 16 kHz mono WAV…",
        stage_percent=0.0,
        overall_percent=2.0,
    )
    _append_log(f"Preparing working WAV from: {source.name}")

    proc: subprocess.Popen[str] | None = None
    recent_output: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            _cancel_if_requested()
            line = raw_line.strip()
            if not line:
                continue
            recent_output.append(line)
            recent_output = recent_output[-20:]

            if line.startswith("out_time_us=") and duration:
                try:
                    seconds = int(line.split("=", 1)[1]) / 1_000_000.0
                    percent = max(0.0, min(99.0, seconds * 100.0 / duration))
                    _update(
                        stage_percent=round(percent, 1),
                        overall_percent=round(2.0 + 3.0 * percent / 100.0, 1),
                    )
                except (TypeError, ValueError):
                    pass

        returncode = proc.wait()
        if returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
            detail = "\n".join(recent_output[-8:])
            raise RuntimeError(
                "ffmpeg could not extract a usable audio stream from this file."
                + (f"\n{detail}" if detail else "")
            )

        _update(stage_percent=100.0, overall_percent=5.0)
        _append_log(f"Working WAV ready: {destination.name}")
    except JobCancelled:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        destination.unlink(missing_ok=True)
        raise
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _run_job(config: dict[str, Any]) -> None:
    caffeinate_proc: subprocess.Popen[Any] | None = None
    project_dir: Path | None = None
    work_dir: Path | None = None
    exports_complete = False
    final_state: dict[str, Any] = {}
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
            raise FileNotFoundError(f"Media file not found: {input_path}")

        projects_dir.mkdir(parents=True, exist_ok=True)
        # Reserve a dated folder; never reuse an existing project.
        project_dir = create_project_dir(projects_dir, input_path.stem,
                                         job_date=date.fromisoformat(config["job_date"]))
        _update(projects_dir=str(projects_dir), project_dir=str(project_dir))
        _append_log(f"Project: {project_dir}")

        source_for_job = input_path
        if bool(config.get("copy_original", False)):
            copied_input = project_dir / input_path.name
            _copy_source_with_progress(input_path, copied_input)
            source_for_job = copied_input
            _append_log(f"Original media copied into project: {copied_input.name}")
        else:
            _append_log("Original media left in place")

        speaker_count = int(config["speaker_count"]) if config.get("speaker_count") else None

        hf_token = _get_hf_token()
        if speaker_count != 1 and not hf_token:
            raise RuntimeError(
                "No Hugging Face token is configured. Save one in the app settings (or set "
                "HF_TOKEN when running from source) for multi-speaker pyannote diarization."
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

        work_dir = project_dir / ".nixscribe-work"
        working_input = work_dir / "audio-16k-mono.wav"
        _extract_audio_to_wav(source_for_job, working_input)
        mp3_path = None
        if config.get("save_mp3", True):
            _update(stage="Saving audio", detail="Encoding original audio to 128 kbps MP3…",
                    stage_percent=0.0, overall_percent=5.0)
            mp3_path = export_mp3(source_for_job, mp3_destination(project_dir, input_path.stem),
                                  _cancel_if_requested)
            _append_log(f"MP3 audio saved: {mp3_path.name}")
            _update(stage_percent=100.0, overall_percent=5.0)

        _update(
            status="Running",
            stage="Loading audio",
            detail="Loading the prepared 16 kHz mono WAV…",
            stage_percent=0.0,
            overall_percent=5.0,
            error=None,
        )
        _append_log(f"Input: {input_path}")
        _append_log(f"Projects location: {projects_dir}")
        _append_log(f"Hugging Face cache: {_huggingface_hub_cache()}")
        _append_log(f"PyTorch cache: {_torch_cache()}")
        _append_log(
            f"Profile: model={config['model']}, "
            f"recognition_precision={'FP16' if config['model'] in MLX_REPOSITORIES else config['compute_type']}, "
            f"batch={config['batch_size']}, speakers={config['speaker_count'] or 'Auto'}, "
            f"diarization={config.get('diarization_device', 'cpu')}"
        )
        _cancel_if_requested()

        audio = whisperx.load_audio(str(working_input))
        _update(overall_percent=6.0, stage_percent=100.0)
        _cancel_if_requested()

        requested_model = config["model"]
        model_id = requested_model
        repository = repository_for(model_id)

        def model_stage(stage: str) -> None:
            if stage == "checking":
                _update(stage="Checking model", detail=f"Checking whether {model_id} is already saved…",
                        stage_percent=0.0, overall_percent=6.0)
            elif stage == "downloading":
                _update(stage="Downloading model",
                        detail=f"Downloading {model_id} from Hugging Face. First use can take a while.",
                        stage_percent=0.0, overall_percent=6.0,
                        live_position="Downloading the speech recognition model. No audio is being uploaded.")
                _append_log(f"Model files missing; downloading {repository} to the shared Hugging Face cache")
            elif stage == "cached":
                _append_log(f"Model already downloaded: {repository}")
            elif stage == "downloaded":
                _append_log(f"Model download complete: {repository}")

        def prepare_selected_model():
            _append_log(f"Checking local model cache: {repository}")
            path = prepare_model(model_id, work_dir, model_stage, _cancel_if_requested)
            _append_log(f"Model snapshot: {path}")
            _cancel_if_requested()
            return path

        def prompt_for_attempt():
            plan = require_prompt(prepare_prompt(
                model_id, config["initial_prompt"], config["hotwords"],
                snapshot=model_path))
            _append_log("Recognition prompt: " + json.dumps(prompt_metadata(plan), sort_keys=True)
                        + "; condition_on_previous_text=False")
            return plan

        is_mlx = model_id in MLX_REPOSITORIES
        if is_mlx:
            attempt_started = time.monotonic()
            try:
                model_path = prepare_selected_model()
                prompt_plan = prompt_for_attempt()
                asr_started = time.monotonic()
                _update(stage="GPU recognition", detail="Loading the cached MLX model and transcribing on Apple GPU…",
                        stage_percent=0.0, overall_percent=8.0,
                        live_position="MLX GPU decoding; transcript will appear when this pass completes.")
                _append_log(f"MLX Apple GPU model: {MLX_REPOSITORIES[model_id]} (configured FP16)")
                result = transcribe_mlx(
                    working_input, model_id, model_path, len(audio) / 16000.0,
                    prompt_plan["initial_prompt"], _cancel_if_requested,
                )
                _update(stage_percent=100.0, overall_percent=66.0)
            except (JobCancelled, PromptError):
                raise
            except Exception as exc:
                _cancel_if_requested()
                _append_log(f"MLX attempt failed after {time.monotonic() - attempt_started:.1f}s: "
                            f"{type(exc).__name__}: {exc}")
                model_id = MLX_CPU_FALLBACK[requested_model]
                repository = repository_for(model_id)
                _append_log(f"Retrying speech recognition with {model_id} (CPU, float32)")
                is_mlx = False
                _torch_cleanup(torch)

        if not is_mlx:
            model_path = prepare_selected_model()
            prompt_plan = prompt_for_attempt()
            asr_started = time.monotonic()
            compute_type = "float32" if requested_model in MLX_REPOSITORIES else config["compute_type"]
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
                "initial_prompt": prompt_plan["initial_prompt"] or None,
                "hotwords": prompt_plan["hotwords"] or None,
                "suppress_tokens": [-1],
                "suppress_numerals": False,
            }

            _update(
                stage="Loading model",
                detail=f"Loading {model_id} on CPU ({compute_type})…",
                stage_percent=0.0,
                overall_percent=6.5,
            )
            _append_log("Loading Whisper ASR model")

            model = whisperx.load_model(
                str(model_path),
                local_files_only=True,
                device="cpu",
                compute_type=compute_type,
                language="en",
                task="transcribe",
                asr_options=asr_options,
                vad_method="pyannote",
                vad_options={"chunk_size": 30, "vad_onset": 0.500, "vad_offset": 0.363},
                threads=int(config.get("threads", 0)) or 4,
                use_auth_token=hf_token,
            )
            _update(stage_percent=100.0, overall_percent=8.0)
            _cancel_if_requested()

            _append_log("Speech recognition started - Faster-Whisper (CPU)")
            transcribe_cb = stage_callback(
                "Transcribing",
                f"Whisper {model_id} is decoding speech…",
                8.0,
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
                )
        _update(
            live_preview=_raw_preview_from_segments(result.get("segments", [])),
            live_position="Transcription pass complete — refining timestamps…",
            live_segment_count=len(result.get("segments", [])),
        )
        _append_log(f"Transcription finished: {len(result.get('segments', []))} ASR segments")
        asr_elapsed = time.monotonic() - asr_started
        audio_seconds = len(audio) / 16000.0
        _append_log(f"ASR elapsed: {asr_elapsed:.1f}s for {audio_seconds:.1f}s audio "
                    f"(real-time factor {asr_elapsed / max(audio_seconds, 0.001):.3f}; includes model loading)")
        recognition = {"requested": requested_model, "actual": model_id, "repository": repository,
                       "engine": "mlx" if is_mlx else "faster-whisper",
                       "device": "apple-gpu" if is_mlx else "cpu",
                       "configured_precision": "float16" if is_mlx else compute_type,
                       "fallback": model_id != requested_model,
                       "prompt": prompt_metadata(prompt_plan)}
        _update(recognition=recognition)
        _append_log(f"Recognition completed: model={model_id}, requested={requested_model}, "
                    f"engine={recognition['engine']}, device={recognition['device']}, "
                    f"configured_precision={recognition['configured_precision']}, "
                    f"fallback={'yes' if recognition['fallback'] else 'no'}")
        if not is_mlx:
            del model
        _torch_cleanup(torch)
        _cancel_if_requested()

        align_started = time.monotonic()
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

        align_elapsed = time.monotonic() - align_started
        requested_device = config.get("diarization_device", default_device())
        combined = requested_device == "combined"
        intelligent = requested_device == "intelligent-auto-v5"
        dual = combined or intelligent
        auto_result = None
        actual_device = "single"
        raw_intervals = {}
        attempts = []
        comparison = None
        aligned_words = [word for seg in result.get("segments", []) for word in seg.get("words", [])]
        diagnostics = config.get("save_diagnostics", False)
        if diagnostics:
            # Only transcript fields, never configuration, paths, tokens or exception text.
            def finite_time(value):
                try:
                    value = float(value)
                    return value if math.isfinite(value) else None
                except (ValueError, TypeError, OverflowError):
                    return None
            aligned_snapshot = [{"word": str(w.get("word", "")),
                                 "start": finite_time(w.get("start")),
                                 "end": finite_time(w.get("end"))} for w in aligned_words]
        diarize_total_started = time.monotonic()

        if speaker_count == 1:
            _update(
                stage="Assigning speaker",
                detail="Single-speaker mode: assigning SPEAKER_00 without diarization…",
                stage_percent=0.0,
                overall_percent=84.0,
            )
            for seg in result.get("segments", []):
                seg["speaker"] = "SPEAKER_00"
                for word in seg.get("words", []):
                    word["speaker"] = "SPEAKER_00"
            speakers = ["SPEAKER_00"]
            _update(stage_percent=100.0, overall_percent=97.0)
            _append_log("Single-speaker mode: pyannote diarization skipped")
        else:
            candidates = fallback_devices(requested_device)
            actual_device = requested_device
            diarize_started = time.monotonic()
            diarize_segments = None

            if "coreml" in candidates:
                _update(stage="Loading diarization",
                        detail="Checking FluidAudio Core ML models…",
                        stage_percent=0.0, overall_percent=83.0)
                _append_log("Speaker diarization requested - FluidAudio v0.18.4 (Core ML / all compute units)")

                def coreml_progress(phase: str, percent: float) -> None:
                    if phase == "preparing":
                        _update(stage="Loading diarization",
                                detail="Preparing Core ML models (first use may download weights)…",
                                stage_percent=0.0, overall_percent=83.0)
                        _append_log("Preparing FluidAudio models; cache miss or cached model load failed")
                    elif phase == "inference":
                        stage_callback("Identifying speakers",
                                       "Detecting speaker turns with Core ML…",
                                       84.0, 89.0 if dual else 97.0)(percent)

                try:
                    import pandas as pd
                    helper = BUNDLED_BIN_DIR / "NixScribeDiarizer"
                    if not getattr(sys, "frozen", False):
                        helper = RESOURCE_ROOT / "native-diarizer" / ".build" / "release" / "NixScribeDiarizer"
                    intervals = run_coreml(helper, working_input, speaker_count,
                                           coreml_progress, _cancel_if_requested)
                    if not intervals and result.get("segments"):
                        raise RuntimeError("Core ML returned no speaker intervals for recognized speech")
                    diarize_segments = pd.DataFrame(intervals, columns=["start", "end", "speaker"])
                    actual_device = "coreml"
                    if dual or diagnostics:
                        raw_intervals["coreml"] = intervals
                        attempts.append({"device": "coreml", "status": "success",
                                         "seconds": time.monotonic() - diarize_started})
                    _append_log("Experimental FluidAudio Core ML diarization completed")
                    if speaker_count is not None and len({r["speaker"] for r in intervals}) != speaker_count:
                        _append_log("Warning: FluidAudio did not return the requested exact speaker count")
                except JobCancelled:
                    raise
                except Exception as exc:
                    _cancel_if_requested()
                    if dual or diagnostics:
                        attempts.append({"device": "coreml", "status": "failed",
                                         "seconds": time.monotonic() - diarize_started,
                                         "error_type": type(exc).__name__})
                    _append_log(f"Core ML diarization failed: {type(exc).__name__}: {exc}")
                    _append_log(f"Core ML attempt elapsed: {time.monotonic() - diarize_started:.1f}s")
                    _append_log("Retrying speaker diarization with Pyannote on Apple GPU (MPS); Core ML output discarded")
                    actual_device = "mps"
                    diarize_started = time.monotonic()

            if (diarize_segments is None or dual) and "mps" in candidates:
                _cancel_if_requested()
                diarize_started = time.monotonic()
                _update(stage="Loading diarization",
                        detail="Loading Pyannote Community-1 on Apple GPU (experimental)…",
                        stage_percent=0.0, overall_percent=89.0 if dual else 83.0)
                _append_log("Speaker diarization attempt - Pyannote Community-1 (Apple GPU / MPS)")

                def mps_progress(phase: str, percent: float) -> None:
                    if phase == "loading":
                        return
                    stage_callback("Identifying speakers",
                                   "Detecting speaker turns on Apple GPU…",
                                   89.0 if dual else 84.0, 97.0)(percent)

                try:
                    import pandas as pd
                    # Send only word timings to the isolated experimental worker.
                    if intelligent and raw_intervals.get("coreml") and aligned_words:
                        def finite_or_none(value):
                            return value if isinstance(value, (int, float)) and math.isfinite(value) else None
                        inputs = {"words": [{"start": finite_or_none(word.get("start")),
                                             **({"end": finite_or_none(word["end"])} if "end" in word else {})}
                                            for word in aligned_words],
                                  "core": raw_intervals["coreml"], "duration": audio_seconds}
                        auto_result = run_mps(working_input, speaker_count, mps_progress,
                                              _cancel_if_requested, auto_inputs=inputs)
                        intervals = auto_result["intervals"]
                        if auto_result.get("labels") is None:
                            _append_log("Intelligent Auto v5 fell back to standard Pyannote: "
                                        + auto_result["fallback_reason"])
                        else:
                            _append_log(f"Intelligent Auto v5: {auto_result['selected']}; "
                                        f"{auto_result['acoustic_changes']} acoustic label changes; "
                                        "one segmentation and one embedding pass")
                    else:
                        if intelligent:
                            _append_log("Intelligent Auto v5 unavailable: missing Core ML intervals or aligned words; using Pyannote")
                        intervals = run_mps(working_input, speaker_count,
                                            mps_progress, _cancel_if_requested)
                    if not intervals and result.get("segments"):
                        raise RuntimeError("GPU diarization returned no speaker intervals for recognized speech")
                    diarize_segments = pd.DataFrame(intervals, columns=["start", "end", "speaker"])
                    actual_device = "combined" if combined and "coreml" in raw_intervals else "mps"
                    if intelligent and auto_result is not None and auto_result.get("labels") is not None:
                        actual_device = "intelligent-auto-v5"
                    if dual or diagnostics:
                        raw_intervals["mps"] = intervals
                        attempts.append({"device": "mps", "status": "success",
                                         "seconds": time.monotonic() - diarize_started})
                    _append_log("Experimental MPS diarization completed")
                except JobCancelled:
                    raise
                except Exception as exc:
                    _cancel_if_requested()
                    if dual or diagnostics:
                        attempts.append({"device": "mps", "status": "failed",
                                         "seconds": time.monotonic() - diarize_started,
                                         "error_type": type(exc).__name__})
                    _append_log(f"GPU diarization failed: {type(exc).__name__}: {exc}")
                    _append_log(f"MPS attempt elapsed: {time.monotonic() - diarize_started:.1f}s")
                    if diarize_segments is not None and not intelligent:
                        actual_device = "coreml"
                    else:
                        if intelligent:
                            diarize_segments = None
                            auto_result = None
                            _append_log("Intelligent Auto v5 unavailable; using standard Pyannote on CPU")
                        _append_log("Retrying speaker diarization on CPU; GPU output discarded")
                        actual_device = "cpu"
                        _update(stage="Loading diarization",
                                detail="Apple GPU unavailable or failed. Retrying on CPU…",
                                stage_percent=0.0, overall_percent=83.0)

            if diarize_segments is None:
                _cancel_if_requested()
                actual_device = "cpu"
                diarize_started = time.monotonic()
                _update(stage="Loading diarization",
                        detail="Loading Pyannote Community-1 on CPU…",
                        stage_percent=0.0, overall_percent=83.0)
                diarize_model = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-community-1",
                    token=hf_token,
                    device="cpu",
                )
                _update(stage_percent=2.0, overall_percent=84.0)
                _append_log("Speaker diarization started - Pyannote Community-1 (CPU)")
                diarize_cb = stage_callback(
                    "Identifying speakers",
                    "Detecting speaker turns on CPU…",
                    84.0, 97.0,
                )
                diarize_kwargs: dict[str, Any] = {
                    "progress_callback": diarize_cb,
                }
                if speaker_count is not None:
                    diarize_kwargs["num_speakers"] = speaker_count
                diarize_segments = diarize_model(str(working_input), **diarize_kwargs)
                del diarize_model
                if diagnostics:
                    raw_intervals["cpu"] = validate_intervals(
                        diarize_segments[["start", "end", "speaker"]].to_dict(orient="records"))
                    attempts.append({"device": "cpu", "status": "success",
                                     "seconds": time.monotonic() - diarize_started})
            diarize_elapsed = time.monotonic() - diarize_total_started
            _append_log(f"Diarization elapsed: {diarize_elapsed:.1f}s "
                        f"(device={actual_device}, requested={requested_device}; includes model loading)")
            if actual_device == "intelligent-auto-v5":
                from intelligent_auto import apply_labels
                _cancel_if_requested()
                apply_labels(result, auto_result["labels"])
            elif actual_device == "combined":
                comparison = apply_reconciliation(result, raw_intervals["coreml"],
                                                  raw_intervals["mps"], _cancel_if_requested)
                _append_log(f"Combined speaker assignment: {sum(d['uncertain'] for d in comparison['decisions'])} "
                            f"uncertain word decisions of {len(comparison['decisions'])}")
            else:
                if combined:
                    _append_log(f"Combined mode degraded to {actual_device}; only one engine available")
                if intelligent:
                    _append_log(f"Intelligent Auto v5 degraded to {actual_device}; standard speaker assignment")
                result = whisperx.assign_word_speakers(
                    diarize_segments,
                    result,
                    fill_nearest=True,
                )
            _torch_cleanup(torch)
            _cancel_if_requested()

            from transcript_format import speaker_segments
            speakers = sorted({
                seg["speaker"] for seg in speaker_segments(result)
                if seg["speaker"] != "UNKNOWN"
            })
            _append_log(
                f"Diarization finished: {len(speakers)} speaker label(s): "
                f"{', '.join(speakers) or 'none'}"
            )

        _update(diarization_actual={"requested": requested_device, "actual": actual_device,
                                    "fallback": actual_device not in (requested_device, "single")})
        from transcript_format import speaker_segments
        export_turns = speaker_segments(result)
        result["audio_duration"] = len(audio) / 16000.0
        _append_log(f"Export speaker turns: {len(export_turns)}; "
                    f"unknown-label turns: {sum(t['speaker'] == 'UNKNOWN' for t in export_turns)}")
        _update(
            live_preview=_final_preview_from_segments(export_turns),
            live_position="Speaker assignment complete; writing TXT and SRT…",
        )

        _update(
            stage="Writing files",
            detail="Writing speaker-labelled TXT and timed SRT…",
            stage_percent=10.0,
            overall_percent=97.3,
        )
        _cancel_if_requested()
        result["language"] = "en"
        txt_path = _write_txt(
            result, input_path, project_dir,
            group_turns=config["txt_group_turns"],
        )
        _update(stage_percent=55.0, overall_percent=98.5)
        _cancel_if_requested()
        srt_path = _write_srt(
            result, input_path, project_dir,
            include_speakers=config["srt_speaker_labels"],
            chars_per_line=config["srt_chars_per_line"],
            max_lines=config["srt_max_lines"],
            max_duration=config["srt_max_duration"],
        )
        exports_complete = True
        _update(stage_percent=85.0, overall_percent=99.2)

        output_files = [str(txt_path), str(srt_path)]
        if diagnostics:
            def package_version(name):
                try:
                    return importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    return None

            try:
                final_words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
                diagnostic_path = write_diagnostics(project_dir / "speaker-diagnostics.json", {
                    "app_version": VERSION,
                    "recognition": recognition,
                    "speaker_identification": {"requested": requested_device, "actual": actual_device,
                                               "requested_count": speaker_count,
                                               "degraded": dual and actual_device not in (requested_device, "single")},
                    "versions": {"whisperx": wx_ver, "torch": str(torch.__version__),
                                 "pyannote.audio": package_version("pyannote.audio"),
                                 "mlx-whisper": package_version("mlx-whisper"), "FluidAudio": "0.18.4"},
                    "models": {"pyannote": "pyannote/speaker-diarization-community-1",
                               "FluidAudio": "OfflineDiarizerModels (default fidelity)"},
                    "seconds": {"audio": audio_seconds, "asr": asr_elapsed, "alignment": align_elapsed,
                                "diarization": diarize_elapsed if speaker_count != 1 else 0.0},
                    "attempts": attempts, "aligned_words": aligned_snapshot,
                    "raw_intervals": raw_intervals, "reconciliation": comparison,
                    "intelligent_auto": ({k: v for k, v in auto_result.items()
                                          if k in ("selected", "acoustic_changes", "fallback_reason",
                                                   "segmentation_calls", "embedding_calls")}
                                         if auto_result is not None else None),
                    "assigned_speakers": [w.get("speaker", "UNKNOWN") for w in final_words],
                })
                output_files.append(str(diagnostic_path))
                _append_log("Private speaker diagnostics written to project folder")
            except Exception as exc:
                _append_log(f"Diagnostic export failed ({type(exc).__name__}); TXT and SRT retained")
        if mp3_path is not None:
            output_files.append(str(mp3_path))
        preview = txt_path.read_text(encoding="utf-8", errors="replace")
        if len(preview) > 120_000:
            preview = preview[:120_000] + "\n\n[Preview truncated in UI; full file is on disk.]\n"

        final_state = dict(
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
        if not exports_complete and project_dir is not None and project_dir.exists():
            try:
                shutil.rmtree(project_dir)
                _append_log("Removed incomplete project folder")
            except Exception:
                _append_log("Could not remove the incomplete project folder")
        _append_log("Job cancelled")
        final_state = dict(
            status="Cancelled",
            stage="Cancelled",
            detail="The transcription was cancelled.",
            error=None,
        )
    except Exception as exc:
        if not exports_complete and project_dir is not None and project_dir.exists():
            try:
                shutil.rmtree(project_dir)
                _append_log("Removed incomplete project folder after error")
            except Exception:
                _append_log("Could not remove the incomplete project folder after error")
        error_text = f"{type(exc).__name__}: {exc}"
        _append_log(error_text)
        _append_log(traceback.format_exc())
        final_state = dict(
            status="Error",
            stage="Error",
            detail=error_text,
            error=error_text,
        )
    finally:
        _remove_work_dir(work_dir)
        if caffeinate_proc is not None:
            try:
                caffeinate_proc.terminate()
                caffeinate_proc.wait(timeout=2)
            except Exception:
                try:
                    caffeinate_proc.kill()
                except Exception:
                    pass
        try:
            _torch_cleanup()
        finally:
            _update(running=False, cancel_requested=False, **final_state)


def _remove_work_dir(work_dir: Path | None) -> None:
    try:
        if work_dir is not None and work_dir.exists():
            shutil.rmtree(work_dir)
            _append_log("Temporary working WAV removed")
    except OSError:
        _append_log("Could not remove temporary working audio directory")


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


@app.get("/branding/icon.png")
def brand_icon():
    path = RESOURCE_ROOT / "assets" / "NixScribe.png"
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="image/png")


@app.get("/")
def index():
    wx = whisperx_version()
    return render_template(
        "index.html",
        default_workdir=str(DEFAULT_WORKDIR),
        default_projects_dir=str(DEFAULT_PROJECTS_DIR),
        models=MODELS,
        model_repositories=MODEL_REPOSITORIES,
        model_descriptions=MODEL_DESCRIPTIONS,
        mlx_models=MLX_MODELS,
        mlx_repositories=MLX_REPOSITORIES,
        default_model=preferred_model(),
        apple_silicon=apple_silicon(),
        app_version=VERSION,
        brand_icon=(RESOURCE_ROOT / "assets" / "NixScribe.png").is_file(),
        whisperx_version=wx or "not installed",
        hf_token_set=bool(_get_hf_token()),
        default_diarization_device=default_device(),
    )


@app.get("/api/status")
def status():
    with state_lock:
        data = dict(STATE)
        data["log"] = list(STATE["log"])
        if data.get("started_at") and data.get("running"):
            data["elapsed_seconds"] = max(0.0, time.time() - float(data["started_at"]))
    data["whisperx_version"] = whisperx_version()
    data["hf_token_set"] = bool(_get_hf_token())
    return jsonify(data)


@app.post("/api/log/copy")
def copy_activity_log():
    # Only accept a same-origin UI request; copy the current log, not arbitrary
    # client-provided clipboard contents. The clipboard stays on this Mac.
    if not request.is_json or request.headers.get("X-NixScribe-Action") != "copy-log":
        return jsonify({"ok": False, "error": "Invalid clipboard request."}), 400
    origin = request.headers.get("Origin")
    if origin and origin != request.host_url.rstrip("/"):
        return jsonify({"ok": False, "error": "Invalid clipboard origin."}), 403
    if sys.platform != "darwin":
        return jsonify({"ok": False, "error": "Native clipboard is only available on macOS."}), 400
    with state_lock:
        entries = list(STATE["log"])
    if not entries:
        return jsonify({"ok": False, "error": "No activity to copy yet."}), 400
    try:
        subprocess.run(["/usr/bin/pbcopy"], input="\n".join(entries) + "\n",
                       encoding="utf-8", capture_output=True, check=True, timeout=5)
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False, "error": "Could not copy to the macOS clipboard."}), 500


@app.post("/api/log/export")
def save_activity_log():
    # Export the current local log snapshot, never an arbitrary client path.
    if not request.is_json or request.headers.get("X-NixScribe-Action") != "export-log":
        return jsonify({"ok": False, "error": "Invalid log export request."}), 400
    origin = request.headers.get("Origin")
    if origin and origin != request.host_url.rstrip("/"):
        return jsonify({"ok": False, "error": "Invalid log export origin."}), 403
    if sys.platform != "darwin":
        return jsonify({"ok": False, "error": "Native log export requires macOS."}), 400
    with state_lock:
        entries = list(STATE["log"])
    if not entries:
        return jsonify({"ok": False, "error": "No activity to export yet."}), 400
    try:
        script = 'POSIX path of (choose folder with prompt "Save a private copy of the current activity log")'
        proc = subprocess.run(["/usr/bin/osascript", "-e", script],
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            return jsonify({"ok": False, "cancelled": True})
        path = export_activity_log(Path(proc.stdout.rstrip("\n")), entries)
        return jsonify({"ok": True, "path": str(path)})
    except Exception:
        return jsonify({"ok": False, "error": "Could not save the activity log."}), 500


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
    script = 'POSIX path of (choose file with prompt "Choose audio or video to transcribe")'
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


@app.post("/api/prompt-preview")
def prompt_preview():
    if not request.is_json or request.headers.get("X-NixScribe-Action") != "prompt-preview":
        return jsonify({"error": "Invalid prompt preview request."}), 400
    origin = request.headers.get("Origin")
    if origin and origin != request.host_url.rstrip("/"):
        return jsonify({"error": "Invalid prompt preview origin."}), 403
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid prompt preview payload."}), 400
    try:
        initial, vocabulary = prompt_fields(payload)
        plan = prepare_prompt(str(payload.get("model", preferred_model())), initial, vocabulary)
        return jsonify(plan)
    except PromptError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/start")
def start_job():
    payload = request.get_json(force=True, silent=False) or {}
    with state_lock:
        if STATE.get("running"):
            return jsonify({"ok": False, "error": "A transcription is already running."}), 409

    input_file = str(payload.get("input_file", "")).strip()
    projects_dir = str(payload.get("projects_dir", str(DEFAULT_PROJECTS_DIR))).strip()
    if not input_file:
        return jsonify({"ok": False, "error": "Choose an audio or video file first."}), 400
    if not projects_dir:
        return jsonify({"ok": False, "error": "Choose where transcription projects should be saved."}), 400

    speaker_raw = str(payload.get("speaker_count", "auto")).strip().lower()
    try:
        speaker_count = None if speaker_raw in {"", "auto"} else int(speaker_raw)
        if speaker_count is not None and not 1 <= speaker_count <= 10:
            raise ValueError
    except ValueError:
        return jsonify({"ok": False, "error": "Speaker count must be Auto or a number from 1 to 10."}), 400

    try:
        srt_width = int(payload.get("srt_chars_per_line", 42))
        srt_lines = int(payload.get("srt_max_lines", 2))
        srt_duration = float(payload.get("srt_max_duration", 6))
    except (ValueError, TypeError):
        return jsonify({"ok": False, "error": "Invalid subtitle formatting options."}), 400
    if (
        srt_width not in (32, 37, 42, 48)
        or srt_lines not in (1, 2)
        or srt_duration not in (3.0, 4.0, 6.0, 8.0)
    ):
        return jsonify({"ok": False, "error": "Unsupported subtitle formatting options."}), 400

    try:
        initial_prompt, hotwords = prompt_fields(payload)
    except PromptError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    config = {
        "input_file": input_file,
        "projects_dir": projects_dir,
        "job_date": date.today().isoformat(),
        "copy_original": payload.get("copy_original", False) is True,
        "save_mp3": payload.get("save_mp3", True) is True,
        "model": str(payload.get("model", preferred_model())),
        "compute_type": str(payload.get("compute_type", "float32")),
        "batch_size": max(1, min(16, int(payload.get("batch_size", 4)))),
        "threads": max(0, min(32, int(payload.get("threads", 0)))),
        "speaker_count": speaker_count,
        "diarization_device": str(payload.get("diarization_device", default_device())),
        "save_diagnostics": payload.get("save_diagnostics", False) is True,
        "txt_group_turns": payload.get("txt_group_turns", True) is True,
        "srt_speaker_labels": payload.get("srt_speaker_labels", False) is True,
        "srt_chars_per_line": srt_width,
        "srt_max_lines": srt_lines,
        "srt_max_duration": srt_duration,
        "initial_prompt": initial_prompt,
        "hotwords": hotwords,
    }

    if config["diarization_device"] not in DIARIZATION_DEVICES:
        return jsonify({"ok": False, "error": "Unsupported speaker identification device."}), 400
    if config["diarization_device"] in {"combined", "intelligent-auto-v5"} and not apple_silicon():
        return jsonify({"ok": False, "error": "Experimental dual-engine speaker identification requires Apple silicon."}), 400
    if config["diarization_device"] == "intelligent-auto-v5" and speaker_count not in (None, 1):
        return jsonify({"ok": False, "error": "Intelligent Auto v5 requires Auto speaker count; use Pyannote for manual counts."}), 400
    if config["model"] not in ALL_MODEL_IDS:
        return jsonify({"ok": False, "error": "Unsupported model selection."}), 400
    if config["model"] in MLX_REPOSITORIES:
        if not apple_silicon():
            return jsonify({"ok": False, "error": "MLX requires an Apple silicon Mac."}), 400
    if config["compute_type"] not in {"float32", "int8"}:
        return jsonify({"ok": False, "error": "Unsupported compute type."}), 400

    try:
        plan = prepare_prompt(config["model"], initial_prompt, hotwords)
        if plan["available"]:
            require_prompt(plan)
        # An uncached CPU tokenizer is checked again after normal model preparation.
    except PromptError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    with state_lock:
        if STATE.get("running"):
            return jsonify({"ok": False, "error": "A transcription is already running."}), 409
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
            recognition=None,
            diarization_actual=None,
            error=None,
            detected_speakers=[],
            settings={k: v for k, v in config.items() if k not in ("initial_prompt", "hotwords")},
        )
        try:
            threading.Thread(target=_run_job, args=(config,), daemon=True).start()
        except RuntimeError as exc:
            _update(running=False, cancel_requested=False, status="Error", stage="Error",
                    detail="Could not start transcription.", error=str(exc))
            return jsonify({"ok": False, "error": "Could not start transcription."}), 500
    return jsonify({"ok": True})


@app.post("/api/quit")
def quit_app():
    """Request native app shutdown without interrupting running jobs."""
    with state_lock:
        if STATE.get("running"):
            return jsonify({
                "ok": False,
                "error": "Finish or cancel the transcription before quitting.",
            }), 409
        callback = NATIVE_QUIT
    if callback is None:
        return jsonify({
            "ok": False,
            "error": "Native Quit is not available in browser/test mode.",
        }), 409
    callback()
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
        issues.append("No Hugging Face token is configured; multi-speaker diarization will not work.")
    return jsonify({"ok": not issues, "issues": issues, "whisperx_version": wx})


def _backend_self_test() -> None:
    """Verify frozen ML imports and the WhisperX API contract used by NixScribe."""
    import inspect

    import torch
    import whisperx
    from whisperx.asr import FasterWhisperPipeline
    from faster_whisper.utils import _MODELS, available_models

    missing_models = MODEL_IDS - set(available_models())
    if missing_models:
        raise RuntimeError(f"Unsupported Faster-Whisper models: {sorted(missing_models)}")
    for model_id, repository in MODEL_REPOSITORIES.items():
        if _MODELS.get(model_id) != repository:
            raise RuntimeError(f"Model repository changed for {model_id}: expected {repository}")
    from whisperx.alignment import align as alignment_impl
    from whisperx.diarize import DiarizationPipeline

    required = {
        "FasterWhisperPipeline.transcribe": (
            FasterWhisperPipeline.transcribe,
            {"audio", "batch_size", "chunk_size", "verbose", "progress_callback"},
        ),
        "whisperx.align": (
            alignment_impl,
            {
                "transcript",
                "model",
                "align_model_metadata",
                "audio",
                "device",
                "interpolate_method",
                "return_char_alignments",
                "progress_callback",
            },
        ),
        "DiarizationPipeline.__init__": (
            DiarizationPipeline.__init__,
            {"model_name", "token", "device"},
        ),
        "DiarizationPipeline.__call__": (
            DiarizationPipeline.__call__,
            {"audio", "num_speakers", "progress_callback"},
        ),
    }

    for name, (callable_obj, expected) in required.items():
        actual = set(inspect.signature(callable_obj).parameters)
        missing = expected - actual
        if missing:
            raise RuntimeError(
                f"Unsupported WhisperX API for {name}; missing parameters: "
                f"{sorted(missing)}"
            )

    print(
        f"Backend self-test OK: torch {torch.__version__}, "
        f"WhisperX {whisperx_version() or 'unknown'}, API contract verified"
    )


def _open_browser() -> None:
    try:
        webbrowser.open(f"http://{APP_HOST}:{APP_PORT}")
    except Exception:
        pass


if __name__ == "__main__":
    if sys.argv[1:2] == ["--clustering-probe"]:
        from clustering_probe import main as probe_main
        probe_main(sys.argv[2:])
        raise SystemExit(0)
    if len(sys.argv) == 4 and sys.argv[1] == "--mps-diarization-worker":
        from diarization_backend import worker_main as diarization_worker
        diarization_worker(Path(sys.argv[2]), Path(sys.argv[3]), _get_hf_token())
        raise SystemExit(0)
    if "--self-test-mps" in sys.argv:
        from diarization_backend import require_mps
        require_mps()
        print("PyTorch Apple GPU arithmetic passed; diarization accuracy is not yet verified")
        raise SystemExit(0)
    if len(sys.argv) == 4 and sys.argv[1] == "--model-download-worker":
        from model_cache import download_worker
        download_worker(sys.argv[2], Path(sys.argv[3]))
        raise SystemExit(0)
    if len(sys.argv) == 4 and sys.argv[1] == "--mlx-worker":
        from mlx_backend import worker_main
        worker_main(Path(sys.argv[2]), Path(sys.argv[3]))
        raise SystemExit(0)
    if "--self-test-mlx-package" in sys.argv:
        from mlx_backend import check_package
        check_package(RESOURCE_ROOT)
        raise SystemExit(0)
    if "--self-test-mlx" in sys.argv:
        from mlx_backend import require_gpu
        require_gpu()
        import mlx_whisper
        assert callable(mlx_whisper.transcribe)
        print("MLX Whisper imports and Apple GPU arithmetic passed")
        raise SystemExit(0)
    if "--self-test-backend" in sys.argv:
        _backend_self_test()
        raise SystemExit(0)
    if "--self-test-native" in sys.argv:
        import webview
        import objc
        import Cocoa
        import WebKit
        from native_app import run_native
        assert callable(run_native) and hasattr(webview, "create_window")
        print("Native macOS WebKit backend imports OK")
        raise SystemExit(0)

    print(f"{APP_NAME}: http://{APP_HOST}:{APP_PORT}")
    print(f"Work directory: {DEFAULT_WORKDIR}")

    if os.environ.get("WHISPERX_UI_NO_BROWSER") == "1" or "--browser" in sys.argv or sys.platform != "darwin":
        if os.environ.get("WHISPERX_UI_NO_BROWSER") != "1":
            threading.Timer(0.8, _open_browser).start()
        app.run(host=APP_HOST, port=APP_PORT, debug=False, threaded=True, use_reloader=False)
    else:
        from native_app import run_native

        def is_running() -> bool:
            with state_lock:
                return bool(STATE.get("running"))

        def register_quit(callback: Callable[[], None] | None) -> None:
            global NATIVE_QUIT
            with state_lock:
                NATIVE_QUIT = callback

        run_native(app, APP_HOST, APP_PORT, is_running, register_quit)
