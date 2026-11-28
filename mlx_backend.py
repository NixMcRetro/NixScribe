"""Isolated experimental MLX Whisper speech recognition.

The subprocess owns the GPU model; it exits before the shared CPU alignment and
diarization stages. No model weights or transcripts are sent to a service.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable

from app_config import MLX_REPOSITORIES


def normalize_segments(raw: dict[str, Any], duration: float) -> dict[str, Any]:
    """Adapt MLX's segment timestamps to WhisperX's alignment input contract."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid audio duration")
    if not isinstance(raw, dict) or not isinstance(raw.get("segments"), list):
        raise ValueError("MLX did not return a segment list")
    segments = []
    for item in raw["segments"]:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("Invalid MLX segment")
        text = item["text"].strip()
        if not text:
            continue
        start, end = float(item["start"]), float(item["end"])
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Non-finite MLX timestamp")
        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end <= start:
            # Do not silently discard recognized speech with invalid timing.
            raise ValueError("MLX returned nonempty speech with invalid timing")
        segments.append({"start": start, "end": end, "text": text})
    segments.sort(key=lambda seg: seg["start"])
    return {"language": "en", "segments": segments}


def require_gpu():
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Apple GPU unavailable. Choose a CPU Whisper model.")
    mx.set_default_device(mx.gpu)
    # Execute an actual GPU operation, not just an import.
    with mx.stream(mx.gpu):
        value = mx.sum(mx.array([1, 2, 3]))
        mx.eval(value)
        if int(value.item()) != 6:
            raise RuntimeError("MLX GPU arithmetic check failed")
    return mx


def worker_main(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["model"] not in MLX_REPOSITORIES:
        raise ValueError("Unsupported MLX model")
    snapshot = Path(request["snapshot"])
    from model_cache import snapshot_ready
    if not snapshot_ready(snapshot, request["model"]):
        raise RuntimeError("Local MLX model is incomplete")
    mx = require_gpu()
    import mlx_whisper

    with mx.stream(mx.gpu):
        result = mlx_whisper.transcribe(
            request["audio"],
            path_or_hf_repo=str(snapshot),
            language="en",
            task="transcribe",
            fp16=True,
            verbose=None,
            initial_prompt=request.get("initial_prompt") or None,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
    # Serialize only the fields the existing aligner needs.
    result = normalize_segments(result, float(request["duration"]))
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def transcribe_mlx(audio_path: Path, model: str, snapshot: Path, duration: float,
                   initial_prompt: str, check_cancel: Callable[[], None]) -> dict:
    if model not in MLX_REPOSITORIES:
        raise ValueError("Unsupported MLX model")
    check_cancel()
    with tempfile.TemporaryDirectory(prefix="mlx-", dir=audio_path.parent) as work:
        work = Path(work)
        request_path = work / "request.json"
        result_path = work / "result.json"
        request_path.write_text(json.dumps({
            "audio": str(audio_path), "model": model, "duration": duration,
            "snapshot": str(snapshot),
            "initial_prompt": initial_prompt,
        }), encoding="utf-8")
        # In a frozen app the executable is NixScribe; in source mode it is Python.
        command = [sys.executable]
        if not getattr(sys, "frozen", False):
            command.append(str(Path(__file__).with_name("app.py")))
        command.extend(["--mlx-worker", str(request_path), str(result_path)])
        with tempfile.TemporaryFile() as output:
            proc = subprocess.Popen(command, stdout=output, stderr=output)
            try:
                while True:
                    check_cancel()
                    try:
                        rc = proc.wait(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                check_cancel()
                if rc != 0 or not result_path.is_file():
                    # Worker stdout/stderr may contain private prompt or speech.
                    raise RuntimeError(
                        f"MLX Whisper worker failed (exit {rc}). You can retry using a CPU model."
                    )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                return normalize_segments(result, duration)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()


def check_package(resource_root: Path) -> None:
    """Check frozen assets without requiring a physical GPU on hosted CI."""
    for name, version in (("mlx", "0.29.3"), ("mlx-metal", "0.29.3"),
                          ("mlx-whisper", "0.4.3")):
        actual = importlib.metadata.version(name)
        if actual != version:
            raise RuntimeError(f"{name}: expected {version}, found {actual}")
    if getattr(sys, "frozen", False):
        mlx_dir = resource_root / "mlx"
        if not list(mlx_dir.rglob("*.metallib")):
            raise RuntimeError("MLX Metal shader library is missing from the bundle")
        if not list(mlx_dir.rglob("libmlx.dylib")):
            raise RuntimeError("MLX native library is missing from the bundle")
    print("MLX package and shader check passed; physical GPU test is separate.")
