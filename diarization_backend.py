"""Experimental Pyannote MPS inference in an isolated subprocess.

Only local audio paths and speaker-count settings are exchanged. The worker
uses the existing Hugging Face token/cache and returns diarization intervals,
not embeddings or audio. Native MPS failures cannot abort the main app.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable

REPOSITORY = "pyannote/speaker-diarization-community-1"
DEVICES = frozenset(("cpu", "mps", "coreml", "combined"))


def default_device() -> str:
    from app_config import apple_silicon
    return "mps" if apple_silicon() else "cpu"


def fallback_devices(requested: str) -> tuple[str, ...]:
    """Keep explicit CPU selections on CPU; progressively retry accelerated jobs."""
    return {
        "coreml": ("coreml", "mps", "cpu"),
        "combined": ("coreml", "mps", "cpu"),
        "mps": ("mps", "cpu"),
        "cpu": ("cpu",),
    }[requested]


def validate_intervals(rows: Any) -> list[dict]:
    if not isinstance(rows, list):
        raise ValueError("Diarization did not return an interval list")
    intervals = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid diarization interval")
        start, end = float(row["start"]), float(row["end"])
        speaker = row["speaker"]
        if (not math.isfinite(start) or not math.isfinite(end) or
                start < 0 or end <= start or not isinstance(speaker, str) or
                not speaker.strip()):
            raise ValueError("Invalid diarization timestamp or speaker")
        intervals.append({"start": start, "end": end, "speaker": speaker})
    return sorted(intervals, key=lambda row: row["start"])


def require_mps():
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable on this Mac")
    x = torch.tensor([1., 2., 3.], device="mps")
    value = x.sum()
    torch.mps.synchronize()
    if float(value.cpu().item()) != 6.0:
        raise RuntimeError("MPS arithmetic self-test failed")
    return torch


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    os.replace(temporary, path)


def worker_main(request_path: Path, result_path: Path, token: str | None) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    torch = require_mps()
    from whisperx.diarize import DiarizationPipeline

    progress = result_path.with_name("progress.json")
    _write_json(progress, {"phase": "loading", "percent": 0.0})
    pipeline = DiarizationPipeline(model_name=REPOSITORY, token=token, device="mps")
    _write_json(progress, {"phase": "inference", "percent": 0.0})

    def report(percent: float) -> None:
        _write_json(progress, {"phase": "inference", "percent": float(percent)})

    kwargs = {"progress_callback": report}
    if request["speaker_count"] is not None:
        kwargs["num_speakers"] = int(request["speaker_count"])
    df = pipeline(request["audio"], **kwargs)
    torch.mps.synchronize()
    rows = df[["start", "end", "speaker"]].to_dict(orient="records")
    _write_json(result_path, {"intervals": validate_intervals(rows)})


def run_mps(audio: Path, speaker_count: int | None,
            on_progress: Callable[[str, float], None],
            check_cancel: Callable[[], None]) -> list[dict]:
    check_cancel()
    with tempfile.TemporaryDirectory(prefix="diarization-mps-", dir=audio.parent) as tmp:
        tmp = Path(tmp)
        request = tmp / "request.json"
        result = tmp / "result.json"
        progress = tmp / "progress.json"
        request.write_text(json.dumps({"audio": str(audio), "speaker_count": speaker_count}),
                           encoding="utf-8")
        command = [sys.executable]
        if not getattr(sys, "frozen", False):
            command.append(str(Path(__file__).with_name("app.py")))
        command.extend(["--mps-diarization-worker", str(request), str(result)])
        # Do not silently run unsupported MPS operators on CPU. If the worker
        # fails, the parent explicitly retries the entire diarization on CPU.
        env = dict(os.environ)
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        with tempfile.TemporaryFile() as log:
            proc = subprocess.Popen(command, stdout=log, stderr=log, env=env)
            try:
                last = None
                while True:
                    check_cancel()
                    if progress.is_file():
                        value = json.loads(progress.read_text(encoding="utf-8"))
                        phase = value["phase"]
                        percent = max(0.0, min(100.0, float(value["percent"])))
                        current = (phase, percent)
                        if current != last:
                            on_progress(phase, percent)
                            last = current
                    try:
                        rc = proc.wait(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                check_cancel()
                if rc != 0 or not result.is_file():
                    log.seek(0, 2)
                    log.seek(max(0, log.tell() - 2000))
                    detail = log.read().decode("utf-8", errors="replace").strip()
                    # Avoid copying any authentication tokens from library errors.
                    import re
                    detail = re.sub(r"hf_[A-Za-z0-9]+", "[redacted]", detail)
                    raise RuntimeError(f"MPS diarization worker exited ({rc})"
                                       + (f": {detail}" if detail else ""))
                rows = validate_intervals(json.loads(result.read_text(encoding="utf-8"))["intervals"])
                on_progress("inference", 100.0)
                return rows
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()


def run_coreml(helper: Path, audio: Path, speaker_count: int | None,
               on_progress: Callable[[str, float], None],
               check_cancel: Callable[[], None]) -> list[dict]:
    """Run the bundled native helper; no recordings are sent to the network."""
    check_cancel()
    if not helper.is_file() or not os.access(helper, os.X_OK):
        raise RuntimeError("Bundled FluidAudio helper is unavailable")
    with tempfile.TemporaryDirectory(prefix="diarization-coreml-", dir=audio.parent) as tmp:
        result = Path(tmp) / "result.json"
        progress = Path(tmp) / "progress.json"
        command = [str(helper), str(audio), str(result),
                   str(speaker_count) if speaker_count is not None else "auto"]
        with tempfile.TemporaryFile() as log:
            proc = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                last = None
                while True:
                    check_cancel()
                    if progress.is_file():
                        value = json.loads(progress.read_text(encoding="utf-8"))
                        current = (str(value["phase"]),
                                   max(0.0, min(100.0, float(value["percent"]))))
                        if current != last:
                            on_progress(*current)
                            last = current
                    try:
                        rc = proc.wait(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                check_cancel()
                if rc != 0 or not result.is_file():
                    log.seek(0, 2)
                    log.seek(max(0, log.tell() - 2000))
                    import re
                    detail = re.sub(r"hf_[A-Za-z0-9]+", "[redacted]",
                                    log.read().decode("utf-8", errors="replace")).strip()
                    raise RuntimeError(f"FluidAudio helper exited ({rc})"
                                       + (f": {detail}" if detail else ""))
                rows = validate_intervals(json.loads(result.read_text(encoding="utf-8"))["intervals"])
                on_progress("inference", 100.0)
                return rows
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
