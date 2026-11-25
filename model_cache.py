"""Resolve the public ASR model into the existing Hugging Face cache.

A local-only snapshot lookup can return an incomplete snapshot, so check its
required files before declaring it ready. No new cache or model copy is made.
Downloads run in an isolated, cancellable worker. No audio or token is passed.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable

from app_config import MODEL_REPOSITORIES, MLX_REPOSITORIES

CPU_PATTERNS = ("config.json", "preprocessor_config.json", "model.bin",
                "tokenizer.json", "vocabulary.*")
MLX_PATTERNS = ("config.json", "model.safetensors", "weights.safetensors",
                "weights.npz")
PREPROCESSOR_REQUIRED = frozenset(("large-v3", "large-v3-turbo",
                                   "distil-large-v3", "distil-large-v3.5"))


def repository_for(model: str) -> str:
    if model in MLX_REPOSITORIES:
        return MLX_REPOSITORIES[model]
    return MODEL_REPOSITORIES[model]


def patterns_for(model: str) -> tuple[str, ...]:
    if model in MLX_REPOSITORIES:
        return MLX_PATTERNS
    if model in MODEL_REPOSITORIES:
        return CPU_PATTERNS
    raise ValueError("Unsupported model")


def nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def snapshot_ready(path: Path, model: str) -> bool:
    if not nonempty(path / "config.json"):
        return False
    if model in MLX_REPOSITORIES:
        return any(nonempty(path / name) for name in MLX_PATTERNS[1:])
    if model not in MODEL_REPOSITORIES:
        return False
    return (
        all(nonempty(path / name) for name in ("model.bin", "tokenizer.json"))
        and any(nonempty(p) for p in path.glob("vocabulary.*"))
        and (model not in PREPROCESSOR_REQUIRED
             or nonempty(path / "preprocessor_config.json"))
    )


def cached_snapshot(model: str) -> Path | None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(snapshot_download(repository_for(model), local_files_only=True,
                                      allow_patterns=list(patterns_for(model))))
    except LocalEntryNotFoundError:
        return None
    return path if snapshot_ready(path, model) else None


def download_worker(model: str, output: Path) -> None:
    from huggingface_hub import snapshot_download

    # Use the exact same HF cache resolution and file filters as the backends.
    path = Path(snapshot_download(repository_for(model),
                                  allow_patterns=list(patterns_for(model))))
    if not snapshot_ready(path, model):
        raise RuntimeError("Model snapshot is incomplete after download")
    output.write_text(json.dumps({"snapshot": str(path)}), encoding="utf-8")


def prepare_model(model: str, work: Path, on_stage: Callable[[str], None],
                  check_cancel: Callable[[], None]) -> Path:
    on_stage("checking")
    check_cancel()
    path = cached_snapshot(model)
    if path is not None:
        check_cancel()
        on_stage("cached")
        return path

    on_stage("downloading")
    check_cancel()
    with tempfile.TemporaryDirectory(prefix="model-download-", dir=work) as tmp:
        result = Path(tmp) / "snapshot.json"
        command = [sys.executable]
        if not getattr(sys, "frozen", False):
            command.append(str(Path(__file__).with_name("app.py")))
        command.extend(["--model-download-worker", model, str(result)])
        with tempfile.TemporaryFile() as log:
            proc = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                while True:
                    check_cancel()
                    try:
                        rc = proc.wait(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                check_cancel()
                if rc != 0 or not result.is_file():
                    log.seek(0, 2)
                    log.seek(max(0, log.tell() - 3000))
                    detail = log.read().decode("utf-8", errors="replace").strip()
                    raise RuntimeError("Could not download the model. Check your internet connection "
                                       "and available disk space." + ("\n" + detail if detail else ""))
                path = Path(json.loads(result.read_text(encoding="utf-8"))["snapshot"])
                if not snapshot_ready(path, model):
                    raise RuntimeError("Downloaded model is incomplete")
                on_stage("downloaded")
                return path
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
