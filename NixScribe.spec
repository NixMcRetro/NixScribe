# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the self-contained Apple-silicon macOS app."""

import os
import runpy
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata, collect_dynamic_libs, collect_data_files

project = Path(os.getcwd()).resolve()
version = runpy.run_path(str(project / "app_config.py"))["VERSION"]
ffmpeg = os.environ.get("WHISPERX_BUNDLED_FFMPEG")
ffprobe = os.environ.get("WHISPERX_BUNDLED_FFPROBE")
if not ffmpeg:
    raise SystemExit("WHISPERX_BUNDLED_FFMPEG is not set")

binaries = [(ffmpeg, "bin")]
if ffprobe and Path(ffprobe).is_file():
    binaries.append((ffprobe, "bin"))

datas = [(str(project / "templates"), "templates")]
hiddenimports = []

# WhisperX and the diarization stack use plugin registries and dynamic imports.
# Collecting the dynamic portions is deliberately conservative: the result is
# larger, but much more reliable as a portable offline transcription app.
collect_packages = [
    "whisperx",
    "faster_whisper",
    "ctranslate2",
    "pyannote.audio",
    "pyannote.core",
    "pyannote.database",
    "pyannote.metrics",
    "webview",
    "mlx",
    "mlx_whisper",
    "tiktoken",
    "tiktoken_ext",
]

for package in collect_packages:
    try:
        d, b, h = collect_all(package)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        try:
            hiddenimports += collect_submodules(package)
        except Exception:
            pass

# The mlx-metal wheel places its native libraries and shaders under mlx/lib.
# Collect these explicitly; normal Python module analysis is not sufficient.
binaries += collect_dynamic_libs("mlx")
datas += collect_data_files("mlx", includes=["**/*.metallib"])
hiddenimports += ["mlx.core", "tiktoken_ext.openai_public"]

# Preserve distribution metadata needed by importlib.metadata and several ML
# packages. Missing packages are harmless because dependency versions differ.
metadata_packages = [
    "whisperx",
    "faster-whisper",
    "ctranslate2",
    "pyannote.audio",
    "pyannote.core",
    "pyannote.database",
    "pyannote.metrics",
    "torch",
    "torchaudio",
    "transformers",
    "huggingface-hub",
    "mlx",
    "mlx-metal",
    "mlx-whisper",
]
for package in metadata_packages:
    try:
        datas += copy_metadata(package, recursive=True)
    except Exception:
        pass

# A few backends are discovered dynamically and are inexpensive to name here.
for package in [
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "soundfile",
    "av",
    "librosa",
]:
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass

# De-duplicate while keeping deterministic ordering.
def unique(items):
    seen = set()
    result = []
    for item in items:
        key = repr(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result

binaries = unique(binaries)
datas = unique(datas)
hiddenimports += [
    "webview.platforms.cocoa",
    "webview.menu",
    "objc",
    "AppKit",
    "Foundation",
    "Cocoa",
    "WebKit",
    "Quartz",
    "Security",
]
hiddenimports = sorted(set(hiddenimports))

a = Analysis(
    [str(project / "app.py")],
    pathex=[str(project)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "matplotlib.tests",
        "numpy.tests",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NixScribe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="NixScribe",
)

app = BUNDLE(
    coll,
    name="NixScribe.app",
    icon=str(project / "assets" / "NixScribe.icns"),
    bundle_identifier="com.nixmcretro.nixscribe",
    info_plist={
        "CFBundleDisplayName": "NixScribe",
        "CFBundleName": "NixScribe",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.utilities",
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
    },
)
